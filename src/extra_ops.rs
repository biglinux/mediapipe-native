//! Operations that only format-3 plans contain: pooling, resizing and the
//! other layers of the hand, pose and holistic graphs. `Plan::check` refuses
//! them in format 1, so the face plans never reach this module.
use crate::{ops, plan::Op};

pub(crate) fn average_pool(op: &Op, x: &[f32], y: &mut [f32]) {
    // Validation restricts this operation to full valid windows, so include-pad
    // and exclude-pad are identical. No hidden ceil mode or truncated divisor.
    let divisor = (op.kh * op.kw) as f32;
    for oy in 0..op.oh {
        for ox in 0..op.ow {
            let start = (oy * op.ow + ox) * op.oc;
            let out = &mut y[start..start + op.oc];
            out.fill(0.0);
            for ky in 0..op.kh {
                for kx in 0..op.kw {
                    let src = ((oy * op.sh + ky) * op.iw + ox * op.sw + kx) * op.ic;
                    for (dst, &v) in out.iter_mut().zip(&x[src..src + op.ic]) {
                        *dst += v;
                    }
                }
            }
            for dst in out {
                *dst /= divisor;
            }
        }
    }
}

/// Bilinear resize with half-pixel centres, following TFLite's reference
/// `ResizeBilinear`: source coordinate `(o + 0.5) * in / out - 0.5`, bounds
/// clamped to the image, weights measured from the clamped lower bound.
pub(crate) fn resize_bilinear(op: &Op, x: &[f32], y: &mut [f32]) {
    // Column coordinates are computed once per block of output columns and
    // reused for every row; the arithmetic per value is unchanged.
    const BLOCK: usize = 64;
    let c = op.oc;
    let line = op.iw * c;
    let axis = |out: usize, input: usize, output: usize| {
        let at = (out as f32 + 0.5) * (input as f32 / output as f32) - 0.5;
        let low = (at.floor() as isize).max(0) as usize;
        let high = (at.ceil() as isize).min(input as isize - 1) as usize;
        (low, high, at - low as f32)
    };
    for start in (0..op.ow).step_by(BLOCK) {
        let n = (op.ow - start).min(BLOCK);
        let mut cols = [(0, 0, 0.0); BLOCK];
        for (k, col) in cols[..n].iter_mut().enumerate() {
            *col = axis(start + k, op.iw, op.ow);
        }
        for oy in 0..op.oh {
            let (y0, y1, dy) = axis(oy, op.ih, op.oh);
            let (top, bottom) = (&x[y0 * line..][..line], &x[y1 * line..][..line]);
            let out = &mut y[(oy * op.ow + start) * c..][..n * c];
            if ops::bilinear_row(out, top, bottom, &cols[..n], dy, c) {
                continue;
            }
            for (&(x0, x1, dx), o) in cols[..n].iter().zip(out.chunks_exact_mut(c)) {
                let (wa, wb) = ((1.0 - dy) * (1.0 - dx), dy * (1.0 - dx));
                let (wd, we) = ((1.0 - dy) * dx, dy * dx);
                let (a, b) = (&top[x0 * c..][..c], &bottom[x0 * c..][..c]);
                let (d, e) = (&top[x1 * c..][..c], &bottom[x1 * c..][..c]);
                for k in 0..c {
                    o[k] = a[k] * wa + b[k] * wb + d[k] * wd + e[k] * we;
                }
            }
        }
    }
}

/// TFLite `DEPTH_TO_SPACE`: channel group `i * b + j` of input pixel (y, x)
/// becomes output pixel (y * b + i, x * b + j).
pub(crate) fn depth_to_space(op: &Op, x: &[f32], y: &mut [f32]) {
    let (b, c) = (op.oh / op.ih, op.oc);
    for iy in 0..op.ih {
        for ix in 0..op.iw {
            for i in 0..b {
                for j in 0..b {
                    let src = (iy * op.iw + ix) * op.ic + (i * b + j) * c;
                    let dst = ((iy * b + i) * op.ow + ix * b + j) * c;
                    y[dst..dst + c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
    }
}

/// Channels `begin..begin + oc` of every pixel.
pub(crate) fn channel_slice(op: &Op, x: &[f32], y: &mut [f32]) {
    for p in 0..op.oh * op.ow {
        let src = p * op.ic + op.begin;
        y[p * op.oc..(p + 1) * op.oc].copy_from_slice(&x[src..src + op.oc]);
    }
}

/// `y[c][w] = x[w][c]` for a one-row tensor.
pub(crate) fn transpose(op: &Op, x: &[f32], y: &mut [f32]) {
    let (width, c) = (op.iw, op.ic);
    for i in 0..width {
        for j in 0..c {
            y[j * width + i] = x[i * c + j];
        }
    }
}

/// Layer normalization over each pixel's channels in the order of TFLite's
/// expanded graph: mean, mean of squared deviations, reciprocal square root of
/// that plus epsilon, times gamma, then `x * s + (-mean) * s`.
pub(crate) fn layer_norm(op: &Op, x: &[f32], y: &mut [f32], w: &[f32]) {
    let c = op.oc;
    let (gamma, epsilon) = (&w[..c], w[c]);
    for p in 0..op.oh * op.ow {
        let v = &x[p * c..][..c];
        let mean = v.iter().sum::<f32>() / c as f32;
        let variance = v.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / c as f32;
        let r = 1.0 / (variance + epsilon).sqrt();
        for ((out, &a), &g) in y[p * c..][..c].iter_mut().zip(v).zip(gamma) {
            let s = r * g;
            *out = a * s + (-mean) * s;
        }
    }
}

/// Subtract each channel's mean over the row's points, then divide by the
/// points' mean distance from that centre. The centred values are staged in
/// `y`, so nothing is allocated.
pub(crate) fn center_scale(op: &Op, x: &[f32], y: &mut [f32]) {
    let (n, c) = (op.iw, op.ic);
    for ch in 0..c {
        let mean = (0..n).map(|i| x[i * c + ch]).sum::<f32>() / n as f32;
        for i in 0..n {
            y[i * c + ch] = x[i * c + ch] - mean;
        }
    }
    let spread = (0..n)
        .map(|i| y[i * c..][..c].iter().map(|v| v * v).sum::<f32>().sqrt())
        .sum::<f32>()
        / n as f32;
    for v in &mut y[..n * c] {
        *v /= spread;
    }
}
