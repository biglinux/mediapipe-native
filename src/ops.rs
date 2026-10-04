//! Numeric kernels for the MediaPipe vision graphs.
//!
//! Every tensor is NHWC — height, then width, then channel, with channel the
//! fastest-moving axis. That one choice decides the shape of everything here:
//!
//! * a 1x1 convolution becomes a matrix multiply over pixels, with the channel
//!   vector contiguous, so no gather and no transpose is ever needed;
//! * a depthwise tap becomes an elementwise multiply-add over a contiguous run
//!   of channels, because the kernel weight is constant per channel;
//! * an activation reads the same bytes the convolution just wrote.
//!
//! Four instruction-set tiers are dispatched at runtime, decided once per
//! process: AVX2 with fused multiply-add, AVX without it, SSE4.1, and a plain
//! scalar fallback. Results are bit-identical across runs and machines within
//! one tier, and the reference outputs pin them; they are not identical across
//! tiers. The tiled pointwise kernels add bias before the dot product while the
//! irregular-width kernels add it afterwards, and tier 3 rounds once per fused
//! multiply-add, so every kernel keeps the summation order of the tier it
//! belongs to.
//!
//! The kernel family and the tier scheme come from the BigLinux
//! `dpdfnet-native` and `deepfilternet-quantized-ladspa` audio work — see
//! `NOTICE`. What is new here is the two-dimensional shape handling and the
//! pointwise micro-kernel, which are tuned for image-sized tensors rather than
//! single spectral frames.

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
static TIER: AtomicU8 = AtomicU8::new(u8::MAX);

/// Widest usable instruction set: 3 = AVX2+FMA, 2 = AVX, 1 = SSE4.1, 0 = scalar.
///
/// Set `MEDIAPIPE_NATIVE_TIER` to cap it, which is how a machine with AVX2 can
/// be made to run an older machine's code path for comparison. Values above
/// what the CPU actually has are ignored.
#[cfg(target_arch = "x86_64")]
pub fn simd_tier() -> u8 {
    let cached = TIER.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached;
    }
    let detected = if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    {
        3
    } else if std::is_x86_feature_detected!("avx") {
        2
    } else if std::is_x86_feature_detected!("sse4.1") {
        1
    } else {
        0
    };
    let tier = std::env::var("MEDIAPIPE_NATIVE_TIER")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .map_or(detected, |cap| cap.min(detected));
    TIER.store(tier, Ordering::Relaxed);
    tier
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_tier() -> u8 {
    0
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn read_mxcsr() -> u32 {
    let mut csr = 0u32;
    // SAFETY: an aligned writable u32 and a read-only observation of MXCSR.
    unsafe {
        std::arch::asm!("stmxcsr [{p}]",p=in(reg)&mut csr,options(nostack,preserves_flags));
    }
    csr
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn write_mxcsr(csr: u32) {
    // SAFETY: callers supply a previously read MXCSR with only FTZ/DAZ changed.
    // Restoring MXCSR can change floating-point exception flags: do not claim
    // preserves_flags. A memory clobber also fences tensor accesses at scope edges.
    unsafe {
        std::arch::asm!("ldmxcsr [{p}]",p=in(reg)&csr,options(nostack));
    }
}

/// A per-call, non-Send scope: never transfer the thread's FP environment.
#[must_use]
pub(crate) struct DenormalGuard {
    #[cfg(target_arch = "x86_64")]
    saved: u32,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl DenormalGuard {
    pub(crate) fn enter() -> Self {
        #[cfg(target_arch = "x86_64")]
        let saved = read_mxcsr();
        #[cfg(target_arch = "x86_64")]
        // SAFETY: keep all reserved bits and the caller's rounding mode.
        unsafe {
            write_mxcsr(saved | 0x8040);
        }
        Self {
            #[cfg(target_arch = "x86_64")]
            saved,
            _not_send: std::marker::PhantomData,
        }
    }
}
impl Drop for DenormalGuard {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: restores exactly the state captured on this same thread.
        unsafe {
            write_mxcsr(self.saved);
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! fma8 {
    (true, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c)
    };
    (false, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_add_ps(std::arch::x86_64::_mm256_mul_ps($a, $b), $c)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! fma4 {
    ($a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm_add_ps(std::arch::x86_64::_mm_mul_ps($a, $b), $c)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vld {
    (8, $p:expr) => {
        std::arch::x86_64::_mm256_loadu_ps($p)
    };
    (4, $p:expr) => {
        std::arch::x86_64::_mm_loadu_ps($p)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vst {
    (8, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm256_storeu_ps($p, $v)
    };
    (4, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm_storeu_ps($p, $v)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vsplat {
    (8, $s:expr) => {
        std::arch::x86_64::_mm256_set1_ps($s)
    };
    (4, $s:expr) => {
        std::arch::x86_64::_mm_set1_ps($s)
    };
}

/// `min(hi, max(lo, v))`: [`clip_inplace`]'s clamp, bit for bit.
#[cfg(target_arch = "x86_64")]
macro_rules! vclip {
    (8, $v:expr, $lo:expr, $hi:expr) => {
        std::arch::x86_64::_mm256_min_ps($hi, std::arch::x86_64::_mm256_max_ps($lo, $v))
    };
    (4, $v:expr, $lo:expr, $hi:expr) => {
        std::arch::x86_64::_mm_min_ps($hi, std::arch::x86_64::_mm_max_ps($lo, $v))
    };
}

/// [`vclip!`] only when `on`: a loop-invariant test is cheaper than two
/// vector operations per store when there is nothing to clamp.
#[cfg(target_arch = "x86_64")]
macro_rules! vclip_if {
    ($on:expr, $lanes:tt, $v:expr, $lo:expr, $hi:expr) => {
        if $on {
            vclip!($lanes, $v, $lo, $hi)
        } else {
            $v
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vfma {
    (8, $fma:tt, $a:expr, $b:expr, $c:expr) => {
        fma8!($fma, $a, $b, $c)
    };
    (4, false, $a:expr, $b:expr, $c:expr) => {
        fma4!($a, $b, $c)
    };
}

#[inline]
fn span(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("kernel dimensions overflow")
}

// ── pointwise (1x1) convolution ─────────────────────────────────────────────

/// [`pointwise_with`] with row-major weights and no clamp, for the tests.
#[cfg(test)]
fn pointwise(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    ci: usize,
    co: usize,
) {
    pointwise_with(y, w, x, bias, count, ci, co, false, None);
}

/// 1x1 convolution over `count` neighbouring pixels: `y[p] = bias + W^T x[p]`,
/// with `W` laid out `[in_channel][out_channel]`.
///
/// Two thirds of a MediaPipe face graph is this one shape, so it gets a
/// dedicated kernel rather than one matrix-vector call per pixel.
///
/// What limits it is not multiply throughput but the load ports. Every input
/// value has to be splashed across a whole vector register before it can be
/// multiplied, and on x86 that splash is itself a load. Holding two 8-wide
/// output blocks per tile lets two weight-row loads and four splashes feed
/// eight vector multiply-adds — six loads per eight products, where the plain
/// one-pixel-at-a-time form needs five loads per four. On a pre-2013 core
/// without fused multiply-add that ratio is the whole difference between
/// fitting a 30 fps budget and missing it.
///
/// Eight output channels use twelve pixels on AVX2 and eight on AVX.
/// Multiples of sixteen use
/// six pixels on AVX2 and four on AVX, with a four-pixel remainder before the
/// scalar tail. Irregular widths use four pixels with 8/4/scalar output tails
/// and add bias last.
///
/// Weights may be stored as 16-column panels,
/// `[co/16][ci][16]`, by `Model::load`, and optionally clamping each output
/// to `clip` as it is stored. Panels are the same products and sums in the
/// same order; only the weight addresses change, so a panel is contiguous in
/// cache instead of `co` floats apart per input channel (with `co` = 128, 256
/// or 1152 that stride maps every row of a panel to a few L1 sets and evicts
/// it). They need `co % 16 == 0` and an AVX tier: the only kernels that read
/// them. The clamp is [`clip_inplace`] bit for bit, without a second pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pointwise_with(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    clip: Option<(f32, f32)>,
) {
    assert!(y.len() >= span(count, co) && x.len() >= span(count, ci) && w.len() >= span(ci, co));
    assert!(!panels || (co.is_multiple_of(16) && simd_tier() >= 2));
    // MAXPS/MINPS against infinities return every value unchanged, NaN and
    // -0 included, so the vector kernels clamp unconditionally.
    let vector_clip = simd_tier() >= 2;
    let bounds = clip
        .filter(|_| vector_clip)
        .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    if let Some(b) = bias {
        assert!(b.len() >= co);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let tiled = 0;
    #[cfg(target_arch = "x86_64")]
    let tiled = {
        let bias_ptr = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        let tier = simd_tier();
        // SAFETY: the asserts above cover every element the bodies touch, and a
        // null bias pointer is only read when `bias` was `Some`.
        unsafe {
            if co.is_multiple_of(16) {
                // Exactly 16 outputs gets its own copy of each tile: a constant
                // `co` drops the panel loop and the strided store addressing,
                // 16% faster for 8->16 on the i3-2375M and 12% on the i5-13400.
                let exact = co == 16;
                match tier {
                    3 => {
                        // Twelve preserves the four-pixel bias-first boundary.
                        let prefix = count / 12 * 12;
                        let (wide, narrow) = if exact {
                            (pw_p6o16_avx2::<16> as Tile16, pw_p4o16_avx2::<16> as Tile16)
                        } else {
                            (pw_p6o16_avx2::<0> as Tile16, pw_p4o16_avx2::<0> as Tile16)
                        };
                        let done = wide(y, w, x, bias_ptr, prefix, ci, co, panels, bounds);
                        done + narrow(
                            &mut y[done * co..],
                            w,
                            &x[done * ci..],
                            bias_ptr,
                            count - done,
                            ci,
                            co,
                            panels,
                            bounds,
                        )
                    }
                    2 if exact => {
                        pw_p4o16_avx::<16>(y, w, x, bias_ptr, count, ci, co, panels, bounds)
                    }
                    // Weights past the i3-2375M's 256 KB L2 (112->672,
                    // 192->1152, 1152->192) measured 8-11% slower with it.
                    2 if co.is_multiple_of(32) && ci * co <= 65536 => {
                        pw_p2o32_avx(y, w, x, bias_ptr, count, ci, co, panels, bounds)
                    }
                    2 => pw_p4o16_avx::<0>(y, w, x, bias_ptr, count, ci, co, panels, bounds),
                    1 => pw_any_sse(y, w, x, bias_ptr, count, ci, co, bounds),
                    _ => 0,
                }
            } else if co == 8 {
                match tier {
                    3 => {
                        // Tier-3 results depend on where the eight-pixel bias-first tiles end.
                        let prefix = count / 24 * 24;
                        let done = pw_p12o8_avx2(y, w, x, bias_ptr, prefix, ci, bounds);
                        done + pw_p8o8_avx2(
                            &mut y[done * 8..],
                            w,
                            &x[done * ci..],
                            bias_ptr,
                            count - done,
                            ci,
                            bounds,
                        )
                    }
                    2 => pw_p8o8_avx(y, w, x, bias_ptr, count, ci, bounds),
                    1 => pw_any_sse(y, w, x, bias_ptr, count, ci, co, bounds),
                    _ => 0,
                }
            } else {
                match tier {
                    3 => pw_any_avx2(y, w, x, bias_ptr, count, ci, co, bounds),
                    2 => pw_any_avx(y, w, x, bias_ptr, count, ci, co, bounds),
                    1 => pw_any_sse(y, w, x, bias_ptr, count, ci, co, bounds),
                    _ => 0,
                }
            }
        }
    };
    pointwise_rest(y, w, x, bias, tiled, count, ci, co, panels, bounds);
    if let Some((low, high)) = clip.filter(|_| !vector_clip) {
        clip_inplace(&mut y[..count * co], low, high);
    }
}

/// Pixels the tiled kernels did not cover, one matrix-vector call each.
// Shape, weights and bias are eight separate facts about one convolution;
// bundling them into a struct would only move the argument list.
#[allow(clippy::too_many_arguments)]
fn pointwise_rest(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    mut from: usize,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    // Up to four leftover pixels share one pass over the weights instead of
    // one pass each, with the arithmetic of `matvec_t` plus a trailing bias.
    #[cfg(target_arch = "x86_64")]
    if co.is_multiple_of(8) && simd_tier() >= 2 {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        while from < count {
            let pixels = (count - from).min(4);
            // SAFETY: `pointwise` checked every pixel below `count` and the
            // bias length; co is a multiple of eight; AVX was detected.
            unsafe {
                let (yr, xr) = (&mut y[from * co..], &x[from * ci..]);
                if simd_tier() == 3 {
                    pw_rest_avx2(yr, w, xr, b, pixels, ci, co, panels, bounds);
                } else {
                    pw_rest_avx(yr, w, xr, b, pixels, ci, co, panels, bounds);
                }
            }
            from += pixels;
        }
        return;
    }
    assert!(!panels, "panel weights reach only the AVX kernels");
    for p in from..count {
        let out = &mut y[p * co..p * co + co];
        matvec_t(out, w, &x[p * ci..p * ci + ci], ci, co);
        if let Some(b) = bias {
            for (v, &bv) in out.iter_mut().zip(b) {
                *v += bv;
            }
        }
        for v in out {
            *v = v.clamp(bounds.0, bounds.1);
        }
    }
}

/// Leftover pixels (at most four) for `co % 8 == 0`: each output is zero,
/// then one multiply-add per input channel in order, then the bias.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_rest_body {
    ($y:expr, $w:expr, $x:expr, $b:expr, $pixels:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: the caller's contract, checked by `pointwise`.
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (pixels, ci, co, panels) = ($pixels, $ci, $co, $panels);
            let mut j = 0;
            // One or two pixels would leave two or four chains waiting on
            // multiply-add latency: take several 16-column blocks at once.
            macro_rules! blocks {
                ($p:literal, $blocks:literal) => {
                    while j + 16 * $blocks <= co {
                        let mut acc = [[_mm256_setzero_ps(); 2 * $blocks]; $p];
                        for i in 0..ci {
                            for m in 0..$blocks {
                                let jb = j + 16 * m;
                                let row = if panels {
                                    wp.add(jb * ci + i * 16)
                                } else {
                                    wp.add(i * co + jb)
                                };
                                let (r0, r1) = (_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8)));
                                for (k, a) in acc.iter_mut().enumerate() {
                                    let v = _mm256_set1_ps(*xp.add(k * ci + i));
                                    a[2 * m] = fma8!($fma, r0, v, a[2 * m]);
                                    a[2 * m + 1] = fma8!($fma, r1, v, a[2 * m + 1]);
                                }
                            }
                        }
                        for (k, a) in acc.iter().enumerate() {
                            for (m, &v) in a.iter().enumerate() {
                                let v = if b.is_null() {
                                    v
                                } else {
                                    _mm256_add_ps(v, _mm256_loadu_ps(b.add(j + 8 * m)))
                                };
                                _mm256_storeu_ps(
                                    yp.add(k * co + j + 8 * m),
                                    vclip_if!(clamps, 8, v, lo, hi),
                                );
                            }
                        }
                        j += 16 * $blocks;
                    }
                };
            }
            match pixels {
                1 => blocks!(1, 4),
                2 => blocks!(2, 2),
                _ => {}
            }
            while j < co {
                let wide = j + 16 <= co;
                // Row `i` of this 16-column block: in place, or in its panel.
                let (base, stride) = if panels { (j * ci, 16) } else { (j, co) };
                let mut low = [_mm256_setzero_ps(); 4];
                let mut high = [_mm256_setzero_ps(); 4];
                for i in 0..ci {
                    let row = wp.add(base + i * stride);
                    let r0 = _mm256_loadu_ps(row);
                    let r1 = if wide {
                        _mm256_loadu_ps(row.add(8))
                    } else {
                        r0
                    };
                    for k in 0..pixels {
                        let v = _mm256_set1_ps(*xp.add(k * ci + i));
                        low[k] = fma8!($fma, r0, v, low[k]);
                        if wide {
                            high[k] = fma8!($fma, r1, v, high[k]);
                        }
                    }
                }
                for k in 0..pixels {
                    let (mut a, mut c) = (low[k], high[k]);
                    if !b.is_null() {
                        a = _mm256_add_ps(a, _mm256_loadu_ps(b.add(j)));
                        if wide {
                            c = _mm256_add_ps(c, _mm256_loadu_ps(b.add(j + 8)));
                        }
                    }
                    _mm256_storeu_ps(yp.add(k * co + j), vclip_if!(clamps, 8, a, lo, hi));
                    if wide {
                        _mm256_storeu_ps(yp.add(k * co + j + 8), vclip_if!(clamps, 8, c, lo, hi));
                    }
                }
                j += if wide { 16 } else { 8 };
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_rest_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    pixels: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    pw_rest_body!(y, w, x, b, pixels, ci, co, panels, bounds, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_rest_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    pixels: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    pw_rest_body!(y, w, x, b, pixels, ci, co, panels, bounds, false)
}

/// Register tile by sixteen output channels. Returns the pixel count handled.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_o16_body {
    ($y:expr, $w:expr, $x:expr, $b:expr, $count:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $pixels:expr, $fma:tt) => {
        pw_o16_body!($y, $w, $x, $b, $count, $ci, $co, $panels, $bounds, $pixels, $fma, false)
    };
    ($y:expr, $w:expr, $x:expr, $b:expr, $count:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $pixels:expr, $fma:tt, $skew:expr) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        // SAFETY: bounds are the caller's contract, checked by `pointwise`.
        unsafe {
            let (y, w, x, b, count, ci, co, panels) = ($y, $w, $x, $b, $count, $ci, $co, $panels);
            let (floor, ceil) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (yp, wp, xp) = (y.as_mut_ptr(), w.as_ptr(), x.as_ptr());
            // One tile: `$pixels` pixels by the 16 output channels at `jb`;
            // the tile's pixel `k` is read at `src + k * step`.
            let tile = |src: *const f32, step: usize, p: usize, jb: usize| {
                let (lo, hi) = if b.is_null() {
                    (_mm256_setzero_ps(), _mm256_setzero_ps())
                } else {
                    (_mm256_loadu_ps(b.add(jb)), _mm256_loadu_ps(b.add(jb + 8)))
                };
                let mut low = [lo; $pixels];
                let mut high = [hi; $pixels];
                let (base, stride) = if panels { (jb * ci, 16) } else { (jb, co) };
                for i in 0..ci {
                    let row = wp.add(base + i * stride);
                    let r0 = _mm256_loadu_ps(row);
                    let r1 = _mm256_loadu_ps(row.add(8));
                    for k in 0..$pixels {
                        let value = _mm256_set1_ps(*src.add(k * step + i));
                        low[k] = fma8!($fma, r0, value, low[k]);
                        high[k] = fma8!($fma, r1, value, high[k]);
                    }
                }
                for k in 0..$pixels {
                    let (a, c) = (
                        vclip_if!(clamps, 8, low[k], floor, ceil),
                        vclip_if!(clamps, 8, high[k], floor, ceil),
                    );
                    _mm256_storeu_ps(yp.add((p + k) * co + jb), a);
                    _mm256_storeu_ps(yp.add((p + k) * co + jb + 8), c);
                }
            };
            let done = count / $pixels * $pixels;
            // Sandy Bridge cannot serve two L1 reads in one cycle from the
            // same 16-byte bank of different lines (address bits 4-6). A pixel
            // stride of a multiple of 128 bytes puts every broadcast of a tile
            // in one bank; copied at a stride of ci + 4 floats, the tile's
            // pixels fall in different banks. Same values, so same sums. The
            // copy paid for itself from four 16-column panels per pixel tile
            // (two measured slower on the i3-2375M); repeated for every panel,
            // as the order below for co > count would need, it cost more than
            // the conflicts it removes.
            let skew = $skew && ci.is_multiple_of(32) && ci <= SKEW_MAX_CI && co >= 64;
            let stride = ci + 4;
            const COPY: usize = if $skew {
                $pixels * (SKEW_MAX_CI + 4)
            } else {
                0
            };
            let mut copy = [std::mem::MaybeUninit::<f32>::uninit(); COPY];
            let cp = copy.as_mut_ptr().cast::<f32>();
            // SAFETY (with `skew`): pixel p + k < count lies in x, and
            // k * stride + ci <= COPY because ci <= SKEW_MAX_CI. A tile reads
            // only the ci floats copied for each of its pixels.
            let gather = |p: usize| {
                for k in 0..$pixels {
                    std::ptr::copy_nonoverlapping(xp.add((p + k) * ci), cp.add(k * stride), ci);
                }
            };
            // Same tiles either way. When the weights outweigh the pixels,
            // finish each 16-column panel across every pixel while it is in
            // L1, instead of streaming all weights once per pixel tile.
            if co > count {
                for jb in (0..co).step_by(16) {
                    for p in (0..done).step_by($pixels) {
                        tile(xp.add(p * ci), ci, p, jb);
                    }
                }
            } else {
                for p in (0..done).step_by($pixels) {
                    if skew {
                        gather(p);
                    }
                    for jb in (0..co).step_by(16) {
                        if skew {
                            tile(cp.cast_const(), stride, p, jb);
                        } else {
                            tile(xp.add(p * ci), ci, p, jb);
                        }
                    }
                }
            }
            done
        }
    }};
}

/// Register tile by eight output channels, for `co == 8`.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_o8_body {
    ($y:expr, $w:expr, $x:expr, $b:expr, $count:expr, $ci:expr, $bounds:expr, $pixels:expr, $fma:tt) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        // SAFETY: bounds are the caller's contract, checked by `pointwise`.
        unsafe {
            let (y, w, x, b, count, ci) = ($y, $w, $x, $b, $count, $ci);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (yp, wp, xp) = (y.as_mut_ptr(), w.as_ptr(), x.as_ptr());
            let seed = if b.is_null() {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(b)
            };
            let mut p = 0usize;
            while p + $pixels <= count {
                let mut acc = [seed; $pixels];
                for i in 0..ci {
                    let r = _mm256_loadu_ps(wp.add(i * 8));
                    for k in 0..$pixels {
                        acc[k] = fma8!($fma, r, _mm256_set1_ps(*xp.add((p + k) * ci + i)), acc[k]);
                    }
                }
                for k in 0..$pixels {
                    _mm256_storeu_ps(yp.add((p + k) * 8), vclip_if!(clamps, 8, acc[k], lo, hi));
                }
                p += $pixels;
            }
            p
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p6o16_avx2<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 6, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p4o16_avx2<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 4, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p4o16_avx<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 4, false, true)
}

/// Register tile of two pixels by 32 output channels, for `co % 32 == 0` on
/// AVX: each broadcast input feeds four registers instead of two. On the
/// i3-2375M it is 2-14% faster than `pw_p4o16_avx` while the weights fit in
/// L2. Each output keeps the bias-first, input-ordered sum: the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p2o32_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    use std::arch::x86_64::*;
    // SAFETY: bounds are the caller's contract, checked by `pointwise`;
    // co % 32 == 0, so every 32-column tile lies inside a row of y and w.
    unsafe {
        let (floor, ceil) = (_mm256_set1_ps(bounds.0), _mm256_set1_ps(bounds.1));
        let clamps = bounds != (f32::NEG_INFINITY, f32::INFINITY);
        let (yp, wp, xp) = (y.as_mut_ptr(), w.as_ptr(), x.as_ptr());
        let tile = |p: usize, jb: usize| {
            let seed = |m: usize| {
                if b.is_null() {
                    _mm256_setzero_ps()
                } else {
                    _mm256_loadu_ps(b.add(jb + 8 * m))
                }
            };
            let mut acc = [[seed(0), seed(1), seed(2), seed(3)]; 2];
            // Two 16-column panels when the weights were repacked.
            let (left, right, stride) = if panels {
                (jb * ci, (jb + 16) * ci, 16)
            } else {
                (jb, jb + 16, co)
            };
            let (x0, x1) = (xp.add(p * ci), xp.add((p + 1) * ci));
            for i in 0..ci {
                let (l, r) = (wp.add(left + i * stride), wp.add(right + i * stride));
                let rows = [
                    _mm256_loadu_ps(l),
                    _mm256_loadu_ps(l.add(8)),
                    _mm256_loadu_ps(r),
                    _mm256_loadu_ps(r.add(8)),
                ];
                let values = [_mm256_set1_ps(*x0.add(i)), _mm256_set1_ps(*x1.add(i))];
                for (a, &value) in acc.iter_mut().zip(&values) {
                    for (v, &row) in a.iter_mut().zip(&rows) {
                        *v = fma8!(false, row, value, *v);
                    }
                }
            }
            for (k, a) in acc.iter().enumerate() {
                for (m, &v) in a.iter().enumerate() {
                    let v = vclip_if!(clamps, 8, v, floor, ceil);
                    _mm256_storeu_ps(yp.add((p + k) * co + jb + 8 * m), v);
                }
            }
        };
        // The pixels `pw_p4o16_avx` would cover: the rest adds its bias last.
        let done = count / 4 * 4;
        // As in `pw_o16_body`, keep a panel in L1 when weights outweigh pixels.
        if co > count {
            for jb in (0..co).step_by(32) {
                for p in (0..done).step_by(2) {
                    tile(p, jb);
                }
            }
        } else {
            for p in (0..done).step_by(2) {
                for jb in (0..co).step_by(32) {
                    tile(p, jb);
                }
            }
        }
        done
    }
}

#[cfg(target_arch = "x86_64")]
type Tile16 = unsafe fn(
    &mut [f32],
    &[f32],
    &[f32],
    *const f32,
    usize,
    usize,
    usize,
    bool,
    (f32, f32),
) -> usize;

/// Widest input copied into the bank-skewed tile buffer of `pw_o16_body`.
#[cfg(target_arch = "x86_64")]
const SKEW_MAX_CI: usize = 1152;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pw_p12o8_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    bounds: (f32, f32),
) -> usize {
    pw_o8_body!(y, w, x, b, count, ci, bounds, 12, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pw_p8o8_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    bounds: (f32, f32),
) -> usize {
    pw_o8_body!(y, w, x, b, count, ci, bounds, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pw_p8o8_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    bounds: (f32, f32),
) -> usize {
    pw_o8_body!(y, w, x, b, count, ci, bounds, 8, false)
}

// ── transposed matrix-vector, plain and four-pixel ──────────────────────────

/// `y[0..n] = sum_i x[i] * a[i*n ..]`, with `a` laid out `[m][n]`.
///
/// Each output block is held in a register across the whole reduction, so the
/// result is stored once instead of being read back `m` times, and there is no
/// horizontal sum at the end.
pub(crate) fn matvec_t(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m && a.len() >= span(m, n));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the assert above covers every element the bodies touch.
    unsafe {
        match simd_tier() {
            3 => return mvt_avx2(y, a, x, m, n),
            2 => return mvt_avx(y, a, x, m, n),
            1 => return mvt_sse(y, a, x, m, n),
            _ => {}
        }
    }
    for j in 0..n {
        let mut s = 0.0f32;
        for i in 0..m {
            s += x[i] * a[i * n + j];
        }
        y[j] = s;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_body {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $fma:tt) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        // SAFETY: bounds are the caller's contract, checked by `matvec_t`.
        unsafe {
            let (y, a, x, m, n) = ($y, $a, $x, $m, $n);
            let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
            let mut j = 0usize;
            // Eight blocks keep eight independent chains in flight; four left
            // the reduction waiting on multiply-add latency. Each output still
            // sums over `i` in order.
            while j + 64 <= n {
                let mut acc = [_mm256_setzero_ps(); 8];
                for i in 0..m {
                    let s = _mm256_set1_ps(*xp.add(i));
                    let row = ap.add(i * n + j);
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = fma8!($fma, _mm256_loadu_ps(row.add(8 * k)), s, *a);
                    }
                }
                for (k, a) in acc.into_iter().enumerate() {
                    _mm256_storeu_ps(yp.add(j + 8 * k), a);
                }
                j += 64;
            }
            while j + 32 <= n {
                let mut a0 = _mm256_setzero_ps();
                let mut a1 = _mm256_setzero_ps();
                let mut a2 = _mm256_setzero_ps();
                let mut a3 = _mm256_setzero_ps();
                for i in 0..m {
                    let s = _mm256_set1_ps(*xp.add(i));
                    let row = ap.add(i * n + j);
                    a0 = fma8!($fma, _mm256_loadu_ps(row), s, a0);
                    a1 = fma8!($fma, _mm256_loadu_ps(row.add(8)), s, a1);
                    a2 = fma8!($fma, _mm256_loadu_ps(row.add(16)), s, a2);
                    a3 = fma8!($fma, _mm256_loadu_ps(row.add(24)), s, a3);
                }
                _mm256_storeu_ps(yp.add(j), a0);
                _mm256_storeu_ps(yp.add(j + 8), a1);
                _mm256_storeu_ps(yp.add(j + 16), a2);
                _mm256_storeu_ps(yp.add(j + 24), a3);
                j += 32;
            }
            while j + 8 <= n {
                let mut acc = _mm256_setzero_ps();
                for i in 0..m {
                    acc = fma8!(
                        $fma,
                        _mm256_loadu_ps(ap.add(i * n + j)),
                        _mm256_set1_ps(*xp.add(i)),
                        acc
                    );
                }
                _mm256_storeu_ps(yp.add(j), acc);
                j += 8;
            }
            while j < n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += *xp.add(i) * *ap.add(i * n + j);
                }
                *yp.add(j) = s;
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    mvt_body!(y, a, x, m, n, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    mvt_body!(y, a, x, m, n, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_sse(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    // SAFETY: bounds are the caller's contract, checked by `matvec_t`.
    unsafe {
        let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
        let mut j = 0usize;
        while j + 4 <= n {
            let mut acc = _mm_setzero_ps();
            for i in 0..m {
                acc = fma4!(
                    _mm_loadu_ps(ap.add(i * n + j)),
                    _mm_set1_ps(*xp.add(i)),
                    acc
                );
            }
            _mm_storeu_ps(yp.add(j), acc);
            j += 4;
        }
        while j < n {
            let mut s = 0.0f32;
            for i in 0..m {
                s += *xp.add(i) * *ap.add(i * n + j);
            }
            *yp.add(j) = s;
            j += 1;
        }
    }
}

/// `y[0..n] += sum_i x[i] * a[i*n ..]`, one pixel at a time.
///
/// The accumulating twin of [`matvec_t`], for the leftover pixels at the end
/// of a row that the four-at-a-time kernel cannot cover, and for layers whose
/// output is a single pixel.
pub(crate) fn matvec_t_acc(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m && a.len() >= span(m, n));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the assert above covers every element the bodies touch.
    unsafe {
        let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
        match simd_tier() {
            3 => return mvt_acc_avx2(yp, ap, xp, m, n),
            2 => return mvt_acc_avx(yp, ap, xp, m, n),
            1 => return mvt_acc_sse(yp, ap, xp, m, n),
            _ => {}
        }
    }
    for j in 0..n {
        let mut s = y[j];
        for i in 0..m {
            s += x[i] * a[i * n + j];
        }
        y[j] = s;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_acc_body {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds are the caller's contract, checked by `matvec_t_acc`.
        unsafe {
            let (yp, ap, xp) = ($y, $a, $x);
            let (m, n) = ($m, $n);
            let mut j = 0usize;
            // Eight independent chains, as in `mvt_body`; same order per output.
            while j + 8 * $lanes <= n {
                let mut acc = [vld!($lanes, yp.add(j)); 8];
                for (k, a) in acc.iter_mut().enumerate().skip(1) {
                    *a = vld!($lanes, yp.add(j + k * $lanes));
                }
                for i in 0..m {
                    let s = vsplat!($lanes, *xp.add(i));
                    let row = ap.add(i * n + j);
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = vfma!($lanes, $fma, vld!($lanes, row.add(k * $lanes)), s, *a);
                    }
                }
                for (k, a) in acc.into_iter().enumerate() {
                    vst!($lanes, yp.add(j + k * $lanes), a);
                }
                j += 8 * $lanes;
            }
            while j + 4 * $lanes <= n {
                let mut a0 = vld!($lanes, yp.add(j));
                let mut a1 = vld!($lanes, yp.add(j + $lanes));
                let mut a2 = vld!($lanes, yp.add(j + 2 * $lanes));
                let mut a3 = vld!($lanes, yp.add(j + 3 * $lanes));
                for i in 0..m {
                    let s = vsplat!($lanes, *xp.add(i));
                    let row = ap.add(i * n + j);
                    a0 = vfma!($lanes, $fma, vld!($lanes, row), s, a0);
                    a1 = vfma!($lanes, $fma, vld!($lanes, row.add($lanes)), s, a1);
                    a2 = vfma!($lanes, $fma, vld!($lanes, row.add(2 * $lanes)), s, a2);
                    a3 = vfma!($lanes, $fma, vld!($lanes, row.add(3 * $lanes)), s, a3);
                }
                vst!($lanes, yp.add(j), a0);
                vst!($lanes, yp.add(j + $lanes), a1);
                vst!($lanes, yp.add(j + 2 * $lanes), a2);
                vst!($lanes, yp.add(j + 3 * $lanes), a3);
                j += 4 * $lanes;
            }
            while j + $lanes <= n {
                let mut acc = vld!($lanes, yp.add(j));
                for i in 0..m {
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, ap.add(i * n + j)),
                        vsplat!($lanes, *xp.add(i)),
                        acc
                    );
                }
                vst!($lanes, yp.add(j), acc);
                j += $lanes;
            }
            while j < n {
                let mut s = *yp.add(j);
                for i in 0..m {
                    s += *xp.add(i) * *ap.add(i * n + j);
                }
                *yp.add(j) = s;
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_acc_avx2(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_acc_avx(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_acc_sse(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 4, false)
}

/// Four pixels sharing one weight matrix, accumulating into `ys`.
///
/// Used by convolutions with a kernel bigger than one pixel, where each tap is
/// a separate pass and the running sum has to stay in memory between taps.
/// Four independent accumulators keep the multiply pipeline busy even when the
/// output is only 8 channels wide, where a single accumulator would stall
/// waiting on its own previous result.
///
/// # Safety
/// `a` must hold `m*n` floats, each `xs[k]` at least `m`, each `ys[k]` at
/// least `n`, and the four output ranges must not overlap each other or
/// either input/weight range. Address products must fit `usize`.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn matvec_t_acc_x4(
    ys: [*mut f32; 4],
    a: *const f32,
    xs: [*const f32; 4],
    m: usize,
    n: usize,
) {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        match simd_tier() {
            3 => mvt_x4_avx2(ys, a, xs, m, n),
            2 => mvt_x4_avx(ys, a, xs, m, n),
            1 => mvt_x4_sse(ys, a, xs, m, n),
            _ => {
                for p in 0..4 {
                    for j in 0..n {
                        let mut acc = *ys[p].add(j);
                        for i in 0..m {
                            acc += *xs[p].add(i) * *a.add(i * n + j);
                        }
                        *ys[p].add(j) = acc;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_x4_body {
    ($ys:expr, $a:expr, $xs:expr, $m:expr, $n:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds and non-overlap are the caller's contract.
        unsafe {
            let (ys, a, xs, m, n) = ($ys, $a, $xs, $m, $n);
            let mut j = 0usize;
            while j + $lanes <= n {
                let mut a0 = vld!($lanes, ys[0].add(j));
                let mut a1 = vld!($lanes, ys[1].add(j));
                let mut a2 = vld!($lanes, ys[2].add(j));
                let mut a3 = vld!($lanes, ys[3].add(j));
                for i in 0..m {
                    let row = vld!($lanes, a.add(i * n + j));
                    a0 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[0].add(i)), a0);
                    a1 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[1].add(i)), a1);
                    a2 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[2].add(i)), a2);
                    a3 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[3].add(i)), a3);
                }
                vst!($lanes, ys[0].add(j), a0);
                vst!($lanes, ys[1].add(j), a1);
                vst!($lanes, ys[2].add(j), a2);
                vst!($lanes, ys[3].add(j), a3);
                j += $lanes;
            }
            while j < n {
                for k in 0..4 {
                    let mut s = *ys[k].add(j);
                    for i in 0..m {
                        s += *xs[k].add(i) * *a.add(i * n + j);
                    }
                    *ys[k].add(j) = s;
                }
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_x4_avx2(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_x4_avx(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_x4_sse(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 4, false)
}

// ── depthwise ───────────────────────────────────────────────────────────────

/// One depthwise kernel tap: `y[p*c + ch] += x[p*step + ch] * w[ch]`.
///
/// `step` is the distance in floats between neighbouring source pixels, which
/// is `c` for a unit stride and a multiple of it when the convolution skips
/// pixels. The weight is constant per channel, so a whole channel vector is
/// held in a register and reused across every pixel in the run.
pub(crate) fn depthwise_tap(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    count: usize,
    c: usize,
    step: usize,
) {
    assert!(y.len() >= span(count, c) && w.len() >= c);
    assert!(
        count == 0
            || x.len()
                >= span(count - 1, step)
                    .checked_add(c)
                    .expect("input span overflows")
    );
    #[cfg(target_arch = "x86_64")]
    if c.is_multiple_of(8) {
        // SAFETY: the asserts above cover every element the bodies touch.
        unsafe {
            match simd_tier() {
                3 => return dw_avx2(y, x, w, count, c, step),
                2 => return dw_avx(y, x, w, count, c, step),
                1 => return dw_sse(y, x, w, count, c, step),
                _ => {}
            }
        }
    }
    for p in 0..count {
        for ch in 0..c {
            y[p * c + ch] += x[p * step + ch] * w[ch];
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! dw_body {
    ($y:expr, $x:expr, $w:expr, $count:expr, $c:expr, $step:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds are the caller's contract, checked by `depthwise_tap`.
        unsafe {
            let (y, x, w, count, c, step) = ($y, $x, $w, $count, $c, $step);
            let (yp, xp, wp) = (y.as_mut_ptr(), x.as_ptr(), w.as_ptr());
            for p in 0..count {
                let (dst, src) = (p * c, p * step);
                let mut ch = 0usize;
                while ch + $lanes <= c {
                    let acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + ch)),
                        vld!($lanes, wp.add(ch)),
                        vld!($lanes, yp.add(dst + ch))
                    );
                    vst!($lanes, yp.add(dst + ch), acc);
                    ch += $lanes;
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dw_avx2(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dw_avx(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dw_sse(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 4, false)
}

// ── elementwise ─────────────────────────────────────────────────────────────

/// Copy a per-channel bias across `count` neighbouring pixels.
pub(crate) fn fill_bias(y: &mut [f32], bias: Option<&[f32]>, count: usize, c: usize) {
    let n = span(count, c);
    assert!(y.len() >= n);
    if n == 0 {
        return;
    }
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    match bias {
        None => y[..n].fill(0.0),
        Some(b) => {
            for p in 0..count {
                y[p * c..p * c + c].copy_from_slice(&b[..c]);
            }
        }
    }
}

/// One output row of a bilinear resize for `c % 8 == 0` on AVX tiers:
/// `o = a * wa + b * wb + d * wd + e * we` per channel, left to right with
/// separate multiplies and adds, the scalar loop's arithmetic lane for lane.
/// `cols` holds each output column's two source columns and its weight.
/// Returns false when it did nothing.
pub(crate) fn bilinear_row(
    out: &mut [f32],
    top: &[f32],
    bottom: &[f32],
    cols: &[(usize, usize, f32)],
    dy: f32,
    c: usize,
) -> bool {
    if !c.is_multiple_of(8) || simd_tier() < 2 {
        return false;
    }
    assert!(out.len() >= span(cols.len(), c));
    for &(x0, x1, _) in cols {
        let last = x0.max(x1) * c + c;
        assert!(top.len() >= last && bottom.len() >= last);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: AVX detected; every source pixel and output row checked above.
    unsafe {
        bilinear_row_avx(out, top, bottom, cols, dy, c);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn bilinear_row_avx(
    out: &mut [f32],
    top: &[f32],
    bottom: &[f32],
    cols: &[(usize, usize, f32)],
    dy: f32,
    c: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: bilinear_row checked every index below.
    unsafe {
        let (op, tp, bp) = (out.as_mut_ptr(), top.as_ptr(), bottom.as_ptr());
        for (p, &(x0, x1, dx)) in cols.iter().enumerate() {
            let (wa, wb) = ((1.0 - dy) * (1.0 - dx), dy * (1.0 - dx));
            let (wd, we) = ((1.0 - dy) * dx, dy * dx);
            let [wa, wb, wd, we] = [wa, wb, wd, we].map(|v| _mm256_set1_ps(v));
            for k in (0..c).step_by(8) {
                let a = _mm256_mul_ps(_mm256_loadu_ps(tp.add(x0 * c + k)), wa);
                let b = _mm256_mul_ps(_mm256_loadu_ps(bp.add(x0 * c + k)), wb);
                let d = _mm256_mul_ps(_mm256_loadu_ps(tp.add(x1 * c + k)), wd);
                let e = _mm256_mul_ps(_mm256_loadu_ps(bp.add(x1 * c + k)), we);
                let v = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a, b), d), e);
                _mm256_storeu_ps(op.add(p * c + k), v);
            }
        }
    }
}

/// [`add_with`] without a clamp, for the tests.
#[cfg(test)]
fn add(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize) {
    add_with(y, x1, x2, n, None);
}

/// `y = x1 + x2`, then optionally [`clip_inplace`], in one pass: both split vector
/// and scalar elements at the same index, so each element gets the same
/// operations in the same order.
pub(crate) fn add_with(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize, clip: Option<(f32, f32)>) {
    assert!(y.len() >= n && x1.len() >= n && x2.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        // SAFETY: the assert above covers every element read and written.
        unsafe {
            return add_avx(y, x1, x2, n, bounds);
        }
    }
    for i in 0..n {
        y[i] = x1[i] + x2[i];
    }
    if let Some((low, high)) = clip {
        clip_inplace(&mut y[..n], low, high);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_avx(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize, bounds: (f32, f32)) {
    use std::arch::x86_64::*;
    // SAFETY: `add_with` asserted all three slices hold at least `n` floats.
    unsafe {
        let (yp, ap, bp) = (y.as_mut_ptr(), x1.as_ptr(), x2.as_ptr());
        let (lo, hi) = (_mm256_set1_ps(bounds.0), _mm256_set1_ps(bounds.1));
        let clamps = bounds != (f32::NEG_INFINITY, f32::INFINITY);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_add_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)));
            _mm256_storeu_ps(yp.add(i), vclip_if!(clamps, 8, v, lo, hi));
            i += 8;
        }
        while i < n {
            *yp.add(i) = (*ap.add(i) + *bp.add(i)).clamp(bounds.0, bounds.1);
            i += 1;
        }
    }
}

/// Parametric ReLU in place: negative values are scaled by a learned factor
/// that differs per channel. `slope` holds one factor per channel and repeats
/// every `c` elements, which is exactly how NHWC lays the data out.
pub(crate) fn prelu_inplace(y: &mut [f32], slope: &[f32], n: usize, c: usize) {
    assert!(y.len() >= n && slope.len() >= c && (n == 0 || c > 0));
    let src = y.as_ptr();
    // SAFETY: the source is the destination itself, checked above.
    unsafe { prelu_from::<false>(y, src, slope, n, c) }
}

/// `y = x`, then [`prelu_inplace`], in one pass: every element is read once
/// and written once, with the same comparison and product.
pub(crate) fn prelu(y: &mut [f32], x: &[f32], slope: &[f32], n: usize, c: usize) {
    assert!(y.len() >= n && x.len() >= n && slope.len() >= c && (n == 0 || c > 0));
    // SAFETY: both extents checked above.
    unsafe { prelu_from::<true>(y, x.as_ptr(), slope, n, c) }
}

/// `COPY` false means `src` is `y` itself: that instance reads through `y`,
/// so it compiles to the in-place kernel it always was.
///
/// # Safety
/// `src` must be readable for `n` floats: `y` itself or a buffer that does
/// not partially overlap it. `y`, `slope` and `c` as checked by the callers.
unsafe fn prelu_from<const COPY: bool>(
    y: &mut [f32],
    src: *const f32,
    slope: &[f32],
    n: usize,
    c: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: bounds cover the complete tensor and one slope vector.
        // Vector loops stay within each pixel; irregular/partial tails are scalar.
        unsafe {
            return match c {
                8 => prelu_avx::<8, COPY>(y, src, slope, n, c),
                16 => prelu_avx::<16, COPY>(y, src, slope, n, c),
                32 => prelu_avx::<32, COPY>(y, src, slope, n, c),
                64 => prelu_avx::<64, COPY>(y, src, slope, n, c),
                128 => prelu_avx::<128, COPY>(y, src, slope, n, c),
                256 => prelu_avx::<256, COPY>(y, src, slope, n, c),
                _ => prelu_avx::<0, COPY>(y, src, slope, n, c),
            };
        }
    }
    for i in 0..n {
        // SAFETY: i < n, the caller's contract for `src`.
        let v = unsafe { *src.add(i) };
        y[i] = if v < 0.0 { v * slope[i % c] } else { v };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn prelu_avx<const C: usize, const COPY: bool>(
    y: &mut [f32],
    src: *const f32,
    slope: &[f32],
    n: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let c = if C == 0 { c } else { C };
    if n == 0 {
        return;
    }
    // SAFETY: `prelu_from`'s callers checked the lengths and c > 0 for
    // nonempty input. Vector loads stay inside one pixel; each tail stays
    // inside slope.
    unsafe {
        let yp = y.as_mut_ptr();
        let (xp, sp) = (if COPY { src } else { yp.cast_const() }, slope.as_ptr());
        let zero = _mm256_setzero_ps();
        let full = n / c * c;
        let mut base = 0;
        while base < full {
            let mut ch = 0;
            while ch + 8 <= c {
                let v = _mm256_loadu_ps(xp.add(base + ch));
                let scaled = _mm256_mul_ps(v, _mm256_loadu_ps(sp.add(ch)));
                let negative = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
                store8_full(yp.add(base + ch), _mm256_blendv_ps(v, scaled, negative));
                ch += 8;
            }
            while ch < c {
                prelu_scalar::<COPY>(yp.add(base + ch), xp.add(base + ch), *sp.add(ch));
                ch += 1;
            }
            base += c;
        }
        for i in full..n {
            prelu_scalar::<COPY>(yp.add(i), xp.add(i), *sp.add(i - full));
        }
    }
}

/// One scalar PReLU element; in place it stores only a negative value, as the
/// in-place kernel always did.
///
/// # Safety
/// Both pointers valid for one float; equal when `COPY` is false.
#[inline(always)]
unsafe fn prelu_scalar<const COPY: bool>(y: *mut f32, x: *const f32, slope: f32) {
    // SAFETY: the caller's contract.
    unsafe {
        let v = *x;
        if v < 0.0 {
            *y = v * slope;
        } else if COPY {
            *y = v;
        }
    }
}

/// `y = max(y, x)`.
pub(crate) fn max_into(y: &mut [f32], x: &[f32], n: usize) {
    assert!(y.len() >= n && x.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: the assert above covers every element read and written.
        unsafe {
            return max_avx(y, x, n);
        }
    }
    for i in 0..n {
        if x[i] > y[i] {
            y[i] = x[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_avx(y: &mut [f32], x: &[f32], n: usize) {
    use std::arch::x86_64::*;
    // SAFETY: `max_into` asserted both slices hold at least `n` floats.
    unsafe {
        let (yp, xp) = (y.as_mut_ptr(), x.as_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(
                yp.add(i),
                _mm256_max_ps(_mm256_loadu_ps(yp.add(i)), _mm256_loadu_ps(xp.add(i))),
            );
            i += 8;
        }
        while i < n {
            if *xp.add(i) > *yp.add(i) {
                *yp.add(i) = *xp.add(i);
            }
            i += 1;
        }
    }
}

/// Complete, non-overlapping 2x2 NHWC pooling. Returns false when this tier
/// should keep using the general path. Channel tails retain scalar comparison
/// semantics; vector lanes retain MAXPS operand order (including NaNs/zeros).
pub(crate) fn maxpool_2x2(x: &[f32], y: &mut [f32], oh: usize, ow: usize, c: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        let count = oh
            .checked_mul(ow)
            .and_then(|n| n.checked_mul(c))
            .expect("pool overflow");
        assert!(x.len() >= count.checked_mul(4).expect("pool overflow") && y.len() >= count);
        // SAFETY: full 2x2 windows, extents checked above, disjoint Rust slices,
        // and AVX support established by the process-wide dispatch.
        unsafe { maxpool_2x2_avx(x, y, oh, ow, c) };
        return true;
    }
    let _ = (x, y, oh, ow, c);
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn maxpool_2x2_avx(x: &[f32], y: &mut [f32], oh: usize, ow: usize, c: usize) {
    use std::arch::x86_64::*;
    // SAFETY: the wrapper validated all four input pixels and output extents.
    unsafe {
        let row_stride = 2 * ow * c;
        let vector_end = c / 8 * 8;
        for row in 0..oh {
            for col in 0..ow {
                let top = x.as_ptr().add(2 * row * row_stride + 2 * col * c);
                let bottom = top.add(row_stride);
                let dst = y.as_mut_ptr().add((row * ow + col) * c);
                let mut ch = 0;
                while ch < vector_end {
                    let mut value = _mm256_loadu_ps(top.add(ch));
                    value = _mm256_max_ps(value, _mm256_loadu_ps(top.add(c + ch)));
                    value = _mm256_max_ps(value, _mm256_loadu_ps(bottom.add(ch)));
                    value = _mm256_max_ps(value, _mm256_loadu_ps(bottom.add(c + ch)));
                    _mm256_storeu_ps(dst.add(ch), value);
                    ch += 8;
                }
                while ch < c {
                    let mut value = *top.add(ch);
                    for next in [*top.add(c + ch), *bottom.add(ch), *bottom.add(c + ch)] {
                        if next > value {
                            value = next;
                        }
                    }
                    *dst.add(ch) = value;
                    ch += 1;
                }
            }
        }
    }
}

/// Elementwise sum followed by PReLU, without materializing the intermediate.
/// The addition and multiplication stay separate operations, in that order,
/// so the result matches an add followed by [`prelu_inplace`] bit for bit.
pub(crate) fn add_prelu(y: &mut [f32], x1: &[f32], x2: &[f32], slope: &[f32], n: usize, c: usize) {
    assert!(y.len() >= n && x1.len() >= n && x2.len() >= n);
    assert!(slope.len() >= c && (n == 0 || c > 0));
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: lengths above cover every read/write; slices cannot alias.
        unsafe {
            return match c {
                8 => add_prelu_avx::<8>(y, x1, x2, slope, n, c),
                16 => add_prelu_avx::<16>(y, x1, x2, slope, n, c),
                32 => add_prelu_avx::<32>(y, x1, x2, slope, n, c),
                64 => add_prelu_avx::<64>(y, x1, x2, slope, n, c),
                128 => add_prelu_avx::<128>(y, x1, x2, slope, n, c),
                256 => add_prelu_avx::<256>(y, x1, x2, slope, n, c),
                _ => add_prelu_avx::<0>(y, x1, x2, slope, n, c),
            };
        }
    }
    for i in 0..n {
        let v = x1[i] + x2[i];
        y[i] = if v < 0.0 { v * slope[i % c] } else { v };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_prelu_avx<const C: usize>(
    y: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    slope: &[f32],
    n: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    let c = if C == 0 { c } else { C };
    if n == 0 {
        return;
    }
    // SAFETY: validated in add_prelu. All vector loads stay within a pixel.
    unsafe {
        let (yp, ap, bp, sp) = (y.as_mut_ptr(), x1.as_ptr(), x2.as_ptr(), slope.as_ptr());
        let zero = _mm256_setzero_ps();
        let mut base = 0;
        while base < n {
            let len = c.min(n - base);
            let mut ch = 0;
            while ch + 8 <= len {
                let i = base + ch;
                let v = _mm256_add_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)));
                let scaled = _mm256_mul_ps(v, _mm256_loadu_ps(sp.add(ch)));
                let negative = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
                _mm256_storeu_ps(yp.add(i), _mm256_blendv_ps(v, scaled, negative));
                ch += 8;
            }
            while ch < len {
                let i = base + ch;
                let v = *ap.add(i) + *bp.add(i);
                *yp.add(i) = if v < 0.0 { v * *sp.add(ch) } else { v };
                ch += 1;
            }
            base += len;
        }
    }
}

/// `y = relu(x1 + x2)` over `pixels` NHWC pixels of `oc` channels, where `x1`
/// stores only its first `ic` channels and the rest are the zeros a channel
/// pad would have written. It stands in for Padc→Add→Relu or Add→Relu with the
/// same bits: the pad zero stays the first operand, and exactly like [`relu`]
/// over the whole `pixels * oc` tensor, AVX tiers give every element before
/// its last complete eight MAXPS semantics (NaN and -0 become +0) and the
/// remainder the scalar comparison.
///
/// # Safety
/// `x1` must be readable for `pixels * ic` floats, `x2` for `pixels * oc`, and
/// `y` writable for `pixels * oc`. `y` may be `x2`, or `x1` when `ic == oc`,
/// element for element; no other overlap is allowed.
pub(crate) unsafe fn add_relu(
    y: *mut f32,
    x1: *const f32,
    x2: *const f32,
    pixels: usize,
    ic: usize,
    oc: usize,
) {
    assert!(ic <= oc);
    let n = span(pixels, oc);
    // Without padding the pixels are one run; no per-pixel tails.
    let (pixels, ic, oc) = if ic == oc {
        (1, n, n)
    } else {
        (pixels, ic, oc)
    };
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: the caller's contract; AVX was detected.
        return unsafe { add_relu_avx(y, x1, x2, pixels, ic, oc) };
    }
    // SAFETY: every index below is within the extents the caller guarantees.
    unsafe {
        for p in 0..pixels {
            for ch in 0..oc {
                let a = if ch < ic { *x1.add(p * ic + ch) } else { 0.0 };
                let v = a + *x2.add(p * oc + ch);
                *y.add(p * oc + ch) = if v < 0.0 { 0.0 } else { v };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_relu_avx(
    y: *mut f32,
    x1: *const f32,
    x2: *const f32,
    pixels: usize,
    ic: usize,
    oc: usize,
) {
    use std::arch::x86_64::*;
    let vector_end = pixels * oc / 8 * 8;
    let relu_at = |v: f32, i: usize| {
        if i < vector_end {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        } else if v < 0.0 {
            0.0
        } else {
            v
        }
    };
    // SAFETY: `add_relu`'s contract. Each vector is loaded before it is
    // stored, so `y` aliasing an input index for index is harmless.
    unsafe {
        let (zero8, zero4) = (_mm256_setzero_ps(), _mm_setzero_ps());
        for p in 0..pixels {
            let (a, b, d, base) = (x1.add(p * ic), x2.add(p * oc), y.add(p * oc), p * oc);
            let mut ch = 0;
            while ch < oc {
                let real = ch < ic;
                let end = if real { ic } else { oc };
                if ch + 8 <= end && base + ch + 8 <= vector_end {
                    let first = if real {
                        _mm256_loadu_ps(a.add(ch))
                    } else {
                        zero8
                    };
                    let v = _mm256_add_ps(first, _mm256_loadu_ps(b.add(ch)));
                    _mm256_storeu_ps(d.add(ch), _mm256_max_ps(v, zero8));
                    ch += 8;
                } else if ch + 4 <= end && base + ch + 4 <= vector_end {
                    let first = if real { _mm_loadu_ps(a.add(ch)) } else { zero4 };
                    let v = _mm_add_ps(first, _mm_loadu_ps(b.add(ch)));
                    _mm_storeu_ps(d.add(ch), _mm_max_ps(v, zero4));
                    ch += 4;
                } else {
                    let first = if real { *a.add(ch) } else { 0.0 };
                    *d.add(ch) = relu_at(first + *b.add(ch), base + ch);
                    ch += 1;
                }
            }
        }
    }
}

/// Clamp every value to `[low, high]` in place, exactly as [`f32::clamp`]
/// (and the standalone `Clip` operation) does.
pub(crate) fn clip_inplace(y: &mut [f32], low: f32, high: f32) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: AVX detected; the body stays inside `y`.
        return unsafe { clip_avx(y, low, high) };
    }
    for v in y {
        *v = v.clamp(low, high);
    }
}

/// `min(high, max(low, v))` with these operand orders is `f32::clamp` bit for
/// bit: MAXPS/MINPS return their second operand for a NaN or a tie, so NaN
/// and -0 pass through as `clamp` leaves them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn clip_avx(y: &mut [f32], low: f32, high: f32) {
    use std::arch::x86_64::*;
    let (lo, hi) = (_mm256_set1_ps(low), _mm256_set1_ps(high));
    let (chunks, rest) = y.as_chunks_mut::<8>();
    for chunk in chunks {
        // SAFETY: each chunk is exactly eight floats.
        unsafe {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_min_ps(hi, _mm256_max_ps(lo, v)));
        }
    }
    for v in rest {
        *v = v.clamp(low, high);
    }
}

/// Copy and apply ReLU in one pass, preserving the existing tier semantics.
pub(crate) fn relu(y: &mut [f32], x: &[f32], n: usize) {
    assert!(y.len() >= n && x.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: caller's nonoverlapping slices and bounds checked above.
        unsafe {
            return relu_copy_avx(y, x, n);
        }
    }
    for i in 0..n {
        y[i] = if x[i] < 0.0 { 0.0 } else { x[i] };
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn relu_copy_avx(y: &mut [f32], x: &[f32], n: usize) {
    use std::arch::x86_64::*;
    // SAFETY: checked by relu. The scalar tail never reads past n.
    unsafe {
        let (yp, xp) = (y.as_mut_ptr(), x.as_ptr());
        let mut i = 0;
        let zero = _mm256_setzero_ps();
        while i + 8 <= n {
            _mm256_storeu_ps(yp.add(i), _mm256_max_ps(_mm256_loadu_ps(xp.add(i)), zero));
            i += 8;
        }
        while i < n {
            let v = *xp.add(i);
            *yp.add(i) = if v < 0.0 { 0.0 } else { v };
            i += 1;
        }
    }
}

/// One output row of a 5-wide, stride-1 depthwise window, borders included,
/// `$pixels` pixels at a time: each input column of a tile is loaded once per
/// kernel row and feeds every pixel whose window covers it. A column outside
/// the image is skipped, so every output takes bias, then its live taps in
/// row and column order: the sums of the tap-by-tap tiles.
#[cfg(target_arch = "x86_64")]
macro_rules! dw5_row_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$ow:expr,$iw:expr,$pl:expr,$c:expr,$row:expr,$rows:expr,$bounds:expr,$pixels:literal,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: checked by depthwise_row5; every loaded column is inside
        // the image and every stored pixel inside the row.
        unsafe {
            let (yp, xp, wp, b) = ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b);
            let (ow, iw, pl, c, row, (ky0, ky1)) = ($ow, $iw, $pl, $c, $row, $rows);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            for first in (0..ow).step_by($pixels) {
                let left = first as isize - pl;
                let mut ch = 0;
                while ch < c {
                    let mut acc = [_mm256_setzero_ps(); $pixels];
                    if !b.is_null() {
                        acc = [_mm256_loadu_ps(b.add(ch)); $pixels];
                    }
                    // Two copies of the taps: only border tiles test columns.
                    macro_rules! taps {
                        ($border:expr) => {
                            for ky in ky0..ky1 {
                                let taps = wp.add(ky * 5 * c + ch);
                                let k = [
                                    _mm256_loadu_ps(taps),
                                    _mm256_loadu_ps(taps.add(c)),
                                    _mm256_loadu_ps(taps.add(2 * c)),
                                    _mm256_loadu_ps(taps.add(3 * c)),
                                    _mm256_loadu_ps(taps.add(4 * c)),
                                ];
                                let src = xp.add((ky - ky0) * row + ch);
                                for t in 0..$pixels + 4 {
                                    let j = left + t as isize;
                                    if $border && (j < 0 || j >= iw as isize) {
                                        continue;
                                    }
                                    let v = _mm256_loadu_ps(src.add(j as usize * c));
                                    for (kx, weight) in k.iter().enumerate() {
                                        if kx <= t && t - kx < $pixels {
                                            acc[t - kx] = fma8!($fma, v, *weight, acc[t - kx]);
                                        }
                                    }
                                }
                            }
                        };
                    }
                    if left >= 0 && left as usize + $pixels + 4 <= iw {
                        taps!(false);
                    } else {
                        taps!(true);
                    }
                    for (i, v) in acc.iter().enumerate() {
                        let v = if clamps { vclip!(8, *v, lo, hi) } else { *v };
                        _mm256_storeu_ps(yp.add((first + i) * c + ch), v);
                    }
                    ch += 8;
                }
            }
        }
    }};
}

/// A whole output row of a 5x5, stride-1 depthwise layer whose width is a
/// multiple of seven or eight. `x` starts at the first live input row and
/// column 0; `rows` are the live kernel rows. Returns false when unsupported.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_row5(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    ow: usize,
    iw: usize,
    pl: isize,
    c: usize,
    row: usize,
    rows: (usize, usize),
    clip: Option<(f32, f32)>,
) -> bool {
    let pixels = if ow.is_multiple_of(7) { 7 } else { 8 };
    if simd_tier() < 2 || !c.is_multiple_of(8) || !ow.is_multiple_of(pixels) || rows.0 >= rows.1 {
        return false;
    }
    assert!(rows.1 <= 5 && y.len() >= ow * c && w.len() >= 25 * c && row >= iw * c);
    assert!(x.len() >= (rows.1 - rows.0 - 1) * row + iw * c);
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: bounds above, lane divisibility, and feature dispatch.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match (simd_tier(), pixels) {
            (3, 7) => dw5_row_avx2::<7>(y, x, w, b, ow, iw, pl, c, row, rows, bounds),
            (3, _) => dw5_row_avx2::<8>(y, x, w, b, ow, iw, pl, c, row, rows, bounds),
            (_, 7) => dw5_row_avx::<7>(y, x, w, b, ow, iw, pl, c, row, rows, bounds),
            _ => dw5_row_avx::<8>(y, x, w, b, ow, iw, pl, c, row, rows, bounds),
        }
        return true;
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw5_row_avx2<const P: usize>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    ow: usize,
    iw: usize,
    pl: isize,
    c: usize,
    row: usize,
    rows: (usize, usize),
    bounds: (f32, f32),
) {
    if P == 7 {
        dw5_row_body!(y, x, w, b, ow, iw, pl, c, row, rows, bounds, 7, true)
    } else {
        dw5_row_body!(y, x, w, b, ow, iw, pl, c, row, rows, bounds, 8, true)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw5_row_avx<const P: usize>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    ow: usize,
    iw: usize,
    pl: isize,
    c: usize,
    row: usize,
    rows: (usize, usize),
    bounds: (f32, f32),
) {
    if P == 7 {
        dw5_row_body!(y, x, w, b, ow, iw, pl, c, row, rows, bounds, 7, false)
    } else {
        dw5_row_body!(y, x, w, b, ow, iw, pl, c, row, rows, bounds, 8, false)
    }
}

// Whole 3x3 depthwise windows. A pixel's accumulator is stored only once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_3x3_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    c: usize,
    row_stride: usize,
    step: usize,
    clip: Option<(f32, f32)>,
) -> bool {
    if count == 0 || c == 0 {
        return false;
    }
    // MAXPS/MINPS against infinities return every value unchanged.
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    assert!(y.len() >= count.checked_mul(c).expect("depthwise output overflow"));
    let last = (count - 1)
        .checked_mul(step)
        .and_then(|n| n.checked_add(span(2, row_stride)))
        .and_then(|n| n.checked_add(span(3, c)))
        .expect("depthwise input overflow");
    assert!(x.len() >= last && w.len() >= span(9, c));
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: bounds above, lane divisibility below, and feature dispatch.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match simd_tier() {
            3 if c.is_multiple_of(8) => {
                match (c, step) {
                    (8, 8) => dw3_avx2::<8>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                    (16, 16) => {
                        dw3_avx2::<16>(y, x, w, b, count, c, row_stride, step, (0, c), bounds)
                    }
                    _ => dw3_avx2::<0>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                }
                return true;
            }
            2 if c.is_multiple_of(8) => {
                match (c, step) {
                    (8, 8) => dw3_avx::<8>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                    (16, 16) => {
                        dw3_avx::<16>(y, x, w, b, count, c, row_stride, step, (0, c), bounds)
                    }
                    _ => dw3_avx::<0>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                }
                return true;
            }
            // Four channels past a multiple of eight have always taken the
            // SSE body, unfused on every tier. AVX without FMA computes the
            // same lanes, so only the last four channels stay 128-bit.
            2..=3 if c.is_multiple_of(4) => {
                let split = c - 4;
                dw3_avx::<0>(y, x, w, b, count, c, row_stride, step, (0, split), bounds);
                dw3_sse::<0>(y, x, w, b, count, c, row_stride, step, (split, c), bounds);
                return true;
            }
            1 if c.is_multiple_of(4) => {
                match (c, step) {
                    (8, 8) => dw3_sse::<8>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                    (16, 16) => {
                        dw3_sse::<16>(y, x, w, b, count, c, row_stride, step, (0, c), bounds)
                    }
                    _ => dw3_sse::<0>(y, x, w, b, count, c, row_stride, step, (0, c), bounds),
                }
                return true;
            }
            _ => {}
        }
    }
    false
}

/// A 1x1, unpadded, unit-stride depthwise layer (a folded per-channel scale
/// and shift) over `count` pixels, on AVX tiers with `c % 8 == 0`: each output
/// is its bias, then one `x * w + acc`, the one-tap window's arithmetic, with
/// the weight and bias held in registers across the pixels. Returns false
/// when it did nothing.
pub(crate) fn depthwise_1x1(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    c: usize,
    clip: Option<(f32, f32)>,
) -> bool {
    if c == 0 || !c.is_multiple_of(8) || simd_tier() < 2 {
        return false;
    }
    let n = span(count, c);
    assert!(y.len() >= n && x.len() >= n && w.len() >= c);
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: extents checked above; AVX detected; c is a multiple of eight.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        if simd_tier() == 3 {
            dw1_avx2(y, x, w, b, count, c, bounds);
        } else {
            dw1_avx(y, x, w, b, count, c, bounds);
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
macro_rules! dw1_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$count:expr,$c:expr,$bounds:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: checked by depthwise_1x1.
        unsafe {
            let (yp, xp, wp, b, count, c) =
                ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b, $count, $c);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            for ch in (0..c).step_by(8) {
                let weight = _mm256_loadu_ps(wp.add(ch));
                let seed = if b.is_null() {
                    _mm256_setzero_ps()
                } else {
                    _mm256_loadu_ps(b.add(ch))
                };
                for p in 0..count {
                    let v = fma8!($fma, _mm256_loadu_ps(xp.add(p * c + ch)), weight, seed);
                    _mm256_storeu_ps(yp.add(p * c + ch), vclip_if!(clamps, 8, v, lo, hi));
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dw1_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    bounds: (f32, f32),
) {
    dw1_body!(y, x, w, b, count, c, bounds, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dw1_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    bounds: (f32, f32),
) {
    dw1_body!(y, x, w, b, count, c, bounds, false)
}

/// The live part of `kh`-by-`kw` depthwise windows, on AVX tiers with
/// `c % 8 == 0`. Returns false when it did nothing.
///
/// Every pixel of the run reads kernel rows `rows.0..rows.1` and columns
/// `cols.0..cols.1` of a kernel `kw` wide; `x` starts under tap
/// (`rows.0`, `cols.0`) of the first pixel. A border pixel simply has fewer
/// live taps, the ones the fallback does not skip as padding. Each output
/// keeps exactly the fallback's arithmetic: seeded with the bias, then one
/// `x * w + acc` per live tap in row-major order, fused on tier 3 and separate
/// on tier 2, as [`depthwise_tap`] does for these widths. It is stored once,
/// instead of read and written back once per tap.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_window(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    c: usize,
    row_stride: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    clip: Option<(f32, f32)>,
) -> bool {
    if count == 0 || c == 0 || !c.is_multiple_of(8) {
        return false;
    }
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    let (rows, cols) = if rows.0 < rows.1 && cols.0 < cols.1 {
        (rows, cols)
    } else {
        ((0, 0), (0, 0)) // no live tap: every output is its bias
    };
    assert!(cols.1 <= kw && y.len() >= count.checked_mul(c).expect("depthwise output overflow"));
    if rows.0 < rows.1 {
        let last = (count - 1)
            .checked_mul(step)
            .and_then(|n| n.checked_add(span(rows.1 - rows.0 - 1, row_stride)))
            .and_then(|n| n.checked_add(span(cols.1 - cols.0, c)))
            .expect("depthwise input overflow");
        assert!(x.len() >= last && w.len() >= span((rows.1 - 1) * kw + cols.1, c));
    }
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: bounds above, lane divisibility, and feature dispatch.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match simd_tier() {
            3 => {
                dwk_avx2(
                    y, x, w, b, count, c, row_stride, step, kw, rows, cols, bounds,
                );
                return true;
            }
            2 => {
                dwk_avx(
                    y, x, w, b, count, c, row_stride, step, kw, rows, cols, bounds,
                );
                return true;
            }
            _ => {}
        }
    }
    false
}

/// `$pixels` neighbouring pixels by `$blocks` eight-channel blocks at `ch`:
/// eight independent chains however the run is shaped.
#[cfg(target_arch = "x86_64")]
macro_rules! dwk_tile {
    ($yp:expr,$xp:expr,$wp:expr,$b:expr,$p:expr,$ch:expr,$c:expr,$row:expr,$step:expr,$kw:expr,$rows:expr,$cols:expr,$clip:expr,$pixels:literal,$blocks:literal,$fma:tt) => {{
        let (yp, xp, wp, b, p, ch, c, row, step, kw) =
            ($yp, $xp, $wp, $b, $p, $ch, $c, $row, $step, $kw);
        let ((ky0, ky1), (kx0, kx1)) = ($rows, $cols);
        let mut acc = [[_mm256_setzero_ps(); $blocks]; $pixels];
        if !b.is_null() {
            for a in acc.iter_mut() {
                for (m, v) in a.iter_mut().enumerate() {
                    *v = _mm256_loadu_ps(b.add(ch + 8 * m));
                }
            }
        }
        for ky in ky0..ky1 {
            for kx in kx0..kx1 {
                let tap = wp.add((ky * kw + kx) * c + ch);
                let src = xp.add(p * step + (ky - ky0) * row + (kx - kx0) * c + ch);
                for m in 0..$blocks {
                    let weight = _mm256_loadu_ps(tap.add(8 * m));
                    for (k, a) in acc.iter_mut().enumerate() {
                        let v = _mm256_loadu_ps(src.add(k * step + 8 * m));
                        a[m] = fma8!($fma, v, weight, a[m]);
                    }
                }
            }
        }
        for (k, a) in acc.iter().enumerate() {
            for (m, v) in a.iter().enumerate() {
                let (lo, hi, clamps) = $clip;
                let v = if clamps { vclip!(8, *v, lo, hi) } else { *v };
                _mm256_storeu_ps(yp.add((p + k) * c + ch + 8 * m), v);
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! dwk_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$count:expr,$c:expr,$row:expr,$step:expr,$kw:expr,$rows:expr,$cols:expr,$bounds:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: checked by depthwise_window; every live tap lies inside it.
        unsafe {
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let clip = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1), clamps);
            let (yp, xp, wp, b) = ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b);
            let (count, c, row, step, kw, rows, cols) =
                ($count, $c, $row, $step, $kw, $rows, $cols);
            // Fewer pixels left (the end of a run, or a lone border column)
            // trade pixels for channel blocks, keeping eight chains.
            macro_rules! run {
                ($pixels:literal, $blocks:literal, $p:expr, $ch:expr) => {
                    dwk_tile!(
                        yp, xp, wp, b, $p, $ch, c, row, step, kw, rows, cols, clip, $pixels,
                        $blocks, $fma
                    )
                };
            }
            macro_rules! pixels {
                ($pixels:literal, $blocks:literal, $p:expr) => {{
                    let mut ch = 0;
                    while ch + 8 * $blocks <= c {
                        run!($pixels, $blocks, $p, ch);
                        ch += 8 * $blocks;
                    }
                    while ch < c {
                        run!($pixels, 1, $p, ch);
                        ch += 8;
                    }
                }};
            }
            let mut p = 0;
            // 5-wide windows with every column live, at stride one: eight
            // neighbouring pixels read twelve input columns per kernel row,
            // each loaded once instead of once per tap (17 loads per 40
            // products instead of 45). Every accumulator still takes bias,
            // then its live rows in order, each left to right: the generic
            // tile's sums. Ten pixels (all 16 registers on AVX2) measured no
            // faster.
            if (kw, cols, step) == (5, (0, 5), c) && rows.0 < rows.1 {
                while p + 8 <= count {
                    let mut ch = 0;
                    while ch < c {
                        let mut acc = [_mm256_setzero_ps(); 8];
                        if !b.is_null() {
                            acc = [_mm256_loadu_ps(b.add(ch)); 8];
                        }
                        for ky in rows.0..rows.1 {
                            let taps = wp.add(ky * 5 * c + ch);
                            let k = [
                                _mm256_loadu_ps(taps),
                                _mm256_loadu_ps(taps.add(c)),
                                _mm256_loadu_ps(taps.add(2 * c)),
                                _mm256_loadu_ps(taps.add(3 * c)),
                                _mm256_loadu_ps(taps.add(4 * c)),
                            ];
                            let src = xp.add(p * c + (ky - rows.0) * row + ch);
                            for j in 0..12 {
                                let v = _mm256_loadu_ps(src.add(j * c));
                                for (kx, weight) in k.iter().enumerate() {
                                    if kx <= j && j - kx < 8 {
                                        acc[j - kx] = fma8!($fma, v, *weight, acc[j - kx]);
                                    }
                                }
                            }
                        }
                        for (i, v) in acc.iter().enumerate() {
                            let (lo, hi, clamps) = clip;
                            let v = if clamps { vclip!(8, *v, lo, hi) } else { *v };
                            _mm256_storeu_ps(yp.add((p + i) * c + ch), v);
                        }
                        ch += 8;
                    }
                    p += 8;
                }
            }
            while p + 8 <= count {
                pixels!(8, 1, p);
                p += 8;
            }
            if p + 4 <= count {
                pixels!(4, 2, p);
                p += 4;
            }
            if p + 2 <= count {
                pixels!(2, 4, p);
                p += 2;
            }
            if p < count {
                pixels!(1, 8, p);
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dwk_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    bounds: (f32, f32),
) {
    dwk_body!(y, x, w, b, count, c, row, step, kw, rows, cols, bounds, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn dwk_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    bounds: (f32, f32),
) {
    dwk_body!(y, x, w, b, count, c, row, step, kw, rows, cols, bounds, false)
}

#[cfg(target_arch = "x86_64")]
macro_rules! dw3_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$count:expr,$c:expr,$row:expr,$step:expr,$chans:expr,$bounds:expr,$lanes:tt,$fma:tt) => {{
        // SAFETY: checked by depthwise_3x3_interior. Each lane belongs to one
        // pixel, and each of the nine source vectors is in the validated window.
        unsafe {
            let (lo, hi) = (vsplat!($lanes, $bounds.0), vsplat!($lanes, $bounds.1));
            // Only a real clamp pays for it: nine taps share each store.
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            macro_rules! out {
                ($v:expr) => {
                    if clamps {
                        vclip!($lanes, $v, lo, hi)
                    } else {
                        $v
                    }
                };
            }
            let (yp, xp, wp, b) = ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b);
            let (count, c, row, step) = ($count, $c, $row, $step);
            let (mut ch, end) = $chans;
            while ch < end {
                let seed = if b.is_null() {
                    vsplat!($lanes, 0.0)
                } else {
                    vld!($lanes, b.add(ch))
                };
                let k0 = vld!($lanes, wp.add(0 * c + ch));
                let k1 = vld!($lanes, wp.add(1 * c + ch));
                let k2 = vld!($lanes, wp.add(2 * c + ch));
                let k3 = vld!($lanes, wp.add(3 * c + ch));
                let k4 = vld!($lanes, wp.add(4 * c + ch));
                let k5 = vld!($lanes, wp.add(5 * c + ch));
                let k6 = vld!($lanes, wp.add(6 * c + ch));
                let k7 = vld!($lanes, wp.add(7 * c + ch));
                let k8 = vld!($lanes, wp.add(8 * c + ch));
                let mut p = 0;
                while p + 4 <= count {
                    let (mut a0, mut a1, mut a2, mut a3) = (seed, seed, seed, seed);
                    let src = p * step + ch;
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 0 * row + 0 * c)),
                        k0,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 0 * row + 0 * c)),
                        k0,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 0 * row + 0 * c)),
                        k0,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 0 * row + 0 * c)),
                        k0,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 0 * row + 1 * c)),
                        k1,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 0 * row + 1 * c)),
                        k1,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 0 * row + 1 * c)),
                        k1,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 0 * row + 1 * c)),
                        k1,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 0 * row + 2 * c)),
                        k2,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 0 * row + 2 * c)),
                        k2,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 0 * row + 2 * c)),
                        k2,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 0 * row + 2 * c)),
                        k2,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 1 * row + 0 * c)),
                        k3,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 1 * row + 0 * c)),
                        k3,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 1 * row + 0 * c)),
                        k3,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 1 * row + 0 * c)),
                        k3,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 1 * row + 1 * c)),
                        k4,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 1 * row + 1 * c)),
                        k4,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 1 * row + 1 * c)),
                        k4,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 1 * row + 1 * c)),
                        k4,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 1 * row + 2 * c)),
                        k5,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 1 * row + 2 * c)),
                        k5,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 1 * row + 2 * c)),
                        k5,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 1 * row + 2 * c)),
                        k5,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 2 * row + 0 * c)),
                        k6,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 2 * row + 0 * c)),
                        k6,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 2 * row + 0 * c)),
                        k6,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 2 * row + 0 * c)),
                        k6,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 2 * row + 1 * c)),
                        k7,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 2 * row + 1 * c)),
                        k7,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 2 * row + 1 * c)),
                        k7,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 2 * row + 1 * c)),
                        k7,
                        a3
                    );
                    a0 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * step + 2 * row + 2 * c)),
                        k8,
                        a0
                    );
                    a1 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * step + 2 * row + 2 * c)),
                        k8,
                        a1
                    );
                    a2 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * step + 2 * row + 2 * c)),
                        k8,
                        a2
                    );
                    a3 = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 3 * step + 2 * row + 2 * c)),
                        k8,
                        a3
                    );
                    vst!($lanes, yp.add((p + 0) * c + ch), out!(a0));
                    vst!($lanes, yp.add((p + 1) * c + ch), out!(a1));
                    vst!($lanes, yp.add((p + 2) * c + ch), out!(a2));
                    vst!($lanes, yp.add((p + 3) * c + ch), out!(a3));
                    p += 4;
                }
                while p < count {
                    let mut acc = seed;
                    let src = p * step + ch;
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * row + 0 * c)),
                        k0,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * row + 1 * c)),
                        k1,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 0 * row + 2 * c)),
                        k2,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * row + 0 * c)),
                        k3,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * row + 1 * c)),
                        k4,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 1 * row + 2 * c)),
                        k5,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * row + 0 * c)),
                        k6,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * row + 1 * c)),
                        k7,
                        acc
                    );
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + 2 * row + 2 * c)),
                        k8,
                        acc
                    );
                    vst!($lanes, yp.add(p * c + ch), out!(acc));
                    p += 1;
                }
                ch += $lanes;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_avx2<const C: usize>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    chans: (usize, usize),
    bounds: (f32, f32),
) {
    // Hot stride-one shapes collapse address arithmetic and make horizontal
    // input reuse visible to LLVM; the generic path handles every other shape.
    let c = if C == 0 { c } else { C };
    let step = if C == 0 { step } else { C };
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, chans, bounds, 8, true)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, chans, bounds, 8, true)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_avx<const C: usize>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    chans: (usize, usize),
    bounds: (f32, f32),
) {
    // Hot stride-one shapes collapse address arithmetic and make horizontal
    // input reuse visible to LLVM; the generic path handles every other shape.
    let c = if C == 0 { c } else { C };
    let step = if C == 0 { step } else { C };
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, chans, bounds, 8, false)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, chans, bounds, 8, false)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_sse<const C: usize>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    chans: (usize, usize),
    bounds: (f32, f32),
) {
    // Hot stride-one shapes collapse address arithmetic and make horizontal
    // input reuse visible to LLVM; the generic path handles every other shape.
    let c = if C == 0 { c } else { C };
    let step = if C == 0 { step } else { C };
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, chans, bounds, 4, false)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, chans, bounds, 4, false)
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! vadd {
    (8,$a:expr,$b:expr) => {
        std::arch::x86_64::_mm256_add_ps($a, $b)
    };
    (4,$a:expr,$b:expr) => {
        std::arch::x86_64::_mm_add_ps($a, $b)
    };
}

// Pixel batching also for widths 24, 28, 36, 42, 88, etc. on the SSE tier;
// AVX tiers use [`pw_any8_body`]. Only the last `count % 4` pixels fall back
// to matrix-vector calls.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_any_body {
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$lanes:tt,$fma:tt) => {{
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (low, high) = $bounds;
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (lo, hi) = (vsplat!($lanes, low), vsplat!($lanes, high));
            let (count, ci, co) = ($count, $ci, $co);
            let mut p = 0;
            while p + 4 <= count {
                let mut j = 0;
                // Two column blocks give eight independent chains, enough to
                // hide multiply-add latency; four stalled on it. Each output
                // still sums over `i` in order and adds bias last.
                while j + 2 * $lanes <= co {
                    let z = vsplat!($lanes, 0.0);
                    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
                    for i in 0..ci {
                        let row = vld!($lanes, wp.add(i * co + j));
                        let next = vld!($lanes, wp.add(i * co + j + $lanes));
                        let s0 = vsplat!($lanes, *xp.add(p * ci + i));
                        let s1 = vsplat!($lanes, *xp.add((p + 1) * ci + i));
                        let s2 = vsplat!($lanes, *xp.add((p + 2) * ci + i));
                        let s3 = vsplat!($lanes, *xp.add((p + 3) * ci + i));
                        a0 = vfma!($lanes, $fma, row, s0, a0);
                        c0 = vfma!($lanes, $fma, next, s0, c0);
                        a1 = vfma!($lanes, $fma, row, s1, a1);
                        c1 = vfma!($lanes, $fma, next, s1, c1);
                        a2 = vfma!($lanes, $fma, row, s2, a2);
                        c2 = vfma!($lanes, $fma, next, s2, c2);
                        a3 = vfma!($lanes, $fma, row, s3, a3);
                        c3 = vfma!($lanes, $fma, next, s3, c3);
                    }
                    if !b.is_null() {
                        let bias = vld!($lanes, b.add(j));
                        let more = vld!($lanes, b.add(j + $lanes));
                        a0 = vadd!($lanes, a0, bias);
                        a1 = vadd!($lanes, a1, bias);
                        a2 = vadd!($lanes, a2, bias);
                        a3 = vadd!($lanes, a3, bias);
                        c0 = vadd!($lanes, c0, more);
                        c1 = vadd!($lanes, c1, more);
                        c2 = vadd!($lanes, c2, more);
                        c3 = vadd!($lanes, c3, more);
                    }
                    vst!(
                        $lanes,
                        yp.add(p * co + j),
                        vclip_if!(clamps, $lanes, a0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j),
                        vclip_if!(clamps, $lanes, a1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j),
                        vclip_if!(clamps, $lanes, a2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j),
                        vclip_if!(clamps, $lanes, a3, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add(p * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c3, lo, hi)
                    );
                    j += 2 * $lanes;
                }
                if j + $lanes <= co {
                    let z = vsplat!($lanes, 0.0);
                    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                    for i in 0..ci {
                        let row = vld!($lanes, wp.add(i * co + j));
                        a0 = vfma!($lanes, $fma, row, vsplat!($lanes, *xp.add(p * ci + i)), a0);
                        a1 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 1) * ci + i)),
                            a1
                        );
                        a2 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 2) * ci + i)),
                            a2
                        );
                        a3 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 3) * ci + i)),
                            a3
                        );
                    }
                    // Bias is deliberately last, as in `matvec_t`: adding it
                    // first would change the f32 results of this tier.
                    if !b.is_null() {
                        let bias = vld!($lanes, b.add(j));
                        a0 = vadd!($lanes, a0, bias);
                        a1 = vadd!($lanes, a1, bias);
                        a2 = vadd!($lanes, a2, bias);
                        a3 = vadd!($lanes, a3, bias);
                    }
                    vst!(
                        $lanes,
                        yp.add(p * co + j),
                        vclip_if!(clamps, $lanes, a0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j),
                        vclip_if!(clamps, $lanes, a1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j),
                        vclip_if!(clamps, $lanes, a2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j),
                        vclip_if!(clamps, $lanes, a3, lo, hi)
                    );
                    j += $lanes;
                }
                // Four pixels side by side, so four sums advance together;
                // each keeps its own in-order scalar reduction.
                while j < co {
                    let mut acc = [0.0f32; 4];
                    for i in 0..ci {
                        let weight = *wp.add(i * co + j);
                        for (k, sum) in acc.iter_mut().enumerate() {
                            *sum += *xp.add((p + k) * ci + i) * weight;
                        }
                    }
                    for (k, mut sum) in acc.into_iter().enumerate() {
                        if !b.is_null() {
                            sum += *b.add(j);
                        }
                        *yp.add((p + k) * co + j) = sum.clamp(low, high);
                    }
                    j += 1;
                }
                p += 4;
            }
            p
        }
    }};
}

/// [`pw_any_body`] on eight lanes: `$pixels` pixels per tile by sixteen, then
/// eight, output columns, and the last `co % 8` columns in masked lanes.
/// A four-lane or scalar tail would multiply and add separately in input
/// order too, so eight masked lanes doing the same are the same arithmetic,
/// for as many pixels at once as there are chains to hide.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_any8_body {
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,$fma:tt) => {
        pw_any8_body!(@tiles $y, $w, $x, $b, $count, $ci, $co, $bounds, $pixels, [], $fma)
    };
    // `$pixels24`: tile height of the optional 24-column block.
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,$pixels24:literal,$fma:tt) => {
        pw_any8_body!(@tiles $y, $w, $x, $b, $count, $ci, $co, $bounds, $pixels, [$pixels24], $fma)
    };
    (@tiles $y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,[$($pixels24:literal)?],$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: pointwise checked all lengths; masked lanes read and write
        // only columns below `co`.
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (count, ci, co) = ($count, $ci, $co);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let done = count / $pixels * $pixels;
            let wide = co / 8 * 8;
            let bias = |j: usize| {
                if b.is_null() {
                    _mm256_setzero_ps()
                } else {
                    _mm256_loadu_ps(b.add(j))
                }
            };
            // One tile: `$pixels` pixels by the 16 columns at `j`.
            let block16 = |p: usize, j: usize| {
                let mut low = [_mm256_setzero_ps(); $pixels];
                let mut high = [_mm256_setzero_ps(); $pixels];
                for i in 0..ci {
                    let r0 = _mm256_loadu_ps(wp.add(i * co + j));
                    let r1 = _mm256_loadu_ps(wp.add(i * co + j + 8));
                    for k in 0..$pixels {
                        let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                        low[k] = fma8!($fma, r0, s, low[k]);
                        high[k] = fma8!($fma, r1, s, high[k]);
                    }
                }
                // Bias is deliberately last, as in `matvec_t`.
                let (b0, b1) = if b.is_null() {
                    (low, high)
                } else {
                    let (u, v) = (bias(j), bias(j + 8));
                    (
                        low.map(|a| _mm256_add_ps(a, u)),
                        high.map(|a| _mm256_add_ps(a, v)),
                    )
                };
                for k in 0..$pixels {
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j),
                        vclip_if!(clamps, 8, b0[k], lo, hi),
                    );
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j + 8),
                        vclip_if!(clamps, 8, b1[k], lo, hi),
                    );
                }
            };
            // The same for the last eight full columns, when `wide % 16 == 8`.
            let block8 = |p: usize, j: usize| {
                let mut acc = [_mm256_setzero_ps(); $pixels];
                for i in 0..ci {
                    let r = _mm256_loadu_ps(wp.add(i * co + j));
                    for k in 0..$pixels {
                        let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                        acc[k] = fma8!($fma, r, s, acc[k]);
                    }
                }
                if !b.is_null() {
                    let u = bias(j);
                    acc = acc.map(|a| _mm256_add_ps(a, u));
                }
                for k in 0..$pixels {
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j),
                        vclip_if!(clamps, 8, acc[k], lo, hi),
                    );
                }
            };
            // Mutated only by the optional 24-column block below.
            #[allow(unused_mut)]
            let (mut pairs, mut single, mut covered) = (wide / 16 * 16, wide % 16 == 8, done);
            $(
                // Widths of 16k + 8 finish with one 24-column block: the 16-
                // and 8-column blocks it replaces each reloaded the same
                // broadcasts. Only where it measured faster: AVX without FMA.
                if single && pairs >= 16 {
                    let j = pairs - 16;
                    (pairs, single) = (j, false);
                    covered = count / $pixels24 * $pixels24;
                    for p in (0..covered).step_by($pixels24) {
                        let mut c0 = [_mm256_setzero_ps(); $pixels24];
                        let mut c1 = [_mm256_setzero_ps(); $pixels24];
                        let mut c2 = [_mm256_setzero_ps(); $pixels24];
                        for i in 0..ci {
                            let r0 = _mm256_loadu_ps(wp.add(i * co + j));
                            let r1 = _mm256_loadu_ps(wp.add(i * co + j + 8));
                            let r2 = _mm256_loadu_ps(wp.add(i * co + j + 16));
                            for k in 0..$pixels24 {
                                let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                                c0[k] = fma8!($fma, r0, s, c0[k]);
                                c1[k] = fma8!($fma, r1, s, c1[k]);
                                c2[k] = fma8!($fma, r2, s, c2[k]);
                            }
                        }
                        for k in 0..$pixels24 {
                            for (m, a) in [c0[k], c1[k], c2[k]].into_iter().enumerate() {
                                // Bias is deliberately last, as in `matvec_t`.
                                let a = if b.is_null() { a } else { _mm256_add_ps(a, bias(j + 8 * m)) };
                                _mm256_storeu_ps(
                                    yp.add((p + k) * co + j + 8 * m),
                                    vclip_if!(clamps, 8, a, lo, hi),
                                );
                            }
                        }
                    }
                }
            )?
            for p in (0..done).step_by($pixels) {
                for j in (0..pairs).step_by(16) {
                    block16(p, j);
                }
                if single {
                    block8(p, pairs);
                }
            }
            if wide < co {
                let left = (co - wide) as i32;
                let mask = _mm256_setr_epi32(
                    -i32::from(left > 0),
                    -i32::from(left > 1),
                    -i32::from(left > 2),
                    -i32::from(left > 3),
                    -i32::from(left > 4),
                    -i32::from(left > 5),
                    -i32::from(left > 6),
                    0,
                );
                let tail_bias = if b.is_null() {
                    None
                } else {
                    Some(_mm256_maskload_ps(b.add(wide), mask))
                };
                macro_rules! tail {
                    ($n:literal, $p:expr) => {{
                        let mut acc = [_mm256_setzero_ps(); $n];
                        for i in 0..ci {
                            let r = _mm256_maskload_ps(wp.add(i * co + wide), mask);
                            for (k, a) in acc.iter_mut().enumerate() {
                                let s = _mm256_set1_ps(*xp.add(($p + k) * ci + i));
                                *a = fma8!(false, r, s, *a);
                            }
                        }
                        for (k, a) in acc.into_iter().enumerate() {
                            let a = tail_bias.map_or(a, |u| _mm256_add_ps(a, u));
                            _mm256_maskstore_ps(
                                yp.add(($p + k) * co + wide),
                                mask,
                                vclip_if!(clamps, 8, a, lo, hi),
                            );
                        }
                    }};
                }
                let mut p = 0;
                while p + 8 <= done {
                    tail!(8, p);
                    p += 8;
                }
                if p + 4 <= done {
                    tail!(4, p);
                    p += 4;
                }
                while p < done {
                    tail!(1, p);
                    p += 1;
                }
            }
            // `pointwise_rest` recomputes every column of the pixels past
            // `covered` with the same arithmetic.
            done.min(covered)
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    pw_any8_body!(y, w, x, b, count, ci, co, bounds, 6, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    pw_any8_body!(y, w, x, b, count, ci, co, bounds, 4, 3, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_sse(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    // SAFETY: pointwise checked all lengths before dispatch.
    pw_any_body!(y, w, x, b, count, ci, co, bounds, 4, false)
}

/// One output channel (a segmentation or score head): eight pixels per AVX
/// register. Every lane is one pixel and accumulates exactly as the tap
/// fallback does, so the bits match it: bias, then for each tap and each input
/// channel in order a separate multiply and add. An 8x8 transpose turns eight
/// pixels' channel vectors into eight channel vectors across those pixels.
fn single_output_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &crate::plan::Op,
    count: usize,
) -> usize {
    // At unit stride the last group of eight may overlap the one before it,
    // recomputing identical sums, so the whole run is covered.
    let planar = op.sw == 1 && op.kw <= 9 && op.kh * op.ic <= PLANES;
    let count = if planar || count < 8 {
        count
    } else {
        count / 8 * 8
    };
    if count < 8 || !op.ic.is_multiple_of(8) || simd_tier() < 2 {
        return 0;
    }
    let (row, pixel) = (span(op.iw, op.ic), span(op.sw, op.ic));
    let reach = span(op.kh - 1, row) + span(count - 1, pixel) + span(op.kw, op.ic);
    assert!(y.len() >= count && x.len() >= reach && w.len() >= span(op.kh * op.kw, op.ic));
    let seed = bias.map_or(0.0, |b| b[0]);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: extents asserted above; AVX detected; ic is a multiple of 8.
    unsafe {
        single_output_avx(y, x, w, seed, op, count, row, pixel);
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn single_output_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    seed: f32,
    op: &crate::plan::Op,
    count: usize,
    row: usize,
    pixel: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: single_output_interior checked every window and weight.
    unsafe {
        let (yp, xp, wp) = (y.as_mut_ptr(), x.as_ptr(), w.as_ptr());
        if pixel == op.ic && op.kw <= 9 && op.kh * op.ic <= PLANES {
            return single_output_planar_avx(yp, xp, wp, seed, op, count, row);
        }
        debug_assert!(count.is_multiple_of(8));
        for p in (0..count).step_by(8) {
            let mut acc = _mm256_set1_ps(seed);
            for ky in 0..op.kh {
                for kx in 0..op.kw {
                    let tap = wp.add((ky * op.kw + kx) * op.ic);
                    let base = xp.add(ky * row + p * pixel + kx * op.ic);
                    for chunk in (0..op.ic).step_by(8) {
                        let r: [__m256; 8] =
                            std::array::from_fn(|k| _mm256_loadu_ps(base.add(k * pixel + chunk)));
                        let (t0, t1) = (
                            _mm256_unpacklo_ps(r[0], r[1]),
                            _mm256_unpackhi_ps(r[0], r[1]),
                        );
                        let (t2, t3) = (
                            _mm256_unpacklo_ps(r[2], r[3]),
                            _mm256_unpackhi_ps(r[2], r[3]),
                        );
                        let (t4, t5) = (
                            _mm256_unpacklo_ps(r[4], r[5]),
                            _mm256_unpackhi_ps(r[4], r[5]),
                        );
                        let (t6, t7) = (
                            _mm256_unpacklo_ps(r[6], r[7]),
                            _mm256_unpackhi_ps(r[6], r[7]),
                        );
                        let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                        let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                        let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                        let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                        let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                        let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                        let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                        let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                        let channels = [
                            _mm256_permute2f128_ps::<0x20>(s0, s4),
                            _mm256_permute2f128_ps::<0x20>(s1, s5),
                            _mm256_permute2f128_ps::<0x20>(s2, s6),
                            _mm256_permute2f128_ps::<0x20>(s3, s7),
                            _mm256_permute2f128_ps::<0x31>(s0, s4),
                            _mm256_permute2f128_ps::<0x31>(s1, s5),
                            _mm256_permute2f128_ps::<0x31>(s2, s6),
                            _mm256_permute2f128_ps::<0x31>(s3, s7),
                        ];
                        for (i, v) in channels.into_iter().enumerate() {
                            let weight = _mm256_set1_ps(*tap.add(chunk + i));
                            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, weight));
                        }
                    }
                }
            }
            _mm256_storeu_ps(yp.add(p), acc);
        }
    }
}

/// Input rows of one output chunk, transposed: `kh * ic` planes (one per
/// kernel row and input channel), each `CHUNK + 16` pixels long: a chunk
/// of up to `CHUNK + 8` outputs plus up to eight more kernel columns.
const PLANES: usize = 48;
const CHUNK: usize = 64;

/// [`single_output_avx`] at unit stride. Instead of transposing every 8x8
/// block once per tap, each input row of a chunk is transposed once into
/// channel planes, where a tap `kx` columns to the right is the same plane
/// read `kx` floats later. Four register groups advance together, so the
/// per-lane chain of separate multiplies and adds does not wait on itself.
/// Each lane still sums bias, then every tap and channel in order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn single_output_planar_avx(
    yp: *mut f32,
    xp: *const f32,
    wp: *const f32,
    seed: f32,
    op: &crate::plan::Op,
    count: usize,
    row: usize,
) {
    use std::arch::x86_64::*;
    const STRIDE: usize = CHUNK + 16;
    let (ic, kh, kw) = (op.ic, op.kh, op.kw);
    // Only columns written below are ever read, so no initialization.
    let mut planes = [std::mem::MaybeUninit::<f32>::uninit(); PLANES * STRIDE];
    let pp = planes.as_mut_ptr().cast::<f32>();
    // SAFETY: the caller's windows cover every column read here (a chunk of
    // n outputs reads n + kw - 1 input columns), and planes hold kh*ic rows of
    // at least n + kw - 1 <= STRIDE floats. Every chunk has n >= 8 outputs:
    // count >= 8, and a remainder shorter than eight joins the chunk before.
    unsafe {
        let mut q = 0;
        while q < count {
            let n = if count - q <= CHUNK + 8 {
                count - q
            } else {
                CHUNK
            };
            let cols = n + kw - 1;
            for ky in 0..kh {
                let src = xp.add(ky * row + q * ic);
                for chunk in (0..ic).step_by(8) {
                    let dst = pp.add((ky * ic + chunk) * STRIDE);
                    let mut col = 0;
                    while col + 8 <= cols {
                        let r: [__m256; 8] = std::array::from_fn(|k| {
                            _mm256_loadu_ps(src.add((col + k) * ic + chunk))
                        });
                        let (t0, t1) = (
                            _mm256_unpacklo_ps(r[0], r[1]),
                            _mm256_unpackhi_ps(r[0], r[1]),
                        );
                        let (t2, t3) = (
                            _mm256_unpacklo_ps(r[2], r[3]),
                            _mm256_unpackhi_ps(r[2], r[3]),
                        );
                        let (t4, t5) = (
                            _mm256_unpacklo_ps(r[4], r[5]),
                            _mm256_unpackhi_ps(r[4], r[5]),
                        );
                        let (t6, t7) = (
                            _mm256_unpacklo_ps(r[6], r[7]),
                            _mm256_unpackhi_ps(r[6], r[7]),
                        );
                        let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                        let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                        let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                        let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                        let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                        let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                        let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                        let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                        let channels = [
                            _mm256_permute2f128_ps::<0x20>(s0, s4),
                            _mm256_permute2f128_ps::<0x20>(s1, s5),
                            _mm256_permute2f128_ps::<0x20>(s2, s6),
                            _mm256_permute2f128_ps::<0x20>(s3, s7),
                            _mm256_permute2f128_ps::<0x31>(s0, s4),
                            _mm256_permute2f128_ps::<0x31>(s1, s5),
                            _mm256_permute2f128_ps::<0x31>(s2, s6),
                            _mm256_permute2f128_ps::<0x31>(s3, s7),
                        ];
                        for (i, v) in channels.into_iter().enumerate() {
                            _mm256_storeu_ps(dst.add(i * STRIDE + col), v);
                        }
                        col += 8;
                    }
                    for col in col..cols {
                        for i in 0..8 {
                            *dst.add(i * STRIDE + col) = *src.add(col * ic + chunk + i);
                        }
                    }
                }
            }
            // One lane per pixel; a group of eight pixels starts at `g`.
            macro_rules! groups {
                ($n:literal, $g:expr) => {{
                    let mut acc = [_mm256_set1_ps(seed); $n];
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let tap = wp.add((ky * kw + kx) * ic);
                            for i in 0..ic {
                                let weight = _mm256_set1_ps(*tap.add(i));
                                let plane = pp.add((ky * ic + i) * STRIDE + $g + kx);
                                for (k, a) in acc.iter_mut().enumerate() {
                                    let v = _mm256_loadu_ps(plane.add(8 * k));
                                    *a = _mm256_add_ps(*a, _mm256_mul_ps(v, weight));
                                }
                            }
                        }
                    }
                    for (k, a) in acc.into_iter().enumerate() {
                        _mm256_storeu_ps(yp.add(q + $g + 8 * k), a);
                    }
                }};
            }
            let mut g = 0;
            while g + 32 <= n {
                groups!(4, g);
                g += 32;
            }
            while g + 8 <= n {
                groups!(1, g);
                g += 8;
            }
            if g < n {
                groups!(1, n - 8);
            }
            q += n;
        }
    }
}

/// RGB windows tiled across pixels; the spatial reduction stays in registers.
pub(crate) fn rgb_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &crate::plan::Op,
    count: usize,
) -> usize {
    // Three pixels leave room for the RGB broadcasts and 24-channel weights
    // without spilling accumulators on AVX2. Other kernels retain four pixels.
    let pixels = match (simd_tier(), op.oc) {
        (_, 8) => 8,
        (_, 24) | (_, 32) => 3,
        _ => 4,
    };
    if count < pixels || op.ic != 3 || !matches!(op.oc, 8 | 16 | 24 | 32) {
        return 0;
    }
    let count = count / pixels * pixels;
    assert!(y.len() >= count * op.oc && w.len() >= op.kh * op.kw * 3 * op.oc);
    assert!(x.len() >= ((op.kh - 1) * op.iw + (op.kw - 1) + (count - 1) * op.sw) * 3 + 3);
    if let Some(b) = bias {
        assert!(b.len() >= op.oc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: complete windows and output sizes checked above, and each
        // specialization is reached only after the matching ISA detection.
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match (simd_tier(), op.oc) {
            (3, 16) => rgb16_avx2(y, x, w, b, op, count),
            (2, 16) => rgb16_avx(y, x, w, b, op, count),
            (3, 24) => rgb24_avx2(y, x, w, b, op, count),
            (2, 24) => rgb24_avx(y, x, w, b, op, count),
            (3, 8) => rgb8_avx2(y, x, w, b, op, count),
            (2, 8) => rgb8_avx(y, x, w, b, op, count),
            (3, 32) => rgb32_avx2(y, x, w, b, op, count),
            (2, 32) => rgb32_avx(y, x, w, b, op, count),
            _ => return 0,
        }
        count
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}
#[cfg(target_arch = "x86_64")]
macro_rules! rgb_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$op:expr,$count:expr,$blocks:expr,$pixels:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        unsafe {
            let (yp, xp, wp, b, op, count) =
                ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b, $op, $count);
            let co = $blocks * 8;
            let mut p = 0;
            while p < count {
                let mut acc = [_mm256_setzero_ps(); $pixels * $blocks];
                if !b.is_null() {
                    for j in 0..$blocks {
                        let seed = _mm256_loadu_ps(b.add(j * 8));
                        for k in 0..$pixels {
                            acc[k * $blocks + j] = seed;
                        }
                    }
                }
                for ky in 0..op.kh {
                    for kx in 0..op.kw {
                        for ic in 0..3 {
                            let src = ((ky * op.iw + kx) + p * op.sw) * 3 + ic;
                            let weight = ((ky * op.kw + kx) * 3 + ic) * co;
                            for j in 0..$blocks {
                                let v = _mm256_loadu_ps(wp.add(weight + j * 8));
                                for k in 0..$pixels {
                                    acc[k * $blocks + j] = fma8!(
                                        $fma,
                                        v,
                                        _mm256_set1_ps(*xp.add(src + k * op.sw * 3)),
                                        acc[k * $blocks + j]
                                    );
                                }
                            }
                        }
                    }
                }
                for k in 0..$pixels {
                    for j in 0..$blocks {
                        _mm256_storeu_ps(yp.add((p + k) * co + j * 8), acc[k * $blocks + j]);
                    }
                }
                p += $pixels;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb16_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 2, 4, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb16_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 2, 4, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb24_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb24_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb8_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 1, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb8_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 1, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb32_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 4, 3, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb32_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 4, 3, false)
}

// Full 2x2 stride-two reductions in 4-pixel x 16-output-channel tiles.
// This is the same load-sharing ratio as the pointwise fast path, but keeps
// accumulators live across all spatial taps instead of spilling after each tap.
pub(crate) fn spatial_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &crate::plan::Op,
    count: usize,
) -> usize {
    if op.ic == 3 {
        return rgb_interior(y, x, w, bias, op, count);
    }
    if op.oc == 1 {
        return single_output_interior(y, x, w, bias, op, count);
    }
    if op.kh != 2 || op.kw != 2 || op.sh != 2 || op.sw != 2 || op.oc == 0 {
        return 0;
    }
    if op.oc.is_multiple_of(8) && !op.oc.is_multiple_of(16) {
        return spatial8_interior(y, x, w, bias, op, count);
    }
    if count < 4 || !op.oc.is_multiple_of(16) {
        return 0;
    }
    let count = count / 4 * 4;
    assert!(y.len() >= count * op.oc && w.len() >= 4 * op.ic * op.oc);
    assert!(x.len() >= (op.iw + (count - 1) * 2 + 2) * op.ic);
    if let Some(b) = bias {
        assert!(b.len() >= op.oc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: all full windows, weights and destinations have been checked;
        // 16-channel panels and four pixels are exact divisors of this call.
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match simd_tier() {
            3 => spatial_avx2(y, x, w, b, op, count),
            2 => spatial_avx(y, x, w, b, op, count),
            _ => return 0,
        }
        count
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// [`spatial_interior`] for `oc % 16 == 8`: eight pixels by one eight-channel
/// block, eight chains. The same bias seed and `ky → kx → channel` sequence per
/// output as the tap fallback, fused only where it fuses.
fn spatial8_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &crate::plan::Op,
    count: usize,
) -> usize {
    let count = count / 8 * 8;
    if count == 0 || simd_tier() < 2 {
        return 0;
    }
    assert!(y.len() >= span(count, op.oc) && w.len() >= span(4 * op.ic, op.oc));
    assert!(x.len() >= (op.iw + (count - 1) * 2 + 2) * op.ic);
    if let Some(b) = bias {
        assert!(b.len() >= op.oc);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: full windows, weights, bias and destinations checked above;
    // AVX detected; oc is a multiple of eight.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        if simd_tier() == 3 {
            spatial8_avx2(y, x, w, b, op, count);
        } else {
            spatial8_avx(y, x, w, b, op, count);
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
macro_rules! spatial8_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$op:expr,$count:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: entered only through spatial8_interior.
        unsafe {
            let (yp, xp, wp, b, op, count) =
                ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b, $op, $count);
            let (ci, co) = (op.ic, op.oc);
            for p in (0..count).step_by(8) {
                for j in (0..co).step_by(8) {
                    let seed = if b.is_null() {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(b.add(j))
                    };
                    let mut acc = [seed; 8];
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let src = xp.add((ky * op.iw + kx + p * 2) * ci);
                            let wt = wp.add((ky * 2 + kx) * ci * co + j);
                            for i in 0..ci {
                                let weight = _mm256_loadu_ps(wt.add(i * co));
                                for (k, a) in acc.iter_mut().enumerate() {
                                    let v = _mm256_set1_ps(*src.add(i + 2 * k * ci));
                                    *a = fma8!($fma, weight, v, *a);
                                }
                            }
                        }
                    }
                    for (k, a) in acc.into_iter().enumerate() {
                        _mm256_storeu_ps(yp.add((p + k) * co + j), a);
                    }
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn spatial8_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    spatial8_body!(y, x, w, b, op, count, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn spatial8_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    spatial8_body!(y, x, w, b, op, count, false)
}

#[cfg(target_arch = "x86_64")]
macro_rules! spatial_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$op:expr,$count:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: private wrappers are entered only through spatial_interior.
        unsafe {
            let (yp, xp, wp, b, op, count) =
                ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b, $op, $count);
            let (ci, co) = (op.ic, op.oc);
            for p in (0..count).step_by(4) {
                for j in (0..co).step_by(16) {
                    let seed0 = if b.is_null() {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(b.add(j))
                    };
                    let seed1 = if b.is_null() {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(b.add(j + 8))
                    };
                    let (mut a0, mut a1, mut a2, mut a3) = (seed0, seed0, seed0, seed0);
                    let (mut b0, mut b1, mut b2, mut b3) = (seed1, seed1, seed1, seed1);
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let src = (ky * op.iw + kx + p * 2) * ci;
                            let wt = (ky * 2 + kx) * ci * co + j;
                            for i in 0..ci {
                                let w0 = _mm256_loadu_ps(wp.add(wt + i * co));
                                let w1 = _mm256_loadu_ps(wp.add(wt + i * co + 8));
                                let x0 = _mm256_set1_ps(*xp.add(src + i));
                                let x1 = _mm256_set1_ps(*xp.add(src + i + 2 * ci));
                                let x2 = _mm256_set1_ps(*xp.add(src + i + 4 * ci));
                                let x3 = _mm256_set1_ps(*xp.add(src + i + 6 * ci));
                                a0 = fma8!($fma, w0, x0, a0);
                                b0 = fma8!($fma, w1, x0, b0);
                                a1 = fma8!($fma, w0, x1, a1);
                                b1 = fma8!($fma, w1, x1, b1);
                                a2 = fma8!($fma, w0, x2, a2);
                                b2 = fma8!($fma, w1, x2, b2);
                                a3 = fma8!($fma, w0, x3, a3);
                                b3 = fma8!($fma, w1, x3, b3);
                            }
                        }
                    }
                    _mm256_storeu_ps(yp.add(p * co + j), a0);
                    _mm256_storeu_ps(yp.add(p * co + j + 8), b0);
                    _mm256_storeu_ps(yp.add((p + 1) * co + j), a1);
                    _mm256_storeu_ps(yp.add((p + 1) * co + j + 8), b1);
                    _mm256_storeu_ps(yp.add((p + 2) * co + j), a2);
                    _mm256_storeu_ps(yp.add((p + 2) * co + j + 8), b2);
                    _mm256_storeu_ps(yp.add((p + 3) * co + j), a3);
                    _mm256_storeu_ps(yp.add((p + 3) * co + j + 8), b3);
                }
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn spatial_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    spatial_body!(y, x, w, b, op, count, true)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn spatial_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &crate::plan::Op,
    count: usize,
) {
    spatial_body!(y, x, w, b, op, count, false)
}

// LLVM 22 can turn a load/blend/store of the same address into VMASKMOVPS.
// Explicit VMOVUPS prevents this expensive masked-store lowering while keeping
// all bits (including positive subnormals and signed zeros) unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn store8_full(dst: *mut f32, value: std::arch::x86_64::__m256) {
    // SAFETY: caller supplies 8 writable floats; unaligned VMOVUPS is allowed.
    // The assembly writes memory, so neither nomem nor readonly is specified.
    unsafe {
        std::arch::asm!("vmovups [{dst}], {value}", dst=in(reg) dst,
            value=in(ymm_reg) value, options(nostack,preserves_flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight textbook definition, used to check every dispatched kernel.
    fn reference_matvec(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| (0..m).map(|i| x[i] * a[i * n + j]).sum())
            .collect()
    }

    fn ramp(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

    #[test]
    fn matvec_matches_the_definition_across_widths() {
        for (m, n) in [(3, 8), (16, 8), (16, 16), (8, 32), (7, 13), (64, 128)] {
            let a = ramp(m * n, 0.1);
            let x = ramp(m, 2.0);
            let mut y = vec![0.0; n];
            matvec_t(&mut y, &a, &x, m, n);
            for (got, want) in y.iter().zip(reference_matvec(&a, &x, m, n)) {
                assert!((got - want).abs() < 1e-4, "{got} vs {want} for {m}x{n}");
            }
        }
    }

    #[test]
    fn pointwise_matches_per_pixel_matvec() {
        for (ci, co) in [(8, 16), (16, 8), (16, 32), (3, 16), (128, 64), (5, 7)] {
            let count = 11; // deliberately not a multiple of any tile width
            let w = ramp(ci * co, 0.3);
            let x = ramp(count * ci, 1.1);
            let bias = ramp(co, 4.0);
            let mut y = vec![0.0; count * co];
            pointwise(&mut y, &w, &x, Some(&bias), count, ci, co);
            for p in 0..count {
                let want = reference_matvec(&w, &x[p * ci..], ci, co);
                for j in 0..co {
                    let got = y[p * co + j];
                    let expect = want[j] + bias[j];
                    assert!(
                        (got - expect).abs() < 1e-4,
                        "pixel {p} channel {j}: {got} vs {expect} for {ci}->{co}"
                    );
                }
            }
        }
    }

    #[test]
    fn irregular_pointwise_widths_match_per_pixel_matvec_exactly() {
        let _scope = DenormalGuard::enter();
        for (ci, co) in [(12, 7), (144, 24), (32, 39), (240, 40), (384, 97)] {
            for count in [1, 4, 6, 13, 50] {
                let w = ramp(ci * co, 0.3);
                let x = ramp(count * ci, 1.1);
                let bias = ramp(co, 4.0);
                let mut got = vec![0.0; count * co];
                pointwise(&mut got, &w, &x, Some(&bias), count, ci, co);
                let mut want = vec![0.0; count * co];
                for p in 0..count {
                    let out = &mut want[p * co..][..co];
                    matvec_t(out, &w, &x[p * ci..][..ci], ci, co);
                    for (v, b) in out.iter_mut().zip(&bias) {
                        *v += b;
                    }
                }
                assert_eq!(bits(&got), bits(&want), "{ci}->{co} count={count}");
            }
        }
    }

    #[test]
    fn depthwise_tap_honours_the_pixel_step() {
        let (c, count, step) = (8, 5, 16);
        let w = ramp(c, 0.9);
        let x = ramp((count - 1) * step + c, 2.5);
        let mut y = vec![1.0; count * c];
        depthwise_tap(&mut y, &x, &w, count, c, step);
        for p in 0..count {
            for ch in 0..c {
                let want = 1.0 + x[p * step + ch] * w[ch];
                assert!((y[p * c + ch] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn prelu_scales_only_negative_values() {
        let c = 8;
        let slope = vec![0.25f32; c];
        let mut y: Vec<f32> = (0..c * 3).map(|i| i as f32 - 12.0).collect();
        let before = y.clone();
        let n = y.len();
        prelu_inplace(&mut y, &slope, n, c);
        for (got, was) in y.iter().zip(&before) {
            let want = if *was < 0.0 { was * 0.25 } else { *was };
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn depthwise_whole_windows_match_nine_taps_exactly() {
        let _scope = DenormalGuard::enter();
        for c in [4, 8, 16, 24, 28, 32, 36, 64, 128] {
            for count in [1, 2, 3, 4, 5, 7, 8, 9, 17] {
                for stride in [1, 2] {
                    let row = ((count - 1) * stride + 3) * c;
                    let x = ramp(row * 3, 0.7);
                    let w = ramp(9 * c, 1.3);
                    let b = ramp(c, 3.2);
                    for bias in [None, Some(b.as_slice())] {
                        let mut want = vec![0.0; count * c];
                        fill_bias(&mut want, bias, count, c);
                        for ky in 0..3 {
                            for kx in 0..3 {
                                depthwise_tap(
                                    &mut want,
                                    &x[ky * row + kx * c..],
                                    &w[(ky * 3 + kx) * c..],
                                    count,
                                    c,
                                    stride * c,
                                );
                            }
                        }
                        let mut got = vec![123.0; count * c + 8];
                        if depthwise_3x3_interior(
                            &mut got,
                            &x,
                            &w,
                            bias,
                            count,
                            c,
                            row,
                            stride * c,
                            None,
                        ) {
                            assert_eq!(
                                bits(&got[..count * c]),
                                bits(&want),
                                "c={c} count={count} stride={stride}"
                            );
                            assert!(got[count * c..].iter().all(|&v| v == 123.0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_windows_of_any_size_match_the_tap_fallback_exactly() {
        let _scope = DenormalGuard::enter();
        for (kh, kw) in [(5, 5), (3, 5), (1, 1), (2, 2), (7, 7)] {
            for c in [8, 16, 24, 32, 64] {
                for count in [1, 3, 7, 8, 9, 16, 17] {
                    for stride in [1, 2] {
                        let row = ((count - 1) * stride + kw) * c;
                        let x = ramp(row * kh, 0.7);
                        let w = ramp(kh * kw * c, 1.3);
                        let b = ramp(c, 3.2);
                        for bias in [None, Some(b.as_slice())] {
                            let mut want = vec![0.0; count * c];
                            fill_bias(&mut want, bias, count, c);
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    depthwise_tap(
                                        &mut want,
                                        &x[ky * row + kx * c..],
                                        &w[(ky * kw + kx) * c..],
                                        count,
                                        c,
                                        stride * c,
                                    );
                                }
                            }
                            let mut got = vec![123.0; count * c + 8];
                            let done = depthwise_window(
                                &mut got,
                                &x,
                                &w,
                                bias,
                                count,
                                c,
                                row,
                                stride * c,
                                kw,
                                (0, kh),
                                (0, kw),
                                None,
                            );
                            assert_eq!(done, simd_tier() >= 2, "{kh}x{kw} c={c}");
                            if done {
                                assert_eq!(
                                    bits(&got[..count * c]),
                                    bits(&want),
                                    "{kh}x{kw} c={c} count={count} stride={stride}"
                                );
                                assert!(got[count * c..].iter().all(|&v| v == 123.0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_border_windows_match_the_tap_fallback_exactly() {
        let _scope = DenormalGuard::enter();
        let (kh, kw) = (5, 5);
        // Every live sub-window a padded border can leave, including none.
        for rows in [(0, 5), (2, 5), (0, 3), (1, 4), (3, 3)] {
            for cols in [(0, 5), (2, 5), (0, 2), (1, 3), (4, 5), (0, 0)] {
                for c in [8, 24, 32, 40, 64] {
                    for (count, step) in [1, 2, 9, 12].into_iter().flat_map(|n| [(n, 1), (n, 2)]) {
                        let (nr, nc) = (rows.1 - rows.0, cols.1 - cols.0);
                        let row = ((count - 1) * step + kw) * c;
                        let mut x = ramp(row * kh, 0.4);
                        x[row + 3 * c + 5] = f32::NAN;
                        let w = ramp(kh * kw * c, 1.9);
                        let b = ramp(c, 2.6);
                        let clips = [None, Some((-0.1, 0.2))];
                        for (bias, clip) in [None, Some(b.as_slice())].into_iter().zip(clips) {
                            let mut want = vec![0.0; count * c];
                            fill_bias(&mut want, bias, count, c);
                            for ky in 0..nr {
                                for kx in 0..nc {
                                    depthwise_tap(
                                        &mut want,
                                        &x[ky * row + kx * c..],
                                        &w[((rows.0 + ky) * kw + cols.0 + kx) * c..],
                                        count,
                                        c,
                                        step * c,
                                    );
                                }
                            }
                            if let Some((low, high)) = clip {
                                clip_inplace(&mut want, low, high);
                            }
                            let mut got = vec![123.0; count * c + 8];
                            let src = if nr > 0 && nc > 0 { &x[..] } else { &x[..0] };
                            let done = depthwise_window(
                                &mut got,
                                src,
                                &w,
                                bias,
                                count,
                                c,
                                row,
                                step * c,
                                kw,
                                rows,
                                cols,
                                clip,
                            );
                            assert_eq!(done, simd_tier() >= 2);
                            if done {
                                assert_eq!(
                                    bits(&got[..count * c]),
                                    bits(&want),
                                    "rows={rows:?} cols={cols:?} c={c} count={count} step={step}"
                                );
                                assert!(got[count * c..].iter().all(|&v| v == 123.0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn whole_5x5_rows_match_the_tap_fallback_exactly() {
        let _scope = DenormalGuard::enter();
        for (ow, iw) in [
            (7, 7),
            (8, 8),
            (14, 14),
            (16, 16),
            (21, 21),
            (24, 24),
            (7, 9),
            (8, 6),
        ] {
            for pl in [0, 1, 2, 3] {
                for rows in [(0, 5), (2, 5), (0, 3), (1, 2)] {
                    for c in [8, 24] {
                        let (nr, row) = (rows.1 - rows.0, iw * c);
                        let mut x = ramp(nr * row, 0.4);
                        x[row / 2 + 5] = f32::NAN;
                        let w = ramp(25 * c, 1.9);
                        let b = ramp(c, 2.6);
                        let clips = [None, Some((-0.1, 0.2))];
                        for (bias, clip) in [None, Some(b.as_slice())].into_iter().zip(clips) {
                            let mut want = vec![0.0; ow * c];
                            fill_bias(&mut want, bias, ow, c);
                            for ox in 0..ow {
                                for ky in 0..nr {
                                    for kx in 0..5 {
                                        let ix = ox as isize - pl + kx as isize;
                                        if (0..iw as isize).contains(&ix) {
                                            depthwise_tap(
                                                &mut want[ox * c..],
                                                &x[ky * row + ix as usize * c..],
                                                &w[((rows.0 + ky) * 5 + kx) * c..],
                                                1,
                                                c,
                                                c,
                                            );
                                        }
                                    }
                                }
                            }
                            if let Some((low, high)) = clip {
                                clip_inplace(&mut want, low, high);
                            }
                            let mut got = vec![123.0; ow * c + 8];
                            let done = depthwise_row5(
                                &mut got, &x, &w, bias, ow, iw, pl, c, row, rows, clip,
                            );
                            assert_eq!(done, simd_tier() >= 2);
                            if done {
                                assert_eq!(
                                    bits(&got[..ow * c]),
                                    bits(&want),
                                    "ow={ow} iw={iw} pl={pl} rows={rows:?} c={c}"
                                );
                                assert!(got[ow * c..].iter().all(|&v| v == 123.0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn single_output_windows_sum_like_the_tap_fallback() {
        let _scope = DenormalGuard::enter();
        for (kh, kw, ic) in [(3, 3, 8), (3, 3, 16), (2, 2, 24), (1, 1, 8)] {
            for stride in [1, 2] {
                for count in [8, 9, 17, 24, 40, 72, 137] {
                    let iw = (count - 1) * stride + kw;
                    let op: crate::plan::Op = serde_json::from_value(serde_json::json!({
                        "kind": "conv", "x": 0, "y": 1, "ih": kh, "iw": iw, "ic": ic,
                        "oh": 1, "ow": count, "oc": 1, "kh": kh, "kw": kw,
                        "sh": stride, "sw": stride,
                    }))
                    .unwrap();
                    let x = ramp(kh * iw * ic, 0.4);
                    let w = ramp(kh * kw * ic, 1.7);
                    let mut got = vec![7.0; count + 1];
                    let done = spatial_interior(&mut got, &x, &w, Some(&[0.3]), &op, count);
                    let planar = stride == 1 && kh * ic <= PLANES;
                    let covered = if planar { count } else { count / 8 * 8 };
                    assert_eq!(done, if simd_tier() >= 2 { covered } else { 0 });
                    for (p, &value) in got.iter().enumerate().take(done) {
                        let mut want = 0.3f32;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                for i in 0..ic {
                                    let src = (ky * iw + p * stride + kx) * ic + i;
                                    want += x[src] * w[(ky * kw + kx) * ic + i];
                                }
                            }
                        }
                        assert_eq!(value.to_bits(), want.to_bits(), "{kh}x{kw} ic={ic} p={p}");
                    }
                    assert!(got[done..].iter().all(|&v| v == 7.0));
                }
            }
        }
    }

    #[test]
    fn rgb_windows_match_the_tap_fallback_exactly() {
        let _scope = DenormalGuard::enter();
        for oc in [8, 16, 24, 32] {
            for (k, stride) in [(3, 2), (5, 2), (3, 1)] {
                for count in [3, 4, 7, 8, 13, 17] {
                    let iw = (count - 1) * stride + k;
                    let op: crate::plan::Op = serde_json::from_value(serde_json::json!({
                        "kind": "conv", "x": 0, "y": 1, "ih": k, "iw": iw, "ic": 3,
                        "oh": 1, "ow": count, "oc": oc, "kh": k, "kw": k,
                        "sh": stride, "sw": stride,
                    }))
                    .unwrap();
                    let x = ramp(k * iw * 3, 0.8);
                    let w = ramp(k * k * 3 * oc, 2.1);
                    let b = ramp(oc, 0.2);
                    let mut got = vec![7.0; count * oc + 1];
                    let done = spatial_interior(&mut got, &x, &w, Some(&b), &op, count);
                    for p in 0..done {
                        let mut want = b.clone();
                        for ky in 0..k {
                            for kx in 0..k {
                                let src = (ky * iw + p * stride + kx) * 3;
                                let tap = &w[(ky * k + kx) * 3 * oc..][..3 * oc];
                                matvec_t_acc(&mut want, tap, &x[src..src + 3], 3, oc);
                            }
                        }
                        assert_eq!(
                            bits(&got[p * oc..][..oc]),
                            bits(&want),
                            "oc={oc} {k}x{k} p={p}"
                        );
                    }
                    assert!(got[done * oc..].iter().all(|&v| v == 7.0));
                }
            }
        }
    }

    #[test]
    fn pointwise_clamps_as_it_stores_like_a_separate_clip() {
        let _scope = DenormalGuard::enter();
        for (ci, co) in [
            (16, 8),
            (5, 16),
            (24, 32),
            (32, 64),
            (9, 24),
            (12, 7),
            (3, 40),
        ] {
            for count in [1, 5, 13, 30] {
                let w: Vec<f32> = ramp(ci * co, 0.3).iter().map(|v| v * 9.0).collect();
                let mut x = ramp(count * ci, 1.1);
                x[(count / 2) * ci + 1] = f32::NAN; // one pixel turns every output into NaN
                let bias = ramp(co, 4.0);
                let panels = co % 16 == 0 && simd_tier() >= 2;
                let packed: Vec<f32> = if panels {
                    (0..co / 16)
                        .flat_map(|j| {
                            (0..ci).flat_map(move |i| (0..16).map(move |k| (i, j * 16 + k)))
                        })
                        .map(|(i, j)| w[i * co + j])
                        .collect()
                } else {
                    w.clone()
                };
                let mut want = vec![0.0; count * co];
                pointwise(&mut want, &w, &x, Some(&bias), count, ci, co);
                clip_inplace(&mut want, 0.0, 6.0);
                let mut got = vec![0.0; count * co];
                let clip = Some((0.0, 6.0));
                pointwise_with(
                    &mut got,
                    &packed,
                    &x,
                    Some(&bias),
                    count,
                    ci,
                    co,
                    panels,
                    clip,
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{ci}->{co} count={count} panels={panels}"
                );
            }
        }
    }

    #[test]
    fn bilinear_rows_blend_like_the_scalar_loop() {
        for c in [8, 16, 32] {
            let mut top = ramp(9 * c, 0.2);
            let bottom = ramp(9 * c, 1.4);
            top[3 * c + 2] = f32::NAN;
            let cols: Vec<(usize, usize, f32)> = (0..17)
                .map(|k| (k / 2, (k / 2 + 1).min(8), 0.25 + 0.5 * (k % 2) as f32))
                .collect();
            let dy = 0.375;
            let mut got = vec![0.0; cols.len() * c];
            if !bilinear_row(&mut got, &top, &bottom, &cols, dy, c) {
                assert!(simd_tier() < 2);
                continue;
            }
            let mut want = vec![0.0; cols.len() * c];
            for (&(x0, x1, dx), o) in cols.iter().zip(want.chunks_exact_mut(c)) {
                let (wa, wb) = ((1.0 - dy) * (1.0 - dx), dy * (1.0 - dx));
                let (wd, we) = ((1.0 - dy) * dx, dy * dx);
                for k in 0..c {
                    o[k] = top[x0 * c + k] * wa
                        + bottom[x0 * c + k] * wb
                        + top[x1 * c + k] * wd
                        + bottom[x1 * c + k] * we;
                }
            }
            assert_eq!(bits(&got), bits(&want), "c={c}");
        }
    }

    #[test]
    fn two_by_two_windows_match_the_tap_fallback_exactly() {
        let _scope = DenormalGuard::enter();
        for (ci, co) in [(8, 8), (32, 8), (16, 24), (8, 16), (64, 32)] {
            for count in [4, 8, 9, 17, 64] {
                let iw = (count - 1) * 2 + 2;
                let op: crate::plan::Op = serde_json::from_value(serde_json::json!({
                    "kind": "conv", "x": 0, "y": 1, "ih": 2, "iw": iw, "ic": ci,
                    "oh": 1, "ow": count, "oc": co, "kh": 2, "kw": 2, "sh": 2, "sw": 2,
                }))
                .unwrap();
                let mut x = ramp(2 * iw * ci, 0.6);
                x[ci + 3] = f32::NAN;
                let w = ramp(4 * ci * co, 1.7);
                let b = ramp(co, 0.9);
                let mut got = vec![7.0; count * co + 1];
                let done = spatial_interior(&mut got, &x, &w, Some(&b), &op, count);
                for p in 0..done {
                    let mut want = b.clone();
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let src = (ky * iw + p * 2 + kx) * ci;
                            let tap = &w[(ky * 2 + kx) * ci * co..][..ci * co];
                            matvec_t_acc(&mut want, tap, &x[src..src + ci], ci, co);
                        }
                    }
                    assert_eq!(bits(&got[p * co..][..co]), bits(&want), "{ci}->{co} p={p}");
                }
                assert!(got[done * co..].iter().all(|&v| v == 7.0));
            }
        }
    }

    #[test]
    fn one_by_one_depthwise_matches_the_one_tap_window_exactly() {
        let _scope = DenormalGuard::enter();
        for c in [8, 16, 32, 64] {
            for count in [1, 7, 64] {
                let mut x = ramp(count * c, 0.3);
                x[c / 2] = f32::NAN;
                x[count * c - 1] = -0.0;
                let w = ramp(c, 1.2);
                let b = ramp(c, 2.2);
                for (bias, clip) in [(None, None), (Some(b.as_slice()), Some((-0.1, 0.2)))] {
                    let mut want = vec![0.0; count * c];
                    let row = count * c;
                    if !depthwise_window(
                        &mut want,
                        &x,
                        &w,
                        bias,
                        count,
                        c,
                        row,
                        c,
                        1,
                        (0, 1),
                        (0, 1),
                        clip,
                    ) {
                        assert!(simd_tier() < 2);
                        continue;
                    }
                    let mut got = vec![0.0; count * c];
                    assert!(depthwise_1x1(&mut got, &x, &w, bias, count, c, clip));
                    assert_eq!(bits(&got), bits(&want), "c={c} count={count}");
                }
            }
        }
    }

    #[test]
    fn add_then_clip_in_one_pass_matches_two_passes() {
        let special = [
            f32::NAN,
            -0.0,
            0.0,
            6.0,
            -1.0,
            7.5,
            f32::INFINITY,
            3.25,
            -f32::INFINITY,
        ];
        for n in [1, 7, 8, 13, 45] {
            let a: Vec<f32> = (0..n).map(|i| special[i % special.len()]).collect();
            let b: Vec<f32> = (0..n)
                .map(|i| special[(i * 5 + 2) % special.len()] * 0.5)
                .collect();
            let mut want = vec![0.0; n];
            add(&mut want, &a, &b, n);
            clip_inplace(&mut want, 0.0, 6.0);
            let mut got = vec![0.0; n];
            add_with(&mut got, &a, &b, n, Some((0.0, 6.0)));
            assert_eq!(bits(&got), bits(&want), "n={n}");
        }
    }

    #[test]
    fn prelu_in_one_pass_matches_copy_then_in_place() {
        let special = [
            f32::NAN,
            -f32::NAN,
            -0.0,
            0.0,
            -1.5,
            2.0,
            f32::NEG_INFINITY,
            -1e-40,
            3.0,
        ];
        for c in [8, 16, 12, 64, 5] {
            for pixels in [1, 3] {
                let n = c * pixels + 3;
                let x: Vec<f32> = (0..n).map(|i| special[i % special.len()]).collect();
                let slope = ramp(c, 0.7);
                let mut want = x.clone();
                prelu_inplace(&mut want, &slope, n, c);
                let mut got = vec![9.0; n];
                prelu(&mut got, &x, &slope, n, c);
                assert_eq!(bits(&got), bits(&want), "c={c} n={n}");
            }
        }
    }

    #[test]
    fn vector_clip_is_clamp_bit_for_bit() {
        let special = [
            f32::NAN,
            -f32::NAN,
            -0.0,
            0.0,
            6.0,
            -1.0,
            7.5,
            f32::INFINITY,
            3.25,
        ];
        let values: Vec<f32> = (0..45).map(|i| special[i % special.len()]).collect();
        let mut got = values.clone();
        clip_inplace(&mut got, 0.0, 6.0);
        for (&g, &v) in got.iter().zip(&values) {
            assert_eq!(g.to_bits(), v.clamp(0.0, 6.0).to_bits(), "{v}");
        }
    }

    #[test]
    fn pointwise_tiles_preserve_bias_order_and_tail_bits() {
        let _scope = DenormalGuard::enter();
        let tier = simd_tier();
        for ci in [1, 3, 8, 13, 16, 32] {
            for co in [1, 4, 6, 8, 16, 24, 28, 32, 36, 42, 48, 88] {
                for count in [1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 17, 24, 32, 50] {
                    let x = ramp(count * ci, 0.6);
                    let w = ramp(ci * co, 0.9);
                    let b = ramp(co, 2.1);
                    for bias in [None, Some(b.as_slice())] {
                        let mut got = vec![123.0; count * co + 8];
                        pointwise(&mut got, &w, &x, bias, count, ci, co);
                        let tiled = if tier >= 2 && co % 16 == 0 {
                            count / 4 * 4
                        } else if tier >= 2 && co == 8 {
                            count / 8 * 8
                        } else {
                            0
                        };
                        for p in 0..count {
                            for j in 0..co {
                                let before = p < tiled;
                                let mut acc = if before {
                                    bias.map_or(0.0, |b| b[j])
                                } else {
                                    0.0
                                };
                                for i in 0..ci {
                                    let (a, v) = (w[i * co + j], x[p * ci + i]);
                                    acc = if tier == 3 && j < co / 8 * 8 {
                                        a.mul_add(v, acc)
                                    } else {
                                        acc + a * v
                                    };
                                }
                                if !before {
                                    if let Some(b) = bias {
                                        acc += b[j];
                                    }
                                }
                                assert_eq!(
                                    got[p * co + j].to_bits(),
                                    acc.to_bits(),
                                    "ci={ci} co={co} n={count} p={p} ch={j}"
                                );
                            }
                        }
                        assert!(got[count * co..].iter().all(|&v| v == 123.0));
                    }
                }
            }
        }
    }

    #[test]
    fn activation_fusions_preserve_bits_including_partial_pixels() {
        let special = [
            -2.0,
            -0.0,
            0.0,
            1.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            1.0e-20,
            -1.0e-20,
        ];
        for c in [1, 4, 8, 13, 16, 24, 28, 32, 64, 128, 256] {
            let slope = ramp(c, 1.1);
            for n in [0, 1, c - 1, c, c + 1, 3 * c + 7] {
                let x: Vec<_> = (0..n).map(|i| special[i % special.len()]).collect();
                let mut want = x.clone();
                for i in 0..n {
                    if want[i] < 0.0 {
                        want[i] *= slope[i % c];
                    }
                }
                let mut got = x.clone();
                got.extend([123.0; 8]);
                prelu_inplace(&mut got, &slope, n, c);
                assert_eq!(bits(&got[..n]), bits(&want), "prelu c={c} n={n}");
                assert!(got[n..].iter().all(|&v| v == 123.0));
                let x2 = ramp(n, 0.1);
                let mut fused = vec![123.0; n + 8];
                let mut separate = vec![0.0; n];
                add(&mut separate, &x, &x2, n);
                prelu_inplace(&mut separate, &slope, n, c);
                add_prelu(&mut fused, &x, &x2, &slope, n, c);
                assert_eq!(bits(&fused[..n]), bits(&separate), "add/prelu c={c} n={n}");
                assert!(fused[n..].iter().all(|&v| v == 123.0));
                // AVX tiers clamp whole vectors with MAXPS, which returns its
                // second operand (+0) for -0 and NaN; scalar code and the
                // vector tail keep both.
                let vector_end = if simd_tier() >= 2 { n / 8 * 8 } else { 0 };
                let old: Vec<f32> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match (i < vector_end, v > 0.0, v < 0.0) {
                        (true, true, _) | (false, _, false) => v,
                        _ => 0.0,
                    })
                    .collect();
                relu(&mut fused, &x, n);
                assert_eq!(bits(&fused[..n]), bits(&old), "relu c={c} n={n}");
            }
        }
    }

    #[test]
    fn spatial_tiles_match_original_tap_accumulations() {
        let _scope = DenormalGuard::enter();
        for (ci, co, kh, kw) in [
            (3, 16, 3, 3),
            (3, 24, 5, 5),
            (16, 16, 2, 2),
            (32, 64, 2, 2),
            (128, 128, 2, 2),
        ] {
            for count in [1, 3, 4, 5, 6, 7, 8, 9, 12, 13] {
                let iw = (count - 1) * 2 + kw;
                let op: crate::plan::Op = serde_json::from_value(serde_json::json!({
                    "kind":"conv","x":0,"y":1,"ih":kh,"iw":iw,"ic":ci,
                    "oh":1,"ow":count,"oc":co,"kh":kh,"kw":kw,"sh":2,"sw":2
                }))
                .unwrap();
                let x = ramp(kh * iw * ci, 1.2);
                let w = ramp(kh * kw * ci * co, 2.4);
                let b = ramp(co, 3.2);
                for bias in [None, Some(b.as_slice())] {
                    let mut got = vec![123.0; count * co + 8];
                    let done = spatial_interior(&mut got, &x, &w, bias, &op, count);
                    let mut want = vec![0.0; done * co];
                    fill_bias(&mut want, bias, done, co);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            for p in 0..done {
                                matvec_t_acc(
                                    &mut want[p * co..(p + 1) * co],
                                    &w[(ky * kw + kx) * ci * co..][..ci * co],
                                    &x[(ky * iw + kx + p * 2) * ci..][..ci],
                                    ci,
                                    co,
                                );
                            }
                        }
                    }
                    assert_eq!(
                        bits(&got[..done * co]),
                        bits(&want),
                        "spatial {ci}->{co}, count={count}"
                    );
                    assert!(got[done * co..].iter().all(|&v| v == 123.0));
                }
            }
        }
    }

    #[test]
    fn public_kernel_span_overflow_panics_before_any_pointer_access() {
        assert!(std::panic::catch_unwind(|| {
            depthwise_tap(&mut [], &[], &[1.0; 8], usize::MAX / 8 + 1, 8, 8);
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            pointwise(&mut [], &[], &[], None, usize::MAX / 8 + 1, 0, 8);
        })
        .is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn floating_point_scope_restores_mxcsr_even_during_unwind() {
        let original = read_mxcsr();
        let caller = original & !0x8040;
        // SAFETY: clear only two supported control bits; restore on exit.
        unsafe {
            write_mxcsr(caller);
        }
        {
            let _scope = DenormalGuard::enter();
            assert_eq!(read_mxcsr() & 0x8040, 0x8040);
        }
        assert_eq!(read_mxcsr(), caller);
        let _ = std::panic::catch_unwind(|| {
            let _scope = DenormalGuard::enter();
            panic!("test unwind");
        });
        assert_eq!(read_mxcsr(), caller);
        unsafe {
            write_mxcsr(original);
        }
    }
}

#[cfg(test)]
mod pooling_regression {
    use super::*;
    #[test]
    fn complete_pool_preserves_each_tiers_nan_zero_tail_and_tap_order() {
        let specials = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc00001),
            f32::from_bits(0xffc00042),
            1.0,
            -2.0,
        ];
        for c in [1, 3, 7, 8, 9, 16, 24, 28, 32, 36, 42, 64, 128] {
            for (oh, ow) in [(1, 1), (2, 3), (3, 5)] {
                for seed in 0..8 {
                    let x: Vec<f32> = (0..oh * ow * c * 4)
                        .map(|i| specials[(i * 5 + i / 7 + seed) % specials.len()])
                        .collect();
                    let mut expected = vec![0.0f32; oh * ow * c];
                    for r in 0..oh {
                        for col in 0..ow {
                            let at = (r * ow + col) * c;
                            let top = (2 * r * 2 * ow + 2 * col) * c;
                            expected[at..at + c].copy_from_slice(&x[top..top + c]);
                            for off in [c, 2 * ow * c, 2 * ow * c + c] {
                                max_into(
                                    &mut expected[at..at + c],
                                    &x[top + off..top + off + c],
                                    c,
                                );
                            }
                        }
                    }
                    let mut got = vec![f32::NAN; expected.len() + 3];
                    if maxpool_2x2(&x, &mut got, oh, ow, c) {
                        for (a, b) in expected.iter().zip(&got) {
                            assert_eq!(a.to_bits(), b.to_bits(), "channels={c}");
                        }
                        assert!(got[expected.len()..].iter().all(|v| v.is_nan()));
                    }
                }
            }
        }
    }
}
