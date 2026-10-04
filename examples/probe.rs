//! Deterministic full-output corpus for binary-to-binary differential testing.
use mediapipe_native::Model;
use std::{
    io::{BufWriter, Write},
    path::Path,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    // Non-finite outputs stop the corpus unless asked for: a model given a
    // degenerate input (every landmark in one place) may legitimately yield them.
    let allow_nonfinite = args.len() == 4 && args[3] == "--allow-nonfinite";
    if args.len() != 3 && !allow_nonfinite {
        return Err("usage: probe PLAN OUTPUT_DIRECTORY [--allow-nonfinite]".into());
    }
    let dir = Path::new(&args[2]);
    std::fs::create_dir_all(dir)?;
    let mut model = Model::load(Path::new(&args[1]))?;
    let (h, w, c) = model.input_shape();
    let mut image = vec![0.0; h * w * c];
    for case in 0..24u32 {
        let mut state = if case == 0 {
            0x2545_f491u32
        } else {
            case.wrapping_mul(0x9e37_79b9)
        };
        for (i, v) in image.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *v = match case {
                1 => 0.0,
                2 => 1.0,
                3 => 0.5,
                4 => ((i / c / w + i / c % w) % 2) as f32,
                5 => (i / c % w) as f32 / w as f32,
                6 => {
                    if i / c == h * w / 2 {
                        1.0
                    } else {
                        0.0
                    }
                }
                7 => 1.0e-30,
                _ => (state >> 8) as f32 / 16_777_216.0,
            };
        }
        model.input_mut().copy_from_slice(&image);
        model.run();
        for out in 0..model.output_count() {
            let file = std::fs::File::create(dir.join(format!("case{case:02}-out{out}.bin")))?;
            let mut file = BufWriter::new(file);
            for v in model.output(out) {
                if !v.is_finite() && !allow_nonfinite {
                    return Err(format!("nonfinite case {case} output {out}").into());
                }
                file.write_all(&v.to_le_bytes())?;
            }
            file.flush()?;
        }
    }
    println!(
        "24 cases; {} outputs; tier {}",
        model.output_count(),
        mediapipe_native::ops::simd_tier()
    );
    Ok(())
}
