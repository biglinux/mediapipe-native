//! Time a plan on this machine, and leave behind what a parity check needs.
//!
//! Prints the instruction set in use, the memory held, the per-frame latency
//! spread, and how many times the frame loop touched the allocator — which
//! should be zero. With `--profile` it also breaks the frame down by operation
//! shape, which is how you find out where a slow machine is actually spending
//! its time rather than guessing.
//!
//! The input is pseudo-random but fixed, and is written next to the plan along
//! with the results, so `tools/check_parity.py` can feed the same pixels to a
//! reference engine and compare.
//!
//! Usage: `mpbench PLAN [FRAMES] [--profile] [--json]`
//!
//! PLAN is a `.mpplan` file or a directory holding `graph.json` and
//! `weights.bin`. Results are written beside it either way.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use mediapipe_native::{ops, Model};

/// Counts allocations while armed. A frame that allocates can stall on the
/// allocator's own locks, which is exactly the kind of rare long frame that
/// shows up as a dropped one.
struct Counting;
static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: every method forwards to the system allocator unchanged; the counter
// only observes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!(
            "usage: mpbench PLAN [FRAMES] [--profile] [--json]   (PLAN: a .mpplan file or a plan directory)"
        );
        std::process::exit(2);
    };
    let target = Path::new(dir);
    // Results are written beside the plan, whether that is a directory of
    // parts or a single packed file.
    let dir = if target.is_dir() {
        target.to_path_buf()
    } else {
        target.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let mut frames = None;
    for value in args.iter().skip(1) {
        if matches!(value.as_str(), "--profile" | "--json") {
            continue;
        }
        match value.parse::<usize>() {
            Ok(n) if n > 0 && frames.is_none() => frames = Some(n),
            _ => {
                eprintln!("mpbench: expected one positive FRAMES count, --profile or --json");
                std::process::exit(2);
            }
        }
    }
    let frames = frames.unwrap_or(200);
    let profile = args.iter().any(|a| a == "--profile");

    let mut model = match Model::load(target) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mpbench: {e}");
            std::process::exit(1);
        }
    };

    let (h, w, c) = model.input_shape();
    let image = pseudo_image(h * w * c);
    dump(&dir.join("bench_input.bin"), &image);

    model.input_mut().copy_from_slice(&image);
    ARMED.store(true, Ordering::Relaxed);
    model.run();
    ARMED.store(false, Ordering::Relaxed);
    let first_frame_allocations = ALLOCATIONS.swap(0, Ordering::Relaxed);
    for _ in 1..20 {
        model.input_mut().copy_from_slice(&image);
        model.run();
    }

    // Reserved before arming, so the counter reports the model's allocations
    // and not this program's bookkeeping.
    let mut times = Vec::with_capacity(frames);
    let cpu_start = process_cpu_seconds();
    ARMED.store(true, Ordering::Relaxed);
    for _ in 0..frames {
        model.input_mut().copy_from_slice(&image);
        let started = Instant::now();
        model.run();
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    ARMED.store(false, Ordering::Relaxed);
    let cpu_elapsed = process_cpu_seconds() - cpu_start;
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);

    for index in 0..model.output_count() {
        if model.output(index).iter().any(|v| !v.is_finite()) {
            eprintln!("mpbench: non-finite value in output {index}");
            std::process::exit(1);
        }
        let name = sanitise(model.output_name(index));
        dump(
            &dir.join(format!("bench_out_{index}_{name}.bin")),
            model.output(index),
        );
    }

    times.sort_by(f64::total_cmp);
    let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];

    println!(
        "instruction set  tier {} (3 AVX2+FMA, 2 AVX, 1 SSE4.1, 0 scalar)",
        ops::simd_tier()
    );
    println!("operations       {}", model.ops().len());
    println!(
        "memory           {:.2} MB weights + {:.2} MB scratch, process PSS {:.1} MiB",
        model.weight_floats() as f64 * 4.0 / 1e6,
        model.scratch_bytes() as f64 / 1e6,
        resident_mib()
    );
    println!("frame loop       {allocations} allocations");
    println!(
        "latency ms       p50 {:.6}   p95 {:.6}   p99 {:.6}   best {:.6}   worst {:.6}   ({frames} frames)",
        at(0.50),
        at(0.95),
        at(0.99),
        times[0],
        times[times.len() - 1],
    );

    println!("cpu ms/frame     {:.6}", cpu_elapsed * 1e3 / frames as f64);
    if allocations != 0 || first_frame_allocations != 0 {
        eprintln!("mpbench: allocation gate failed");
        std::process::exit(1);
    }
    if args.iter().any(|a| a == "--json") {
        println!(
            "{}",
            serde_json::json!({"tier":ops::simd_tier(), "frames":frames,
            "p50_ms":at(0.5),"p95_ms":at(0.95),"p99_ms":at(0.99),"samples_ms":times,
            "cpu_ms_per_frame":cpu_elapsed*1e3/frames as f64,"allocations":allocations,
            "weights_bytes":model.weight_floats()*4,"scratch_bytes":model.scratch_bytes(), "warmup_frames":20,
            "first_frame_allocations":first_frame_allocations, "samples_sorted":true, "pss_mib":resident_mib()})
        );
    }
    if profile {
        report_profile(&mut model, &image);
    }
}

/// A fixed, repeatable image. The pixel values do not change the cost; the
/// shape does. Being repeatable is what lets a reference engine be fed the
/// same thing.
fn pseudo_image(len: usize) -> Vec<f32> {
    let mut state: u32 = 0x2545_f491;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn dump(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::File::create(path).and_then(|mut f| f.write_all(&bytes)) {
        eprintln!("mpbench: could not write {}: {e}", path.display());
        std::process::exit(1);
    }
}

fn sanitise(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_owned()
}

fn resident_mib() -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/smaps_rollup") else {
        return 0.0;
    };
    let kib: u64 = text
        .lines()
        .filter(|l| l.starts_with("Pss:"))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        .sum();
    kib as f64 / 1024.0
}

/// Per-operation timing, grouped by the shape the kernel actually sees.
///
/// The timer is read around every operation, so the total here runs a little
/// above the frame latency above. It is for comparing rows, not for quoting.
fn report_profile(model: &mut Model, image: &[f32]) {
    const ROUNDS: usize = 40;
    let mut spent: BTreeMap<String, (f64, usize, u64)> = BTreeMap::new();

    let mut per_op = Vec::new();
    for _ in 0..ROUNDS {
        model.input_mut().copy_from_slice(image);
        model.run_timed(&mut per_op);
        let ops = model.ops();
        let mut i = 0;
        while i < ops.len() {
            let covered = model.scheduled_len(i);
            let mut label = describe(&ops[i]);
            if model.has_fused_successor(i) {
                label.push_str(" + add (row-local)");
            } else if covered > 1 {
                for op in &ops[i + 1..i + covered] {
                    label.push_str(" + ");
                    label.push_str(&describe(op));
                }
                label.push_str(" (fused)");
            }
            let row = spent.entry(label).or_insert((0.0, 0, 0));
            row.0 += per_op[i];
            row.1 += 1;
            row.2 += ops[i..i + covered].iter().map(macs).sum::<u64>();
            i += covered;
        }
    }

    let mut rows: Vec<_> = spent.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    let total: f64 = rows.iter().map(|r| r.1 .0).sum();
    println!(
        "\nper-operation self time, ms per frame (total {:.2})",
        total / ROUNDS as f64
    );
    for (label, (ms, count, macs)) in rows.iter().take(16) {
        let per_frame = ms / ROUNDS as f64;
        let rate = if *macs > 0 {
            format!(
                "{:>5.1} G/s",
                (*macs as f64 / ROUNDS as f64) / (per_frame * 1e-3) / 1e9
            )
        } else {
            "         ".to_owned()
        };
        println!(
            "  {per_frame:>7.3}  {:>5.1}%  x{:<3} {rate}  {label}",
            100.0 * ms / total,
            count / ROUNDS
        );
    }
}

fn describe(op: &mediapipe_native::plan::Op) -> String {
    use mediapipe_native::Kind;
    match op.kind {
        Kind::Conv => format!(
            "conv {}x{} stride {} {:>4}->{:<4} at {}x{}",
            op.kh, op.kw, op.sh, op.ic, op.oc, op.oh, op.ow
        ),
        Kind::Dwconv => format!(
            "depthwise {}x{} stride {} {:>4} at {}x{}",
            op.kh, op.kw, op.sh, op.oc, op.oh, op.ow
        ),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn macs(op: &mediapipe_native::plan::Op) -> u64 {
    use mediapipe_native::Kind;
    let spatial = (op.oh * op.ow) as u64;
    match op.kind {
        Kind::Conv => (op.kh * op.kw * op.ic * op.oc) as u64 * spatial,
        Kind::Dwconv => (op.kh * op.kw * op.oc) as u64 * spatial,
        _ => 0,
    }
}

// Linux process CPU clock: counts CPU consumed, not time descheduled by the host.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn process_cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    // SAFETY: `ts` is writable and has the Linux 64-bit timespec ABI.
    let rc = unsafe { clock_gettime(2, &mut ts) };
    if rc == 0 {
        ts.sec as f64 + ts.nsec as f64 * 1e-9
    } else {
        f64::NAN
    }
}
#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn process_cpu_seconds() -> f64 {
    f64::NAN
}
