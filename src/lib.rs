//! Run MediaPipe vision models on the CPU with nothing underneath but Rust.
//!
//! The models this targets — the MediaPipe face detector and face mesh, and
//! the hand, pose and holistic landmarkers — are built from a handful of
//! operations repeated a hundred times: a 1x1 convolution, a depthwise
//! convolution, an activation, an addition. A general
//! inference engine also supports many other graphs and execution strategies.
//! This crate specializes those few operations and reserves its tensor storage
//! at load time; no inference library is linked.
//!
//! # Using it
//!
//! ```no_run
//! use std::path::Path;
//! use mediapipe_native::Model;
//!
//! let mut model = Model::load(Path::new("plans/face_landmarks"))?;
//! let (h, w, c) = model.input_shape();
//! model.input_mut().copy_from_slice(&vec![0.0; h * w * c]);
//! model.run();
//! let landmarks = model.output(0);
//! # Ok::<(), mediapipe_native::Error>(())
//! ```
//!
//! Pixels are expected as `f32`, laid out row by row, and within a row pixel
//! by pixel, with the colour channels of one pixel next to each other. That is
//! the same order MediaPipe's own models use.
//!
//! # Getting a plan
//!
//! A plan is produced offline: `tools/export_tflite.py` converts the `.tflite`
//! files inside MediaPipe task bundles (format 1 for the face models, format 3
//! for the others), and `tools/export_graph.py` converts an ONNX export of the
//! face models.
//! Python, ONNX and TensorFlow are needed for that step only; none of them is
//! a dependency of this crate or of anything that links it.
//!
//! # What it does not do
//!
//! One thread, one image at a time, 32-bit floats throughout. There is no
//! thread pool, no batching and no quantisation. Performance comparisons must
//! use the same CPU, graph, precision and thread budget; docs/performance.md
//! describes the method and the current measurements.

mod extra_ops;
pub mod ops;
pub mod plan;

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::path::Path;
use std::ptr::NonNull;

pub use plan::{Error, Kind, Plan};

/// Cache-line-aligned allocation base. Individual NHWC rows may be unaligned;
/// all generic SIMD paths therefore continue to use unaligned loads/stores.
struct Buffer {
    ptr: NonNull<f32>,
    len: usize,
}

impl Buffer {
    fn new(len: usize) -> Self {
        let layout = Self::layout(len);
        // SAFETY: `layout` has a non-zero size and a valid power-of-two align.
        let raw = unsafe { alloc_zeroed(layout) }.cast::<f32>();
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(
            len.max(1).checked_mul(4).expect("buffer size overflows"),
            64,
        )
        .expect("buffer size overflows")
    }

    /// # Safety
    /// No mutable view of this buffer may exist for the returned lifetime.
    unsafe fn get(&self) -> &[f32] {
        // SAFETY: the allocation holds `len` initialised floats; the caller
        // promises no mutable view overlaps this one.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// # Safety
    /// No other view of this buffer may exist for the returned lifetime.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut [f32] {
        // SAFETY: the allocation holds `len` initialised floats; the caller
        // promises this is the only live view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated by `Buffer::new` with exactly this layout.
        unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.len)) };
    }
}

/// A loaded model with its scratch memory already reserved.
///
/// Running one frame allocates nothing.
pub struct Model {
    plan: Plan,
    /// Held in cache-line-aligned storage, not a plain vector: the exporter
    /// starts every weight block on a 64-byte boundary, and that only buys
    /// aligned vector loads if the block the offsets are measured from is
    /// aligned too.
    weights: Buffer,
    pool: Vec<Buffer>,
    packed_heads: Vec<bool>,
    /// Pointwise layers whose weights `pack_panels` reordered.
    panels: Vec<bool>,
    row_pairs: Vec<bool>,
    /// Depthwise layers whose rows run just ahead of a row-local pair, see
    /// [`can_triple`].
    row_triples: Vec<bool>,
    /// Operations covered by a fused [Padc→]Add→Relu starting here, or 0.
    add_relus: Vec<u8>,
    /// Row-local pairs whose convolution output only their add reads: its
    /// rows go through `rows`, one group at a time, and never reach the pool,
    /// which saves writing back a tensor nothing observes.
    private_rows: Vec<bool>,
    rows: Buffer,
    /// Format-3 plans group pointwise rows into runs of at least
    /// [`GROUP_PIXELS`]: their 7- and 14-pixel rows are too short for the
    /// 12-pixel tiles, while whole large images would fall out of cache before
    /// the folded activation or the paired addition reads them. Format 1
    /// keeps whole-layer scheduling, which its reference outputs pin.
    group_rows: bool,
}

// SAFETY: a `Model` owns its buffers outright — nothing else holds a pointer
// into them, and every method that writes takes `&mut self`. Moving one to
// another thread is therefore sound, which is what lets a caller build a model
// on the main thread and run it on a worker.
unsafe impl Send for Model {}

impl Model {
    /// Load a plan directory: `graph.json` and `weights.bin`, as written by
    /// `tools/export_graph.py`.
    ///
    /// Also accepts a single `.mpplan` container. File decoding allocates
    /// temporary storage; validation completes before tensor buffers are
    /// allocated or kernels execute. Eligible one-pixel heads are repacked
    /// once during loading, without changing the on-disk weights.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let (plan, mut values) = Plan::load(dir)?;
        let packed_heads = pack_heads(&plan, &mut values);
        let panels = pack_panels(&plan, &mut values);
        let row_pairs = plan
            .ops
            .windows(2)
            .map(|ops| can_pair(&ops[0], &ops[1]))
            .chain(std::iter::once(false))
            .take(plan.ops.len())
            .collect::<Vec<_>>();
        let row_triples = plan
            .ops
            .windows(3)
            .map(|ops| can_triple(&ops[0], &ops[1], &ops[2]))
            .chain([false, false])
            .take(plan.ops.len())
            .collect::<Vec<_>>();
        let mut add_relus = vec![0; plan.ops.len()];
        let mut i = 0;
        while i < plan.ops.len() {
            if row_triples[i] {
                i += 3;
                continue;
            }
            if row_pairs[i] {
                i += 2;
                continue;
            }
            add_relus[i] = add_relu_len(&plan, i);
            i += usize::from(add_relus[i]).max(1);
        }
        let group_rows = plan.format == plan::FORMAT_V3;
        let private_rows = (0..plan.ops.len())
            .map(|i| {
                let conv = &plan.ops[i];
                row_pairs[i] && dead_after(&plan, conv.y, i + 2, conv.out_len())
            })
            .collect::<Vec<_>>();
        let rows = private_rows
            .iter()
            .zip(&plan.ops)
            .filter(|(&private, _)| private)
            .map(|(_, conv)| rows_per_group(group_rows, conv) * conv.ow * conv.oc)
            .max()
            .unwrap_or(0);
        let _ = ops::simd_tier(); // Cache environment/CPU discovery before the first frame.
        let weights = Buffer::new(values.len());
        // SAFETY: freshly allocated, no other view exists.
        unsafe { weights.get_mut() }.copy_from_slice(&values);
        let pool = plan.pool.iter().map(|&n| Buffer::new(n)).collect();
        Ok(Self {
            plan,
            weights,
            pool,
            packed_heads,
            panels,
            row_pairs,
            row_triples,
            add_relus,
            private_rows,
            rows: Buffer::new(rows),
            group_rows,
        })
    }

    /// Height, width and channel count of the image this model expects.
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (self.plan.input.h, self.plan.input.w, self.plan.input.c)
    }

    /// The image buffer, to be filled before each call to [`Model::run`].
    ///
    /// Running the model overwrites scratch memory, and the image shares that
    /// memory once it has been consumed. Fill this again for every frame; do
    /// not assume last frame's pixels survived.
    pub fn input_mut(&mut self) -> &mut [f32] {
        let n = self.plan.input.h * self.plan.input.w * self.plan.input.c;
        // SAFETY: `&mut self` rules out any other live view of the pool.
        let buffer = unsafe { self.pool[self.plan.input.slot].get_mut() };
        &mut buffer[..n]
    }

    /// Run one frame.
    pub fn run(&mut self) {
        let _floating_point_scope = ops::DenormalGuard::enter();
        let mut i = 0;
        while i < self.plan.ops.len() {
            if self.row_triples[i] {
                let ops = &self.plan.ops[i..i + 3];
                let (panels, private) = (self.panels[i + 1], self.private_rows[i + 1]);
                self.row_pair(Some(&ops[0]), &ops[1], &ops[2], panels, private);
                i += 3;
            } else if self.row_pairs[i] {
                self.row_pair(
                    None,
                    &self.plan.ops[i],
                    &self.plan.ops[i + 1],
                    self.panels[i],
                    self.private_rows[i],
                );
                i += 2;
            } else if self.add_relus[i] > 0 {
                i += self.add_relu(i);
            } else {
                self.step(&self.plan.ops[i], self.packed_heads[i], self.panels[i]);
                i += 1;
            }
        }
    }

    /// Run one frame, recording how long each operation took in milliseconds.
    ///
    /// `times` is cleared and refilled with one entry per operation, in plan
    /// order, so it lines up with [`Model::ops`]. For a scheduled pair, the
    /// first entry holds the combined duration and the second is zero.
    /// Reading the clock that often
    /// adds a little to the total, so use this to compare operations against
    /// each other, not to quote a frame time.
    #[doc(hidden)] // mpbench's profile; not a stable API
    pub fn run_timed(&mut self, times: &mut Vec<f64>) {
        let _floating_point_scope = ops::DenormalGuard::enter();
        times.clear();
        times.reserve(self.plan.ops.len());
        let mut i = 0;
        while i < self.plan.ops.len() {
            let started = std::time::Instant::now();
            if self.row_triples[i] {
                let ops = &self.plan.ops[i..i + 3];
                let (panels, private) = (self.panels[i + 1], self.private_rows[i + 1]);
                self.row_pair(Some(&ops[0]), &ops[1], &ops[2], panels, private);
                times.push(started.elapsed().as_secs_f64() * 1e3);
                times.extend([0.0, 0.0]); // The depthwise entry owns the whole group.
                i += 3;
            } else if self.row_pairs[i] {
                self.row_pair(
                    None,
                    &self.plan.ops[i],
                    &self.plan.ops[i + 1],
                    self.panels[i],
                    self.private_rows[i],
                );
                times.push(started.elapsed().as_secs_f64() * 1e3);
                times.push(0.0); // The preceding entry owns the whole scheduled pair.
                i += 2;
            } else if self.add_relus[i] > 0 {
                let covered = self.add_relu(i);
                times.push(started.elapsed().as_secs_f64() * 1e3);
                times.resize(times.len() + covered - 1, 0.0);
                i += covered;
            } else {
                self.step(&self.plan.ops[i], self.packed_heads[i], self.panels[i]);
                times.push(started.elapsed().as_secs_f64() * 1e3);
                i += 1;
            }
        }
    }

    /// How many operations, starting at `index`, [`Model::run_timed`] times as
    /// one entry: 3 for a depthwise ahead of a row-local pair, 2 for a
    /// row-local pair, 2 or 3 for a fused [Padc→]Add→Relu, otherwise 1.
    #[doc(hidden)] // mpbench's profile; not a stable API
    pub fn scheduled_len(&self, index: usize) -> usize {
        if self.row_triples.get(index).copied().unwrap_or(false) {
            3
        } else if self.row_pairs.get(index).copied().unwrap_or(false) {
            2
        } else {
            self.add_relus
                .get(index)
                .map_or(1, |&n| usize::from(n).max(1))
        }
    }

    /// Whether a timed entry includes its following residual addition.
    /// For a fused pair, `run_timed` records the total on the convolution and
    /// zero on its successor. Both operations still produce complete tensors.
    #[doc(hidden)] // mpbench's profile; not a stable API
    pub fn has_fused_successor(&self, index: usize) -> bool {
        self.row_pairs.get(index).copied().unwrap_or(false)
    }

    /// A pointwise convolution and its residual add, a group of rows at a
    /// time, optionally with the depthwise layer producing the convolution's
    /// input computed for the same rows just before. Every operation writes
    /// its complete tensor, except a `private` convolution: see `private_rows`.
    fn row_pair(
        &self,
        dw: Option<&plan::Op>,
        conv: &plan::Op,
        add: &plan::Op,
        panels: bool,
        private: bool,
    ) {
        // The add is allowed to reuse the convolution's input allocation. When
        // channels expand, visit rows backwards, like memmove: writing an
        // expanded row must never destroy input belonging to a future row.
        let backwards = add.y == conv.x && conv.oc > conv.ic;
        debug_assert!(
            !(backwards && dw.is_some()),
            "can_triple rejects backwards groups"
        );
        // SAFETY: immutable weights never overlap any scratch allocation.
        let weights = unsafe { self.weights.get() };
        let block = |b: plan::Block| &weights[b.0..b.0 + b.1];
        let rows = rows_per_group(self.group_rows, conv);
        let groups = conv.oh.div_ceil(rows);
        for group in 0..groups {
            let group = if backwards { groups - 1 - group } else { group };
            let first = group * rows;
            let pixels = (rows.min(conv.oh - first)) * conv.ow;
            let (dst, n) = (first * conv.ow * conv.oc, pixels * conv.oc);
            if let Some(dw) = dw {
                // SAFETY: the validated layer has distinct x/y buffers; the
                // view ends before the convolution reads dw.y.
                let (x, y) = unsafe { (self.pool[dw.x].get(), self.pool[dw.y].get_mut()) };
                let (w, b, tail) = (block(dw.w), opt(dw.b, weights), Tail::of(dw, weights));
                depthwise(
                    dw,
                    x,
                    y,
                    w,
                    b,
                    tail,
                    first..first + rows.min(conv.oh - first),
                );
            }
            {
                // SAFETY: the validated convolution has distinct x/y buffers,
                // and `rows` is an allocation of its own. Drop both views
                // BEFORE the add can overwrite conv.x.
                let (x, y) = unsafe {
                    if private {
                        (self.pool[conv.x].get(), &mut self.rows.get_mut()[..n])
                    } else {
                        (
                            self.pool[conv.x].get(),
                            &mut self.pool[conv.y].get_mut()[dst..dst + n],
                        )
                    }
                };
                let src = first * conv.iw * conv.ic;
                let (clip, tail) = Tail::of(conv, weights).fold_clip();
                ops::pointwise_with(
                    y,
                    block(conv.w),
                    &x[src..src + pixels * conv.ic],
                    opt(conv.b, weights),
                    pixels,
                    conv.ic,
                    conv.oc,
                    panels,
                    clip,
                );
                tail.apply(y, conv.oc);
            }
            {
                // SAFETY: add.y is distinct from both add inputs and from
                // `rows`. No view of the previous convolution scope survives.
                let input = |slot: usize| unsafe {
                    if private && slot == conv.y {
                        &self.rows.get()[..n]
                    } else {
                        &self.pool[slot].get()[dst..dst + n]
                    }
                };
                let (x, x2) = (input(add.x), input(add.x2));
                let y = unsafe { &mut self.pool[add.y].get_mut()[dst..dst + n] };
                let clip = opt(add.clip, weights).map(|b| (b[0], b[1]));
                if let Some(slope) = opt(add.act, weights) {
                    ops::add_prelu(y, x, x2, slope, n, conv.oc);
                    if let Some((low, high)) = clip {
                        ops::clip_inplace(y, low, high);
                    }
                } else {
                    ops::add_with(y, x, x2, n, clip);
                }
            }
        }
    }

    /// Run the fused group starting at `first`; returns the operations covered.
    fn add_relu(&self, first: usize) -> usize {
        let covered = usize::from(self.add_relus[first]);
        let ops = &self.plan.ops[first..first + covered];
        let (pad, add, relu) = match ops {
            [pad, add, relu] => (Some(pad), add, relu),
            [add, relu] => (None, add, relu),
            _ => unreachable!("add_relu_len covers two or three operations"),
        };
        let (x1, ic) = pad.map_or((add.x, add.oc), |p| (p.x, p.ic));
        let pixels = add.oh * add.ow;
        let (a, b, d) = (&self.pool[x1], &self.pool[add.x2], &self.pool[relu.y]);
        assert!(a.len >= pixels * ic && b.len >= pixels * add.oc && d.len >= pixels * add.oc);
        // SAFETY: extents asserted above. `add_relu_len` accepted the group
        // only if the output is a distinct buffer, `x2`, or `x1` without
        // padding, so any overlap is index for index. Raw pointers, never
        // slices, because the output may be one of the inputs.
        unsafe {
            ops::add_relu(
                d.ptr.as_ptr(),
                a.ptr.as_ptr(),
                b.ptr.as_ptr(),
                pixels,
                ic,
                add.oc,
            );
        }
        covered
    }

    fn step(&self, op: &plan::Op, packed_head: bool, panels: bool) {
        {
            // SAFETY: `Plan::check` proved that no operation writes a buffer
            // it also reads, so the output view never overlaps an input view,
            // and the caller holds `&mut self`, ruling out views from outside.
            let (x, y) = unsafe { (self.pool[op.x].get(), self.pool[op.y].get_mut()) };
            // SAFETY: the weight buffer is never written after loading.
            let weights = unsafe { self.weights.get() };
            let block = |b: plan::Block| &weights[b.0..b.0 + b.1];
            let act = (op.act.1 > 0).then(|| block(op.act));
            let tail = Tail::of(op, weights);
            match op.kind {
                Kind::Conv if packed_head => {
                    packed_head_convolve(op, x, y, block(op.w), opt(op.b, weights), tail)
                }
                Kind::Conv => {
                    let (w, b) = (block(op.w), opt(op.b, weights));
                    convolve(op, x, y, w, b, tail, self.group_rows, panels)
                }
                Kind::Dwconv => {
                    depthwise(op, x, y, block(op.w), opt(op.b, weights), tail, 0..op.oh)
                }
                Kind::Prelu => ops::prelu(y, x, block(op.w), op.out_len(), op.oc),
                Kind::Relu => {
                    let n = op.out_len();
                    ops::relu(y, x, n);
                }
                Kind::Add => {
                    let n = op.out_len();
                    // SAFETY: as above; `x2` is proved distinct from `y`.
                    let x2 = unsafe { self.pool[op.x2].get() };
                    let clip = opt(op.clip, weights).map(|b| (b[0], b[1]));
                    if let Some(slope) = act {
                        ops::add_prelu(y, x, x2, slope, n, op.oc);
                        if let Some((low, high)) = clip {
                            ops::clip_inplace(&mut y[..n], low, high);
                        }
                    } else {
                        ops::add_with(y, x, x2, n, clip);
                    }
                }
                Kind::Maxpool => maxpool(op, x, y),
                Kind::Padc => pad_channels(op, x, y),
                Kind::Sigmoid => {
                    for i in 0..op.out_len() {
                        y[i] = 1.0 / (1.0 + (-x[i]).exp());
                    }
                }
                Kind::Avgpool => extra_ops::average_pool(op, x, y),
                Kind::Resize => extra_ops::resize_bilinear(op, x, y),
                Kind::Depthtospace => extra_ops::depth_to_space(op, x, y),
                Kind::Channelslice => extra_ops::channel_slice(op, x, y),
                Kind::Transpose => extra_ops::transpose(op, x, y),
                Kind::Layernorm => extra_ops::layer_norm(op, x, y, block(op.w)),
                Kind::Centerscale => extra_ops::center_scale(op, x, y),
                Kind::Constconcat => {
                    let (head, n) = (block(op.w), op.in_len());
                    y[..head.len()].copy_from_slice(head);
                    y[head.len()..head.len() + n].copy_from_slice(&x[..n]);
                }
                Kind::Clip => {
                    let bounds = block(op.w);
                    for (dst, &src) in y[..op.out_len()].iter_mut().zip(x) {
                        *dst = src.clamp(bounds[0], bounds[1]);
                    }
                }
                Kind::Concat => {
                    let head = op.in_len();
                    // SAFETY: as above; `x2` is proved distinct from `y`.
                    let x2 = unsafe { self.pool[op.x2].get() };
                    y[..head].copy_from_slice(&x[..head]);
                    y[head..head + op.n2].copy_from_slice(&x2[..op.n2]);
                }
            }
        }
    }

    /// How many results the model produces.
    pub fn output_count(&self) -> usize {
        self.plan.outputs.len()
    }

    /// One result, in the order the source model declared them.
    ///
    /// # Panics
    /// If `index` is past [`Model::output_count`].
    pub fn output(&self, index: usize) -> &[f32] {
        let out = &self.plan.outputs[index];
        // SAFETY: `&self` rules out any mutable view of the pool.
        let buffer = unsafe { self.pool[out.slot].get() };
        &buffer[..out.n]
    }

    /// One result, by the name the source model gave it.
    pub fn output_named(&self, name: &str) -> Option<&[f32]> {
        let index = self.plan.outputs.iter().position(|o| o.name == name)?;
        Some(self.output(index))
    }

    /// Name of one result, empty when the source model did not give one.
    pub fn output_name(&self, index: usize) -> &str {
        &self.plan.outputs[index].name
    }

    /// Bytes of scratch memory reserved, not counting the weights.
    pub fn scratch_bytes(&self) -> usize {
        (self.plan.pool.iter().sum::<usize>() + self.rows.len) * 4
    }

    /// Floats of weights held.
    pub fn weight_floats(&self) -> usize {
        self.weights.len
    }

    /// The operations, for callers that want to time or report them.
    #[doc(hidden)] // mpbench's profile; not a stable API
    pub fn ops(&self) -> &[plan::Op] {
        &self.plan.ops
    }
}

fn opt(b: plan::Block, weights: &[f32]) -> Option<&[f32]> {
    (b.1 > 0).then(|| &weights[b.0..b.0 + b.1])
}

/// Smallest pixel run a format-3 pointwise call is given.
const GROUP_PIXELS: usize = 48;

/// Output rows per pointwise call: one, or enough for [`GROUP_PIXELS`].
fn rows_per_group(group_rows: bool, op: &plan::Op) -> usize {
    if group_rows {
        GROUP_PIXELS.div_ceil(op.ow).min(op.oh)
    } else {
        1
    }
}

/// Activations folded into a convolution's tail, applied to each finished run
/// of outputs while it is still in cache: PReLU slopes, then a clamp.
#[derive(Clone, Copy)]
struct Tail<'a> {
    slope: Option<&'a [f32]>,
    clip: Option<&'a [f32]>,
}

impl<'a> Tail<'a> {
    fn of(op: &plan::Op, weights: &'a [f32]) -> Self {
        Self {
            slope: opt(op.act, weights),
            clip: opt(op.clip, weights),
        }
    }

    /// Hand the clamp to a kernel that applies it as it stores, when no
    /// PReLU has to run before it; what remains is still to be applied.
    fn fold_clip(self) -> (Option<(f32, f32)>, Self) {
        match (self.slope, self.clip) {
            (None, Some(b)) => (Some((b[0], b[1])), Self { clip: None, ..self }),
            _ => (None, self),
        }
    }

    fn apply(self, y: &mut [f32], channels: usize) {
        if let Some(slope) = self.slope {
            ops::prelu_inplace(y, slope, y.len(), channels);
        }
        if let Some(bounds) = self.clip {
            ops::clip_inplace(y, bounds[0], bounds[1]);
        }
    }
}

/// Output columns for which this kernel column reads a real pixel rather than
/// padding. Hoisting the question out of the inner loop means padding costs a
/// shorter loop instead of a test per pixel.
fn live_columns(op: &plan::Op, kx: usize) -> (usize, usize) {
    let stride = op.sw as isize;
    let shift = op.pl - kx as isize; // source column = out * stride - shift
    let first = if shift <= 0 {
        0
    } else {
        ((shift + stride - 1) / stride) as usize
    };
    let last_source = op.iw as isize - 1 + shift;
    if last_source < 0 {
        return (0, 0);
    }
    let end = ((last_source / stride) as usize + 1).min(op.ow);
    (first.min(end), end)
}

/// Dense convolution.
///
/// A 1x1 kernel is the whole layer in one matrix multiply per output row, so
/// it takes the dedicated pointwise kernel. RGB stems and eligible 2x2 kernels
/// keep complete interior sums in registers over small output tiles. Borders
/// and unsupported shapes retain the tap-at-a-time fallback, seeded with bias.
/// No path changes the per-output reduction order.
///
/// Either way a whole output row is finished before the next begins, so the
/// activation that follows reads it from the nearest cache.
#[allow(clippy::too_many_arguments)]
fn convolve(
    op: &plan::Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    tail: Tail,
    group_rows: bool,
    panels: bool,
) {
    let (ci, co) = (op.ic, op.oc);

    if op.is_pointwise() {
        // Unit stride keeps every row contiguous, so rows can be grouped.
        let rows = rows_per_group(group_rows, op);
        let (clip, tail) = tail.fold_clip();
        for first in (0..op.oh).step_by(rows) {
            let pixels = rows.min(op.oh - first) * op.ow;
            let (dst, src) = (first * op.ow * co, first * op.ow * ci);
            let run = &mut y[dst..dst + pixels * co];
            let x = &x[src..src + pixels * ci];
            ops::pointwise_with(run, w, x, bias, pixels, ci, co, panels, clip);
            tail.apply(run, co);
        }
        return;
    }

    let tap_len = ci * co;
    #[cfg(target_arch = "x86_64")]
    let tiled = ops::simd_tier() >= 1;
    for oy in 0..op.oh {
        let out_row = oy * op.ow;
        let mut interior = (0, 0);
        let top = (oy * op.sh) as isize - op.pt;
        if top >= 0 && top + op.kh as isize <= op.ih as isize {
            let (a, b) = live_columns(op, 0);
            let (d, e) = live_columns(op, op.kw - 1);
            let (first, end) = (a.max(d), b.min(e));
            if first < end {
                let ix = ((first * op.sw) as isize - op.pl) as usize;
                let src = (top as usize * op.iw + ix) * ci;
                let done = ops::spatial_interior(
                    &mut y[(out_row + first) * co..],
                    &x[src..],
                    w,
                    bias,
                    op,
                    end - first,
                );
                interior = (first, first + done);
            }
        }
        ops::fill_bias(&mut y[out_row * co..], bias, interior.0, co);
        ops::fill_bias(
            &mut y[(out_row + interior.1) * co..],
            bias,
            op.ow - interior.1,
            co,
        );
        for ky in 0..op.kh {
            let iy = (oy * op.sh) as isize + ky as isize - op.pt;
            if iy < 0 || iy as usize >= op.ih {
                continue;
            }
            let in_row = iy as usize * op.iw;
            for kx in 0..op.kw {
                let (first, end) = live_columns(op, kx);
                for (first, end) in [(first, end.min(interior.0)), (first.max(interior.1), end)] {
                    if first >= end {
                        continue;
                    }
                    let shift = op.pl - kx as isize;
                    let tap = &w[(ky * op.kw + kx) * tap_len..][..tap_len];
                    let mut ox = first;
                    #[cfg(target_arch = "x86_64")]
                    if tiled {
                        let (xb, yb) = (x.as_ptr(), y.as_mut_ptr());
                        while ox + 4 <= end {
                            let ix = ((ox * op.sw) as isize - shift) as usize;
                            // SAFETY: `live_columns` keeps every source column,
                            // including the three that follow, inside the row; the
                            // four destinations are distinct pixels of one row.
                            unsafe {
                                ops::matvec_t_acc_x4(
                                    [
                                        yb.add((out_row + ox) * co),
                                        yb.add((out_row + ox + 1) * co),
                                        yb.add((out_row + ox + 2) * co),
                                        yb.add((out_row + ox + 3) * co),
                                    ],
                                    tap.as_ptr(),
                                    [
                                        xb.add((in_row + ix) * ci),
                                        xb.add((in_row + ix + op.sw) * ci),
                                        xb.add((in_row + ix + 2 * op.sw) * ci),
                                        xb.add((in_row + ix + 3 * op.sw) * ci),
                                    ],
                                    ci,
                                    co,
                                );
                            }
                            ox += 4;
                        }
                    }
                    while ox < end {
                        let ix = ((ox * op.sw) as isize - shift) as usize;
                        let dst = (out_row + ox) * co;
                        let pixel = &x[(in_row + ix) * ci..(in_row + ix) * ci + ci];
                        ops::matvec_t_acc(&mut y[dst..dst + co], tap, pixel, ci, co);
                        ox += 1;
                    }
                }
            }
        }
        let dst = out_row * co;
        tail.apply(&mut y[dst..dst + op.ow * co], co);
    }
}

/// Depthwise convolution of output rows `rows`: every channel is filtered on
/// its own, so a kernel tap is one multiply-add over a run of pixels with the
/// channel weight held in a register. Each row is computed on its own, so any
/// split of the rows gives the same bits.
fn depthwise(
    op: &plan::Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    tail: Tail,
    rows: std::ops::Range<usize>,
) {
    let c = op.oc;
    let windowed = c.is_multiple_of(8) && ops::simd_tier() >= 2;
    let (clip, rest) = tail.fold_clip();
    let scale = (op.kh, op.kw, op.sh, op.sw, op.pt, op.pl) == (1, 1, 1, 1, 0, 0)
        && (op.ih, op.iw) == (op.oh, op.ow);
    for oy in rows {
        let row_start = oy * op.ow * c;
        let row = row_start..row_start + op.ow * c;
        if windowed
            && scale
            && ops::depthwise_1x1(
                &mut y[row.clone()],
                &x[row.clone()],
                w,
                bias,
                op.ow,
                c,
                clip,
            )
        {
            rest.apply(&mut y[row], c);
            continue;
        }
        if windowed {
            depthwise_row(op, x, &mut y[row_start..], w, bias, oy, clip);
            rest.apply(&mut y[row_start..row_start + op.ow * c], c);
            continue;
        }
        let mut interior = (0, 0);
        let top = (oy * op.sh) as isize - op.pt;
        if top >= 0 && top + op.kh as isize <= op.ih as isize && ops::simd_tier() >= 1 {
            let (a, b) = live_columns(op, 0);
            let (d, e) = live_columns(op, op.kw - 1);
            let (first, end) = (a.max(d), b.min(e));
            if first < end {
                let ix = ((first * op.sw) as isize - op.pl) as usize;
                let src = (top as usize * op.iw + ix) * c;
                let (dst, src) = (&mut y[row_start + first * c..], &x[src..]);
                let (count, row, step) = (end - first, op.iw * c, op.sw * c);
                let done = if (op.kh, op.kw) == (3, 3) {
                    ops::depthwise_3x3_interior(dst, src, w, bias, count, c, row, step, None)
                } else {
                    ops::depthwise_window(
                        dst,
                        src,
                        w,
                        bias,
                        count,
                        c,
                        row,
                        step,
                        op.kw,
                        (0, op.kh),
                        (0, op.kw),
                        None,
                    )
                };
                if done {
                    interior = (first, end);
                }
            }
        }
        // The interior kernel initializes its own accumulators. Seed only
        // boundaries; when unsupported, (0,0) seeds the complete row.
        ops::fill_bias(&mut y[row_start..], bias, interior.0, c);
        ops::fill_bias(
            &mut y[row_start + interior.1 * c..],
            bias,
            op.ow - interior.1,
            c,
        );
        // Padding and unsupported shapes accumulate tap by tap, in kernel order.
        for ky in 0..op.kh {
            let iy = (oy * op.sh) as isize + ky as isize - op.pt;
            if iy < 0 || iy as usize >= op.ih {
                continue;
            }
            for kx in 0..op.kw {
                let (first, end) = live_columns(op, kx);
                for (first, end) in [(first, end.min(interior.0)), (first.max(interior.1), end)] {
                    if first >= end {
                        continue;
                    }
                    let ix = ((first * op.sw) as isize - (op.pl - kx as isize)) as usize;
                    let tap = &w[(ky * op.kw + kx) * c..][..c];
                    let src = (iy as usize * op.iw + ix) * c;
                    let dst = row_start + first * c;
                    ops::depthwise_tap(&mut y[dst..], &x[src..], tap, end - first, c, op.sw * c);
                }
            }
        }
        tail.apply(&mut y[row_start..row_start + op.ow * c], c);
    }
}

/// One depthwise output row in registers, borders included: the columns with
/// a whole kernel window go to one kernel call, each border column (a window
/// of its own) to another, and every call clamps to `clip` as it stores.
fn depthwise_row(
    op: &plan::Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    oy: usize,
    clip: Option<(f32, f32)>,
) {
    let (c, top) = (op.oc, (oy * op.sh) as isize - op.pt);
    // Kernel taps [low, high) that land on real pixels of a `len`-long axis
    // when the window starts at `first`.
    let live = |first: isize, len: usize, k: usize| {
        let low = (-first).clamp(0, k as isize) as usize;
        (
            low,
            ((len as isize - first).clamp(0, k as isize) as usize).max(low),
        )
    };
    let rows = live(top, op.ih, op.kh);
    let (row, step) = (op.iw * c, op.sw * c);
    if (op.kh, op.kw, op.sw) == (5, 5, 1) && rows.0 < rows.1 {
        let iy = (top + rows.0 as isize) as usize;
        let (x, (ow, iw)) = (&x[iy * row..], (op.ow, op.iw));
        if ops::depthwise_row5(y, x, w, bias, ow, iw, op.pl, c, row, rows, clip) {
            return;
        }
    }
    let run = |y: &mut [f32], ox: usize, count: usize| {
        let left = (ox * op.sw) as isize - op.pl;
        let cols = live(left, op.iw, op.kw);
        let dst = &mut y[ox * c..];
        let src = if rows.0 < rows.1 && cols.0 < cols.1 {
            let iy = (top + rows.0 as isize) as usize;
            &x[(iy * op.iw + (left + cols.0 as isize) as usize) * c..]
        } else {
            &x[..0]
        };
        let whole = rows == (0, op.kh) && cols == (0, op.kw);
        let done = (whole && (op.kh, op.kw, op.sw) == (3, 3, 1))
            && ops::depthwise_3x3_interior(dst, src, w, bias, count, c, row, step, clip);
        if !done {
            let kw = op.kw;
            let done =
                ops::depthwise_window(dst, src, w, bias, count, c, row, step, kw, rows, cols, clip);
            assert!(done, "the windowed kernel covers c % 8 == 0 on AVX tiers");
        }
    };
    // Columns where the first and the last kernel column are both live.
    let (a, b) = live_columns(op, 0);
    let (d, e) = live_columns(op, op.kw - 1);
    let (first, end) = if a.max(d) < b.min(e) {
        (a.max(d), b.min(e))
    } else {
        (op.ow, op.ow)
    };
    for ox in (0..first).chain(end..op.ow) {
        run(y, ox, 1);
    }
    if first < end {
        run(y, first, end - first);
    }
}

/// Sliding-window maximum over each channel independently.
fn maxpool(op: &plan::Op, x: &[f32], y: &mut [f32]) {
    // A complete 2x2 window needs one output store, not three read/modify/write
    // passes. Borders, other shapes and non-AVX tiers use the general loop.
    if op.kh == 2
        && op.kw == 2
        && op.sh == 2
        && op.sw == 2
        && op.ih == op.oh * 2
        && op.iw == op.ow * 2
        && ops::maxpool_2x2(x, y, op.oh, op.ow, op.oc)
    {
        return;
    }
    let c = op.oc;
    for oy in 0..op.oh {
        for ox in 0..op.ow {
            let dst = (oy * op.ow + ox) * c;
            let mut seeded = false;
            for ky in 0..op.kh {
                let iy = oy * op.sh + ky;
                if iy >= op.ih {
                    continue;
                }
                for kx in 0..op.kw {
                    let ix = ox * op.sw + kx;
                    if ix >= op.iw {
                        continue;
                    }
                    let src = &x[(iy * op.iw + ix) * c..][..c];
                    if seeded {
                        ops::max_into(&mut y[dst..dst + c], src, c);
                    } else {
                        y[dst..dst + c].copy_from_slice(src);
                        seeded = true;
                    }
                }
            }
        }
    }
}

/// Widen each pixel to more channels, the new ones zero. The image keeps its
/// height and width; only the per-pixel vector grows.
fn pad_channels(op: &plan::Op, x: &[f32], y: &mut [f32]) {
    for p in 0..op.oh * op.ow {
        let (src, dst) = (p * op.ic, p * op.oc);
        y[dst..dst + op.ic].copy_from_slice(&x[src..src + op.ic]);
        y[dst + op.ic..dst + op.oc].fill(0.0);
    }
}

/// Pack a full-image, single-pixel head into K-by-32 panels at load time.
/// This is only a permutation of f32 values. No padding, quantization, or extra
/// persistent weight copy; the original file is never modified.
/// Whether operation `i`'s weights share floats with any other block. Such a
/// region is never repacked, even if the exporter allowed overlapping blocks.
fn weights_shared(plan: &Plan, i: usize) -> bool {
    let (start, end) = (plan.ops[i].w.0, plan.ops[i].w.0 + plan.ops[i].w.1);
    let overlaps = |b: plan::Block| b.1 > 0 && b.0 < end && start < b.0 + b.1;
    plan.ops.iter().enumerate().any(|(j, other)| {
        (i != j && overlaps(other.w))
            || overlaps(other.b)
            || overlaps(other.act)
            || overlaps(other.clip)
    })
}

/// Reorder the weights of eligible pointwise layers into 16-column panels
/// (see `ops::pointwise_with`). A permutation only, on AVX tiers, where
/// every kernel that will read them understands panels.
fn pack_panels(plan: &Plan, values: &mut [f32]) -> Vec<bool> {
    let mut packed = vec![false; plan.ops.len()];
    if ops::simd_tier() < 2 {
        return packed;
    }
    for (i, op) in plan.ops.iter().enumerate() {
        if op.kind != Kind::Conv
            || !op.is_pointwise()
            || !op.oc.is_multiple_of(16)
            || weights_shared(plan, i)
        {
            continue;
        }
        let (start, ci, co) = (op.w.0, op.ic, op.oc);
        let mut panels = Vec::with_capacity(op.w.1);
        for j in (0..co).step_by(16) {
            for row in 0..ci {
                panels.extend_from_slice(&values[start + row * co + j..][..16]);
            }
        }
        values[start..start + op.w.1].copy_from_slice(&panels);
        packed[i] = true;
    }
    packed
}

fn pack_heads(plan: &Plan, values: &mut [f32]) -> Vec<bool> {
    let mut packed = vec![false; plan.ops.len()];
    for (i, op) in plan.ops.iter().enumerate() {
        if op.kind != Kind::Conv
            || op.is_pointwise() // Its old one-pixel fallback adds bias LAST.
            || op.oh != 1
            || op.ow != 1
            || op.kh != op.ih
            || op.kw != op.iw
            || op.pt != 0
            || op.pl != 0
            || op.oc < 128
        {
            continue;
        }
        if weights_shared(plan, i) {
            continue;
        }
        let (start, end) = (op.w.0, op.w.0 + op.w.1);
        let k = op.in_len();
        let co = op.oc;
        let mut tmp = Vec::with_capacity(op.w.1);
        for j in (0..co).step_by(32) {
            let width = (co - j).min(32);
            for row in 0..k {
                tmp.extend_from_slice(&values[start + row * co + j..start + row * co + j + width]);
            }
        }
        values[start..end].copy_from_slice(&tmp);
        packed[i] = true;
    }
    packed
}

fn packed_head_convolve(
    op: &plan::Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    tail: Tail,
) {
    ops::fill_bias(y, bias, 1, op.oc);
    let k = op.in_len();
    for j in (0..op.oc).step_by(32) {
        let width = (op.oc - j).min(32);
        ops::matvec_t_acc(
            &mut y[j..j + width],
            &w[j * k..j * k + k * width],
            x,
            k,
            width,
        );
    }
    tail.apply(&mut y[..op.oc], op.oc);
}

/// Padc→Add→Relu or Add→Relu computed in one pass, when the tensors that pass
/// no longer writes are dead: no later read and no model output can see what
/// they would have held. Shapes within each operation were validated by
/// `Plan::check`. Returns the operations covered, or 0.
fn add_relu_len(plan: &Plan, i: usize) -> u8 {
    let ops = &plan.ops;
    let (pad, rest) = if ops[i].kind == Kind::Padc {
        (Some(&ops[i]), i + 1)
    } else {
        (None, i)
    };
    let (Some(add), Some(relu)) = (ops.get(rest), ops.get(rest + 1)) else {
        return 0;
    };
    let shape = |op: &plan::Op| (op.oh, op.ow, op.oc);
    if add.kind != Kind::Add
        || add.act.1 > 0
        || add.clip.1 > 0
        || relu.kind != Kind::Relu
        || relu.x != add.y
        || shape(relu) != shape(add)
    {
        return 0;
    }
    if let Some(pad) = pad {
        // The pad must be the first addend, and the output must not be the
        // narrower input: those overlap at different indices.
        if add.x != pad.y
            || add.x2 == pad.y
            || relu.y == pad.x
            || shape(pad) != shape(add)
            || pad.ic >= pad.oc
        {
            return 0;
        }
    }
    let end = rest + 2;
    let n = add.out_len();
    let skipped = [Some(add.y), pad.map(|p| p.y).filter(|&slot| slot != relu.y)];
    if skipped
        .into_iter()
        .flatten()
        .all(|slot| dead_after(plan, slot, end, n))
    {
        (end - i) as u8
    } else {
        0
    }
}

/// Whether the first `n` floats of `slot`, left unwritten before operation
/// `from`, can never be observed. Every write starts at element 0, so what
/// is still stale is `clean..n`; a read of `k` floats sees it when `k > clean`.
fn dead_after(plan: &Plan, slot: usize, from: usize, n: usize) -> bool {
    let mut clean = 0;
    for op in &plan.ops[from..] {
        if clean >= n {
            return true;
        }
        // Floats each operation reads from its first and second input.
        let (read, read2) = match op.kind {
            Kind::Conv
            | Kind::Dwconv
            | Kind::Maxpool
            | Kind::Avgpool
            | Kind::Padc
            | Kind::Resize
            | Kind::Depthtospace
            | Kind::Channelslice
            | Kind::Transpose
            | Kind::Layernorm
            | Kind::Centerscale
            | Kind::Constconcat => (op.in_len(), 0),
            Kind::Relu | Kind::Prelu | Kind::Sigmoid | Kind::Clip => {
                (op.in_len().max(op.out_len()), 0)
            }
            Kind::Add => (op.in_len().max(op.out_len()), op.out_len()),
            Kind::Concat => (op.in_len(), op.n2),
        };
        if (op.x == slot && read > clean) || (op.x2 == slot && read2 > clean) {
            return false;
        }
        if op.y == slot {
            clean = clean.max(op.out_len());
        }
    }
    clean >= n
        || !plan
            .outputs
            .iter()
            .any(|out| out.slot == slot && out.n > clean)
}

/// Scheduling only: preserve the full intermediate tensor and all weight
/// values. Shape equality is essential to both row addressing and alias order.
/// Whether depthwise rows can be computed group by group ahead of the
/// row-local pair `conv`, `add` that consumes them, with every tensor ending
/// as the three operations in sequence leave it. The depthwise input is read
/// up to a kernel height past the current group, so neither later output may
/// be written into it; and an add writing over the depthwise output must not
/// expand, or its rows would land on depthwise rows the convolution has not
/// read yet.
fn can_triple(dw: &plan::Op, conv: &plan::Op, add: &plan::Op) -> bool {
    dw.kind == Kind::Dwconv
        && conv.x == dw.y
        && (conv.ih, conv.iw, conv.ic) == (dw.oh, dw.ow, dw.oc)
        && can_pair(conv, add)
        && dw.x != conv.y
        && dw.x != add.y
        && !(add.y == dw.y && conv.oc > conv.ic)
}

fn can_pair(conv: &plan::Op, add: &plan::Op) -> bool {
    conv.is_pointwise()
        && conv.ih == conv.oh
        && conv.iw == conv.ow
        && add.kind == Kind::Add
        && (add.x == conv.y || add.x2 == conv.y)
        && (add.ih, add.iw, add.ic) == (conv.oh, conv.ow, conv.oc)
        && (add.oh, add.ow, add.oc) == (conv.oh, conv.ow, conv.oc)
}

#[cfg(test)]
mod schedule_tests {
    use super::*;
    fn pair_model(ci: usize, co: usize, alias: bool, swapped: bool, act: bool) -> Model {
        let (h, w) = (3, 7);
        let n = h * w * co;
        let dst = if alias { 0 } else { 3 };
        let nw = ci * co;
        let plan: Plan = serde_json::from_value(serde_json::json!({
            "format":plan::FORMAT,"input":{"slot":0,"h":h,"w":w,"c":ci},
            "pool":[h*w*ci.max(co),n,n,n],
            "outputs":[{"slot":dst,"n":n,"name":"sum"},{"slot":1,"n":n,"name":"conv"}],
            "ops":[
                {"kind":"conv","x":0,"y":1,"ih":h,"iw":w,"ic":ci,"oh":h,"ow":w,"oc":co,
                    "kh":1,"kw":1,"sh":1,"sw":1,"w":[0,nw],"b":[nw,co],
                    "act":if act {vec![nw+co,co]}else{vec![0,0]}},
                {"kind":"add","x":if swapped{2}else{1},"x2":if swapped{1}else{2},"y":dst,
                    "ih":h,"iw":w,"ic":co,"oh":h,"ow":w,"oc":co,
                    "act":if act{vec![nw+co*2,co]}else{vec![0,0]}}
            ]
        }))
        .unwrap();
        let weights = Buffer::new(nw + 3 * co);
        unsafe {
            for (i, v) in weights.get_mut().iter_mut().enumerate() {
                *v = (i as f32 * 0.2).sin() * 0.5;
            }
        }
        let pool = plan.pool.iter().map(|&n| Buffer::new(n)).collect();
        Model {
            plan,
            weights,
            pool,
            packed_heads: vec![false; 2],
            panels: vec![false; 2],
            row_pairs: vec![true, false],
            row_triples: vec![false; 2],
            private_rows: vec![false; 2],
            rows: Buffer::new(0),
            add_relus: vec![0; 2],
            group_rows: false,
        }
    }
    fn results(model: &Model) -> Vec<Vec<u32>> {
        (0..model.output_count())
            .map(|o| model.output(o).iter().map(|v| v.to_bits()).collect())
            .collect()
    }
    #[test]
    fn fused_rows_preserve_expanding_and_contracting_aliases_and_intermediates() {
        for (ci, co) in [(3, 16), (8, 16), (16, 8), (16, 16), (24, 28), (32, 64)] {
            for alias in [false, true] {
                for swap in [false, true] {
                    for act in [false, true] {
                        let mut model = pair_model(ci, co, alias, swap, act);
                        assert!(can_pair(&model.plan.ops[0], &model.plan.ops[1]));
                        let image: Vec<_> =
                            (0..3 * 7 * ci).map(|i| (i as f32 * 0.4).cos()).collect();
                        unsafe {
                            model.pool[2].get_mut().fill(0.125);
                        }
                        model.row_pairs[0] = false;
                        model.input_mut().copy_from_slice(&image);
                        model.run();
                        let want = results(&model);
                        model.row_pairs[0] = true;
                        model.input_mut().copy_from_slice(&image);
                        model.run();
                        assert_eq!(
                            results(&model),
                            want,
                            "{ci}->{co} alias={alias} swap={swap} act={act}"
                        );
                        let mut times = Vec::new();
                        model.input_mut().copy_from_slice(&image);
                        model.run_timed(&mut times);
                        assert_eq!(results(&model), want);
                        assert_eq!(times.len(), 2);
                        assert_eq!(times[1], 0.0);
                    }
                }
            }
        }
    }
    #[test]
    fn private_rows_skip_only_a_convolution_output_nothing_reads() {
        for (ci, co) in [(8, 16), (16, 8), (32, 64)] {
            for alias in [false, true] {
                for swap in [false, true] {
                    let mut model = pair_model(ci, co, alias, swap, true);
                    let n = model.plan.ops[0].out_len();
                    assert!(!dead_after(&model.plan, 1, 2, n), "an output is observed");
                    model.plan.outputs.truncate(1);
                    assert!(dead_after(&model.plan, 1, 2, n));
                    let image: Vec<_> = (0..3 * 7 * ci).map(|i| (i as f32 * 0.4).cos()).collect();
                    unsafe {
                        model.pool[2].get_mut().fill(0.125);
                    }
                    model.row_pairs[0] = false;
                    model.input_mut().copy_from_slice(&image);
                    model.run();
                    let want = results(&model);
                    (model.row_pairs[0], model.private_rows[0]) = (true, true);
                    model.rows = Buffer::new(7 * co);
                    // The pool copy of the convolution must be neither read nor needed.
                    unsafe {
                        model.pool[1].get_mut().fill(f32::NAN);
                    }
                    model.input_mut().copy_from_slice(&image);
                    model.run();
                    assert_eq!(
                        results(&model),
                        want,
                        "{ci}->{co} alias={alias} swap={swap}"
                    );
                    assert!(unsafe { model.pool[1].get() }.iter().all(|v| v.is_nan()));
                }
            }
        }
    }
    /// Depthwise (slot 0 → 1, clamped) → pointwise (1 → 2) → add (2 plus
    /// `residual` → `out`), every slot an output. Slot 3 is a spare residual.
    fn triple_model(
        c: usize,
        co: usize,
        k: usize,
        stride: usize,
        residual: usize,
        out: usize,
    ) -> Model {
        let (ih, iw) = (7, 9);
        let pad = (k / 2) as isize;
        let (oh, ow) = ((ih - 1) / stride + 1, (iw - 1) / stride + 1);
        let n = oh * ow * c.max(co);
        let (nd, nc) = (k * k * c, c * co);
        let (db, cw, cb, dc) = (nd, nd + c, nd + c + nc, nd + c + nc + co);
        let plan: Plan = serde_json::from_value(serde_json::json!({
            "format":plan::FORMAT,"input":{"slot":0,"h":ih,"w":iw,"c":c},
            "pool":[(ih * iw * c).max(n),n,n,n,n],
            "outputs":(0..5).map(|s| serde_json::json!({"slot":s,"n":n,"name":s.to_string()}))
                .collect::<Vec<_>>(),
            "ops":[
                {"kind":"dwconv","x":0,"y":1,"ih":ih,"iw":iw,"ic":c,"oh":oh,"ow":ow,"oc":c,
                    "kh":k,"kw":k,"sh":stride,"sw":stride,"pt":pad,"pl":pad,
                    "w":[0,nd],"b":[db,c],"clip":[dc,2]},
                {"kind":"conv","x":1,"y":2,"ih":oh,"iw":ow,"ic":c,"oh":oh,"ow":ow,"oc":co,
                    "kh":1,"kw":1,"sh":1,"sw":1,"w":[cw,nc],"b":[cb,co]},
                {"kind":"add","x":2,"x2":residual,"y":out,
                    "ih":oh,"iw":ow,"ic":co,"oh":oh,"ow":ow,"oc":co}
            ]
        }))
        .unwrap();
        let weights = Buffer::new(dc + 2);
        unsafe {
            let values = weights.get_mut();
            for (i, v) in values.iter_mut().enumerate() {
                *v = (i as f32 * 0.3).sin() * 0.5;
            }
            values[dc..].copy_from_slice(&[-0.3, 0.4]);
        }
        let pool = plan.pool.iter().map(|&n| Buffer::new(n)).collect();
        Model {
            plan,
            weights,
            pool,
            packed_heads: vec![false; 3],
            panels: vec![false; 3],
            row_pairs: vec![false; 3],
            row_triples: vec![false; 3],
            private_rows: vec![false; 3],
            rows: Buffer::new(0),
            add_relus: vec![0; 3],
            group_rows: false,
        }
    }
    #[test]
    fn depthwise_rows_ahead_of_a_pair_match_the_sequential_operations() {
        // (c, co, kernel, stride, residual slot, add output slot)
        let cases = [
            (8, 8, 3, 1, 0, 4),   // residual is the depthwise input
            (8, 8, 3, 1, 3, 1),   // add over the depthwise output
            (16, 8, 5, 1, 3, 1),  // ... contracting
            (16, 24, 5, 1, 3, 4), // expanding into a fresh slot
            (8, 16, 3, 2, 3, 4),  // stride two
            (12, 12, 3, 1, 1, 4), // residual is the depthwise output
        ];
        for (c, co, k, stride, residual, out) in cases {
            for group_rows in [false, true] {
                let mut model = triple_model(c, co, k, stride, residual, out);
                model.group_rows = group_rows;
                let ops = &model.plan.ops;
                assert!(can_triple(&ops[0], &ops[1], &ops[2]), "{c}->{co} k={k}");
                let image: Vec<_> = (0..7 * 9 * c).map(|i| (i as f32 * 0.7).cos()).collect();
                let label = format!("{c}->{co} k={k} s={stride} res={residual} out={out}");
                unsafe {
                    for s in 1..5 {
                        model.pool[s].get_mut().fill(0.125);
                    }
                }
                model.input_mut().copy_from_slice(&image);
                model.run();
                let want = results(&model);
                model.row_triples[0] = true;
                unsafe {
                    for s in 1..5 {
                        model.pool[s].get_mut().fill(0.125);
                    }
                }
                model.input_mut().copy_from_slice(&image);
                model.run();
                assert_eq!(results(&model), want, "{label}");
                let mut times = Vec::new();
                model.input_mut().copy_from_slice(&image);
                model.run_timed(&mut times);
                assert_eq!((times.len(), times[1], times[2]), (3, 0.0, 0.0), "{label}");
                assert_eq!(model.scheduled_len(0), 3);
            }
        }
        // Later rows of the depthwise input must survive the group before.
        let over_input = triple_model(8, 8, 3, 1, 3, 0);
        let expanding_over_output = triple_model(8, 16, 3, 1, 3, 1);
        for model in [over_input, expanding_over_output] {
            let ops = &model.plan.ops;
            assert!(!can_triple(&ops[0], &ops[1], &ops[2]));
        }
    }
    /// [Padc→]Add→Relu over slots 0 (x1) and 1 (x2). `out` is the relu's
    /// slot; `reread` appends a relu that reads the addition's slot again.
    fn add_relu_model(ic: usize, oc: usize, out: usize, reread: bool) -> Model {
        let (h, w) = (3, 5);
        let n = h * w * oc;
        let shape = |kind: &str, x: usize, y: usize, ic: usize| {
            serde_json::json!({"kind":kind,"x":x,"x2":1,"y":y,
                "ih":h,"iw":w,"ic":ic,"oh":h,"ow":w,"oc":oc})
        };
        let mut ops = Vec::new();
        let mut add_x = 0;
        if ic < oc {
            ops.push(shape("padc", 0, 2, ic));
            add_x = 2;
        }
        ops.push(shape("add", add_x, 3, oc));
        ops.push(shape("relu", 3, out, oc));
        let mut outputs = vec![serde_json::json!({"slot":out,"n":n,"name":"y"})];
        if reread {
            ops.push(shape("relu", 3, 4, oc));
            outputs.push(serde_json::json!({"slot":4,"n":n,"name":"again"}));
        }
        let plan: Plan = serde_json::from_value(serde_json::json!({
            "format":plan::FORMAT,"input":{"slot":0,"h":h,"w":w,"c":ic},
            "pool":[n,n,n,n,n],"outputs":outputs,"ops":ops
        }))
        .unwrap();
        let count = plan.ops.len();
        let pool = plan.pool.iter().map(|&n| Buffer::new(n)).collect();
        Model {
            plan,
            weights: Buffer::new(1),
            pool,
            packed_heads: vec![false; count],
            panels: vec![false; count],
            row_pairs: vec![false; count],
            row_triples: vec![false; count],
            private_rows: vec![false; count],
            rows: Buffer::new(0),
            add_relus: vec![0; count],
            group_rows: false,
        }
    }

    #[test]
    fn fused_add_relu_matches_the_separate_operations_bit_for_bit() {
        let special = [
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fa0_0001), // signalling NaN payload
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0e-40,
            -1.0e-40,
            -1.5,
            2.5,
            -0.25,
        ];
        let fill = |model: &Model, seed: usize| {
            for slot in 0..2 {
                // SAFETY: the test holds the only reference to the model.
                let buffer = unsafe { model.pool[slot].get_mut() };
                for (i, v) in buffer.iter_mut().enumerate() {
                    *v = special[(i * (slot + 5) + seed) % special.len()];
                    // Never two NaN addends: which payload survives then is
                    // unspecified (LLVM may commute `fadd`), in either path.
                    if v.is_nan() && i % 2 == slot {
                        *v = 1.0;
                    }
                }
            }
        };
        // Relu output: fresh slot, the first addend, or the second addend.
        for (ic, oc) in [(8, 12), (24, 28), (5, 7), (12, 12), (7, 7)] {
            for out in [4, 0, 1] {
                for reread in [false, true] {
                    if out == 0 && ic < oc {
                        continue; // Rejected shape: see the alias test below.
                    }
                    let mut model = add_relu_model(ic, oc, out, reread);
                    let covered = add_relu_len(&model.plan, 0);
                    assert_eq!(covered, if reread { 0 } else { 2 + u8::from(ic < oc) });
                    for seed in 0..3 {
                        fill(&model, seed);
                        model.add_relus[0] = 0;
                        model.run();
                        let want = results(&model);
                        fill(&model, seed);
                        model.add_relus[0] = covered;
                        model.run();
                        assert_eq!(results(&model), want, "{ic}->{oc} out={out} seed={seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn add_relu_is_not_fused_into_the_narrow_input_or_over_a_live_tensor() {
        // Writing the widened result over its own narrower input would
        // overlap at different indices.
        assert_eq!(add_relu_len(&add_relu_model(8, 12, 0, false).plan, 0), 0);
        // A tensor read after the group must still be written.
        assert_eq!(add_relu_len(&add_relu_model(8, 12, 4, true).plan, 0), 0);
        // A skipped tensor that stays a model output must still be written.
        let mut model = add_relu_model(12, 12, 4, false);
        model.plan.outputs[0].slot = 3;
        assert_eq!(add_relu_len(&model.plan, 0), 0);
    }

    #[test]
    fn packing_does_not_rewrite_pointwise_bias_order_or_shared_weights() {
        let mut model = pair_model(16, 128, false, false, false);
        let op = &mut model.plan.ops[0];
        op.ih = 1;
        op.iw = 1;
        op.oh = 1;
        op.ow = 1;
        let mut values = unsafe { model.weights.get() }.to_vec();
        let before = values.clone();
        assert!(!pack_heads(&model.plan, &mut values)[0]);
        assert_eq!(values, before);
        // A spatial head whose bias overlaps its weights must not be repacked.
        let op = &mut model.plan.ops[0];
        op.ih = 2;
        op.iw = 2;
        op.kh = 2;
        op.kw = 2;
        op.ic = 4;
        op.b = plan::Block(0, 128);
        assert!(!pack_heads(&model.plan, &mut values)[0]);
        assert_eq!(values, before);
    }
}
