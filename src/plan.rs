//! The execution plan: what the exporter wrote, and how it is checked.
//!
//! A plan is a description and a block of numbers. The description lists the
//! operations in the order they run, the shape of every tensor, and where each
//! operation's numbers live. The numbers are 32-bit floats, one flat block.
//!
//! Both travel in a single `.mpplan` file, so a plan sits beside the other
//! model files and a checksum list covers it the same way. The exporter also
//! leaves them unpacked, as `graph.json` and `weights.bin`, which is what you
//! want while a model is being worked on.
//!
//! Nothing in a plan is trusted. It arrives from disk, and a description that
//! disagreed with its numbers could otherwise read past the end of them. So
//! `Plan::load` checks, before anything runs: that every buffer index exists,
//! that every tensor fits the buffer it was assigned, that every block of
//! numbers lies inside the block, that it is the size the operation's
//! shape implies, and that no number is infinite or not-a-number. After that
//! the executor can index without re-checking.

use std::fmt;
use std::path::Path;

use serde::Deserialize;

/// Anything that stops a plan from being loadable.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The file is a plan, but not one this version understands.
    Format(String),
    /// The plan is self-inconsistent, or disagrees with the weight file.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "reading the plan: {e}"),
            Self::Json(e) => write!(f, "parsing graph.json: {e}"),
            Self::Format(m) => write!(f, "unsupported plan: {m}"),
            Self::Invalid(m) => write!(f, "invalid plan: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

/// Plans carry this tag so an older binary refuses a newer file instead of
/// misreading it.
pub const FORMAT: &str = "mediapipe-native-plan/1";
/// Plans exported from MediaPipe TFLite bundles (hand, pose, holistic): adds
/// pooling, clamps, resize, depth-to-space, channel slices and the blendshape
/// layers to format 1.
pub const FORMAT_V3: &str = "mediapipe-native-plan/3";

/// Ceiling on total scratch memory, in floats. The graphs this runs are a few
/// megabytes; anything far past that means a corrupt or hostile file, and
/// refusing beats trying to allocate it.
const MAX_POOL_FLOATS: usize = 64 << 20; // 256 MB

/// What an operation does. Convolution covers the dense and depthwise cases
/// because they differ only in how the weights are indexed.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Dense convolution: every output channel reads every input channel.
    Conv,
    /// Depthwise convolution: each channel is filtered on its own.
    Dwconv,
    /// Parametric ReLU with one learned slope per channel.
    Prelu,
    /// Plain ReLU.
    Relu,
    /// Elementwise sum of two tensors of the same shape.
    Add,
    /// Sliding-window maximum.
    Maxpool,
    /// Append zero channels to reach a wider channel count.
    Padc,
    /// Logistic squashing function.
    Sigmoid,
    /// Join two tensors end to end.
    Concat,
    /// Valid-window average, summing taps in row-major order then dividing.
    Avgpool,
    /// Clamp to the two finite constants in `w`.
    Clip,
    /// Bilinear resize with half-pixel centres, as TFLite's
    /// `RESIZE_BILINEAR(align_corners=false, half_pixel_centers=true)`.
    Resize,
    /// Move `b*b` channel groups into a `b`-by-`b` block of pixels (TFLite
    /// `DEPTH_TO_SPACE`); `b = oh / ih`.
    Depthtospace,
    /// Copy channels `begin..begin + oc` of every pixel.
    Channelslice,
    /// Swap width and channels of a one-row tensor: `y[c][w] = x[w][c]`.
    Transpose,
    /// Layer normalization over each pixel's channels, in the expanded form
    /// TFLite exports: `w` holds gamma (`oc` values) then epsilon; no beta.
    Layernorm,
    /// Landmark normalization of a one-row tensor of points: subtract each
    /// channel's mean over the points, divide by their mean distance from it.
    Centerscale,
    /// The constant block `w`, then the input, end to end.
    Constconcat,
}

/// A position and length inside the weight file, in floats. `[0, 0]` means the
/// operation has none.
#[derive(Deserialize, Clone, Copy, Default, Debug)]
pub struct Block(pub usize, pub usize);

impl Block {
    fn is_empty(self) -> bool {
        self.1 == 0
    }
}

/// One step of the graph.
///
/// Buffers are referred to by index into the scratch pool, not by tensor name:
/// the exporter has already worked out which tensors can share storage.
#[derive(Deserialize, Debug)]
pub struct Op {
    pub kind: Kind,
    /// Scratch buffer holding the first input.
    pub x: usize,
    /// Scratch buffer holding the second input, for `Add` and `Concat`.
    #[serde(default)]
    pub x2: usize,
    /// Scratch buffer the result is written to.
    pub y: usize,
    /// Input height, width and channel count.
    pub ih: usize,
    pub iw: usize,
    pub ic: usize,
    /// Output height, width and channel count.
    pub oh: usize,
    pub ow: usize,
    pub oc: usize,
    /// Kernel height and width.
    #[serde(default)]
    pub kh: usize,
    #[serde(default)]
    pub kw: usize,
    /// Pixels skipped between kernel placements, vertically and horizontally.
    #[serde(default)]
    pub sh: usize,
    #[serde(default)]
    pub sw: usize,
    /// Rows added above, and columns added to the left, of the input.
    #[serde(default)]
    pub pt: isize,
    #[serde(default)]
    pub pl: isize,
    /// Filter weights.
    #[serde(default)]
    pub w: Block,
    /// Per-output-channel offset added after the filter.
    #[serde(default)]
    pub b: Block,
    /// Parametric-ReLU slopes folded into this operation's tail.
    #[serde(default)]
    pub act: Block,
    /// `[low, high]` clamp folded into this operation's tail, after any
    /// slopes (format 3: RELU6).
    #[serde(default)]
    pub clip: Block,
    /// Element count of the second input, for `Concat`.
    #[serde(default)]
    pub n2: usize,
    /// First input channel copied by `Channelslice`.
    #[serde(default)]
    pub begin: usize,
}

impl Op {
    /// Elements written by this operation.
    pub fn out_len(&self) -> usize {
        match self.kind {
            Kind::Concat => self
                .in_len()
                .checked_add(self.n2)
                .expect("concat length overflows"),
            _ => self
                .oh
                .checked_mul(self.ow)
                .and_then(|n| n.checked_mul(self.oc))
                .expect("output shape overflows"),
        }
    }

    /// Elements read from the first input.
    pub fn in_len(&self) -> usize {
        self.ih
            .checked_mul(self.iw)
            .and_then(|n| n.checked_mul(self.ic))
            .expect("input shape overflows")
    }

    /// True when the whole layer is a single matrix multiply over pixels.
    pub fn is_pointwise(&self) -> bool {
        self.kind == Kind::Conv
            && self.kh == 1
            && self.kw == 1
            && self.sh == 1
            && self.sw == 1
            && self.pt == 0
            && self.pl == 0
    }
}

/// Where the image goes in.
#[derive(Deserialize, Debug)]
pub struct Input {
    pub slot: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

/// Where a result comes out.
#[derive(Deserialize, Debug)]
pub struct Output {
    pub slot: usize,
    /// Element count.
    pub n: usize,
    /// Label from the source model, for callers that want to pick by meaning
    /// rather than position.
    #[serde(default)]
    pub name: String,
}

/// A whole model, ready to be turned into a runnable one.
#[derive(Deserialize, Debug)]
pub struct Plan {
    pub format: String,
    /// Human-readable note about where the weights came from.
    #[serde(default)]
    pub source: String,
    pub input: Input,
    pub outputs: Vec<Output>,
    /// Capacity, in floats, of each scratch buffer.
    pub pool: Vec<usize>,
    pub ops: Vec<Op>,
}

/// First bytes of a single-file plan, so a wrong file is rejected by its own
/// contents rather than by its name.
const MAGIC: &[u8; 8] = b"MPNPLAN1";

impl Plan {
    /// Read and check a plan, given either a single `.mpplan` file or a
    /// directory holding `graph.json` and `weights.bin`.
    ///
    /// One file is what ships: it sits beside the other model files, and a
    /// checksum list can cover it the same way. The two-file form is what the
    /// exporter leaves behind while a model is being worked on, because a
    /// readable `graph.json` is worth having then.
    pub fn load(path: &Path) -> Result<(Self, Vec<f32>)> {
        if path.is_dir() {
            let plan: Self = serde_json::from_slice(&std::fs::read(path.join("graph.json"))?)?;
            let weights = decode_weights(&std::fs::read(path.join("weights.bin"))?)?;
            plan.check(weights.len())?;
            plan.check_constants(&weights)?;
            return Ok((plan, weights));
        }
        Self::from_container(&std::fs::read(path)?)
    }

    /// Split a single-file plan into its description and its numbers.
    ///
    /// Layout: the magic above, the length of the description as a 32-bit
    /// little-endian count, the description as JSON, zero padding up to the
    /// next 64-byte boundary, then the weights as little-endian `f32`. The
    /// padding is what lets the weights be used without being copied first.
    fn from_container(bytes: &[u8]) -> Result<(Self, Vec<f32>)> {
        let header = 12;
        if bytes.len() < header || &bytes[..8] != MAGIC {
            return Err(Error::Format("not a .mpplan file".into()));
        }
        let json_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let json_end = header
            .checked_add(json_len)
            .ok_or_else(|| Error::Invalid("plan description length overflows".into()))?;
        if json_end > bytes.len() {
            return Err(Error::Invalid(format!(
                "plan description claims {json_len} bytes, only {} remain",
                bytes.len() - header
            )));
        }
        let plan: Self = serde_json::from_slice(&bytes[header..json_end])?;
        let start = json_end
            .checked_next_multiple_of(64)
            .ok_or_else(|| Error::Invalid("aligned description length overflows".into()))?;
        if start > bytes.len() {
            return Err(Error::Invalid(
                "plan has no weights after its padding".into(),
            ));
        }
        let weights = decode_weights(&bytes[start..])?;
        plan.check(weights.len())?;
        plan.check_constants(&weights)?;
        Ok((plan, weights))
    }

    /// Pack a plan and its weights into the single-file form.
    ///
    /// Used by the exporter; kept here so the writer and the reader of the
    /// layout sit next to each other and cannot drift apart.
    pub fn to_container(json: &[u8], weights: &[u8]) -> Vec<u8> {
        let json_len =
            u32::try_from(json.len()).expect("plan JSON exceeds the format's u32 length");
        let capacity = json
            .len()
            .checked_add(64)
            .and_then(|n| n.checked_add(weights.len()))
            .expect("plan container size overflows");
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&json_len.to_le_bytes());
        out.extend_from_slice(json);
        out.resize(out.len().next_multiple_of(64), 0);
        out.extend_from_slice(weights);
        out
    }

    fn check_constants(&self, weights: &[f32]) -> Result<()> {
        // Block bounds and lengths have already been validated by check().
        for op in &self.ops {
            if op.kind == Kind::Clip && weights[op.w.0] > weights[op.w.0 + 1] {
                return Err(Error::Invalid("reversed clip interval".into()));
            }
            if op.kind == Kind::Layernorm && weights[op.w.0 + op.oc] <= 0.0 {
                return Err(Error::Invalid(
                    "layer normalization epsilon is not positive".into(),
                ));
            }
            if !op.clip.is_empty() && weights[op.clip.0] > weights[op.clip.0 + 1] {
                return Err(Error::Invalid("reversed folded clip interval".into()));
            }
        }
        Ok(())
    }

    /// Every consistency rule, in one place. `weight_floats` is the length of
    /// the weight file.
    fn check(&self, weight_floats: usize) -> Result<()> {
        if ![FORMAT, FORMAT_V3].contains(&self.format.as_str()) {
            return Err(Error::Format(format!(
                "found {:?}, this build reads {FORMAT:?} or {FORMAT_V3:?}",
                self.format
            )));
        }
        if self.pool.is_empty() {
            return Err(Error::Invalid("no scratch buffers".into()));
        }
        let total = self
            .pool
            .iter()
            .try_fold(0usize, |n, &size| n.checked_add(size))
            .ok_or_else(|| Error::Invalid("scratch pool size overflows".into()))?;
        if total > MAX_POOL_FLOATS {
            return Err(Error::Invalid(format!(
                "scratch pool wants {total} floats, ceiling is {MAX_POOL_FLOATS}"
            )));
        }

        let slot = |i: usize, what: &str| -> Result<usize> {
            self.pool.get(i).copied().ok_or_else(|| {
                Error::Invalid(format!(
                    "{what} names buffer {i}, only {} exist",
                    self.pool.len()
                ))
            })
        };

        let input_len = shape_len(&[self.input.h, self.input.w, self.input.c])
            .ok_or_else(|| Error::Invalid("input shape is zero or overflows".into()))?;
        if slot(self.input.slot, "the input")? < input_len {
            return Err(Error::Invalid(
                "the input buffer is smaller than the image".into(),
            ));
        }
        for out in &self.outputs {
            if slot(out.slot, "an output")? < out.n {
                return Err(Error::Invalid(format!(
                    "output {:?} claims {} elements, its buffer holds fewer",
                    out.name, out.n
                )));
            }
        }

        for (i, op) in self.ops.iter().enumerate() {
            let at = |what: &str| format!("op {i} ({:?}) {what}", op.kind);
            if matches!(
                op.kind,
                Kind::Avgpool
                    | Kind::Clip
                    | Kind::Resize
                    | Kind::Depthtospace
                    | Kind::Channelslice
                    | Kind::Transpose
                    | Kind::Layernorm
                    | Kind::Centerscale
                    | Kind::Constconcat
            ) && self.format != FORMAT_V3
            {
                return Err(Error::Invalid(at("requires plan format 3")));
            }
            let input_len = shape_len(&[op.ih, op.iw, op.ic])
                .ok_or_else(|| Error::Invalid(at("input shape is zero or overflows")))?;
            let output_len = shape_len(&[op.oh, op.ow, op.oc])
                .ok_or_else(|| Error::Invalid(at("output shape is zero or overflows")))?;
            if op.kind == Kind::Concat && input_len.checked_add(op.n2) != Some(output_len) {
                return Err(Error::Invalid(at(
                    "concat lengths disagree with output shape",
                )));
            }
            if matches!(
                op.kind,
                Kind::Conv | Kind::Dwconv | Kind::Avgpool | Kind::Maxpool
            ) {
                // The execution loops use signed coordinates. Bound every
                // coordinate and checked product before reaching any SIMD path.
                if [op.kh, op.kw, op.sh, op.sw]
                    .iter()
                    .any(|&n| n == 0 || n > MAX_POOL_FLOATS)
                    || op.pt < 0
                    || op.pl < 0
                    || op.pt as usize > MAX_POOL_FLOATS
                    || op.pl as usize > MAX_POOL_FLOATS
                {
                    return Err(Error::Invalid(at(
                        "has an unsupported kernel, stride or padding",
                    )));
                }
                for (out, stride, kernel) in [(op.oh, op.sh, op.kh), (op.ow, op.sw, op.kw)] {
                    let extent = out.checked_mul(stride).and_then(|n| n.checked_add(kernel));
                    if extent.is_none_or(|n| n > isize::MAX as usize) {
                        return Err(Error::Invalid(at("window coordinates overflow")));
                    }
                }
                if op.sw.checked_mul(op.ic).is_none() {
                    return Err(Error::Invalid(at("pixel stride overflows")));
                }
            }
            if matches!(
                op.kind,
                Kind::Add
                    | Kind::Relu
                    | Kind::Prelu
                    | Kind::Sigmoid
                    | Kind::Clip
                    | Kind::Layernorm
                    | Kind::Centerscale
            ) && (op.ih, op.iw, op.ic) != (op.oh, op.ow, op.oc)
            {
                return Err(Error::Invalid(at("elementwise input/output shapes differ")));
            }
            if op.kind == Kind::Padc && (op.ih != op.oh || op.iw != op.ow) {
                return Err(Error::Invalid(at(
                    "channel padding changes spatial dimensions",
                )));
            }
            if op.kind == Kind::Transpose
                && (op.ih != 1 || op.oh != 1 || op.ow != op.ic || op.oc != op.iw)
            {
                return Err(Error::Invalid(at("transposes more than one row")));
            }
            if op.kind == Kind::Layernorm && op.w.1 != op.oc + 1 {
                return Err(Error::Invalid(at("needs gamma per channel and epsilon")));
            }
            if op.kind == Kind::Centerscale && op.ih != 1 {
                return Err(Error::Invalid(at("normalizes more than one row of points")));
            }
            if op.kind == Kind::Constconcat
                && ((op.ih, op.oh) != (1, 1)
                    || op.ic != op.oc
                    || op.w.1 == 0
                    || op.w.1.checked_add(input_len) != Some(output_len))
            {
                return Err(Error::Invalid(at("invalid constant concatenation")));
            }
            if op.kind == Kind::Resize && op.ic != op.oc {
                return Err(Error::Invalid(at("resize changes channels")));
            }
            if op.kind == Kind::Depthtospace {
                let block = op.oh / op.ih;
                if block < 2
                    || op.oh != op.ih * block
                    || op.ow != op.iw * block
                    || block.checked_mul(block).and_then(|n| n.checked_mul(op.oc)) != Some(op.ic)
                {
                    return Err(Error::Invalid(at("invalid depth-to-space shape")));
                }
            }
            if op.kind == Kind::Channelslice
                && ((op.ih, op.iw) != (op.oh, op.ow)
                    || op.begin.checked_add(op.oc).is_none_or(|end| end > op.ic))
            {
                return Err(Error::Invalid(at("invalid channel slice")));
            }
            if op.kind == Kind::Avgpool
                && (op.ic != op.oc
                    || op.pt != 0
                    || op.pl != 0
                    || op.kh > op.ih
                    || op.kw > op.iw
                    || op.oh != (op.ih - op.kh) / op.sh + 1
                    || op.ow != (op.iw - op.kw) / op.sw + 1)
            {
                return Err(Error::Invalid(at(
                    "average pool requires complete valid windows",
                )));
            }
            if op.kind == Kind::Maxpool {
                if op.ic != op.oc || op.pt != 0 || op.pl != 0 {
                    return Err(Error::Invalid(at(
                        "maxpool changes channels or requests unsupported padding",
                    )));
                }
                if (op.oh - 1) * op.sh >= op.ih || (op.ow - 1) * op.sw >= op.iw {
                    return Err(Error::Invalid(at(
                        "maxpool contains an empty output window",
                    )));
                }
            }
            if op.is_pointwise() && (op.ih != op.oh || op.iw != op.ow) {
                return Err(Error::Invalid(at(
                    "unit-stride pointwise changes spatial dimensions",
                )));
            }
            if !op.act.is_empty() && !matches!(op.kind, Kind::Conv | Kind::Dwconv | Kind::Add) {
                return Err(Error::Invalid(at(
                    "operation does not support a folded activation",
                )));
            }
            if slot(op.x, &at("input"))? < op.in_len() {
                return Err(Error::Invalid(at("reads past its input buffer")));
            }
            if slot(op.y, &at("output"))? < op.out_len() {
                return Err(Error::Invalid(at("writes past its output buffer")));
            }
            // The executor holds a mutable view of the output buffer and a
            // shared view of the inputs at the same time. That is only sound
            // because no operation writes a buffer it is also reading.
            if op.y == op.x || (matches!(op.kind, Kind::Add | Kind::Concat) && op.y == op.x2) {
                return Err(Error::Invalid(at("writes the buffer it reads")));
            }
            if matches!(op.kind, Kind::Add | Kind::Concat) {
                let need = if op.kind == Kind::Add {
                    op.out_len()
                } else {
                    op.n2
                };
                if slot(op.x2, &at("second input"))? < need {
                    return Err(Error::Invalid(at("reads past its second input buffer")));
                }
            }
            if matches!(op.kind, Kind::Conv | Kind::Dwconv) {
                if op.kh == 0 || op.kw == 0 || op.sh == 0 || op.sw == 0 {
                    return Err(Error::Invalid(at("has a zero kernel or stride")));
                }
                let taps = op
                    .kh
                    .checked_mul(op.kw)
                    .ok_or_else(|| Error::Invalid(at("kernel area overflows")))?;
                let want = if op.kind == Kind::Dwconv {
                    if op.ic != op.oc {
                        return Err(Error::Invalid(at("is depthwise but changes channel count")));
                    }
                    taps.checked_mul(op.oc)
                        .ok_or_else(|| Error::Invalid(at("weight shape overflows")))?
                } else {
                    taps.checked_mul(op.ic)
                        .and_then(|n| n.checked_mul(op.oc))
                        .ok_or_else(|| Error::Invalid(at("weight shape overflows")))?
                };
                if op.w.1 != want {
                    return Err(Error::Invalid(format!(
                        "{}: {} weights for a shape needing {want}",
                        at("weights"),
                        op.w.1
                    )));
                }
                if !op.b.is_empty() && op.b.1 != op.oc {
                    return Err(Error::Invalid(at("has a bias of the wrong length")));
                }
            }
            if op.kind == Kind::Clip && op.w.1 != 2 {
                return Err(Error::Invalid(at("wrong number of constants")));
            }
            if op.kind == Kind::Prelu && op.w.1 != op.oc {
                return Err(Error::Invalid(at("has a slope of the wrong length")));
            }
            if !op.act.is_empty() && op.act.1 != op.oc {
                return Err(Error::Invalid(at("has a folded slope of the wrong length")));
            }
            if !op.clip.is_empty()
                && (self.format != FORMAT_V3
                    || op.clip.1 != 2
                    || !matches!(op.kind, Kind::Conv | Kind::Dwconv | Kind::Add))
            {
                return Err(Error::Invalid(at("has an invalid folded clip")));
            }
            if op.kind == Kind::Padc && op.oc < op.ic {
                return Err(Error::Invalid(at("pads to fewer channels than it has")));
            }
            for (block, what) in [
                (op.w, "weights"),
                (op.b, "bias"),
                (op.act, "slopes"),
                (op.clip, "clip"),
            ] {
                if block.is_empty() {
                    continue;
                }
                let end = block.0.checked_add(block.1).ok_or_else(|| {
                    Error::Invalid(at(&format!("{what} block overflows an address")))
                })?;
                if end > weight_floats {
                    return Err(Error::Invalid(format!(
                        "{}: reaches float {end} of a {weight_floats}-float file",
                        at(what)
                    )));
                }
            }
        }
        Ok(())
    }
}

fn shape_len(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(
        1usize,
        |n, &dim| if dim == 0 { None } else { n.checked_mul(dim) },
    )
}

/// Turn the raw weight file into floats, rejecting a truncated file and any
/// value that would poison every later frame.
fn decode_weights(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(Error::Invalid(format!(
            "weights.bin is {} bytes, not a whole number of floats",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    if let Some(pos) = out.iter().position(|v| !v.is_finite()) {
        return Err(Error::Invalid(format!(
            "weights.bin holds {} at float {pos}",
            out[pos]
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> Plan {
        Plan {
            format: FORMAT.to_owned(),
            source: String::new(),
            input: Input {
                slot: 0,
                h: 2,
                w: 2,
                c: 4,
            },
            outputs: vec![Output {
                slot: 1,
                n: 16,
                name: "y".into(),
            }],
            pool: vec![16, 16],
            ops: vec![Op {
                begin: 0,
                clip: Block::default(),
                kind: Kind::Conv,
                x: 0,
                x2: 0,
                y: 1,
                ih: 2,
                iw: 2,
                ic: 4,
                oh: 2,
                ow: 2,
                oc: 4,
                kh: 1,
                kw: 1,
                sh: 1,
                sw: 1,
                pt: 0,
                pl: 0,
                w: Block(0, 16),
                b: Block(16, 4),
                act: Block::default(),
                n2: 0,
            }],
        }
    }

    #[test]
    fn a_consistent_plan_passes() {
        assert!(minimal().check(20).is_ok());
    }

    #[test]
    fn a_weight_block_past_the_file_is_refused() {
        let plan = minimal();
        let err = plan.check(18).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)), "{err}");
    }

    #[test]
    fn a_buffer_that_cannot_hold_its_tensor_is_refused() {
        let mut plan = minimal();
        plan.pool[1] = 8;
        assert!(plan.check(20).is_err());
    }

    #[test]
    fn a_missing_buffer_is_refused() {
        let mut plan = minimal();
        plan.ops[0].y = 7;
        assert!(plan.check(20).is_err());
    }

    #[test]
    fn a_weight_count_that_contradicts_the_shape_is_refused() {
        let mut plan = minimal();
        plan.ops[0].w = Block(0, 12);
        assert!(plan.check(20).is_err());
    }

    #[test]
    fn a_newer_format_is_refused_rather_than_guessed_at() {
        let mut plan = minimal();
        plan.format = FORMAT_V3.into();
        assert!(plan.check(20).is_ok());
        plan.format = "mediapipe-native-plan/4".into();
        assert!(matches!(plan.check(20), Err(Error::Format(_))));
    }

    #[test]
    fn format_three_additions_are_refused_in_format_one() {
        let mut plan = minimal();
        plan.ops[0].clip = Block(0, 2);
        assert!(plan.check(20).is_err(), "folded clip");
        let mut plan = minimal();
        let op = &mut plan.ops[0];
        (op.kind, op.w, op.b, op.oc) = (Kind::Channelslice, Block(0, 0), Block(0, 0), 2);
        plan.outputs[0].n = 8;
        assert!(plan.check(20).is_err(), "channel slice");
        plan.format = FORMAT_V3.into();
        assert!(plan.check(20).is_ok(), "slice valid in format 3");
    }

    #[test]
    fn malformed_format_three_operations_are_refused() {
        // One 2x2x4 -> ... operation with no weights, in a format-3 plan.
        let shaped = |kind: Kind, (oh, ow, oc): (usize, usize, usize), w: Block| {
            let mut plan = minimal();
            plan.format = FORMAT_V3.into();
            let op = &mut plan.ops[0];
            (op.kind, op.oh, op.ow, op.oc, op.w, op.b) = (kind, oh, ow, oc, w, Block::default());
            plan.outputs[0].n = oh * ow * oc;
            plan
        };
        // Four one-row points of four channels after `head` constant floats.
        let concat = |head: usize| {
            let mut plan = shaped(Kind::Constconcat, (1, 5, 4), Block(0, head));
            (plan.ops[0].ih, plan.ops[0].iw, plan.input.h, plan.input.w) = (1, 4, 1, 4);
            plan.pool = vec![20, 20];
            plan
        };
        let bad = [
            (
                "depth to space that does not divide",
                shaped(Kind::Depthtospace, (4, 4, 2), Block(0, 0)),
            ),
            (
                "depth to space of block one",
                shaped(Kind::Depthtospace, (2, 2, 4), Block(0, 0)),
            ),
            (
                "transpose of two rows",
                shaped(Kind::Transpose, (1, 4, 4), Block(0, 0)),
            ),
            ("slice past the channels", {
                let mut plan = shaped(Kind::Channelslice, (2, 2, 2), Block(0, 0));
                plan.ops[0].begin = 3;
                plan
            }),
            (
                "layer norm without epsilon",
                shaped(Kind::Layernorm, (2, 2, 4), Block(0, 4)),
            ),
            ("constant concatenation of the wrong length", concat(3)),
        ];
        for (what, plan) in bad {
            assert!(plan.check(20).is_err(), "{what} was accepted");
        }
        assert!(shaped(Kind::Depthtospace, (4, 4, 1), Block(0, 0))
            .check(20)
            .is_ok());
        assert!(concat(4).check(20).is_ok());
        // Constants: a non-positive epsilon and a reversed folded clip.
        let mut norm = shaped(Kind::Layernorm, (2, 2, 4), Block(0, 5));
        assert!(norm.check(20).is_ok());
        let mut weights = vec![1.0; 20];
        weights[4] = 0.0;
        assert!(norm.check_constants(&weights).is_err());
        norm.ops[0].kind = Kind::Conv;
        (norm.ops[0].w, norm.ops[0].clip) = (Block(0, 16), Block(16, 2));
        let mut weights = vec![0.0; 20];
        (weights[16], weights[17]) = (6.0, 0.0);
        assert!(norm.check(20).is_ok());
        assert!(norm.check_constants(&weights).is_err());
    }

    #[test]
    fn weights_holding_not_a_number_are_refused() {
        let mut raw = 1.0f32.to_le_bytes().to_vec();
        raw.extend_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_weights(&raw).is_err());
    }

    #[test]
    fn a_packed_plan_round_trips() {
        let plan = minimal();
        let json = serde_json::to_vec(&serde_json::json!({
            "format": FORMAT,
            "source": "",
            "input": {"slot": 0, "h": 2, "w": 2, "c": 4},
            "outputs": [{"slot": 1, "n": 16, "name": "y"}],
            "pool": [16, 16],
            "ops": [{
                "kind": "conv", "x": 0, "y": 1,
                "ih": 2, "iw": 2, "ic": 4, "oh": 2, "ow": 2, "oc": 4,
                "kh": 1, "kw": 1, "sh": 1, "sw": 1,
                "w": [0, 16], "b": [16, 4]
            }]
        }))
        .unwrap();
        let weights: Vec<u8> = (0..20u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let packed = Plan::to_container(&json, &weights);
        let (read_back, values) = Plan::from_container(&packed).unwrap();
        assert_eq!(read_back.ops.len(), plan.ops.len());
        assert_eq!(values.len(), 20);
        assert_eq!(values[3], 3.0);
    }

    #[test]
    fn a_file_that_is_not_a_plan_is_refused() {
        assert!(matches!(
            Plan::from_container(b"not a plan at all, really"),
            Err(Error::Format(_))
        ));
    }

    #[test]
    fn a_truncated_weight_file_is_refused() {
        assert!(decode_weights(&[0, 1, 2]).is_err());
    }

    #[test]
    fn arithmetic_overflows_are_rejected_in_release_too() {
        let mut p = minimal();
        p.pool = vec![usize::MAX, usize::MAX];
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.input.h = usize::MAX;
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.ops[0].oh = usize::MAX;
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.ops[0].sw = usize::MAX;
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.ops[0].w = Block(usize::MAX, 16);
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.ops[0].kind = Kind::Concat;
        p.ops[0].n2 = usize::MAX;
        assert!(p.check(20).is_err());
    }
    #[test]
    fn zero_shapes_and_incoherent_operations_are_rejected() {
        for kind in [
            Kind::Conv,
            Kind::Dwconv,
            Kind::Maxpool,
            Kind::Add,
            Kind::Prelu,
            Kind::Padc,
        ] {
            let mut p = minimal();
            p.ops[0].kind = kind;
            p.ops[0].ic = 0;
            assert!(p.check(20).is_err());
        }
        for kind in [Kind::Add, Kind::Relu, Kind::Sigmoid, Kind::Padc] {
            let mut p = minimal();
            p.ops[0].kind = kind;
            p.ops[0].ow = 1;
            assert!(p.check(20).is_err());
        }
        let mut p = minimal();
        p.ops[0].ow = 1;
        assert!(p.check(20).is_err());
        let mut p = minimal();
        p.ops[0].kind = Kind::Prelu;
        p.ops[0].act = Block(16, 4);
        assert!(p.check(20).is_err());
    }
    #[test]
    fn maxpool_empty_windows_and_unsupported_padding_are_rejected() {
        for (stride, pad) in [(0, 0), (3, 0), (1, 1)] {
            let mut p = minimal();
            p.ops[0].kind = Kind::Maxpool;
            p.ops[0].sh = stride;
            p.ops[0].pt = pad;
            assert!(p.check(20).is_err());
        }
    }
}
