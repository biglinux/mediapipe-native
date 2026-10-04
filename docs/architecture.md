# Architecture

The crate does one thing: given a plan and an input tensor, compute the model's
outputs on one CPU thread without allocating. Everything around inference —
camera, cropping policy, tracking, user input — belongs to the application that
embeds it. This page shows where each responsibility lives and which rules the
code enforces.

## The inference core

```text
.mpplan file ──> plan.rs ──> lib.rs (Model) ──> ops.rs / extra_ops.rs
                 validate     load, schedule,    SIMD and scalar kernels
                 everything   run one frame
```

| File | Owns |
|---|---|
| `src/plan.rs` | The plan container and its validation: shapes, sizes, weight offsets, buffer aliasing and which operations each format allows. Nothing reaches a kernel unchecked. |
| `src/lib.rs` | `Model`: aligned weight and scratch storage, the per-layer schedule decided at load time, and `run()`. |
| `src/ops.rs` | Convolution, depthwise, activation and elementwise kernels for four instruction-set tiers, plus tier detection. |
| `src/extra_ops.rs` | Layers only format-3 plans contain: average pooling, resize, depth-to-space, layer normalization and the blendshape model's input layers. |
| `src/bin/mpbench.rs` | Timing, allocation counting and a per-operation profile for one plan. |

`Model::load` may allocate and repack weights (for example into 16-column panels
on AVX tiers, a permutation that keeps the arithmetic identical). `run()` never
allocates, takes `&mut self`, and wraps the frame in a guard that enables
flush-to-zero and restores the caller's MXCSR on exit. Unsafe leaf kernels rely
on the extents their safe wrappers assert and on the tier dispatch having
checked the instruction set; each `unsafe` block states that proof.

Load time also fuses operations when nothing can observe the difference:
Add→Relu and Padc→Add→Relu run as one pass, and a pointwise convolution read
only by its residual add keeps its rows in a scratch buffer, only if no later
operation or model output reads the tensors that are skipped.

## Plan formats

A plan is a JSON description plus a 64-byte-aligned block of `f32` weights in
one file. The description carries a format tag, and a binary refuses a format
or an operation it does not know rather than guessing.

| Format | Produced by | Adds |
|---|---|---|
| `mediapipe-native-plan/1` | `tools/export_tflite.py --format 1` (or `tools/export_graph.py` from ONNX) | convolution, depthwise, ReLU/PReLU, add, max pooling, channel padding, sigmoid and concatenation: the face models |
| `mediapipe-native-plan/3` | `tools/export_tflite.py` | average pooling, clip, resize, depth-to-space, channel slice, transpose, layer normalization, input centring, constant concatenation, folded RELU6 and row-grouped scheduling |

Format 3 is opt-in: format-1 plans keep the schedule and the bits their
reference outputs pin, and `Plan::check` rejects any format-3 operation inside
them.

## Instruction-set tiers

`ops::simd_tier()` picks the widest tier the CPU and OS support, once per
process: 3 = AVX2 + FMA, 2 = AVX, 1 = SSE4.1, 0 = scalar. Tier 2 must keep
working on Sandy Bridge-class CPUs that have AVX but no AVX2, FMA or F16C, so no
tier-2 kernel may use those. `MEDIAPIPE_NATIVE_TIER` caps the choice for testing;
it cannot enable a feature the CPU lacks.

Results are bit-identical across runs within a tier and differ between tiers
(fused multiply-add rounds once; some tiles add bias first, others last). Never
reorder a reduction, bias add or comparison inside a tier: the reference outputs
and the bit-exact differential tests will catch it.
