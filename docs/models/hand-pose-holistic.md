# Hand, pose and holistic models

The models inside the MediaPipe Hand Landmarker and Holistic Landmarker bundles
run on the same executor as the face models: no TFLite, ONNX Runtime or
OpenVINO at run time, one worker, FP32, zero allocations per frame, every SIMD
tier. Costs per frame are in [performance](../performance.md).

## What is covered

| Plan directory | Source model (bundle/file) | Input | Ops | Weights MB | Scratch MB | Outputs (values) |
|---|---|---|---|---|---|---|
| `plans/hand_detector` | `hand_landmarker/hand_detector.tflite` | 192×192×3 | 104 | 4.55 | 3.87 | 36288, 2016 |
| `plans/hand_landmarks` | `hand_landmarker/hand_landmarks_detector.tflite` | 224×224×3 | 63 | 10.87 | 6.82 | 63, 1, 1, 63 |
| `plans/hand_roi_refinement` | `holistic_landmarker/hand_roi_refinement.tflite` | 256×256×3 | 56 | 0.11 | 1.57 | 4 |
| `plans/pose_detector` | `holistic_landmarker/pose_detector.tflite` | 224×224×3 | 97 | 11.86 | 7.58 | 27048, 2254 |
| `plans/pose_landmarks` | `holistic_landmarker/pose_landmarks_detector.tflite` | 256×256×3 | 121 | 5.46 | 12.12 | 195, 1, 65536, 159744, 117 |
| `plans/holistic_face_landmarks` | `holistic_landmarker/face_landmarks_detector.tflite` | 192×192×3 | 67 | 1.11 | 1.77 | 1434, 1 |
| `plans/face_blendshapes` | `holistic_landmarker/face_blendshapes.tflite` | 1×146×2 | 57 | 1.80 | 0.30 | 52 |

Outputs are in the TFLite model's order. Hand landmarks: 21×3 image
landmarks, presence, handedness, 21×3 world landmarks. Pose landmarks: 39×5
landmarks, presence, 256×256 segmentation logits, 64×64×39 heatmap, 39×3 world
landmarks. Detectors: anchor regressors and scores.

Two bundle files need no plan of their own. The Holistic `face_detector.tflite`,
exported with `tools/export_tflite.py`, gives bit-identical outputs to the existing
`plans/face_detector` on all 24 probe cases: it is the same BlazeFace model.
The Holistic `hand_landmarks_detector.tflite` exports to weights and a graph
identical to the Hand Landmarker's. The Holistic face mesh is a different,
192×192 model from the 256×256 `plans/face_landmarks`, hence its own plan.

Not implemented: the graph logic around the models — anchor decoding,
non-maximum suppression, ROI rotation/cropping, landmark smoothing, choosing the
146 landmarks the blendshape model reads, and the task graph that chains them.
As for the face models, that belongs to the application.

## Provenance and licence

Version 1 of each bundle in Google's MediaPipe model storage; on 2026-10-04 the
`latest` URLs served the same files:

| Bundle | URL | SHA-256 |
|---|---|---|
| `hand_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `holistic_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task` | `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8` |
| `face_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

A `.task` file is a zip archive of `.tflite` models:

| File in the bundle | SHA-256 |
|---|---|
| `hand_landmarker/hand_detector.tflite` | `945f713bc23570bd4ed60f848c401dc8eaf95713183d43ba14cf12e467d27a7d` |
| `hand_landmarker/hand_landmarks_detector.tflite` | `6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9` |
| `holistic_landmarker/hand_landmarks_detector.tflite` | `11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3` |
| `holistic_landmarker/hand_roi_refinement.tflite` | `d40b15e15f93f6c909a3cfb881ce16c9ff9aa6d57417a0c906a6624f1f60b60c` |
| `holistic_landmarker/pose_detector.tflite` | `9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c` |
| `holistic_landmarker/pose_landmarks_detector.tflite` | `1150dc68a713b80660b90ef46ce4e85c1c781bb88b6e3512cc64e6a685ba5588` |
| `holistic_landmarker/face_detector.tflite` | `bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8` |
| `holistic_landmarker/face_landmarks_detector.tflite` | `bc5ee5de06d8c3a5465c3155227615b164480a52105a2b3df5748250ab4d914f` |
| `holistic_landmarker/face_blendshapes.tflite` | `4f36dded049db18d76048567439b2a7f58f1daabc00d78bfe8f3ad396a2d2082` |
| `face_landmarker/face_detector.tflite` | `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` |
| `face_landmarker/face_landmarks_detector.tflite` | `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5` |

The official model cards linked from the
Hand Landmarker and Holistic Landmarker pages, "Model Card Hand Tracking
(Lite_Full) with Fairness Oct 2021" and "Model Card BlazePose GHUM 3D", state
"Licensed under Apache License, Version 2.0". The face models follow the Face
Mesh V2 card already used for the face plans. Bundles and plans stay under the
ignored `plans/`; nothing here commits model data.

## How a TFLite model becomes a plan

`tools/export_tflite.py` (development only; needs TensorFlow's bundled TFLite
interpreter and flatbuffer schema) reads the `.tflite` and writes the same
`graph.json` / `weights.bin` / `.mpplan` files as `tools/export_graph.py`,
whose PReLU folding, buffer assignment and container writer it reuses. TFLite
is already NHWC, so reshapes are aliases and filters are only reordered.

- **Weights.** Bundles store float16 weights behind `DEQUANTIZE` and sparse
  ones behind `DENSIFY`. The interpreter materialises them (without the XNNPACK
  delegate, which would leave those tensors unwritten); the exact float32 values
  are exported. Float16 to float32 is exact; nothing is re-rounded.
- **Folds, all exact.** A spatial `PAD` read only by convolutions becomes their
  top/left padding. A per-channel constant `MUL`, with an optional following
  constant `ADD` (inference batch norm), becomes a 1×1 depthwise convolution
  computing `t + x·s`. `FULLY_CONNECTED` is a 1×1 convolution, a spatial `MEAN`
  a whole-image average pool, a leading or whole `STRIDED_SLICE` an alias.
  `RELU6` becomes a clamp to [0, 6] folded into its producer.
- **Blendshapes.** The MLP-Mixer's expanded layer normalization (ten TFLite
  ops) and its input normalization (seven ops) are recognised as exact wired
  patterns and emitted as one `layernorm` / `centerscale` operation each;
  anything wired differently is refused.
- Every other TFLite operation, activation or attribute is refused by name
  rather than approximated.

These plans use format 3 (`mediapipe-native-plan/3`, see
[architecture](../architecture.md#plan-formats)). It adds `avgpool`, `clip`,
`resize` (bilinear, half-pixel centres, TFLite reference arithmetic),
`depthtospace`, `channelslice`, `transpose`, `layernorm`, `centerscale`,
`constconcat` and the folded `clip` field; format-1 plans reject all of them.
Format 3 also groups pointwise rows into runs of at least 48 pixels (7×7 and
14×14 layers otherwise never fill a register tile); format 1 keeps whole-layer
scheduling, which its reference outputs pin.

## Checking a plan

```bash
python3 tools/export_tflite.py plans/_downloads/hand_landmarker/hand_landmarks_detector.tflite \
  plans/hand_landmarks --source "MediaPipe hand_landmarker.task hand_landmarks_detector.tflite"
cargo build --release --locked --example probe
for tier in 0 1 2 3; do
  MEDIAPIPE_NATIVE_TIER=$tier python3 tools/check_tflite_parity.py \
    plans/_downloads/hand_landmarker/hand_landmarks_detector.tflite \
    plans/hand_landmarks/hand_landmarks_detector.mpplan --work artifacts/parity/hand_landmarks/tier$tier
done
```

`tools/check_tflite_parity.py` runs the probe's 24 deterministic images through
the native plan, through TFLite (XNNPACK and reference kernels, one thread) and
through `tools/run_numpy.py` in float64 on the same graph. It fails if the
float64 run disagrees with TFLite beyond 1e-3 of the output's range (the plan
would not describe the model), and passes an output when the native float64
error is within 0.005 or at most twice the worse TFLite kernel's. A fixed gap
alone cannot work here: on these synthetic images the pose segmentation logits
reach ±900, and TFLite's own kernels differ from each other by 0.6.
`--native DIR` checks probe outputs produced on another machine.

For `face_blendshapes` pass `--skip-cases 1,2,3,7`: constant images are
landmark sets with every point in one place, where the model divides zero by
zero. On AVX tiers ReLU turns that NaN into 0, as TFLite does; tiers 0 and 1
use the scalar ReLU, which preserves NaN. Real landmarks never coincide.

`cargo test --test reference` checks all nine plans against tier-0 goldens in
`tests/reference/` with a 0.005 tolerance (segmentation compared as
sigmoid probability), and `--test runtime` proves zero allocations for all of
them. Both fail when a plan is missing.

## Limits

- Parity is measured on synthetic images, not on labelled video: it shows the
  native arithmetic reproduces the model, not how accurate the model is.
- TFLite timings are a same-host reference with its default XNNPACK delegate
  and one thread, measured from Python; they are not an OpenVINO or
  multi-threaded comparison.
- Plans exported from other bundle versions must pass the parity check again.
