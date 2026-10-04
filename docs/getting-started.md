# Getting started

This guide takes you from a clone to a verified build: toolchain, model plans,
the input contract, and the gates that prove a change kept the numbers.

## Toolchain

The crate needs Rust 1.93 or newer and a C linker.

```bash
cargo build --release --locked --bin mpbench --example probe
RUN_DEBUG=1 tools/verify.sh
```

`tools/verify.sh` refuses global `target-cpu` or `target-feature` flags and
runs formatting, Clippy, the test suite once per instruction-set tier the CPU
supports, the doctests, the tier-2 instruction check and the Python tool tests.
It needs the plans below. Each step is bounded by `GATE_TIMEOUT` seconds
(default 900); logs go to `target/verify` or `EVIDENCE_DIR`. For an offline
build, run `cargo vendor` once and add the configuration it prints.

## Model plans

The executor reads *plans*: a model's operations, shapes and weights converted
offline into one `.mpplan` file. Plans are not in Git because the weights keep
their upstream licence (see [NOTICE](../NOTICE)); they live under the ignored
`plans/` directory. Tests that need a plan fail when it is missing, so a green
run never means inference was skipped.

| Plans | Exporter | Source |
|---|---|---|
| `face_detector`, `face_landmarks` | `tools/export_tflite.py --format 1` | `.tflite` files inside the MediaPipe Face Landmarker bundle |
| hand, pose, holistic (seven plans) | `tools/export_tflite.py` | `.tflite` files inside the MediaPipe Tasks bundles |

The hand, pose and holistic bundles download directly from Google's model
storage; [the model guide](models/hand-pose-holistic.md) has the URLs, hashes
and the plan directory names the tests expect. The face plans keep plan format 1,
whose schedule their reference outputs pin:

```bash
mkdir -p plans/_downloads && cd plans/_downloads
curl -fLO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -fLO https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
curl -fLO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
sha256sum *.task   # compare with docs/models/hand-pose-holistic.md
python3 -c "import zipfile; [zipfile.ZipFile(f).extractall(f[:-5]) for f in ('hand_landmarker.task', 'holistic_landmarker.task', 'face_landmarker.task')]"
cd ../..
python3 tools/export_tflite.py plans/_downloads/face_landmarker/face_detector.tflite \
  plans/face_detector --format 1 --source "MediaPipe face_landmarker.task face_detector.tflite"
python3 tools/export_tflite.py plans/_downloads/face_landmarker/face_landmarks_detector.tflite \
  plans/face_landmarks --format 1 --source "MediaPipe face_landmarker.task face_landmarks_detector.tflite"
python3 tools/export_tflite.py plans/_downloads/holistic_landmarker/pose_landmarks_detector.tflite \
  plans/pose_landmarks --source "MediaPipe holistic_landmarker.task pose_landmarks_detector.tflite"
```

TensorFlow (for the exporter) is needed only
for this step. A `.tflite` file is not a plan; never rename one or substitute a
different face model for the one a plan was exported from.

## Input and output contract

`Model::input_shape()` returns height, width and channels of the NHWC input.
Fill the input before every frame: its storage is reused as scratch later in
the graph. Use RGB and each model's own value range; the face detector and the
face mesh expect different normalization, so read the export metadata instead
of applying one normalizer to both.

Outputs borrow model storage until the next mutable call. Copy what you need
before writing the next input. The face mesh returns 478 × XYZ plus global
scalars: there is no per-landmark confidence, and face presence is one value for
the whole face.

## Proving a change kept the numbers

`cargo test --test reference` compares every plan against committed outputs in
`tests/reference/` with a tolerance of 0.005 in output units. For kernel work,
also compare full outputs bit for bit against a baseline build. Results are
bit-identical within one instruction-set tier, not across tiers, so each tier
runs in its own process:

```bash
mkdir -p artifacts
for tier in 0 1 2 3; do
  for plan in face_landmarks/face_landmarks_detector face_detector/face_detector; do
    MEDIAPIPE_NATIVE_TIER=$tier /path/to/baseline-probe plans/$plan.mpplan \
      artifacts/before/$plan/tier$tier
    MEDIAPIPE_NATIVE_TIER=$tier target/release/examples/probe plans/$plan.mpplan \
      artifacts/after/$plan/tier$tier
  done
done
python3 tools/compare_corpus.py artifacts/before artifacts/after \
  --output artifacts/differential.json
```

`MEDIAPIPE_NATIVE_TIER` only caps the tier: on a CPU without AVX2, a cap of 3
still runs tier 2, so record a missing tier as not run rather than emulated.
Synthetic equality proves the arithmetic did not change; it says nothing about
landmark accuracy on real faces.

To check a plan against the original model, `tools/check_tflite_parity.py`
compares the native outputs with TensorFlow Lite and a float64 NumPy run of the
same graph (`tools/run_numpy.py`); `tools/check_parity.py` compares the face
plans against ONNX Runtime using the input `mpbench` writes next to a plan.
