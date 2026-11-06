# mediapipe-native

Run Google MediaPipe's face, hand, pose and selfie-segmentation models on the
CPU from Rust, with no inference runtime underneath. Fast enough for head and
eye input on a 2011 laptop.

- **One small dependency tree.** Only `serde` and `serde_json`. No TensorFlow
  Lite, ONNX Runtime or OpenVINO.
- **Old CPUs are first-class.** Kernels for AVX2+FMA, AVX, SSE4.1 and plain
  Rust, chosen at run time. The AVX path is tuned on Sandy Bridge, which has
  no AVX2, FMA or F16C.
- **Predictable frames.** One thread, buffers reserved at load time, zero
  allocations per frame, from the first frame on.
- **Checkable results.** Committed reference outputs, bit-for-bit comparison
  within each instruction set, and parity tools against TensorFlow Lite and a
  float64 reference.

## What it runs

| Model | Plans | Input |
|---|---|---|
| Face detector (BlazeFace short-range) | `face_detector` | 128×128 |
| Face mesh, 478 points | `face_landmarks` | 256×256 |
| Palm detector, hand landmarks | `hand_detector`, `hand_landmarks` | 192², 224² |
| Pose detector, pose landmarks + segmentation | `pose_detector`, `pose_landmarks` | 224², 256² |
| Holistic face mesh, blendshapes, hand ROI | three plans | 192², 146×2, 256² |
| Selfie segmentation (person mask), square and landscape | `selfie_segmenter`, `selfie_segmenter_landscape` | 256², 256×144 |

The crate runs the networks only: detection decoding, cropping, tracking and
the task graph that chains the models belong to your application.

## How fast is it?

Against LiteRT 2.3 (TensorFlow Lite, with its XNNPACK CPU backend), the
runtime MediaPipe itself uses. Same `.tflite` models, one thread pinned to one
core, CPU time per frame, median of three runs of 300 frames.

| | Laptop | 2011 laptop |
|---|---|---|
| CPU | AMD Ryzen 5 5600H (Zen 3, AVX2) | Intel Core i3-2375M (Sandy Bridge, AVX only, 1.5 GHz) |
| OS | BigLinux with BigDesktop, Linux 7.2 | BigLinux with BigDesktop, Linux 7.2 |

| ms per frame | Laptop: mediapipe-native | Laptop: LiteRT | 2011 laptop: mediapipe-native | 2011 laptop: LiteRT |
|---|---:|---:|---:|---:|
| Face detector | **0.88** | 0.92 | **5.1** | 6.3 |
| Face mesh | **2.64** | 3.52 | **17.3** | 24.9 |
| Palm detector | **6.64** | 8.65 | **43.3** | 60.9 |
| Hand landmarks | **6.78** | 7.24 | **43.6** | 53.0 |
| Pose detector | **8.79** | 9.22 | **56.1** | 67.1 |
| Pose landmarks | **5.07** | 7.09 | **33.9** | 51.8 |
| Holistic face mesh | **0.85** | 1.12 | **5.4** | 7.6 |
| Face blendshapes | 0.71 | **0.61** | 4.7 | 4.7 |
| Hand ROI refinement | **0.38** | 0.73 | **2.6** | 4.0 |
| Selfie segmentation, square | **1.72** | 2.83 | **11.0** | 17.6 |
| Selfie segmentation, landscape | **0.95** | 1.57 | **5.9** | 9.6 |

Faster on ten of the eleven models: 1.04 to 1.9 times on the laptop, 1.2 to
1.6 times on the i3. The blendshapes model, a small MLP-Mixer over 146
landmarks, is the exception: LiteRT wins it on the laptop and ties on the i3.
The face mesh runs at 58 frames a second on one core of a 2011 laptop.

| Install | mediapipe-native | LiteRT |
|---|---:|---:|
| Runtime | linked into your program, under 1 MB | 59 MB, plus numpy and Python for its Python API |
| The eleven plans the package installs | 42 MB | the same models in `.tflite` bundles |

[Performance](docs/performance.md) has the method, every kernel decision and
what was tried and rejected.

## Why is it fast?

- **The pointwise tile.** Two thirds of a MediaPipe graph is 1×1
  convolution, bound by load ports rather than multipliers. The AVX2 tile
  computes six pixels by sixteen channels, so each weight load feeds more
  multiply-adds, and retires about 1.5 FMAs per cycle over a whole frame on
  a Core i5-13400.
- **Weights packed once.** Pointwise weights are repacked at load into
  16-column panels, so a panel never maps onto a handful of L1 sets.
- **Depthwise without waste.** Windows load each input column once per row of
  outputs instead of once per tap, and a depthwise layer feeding a pointwise
  one computes its rows just ahead of them, so they are read from cache.
- **Activations in place.** ReLU, ReLU6, PReLU, hard-swish and sigmoid are
  applied while a convolution's output is still in registers or L1.
- **Sandy Bridge gets its own kernels.** Its L1 cannot serve two reads of the
  same bank in one cycle, and a stride of 32 channels puts every broadcast in
  one bank: those tiles copy their pixels to a skewed buffer first. Narrow
  layers use a two-pixel by 32-channel tile while the weights fit its 256 KB
  L2.
- **Nothing per frame.** Every buffer is reserved at load time, tensors share
  memory by lifetime, and denormals are flushed for the length of a run.

Each of these is bit-identical within an instruction set and was kept only
when whole frames got faster on both an old and a new CPU.

## Build

You need Rust 1.93 or newer and a C linker.

```bash
cargo build --release --locked
```

Model weights are not in the repository: they keep Google's Apache-2.0 licence
and are converted into *plans* (`.mpplan` files) under the ignored `plans/`
directory. [Getting started](docs/getting-started.md) shows how to download the
MediaPipe bundles, export them, and run the tests. The package built by
[`pkgbuild/PKGBUILD`](pkgbuild/PKGBUILD) does the same from pinned bundles and
installs the plans under `/usr/share/mediapipe-native/` with `mpbench`. Then
time a plan:

```bash
target/release/mpbench plans/face_landmarks/face_landmarks_detector.mpplan 300 --json
```

## Use

```rust,no_run
use mediapipe_native::Model;
use std::path::Path;

fn main() -> Result<(), mediapipe_native::Error> {
    // Load once at startup; loading allocates, running never does.
    let mut model = Model::load(Path::new(
        "plans/face_landmarks/face_landmarks_detector.mpplan",
    ))?;
    let (h, w, c) = model.input_shape();
    let rgb = vec![0.0f32; h * w * c]; // your cropped, normalized RGB frame

    // Per frame: refill the input (its storage is reused as scratch), run, read.
    model.input_mut().copy_from_slice(&rgb);
    model.run();
    let landmarks = model.output(0); // 478 × XYZ
    assert_eq!(landmarks.len(), 478 * 3);
    Ok(())
}
```

Inputs are NHWC `f32` in RGB order, in each model's own value range. Outputs
borrow the model until the next mutable call, so copy what you need before
writing the next frame. A model can be built on one thread and run on another.
[Architecture](docs/architecture.md) covers the contracts, plan formats and
instruction-set tiers.

Never ship a binary built with a global `target-cpu=native` or forced target
features: choosing the tier at run time is what keeps one binary correct on
every CPU. `MEDIAPIPE_NATIVE_TIER=0..3` caps the tier for testing.

## Development

```bash
RUN_DEBUG=1 tools/verify.sh
```

runs formatting, Clippy, the tests once per instruction-set tier the CPU
supports, doctests and the Python tool tests. The [documentation index](docs/README.md)
maps every guide, and [CONTRIBUTING](CONTRIBUTING.md) explains what a change
needs to be accepted.

## Credits and license

The models are the work of Google's MediaPipe team: BlazeFace, the face mesh,
MediaPipe Hands, BlazePose, BlazePose GHUM Holistic and MediaPipe Selfie
Segmentation, published under Apache-2.0. [NOTICE](NOTICE) lists their papers
and authors, the code this builds on, and the licences of the models and
dependencies.

The Rust implementation is by Bruno Gonçalves ([BigLinux](https://www.biglinux.com.br))
and is licensed MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). The folded sigmoid and hard-swish follow
XNNPACK's kernels and keep its BSD-3-Clause licence, which NOTICE carries.
