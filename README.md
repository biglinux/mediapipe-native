# mediapipe-native

Run Google MediaPipe's face, hand and pose models on the CPU from Rust, with no
inference runtime underneath — fast enough for head and eye input on a 2011
laptop.

- **One small dependency tree.** Only `serde` and `serde_json`. No TensorFlow
  Lite, ONNX Runtime or OpenVINO.
- **Old CPUs are first-class.** Kernels for AVX2+FMA, AVX, SSE4.1 and scalar,
  chosen at run time. The AVX path works on Sandy Bridge, which has no AVX2,
  FMA or F16C.
- **Predictable frames.** One thread, buffers reserved at load time, zero
  allocations per frame — from the first frame on, and after moving the model to
  another thread.
- **Checkable results.** Committed reference outputs, bit-for-bit comparison
  within each instruction set, and parity tools against TensorFlow Lite and a
  float64 reference.

## What it runs

| Model | Plans | Input | i5-13400 (AVX2) | i3-2375M (AVX) |
|---|---|---|---:|---:|
| Face detector (BlazeFace short-range) | `face_detector` | 128×128 | 0.83 ms | 5.1 ms |
| Face mesh, 478 points | `face_landmarks` | 256×256 | 2.62 ms | 17.2 ms |
| Palm detector, hand landmarks | `hand_detector`, `hand_landmarks` | 192², 224² | 6.27, 6.51 ms | 43.2, 43.8 ms |
| Pose detector, pose landmarks + segmentation | `pose_detector`, `pose_landmarks` | 224², 256² | 8.36, 4.94 ms | 56.1, 33.9 ms |
| Holistic face mesh, blendshapes, hand ROI | three plans | 192², 146×2, 256² | 0.80, 0.64, 0.38 ms | 5.4, 4.7, 2.5 ms |

CPU time per frame on one thread, median of six interleaved runs
([performance](docs/performance.md)). The crate runs the networks only:
detection decoding, cropping, tracking and the task graph that chains the models
belong to your application.

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
installs the nine plans under `/usr/share/mediapipe-native/` with `mpbench`.
Then time a plan:

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
writing the next frame. [Architecture](docs/architecture.md) covers the
contracts, plan formats and instruction-set tiers.

Never ship a binary built with a global `target-cpu=native` or forced target
features: tier selection at run time is what keeps one binary correct on every
CPU. `MEDIAPIPE_NATIVE_TIER=0..3` caps the tier for testing.

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
MediaPipe Hands, BlazePose and BlazePose GHUM Holistic, published under
Apache-2.0. [NOTICE](NOTICE) lists their papers and authors, the code this
builds on, and the licences of the models and dependencies.

The Rust implementation is by Bruno Gonçalves ([BigLinux](https://www.biglinux.com.br))
and is licensed MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)).
