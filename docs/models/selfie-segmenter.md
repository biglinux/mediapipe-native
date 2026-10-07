# Selfie segmenter

MediaPipe's selfie segmenter (landscape) gives a person mask for webcam
background blur. It runs on the same executor as the other models: no TFLite
at run time, one worker, FP32, zero allocations per frame, every SIMD tier.
Costs per frame are in [performance](../performance.md).

## What is covered

| Plan directory | Source model | Input | Ops | Weights MB | Scratch MB | Output (values) |
|---|---|---|---|---|---|---|
| `plans/selfie_segmenter_landscape` | `selfie_segmenter_landscape.tflite` | 144×256×3 | 93 | 0.43 | 2.34 | 36864 |

- **Input:** 144 rows of 256 RGB pixels, NHWC `f32` in [0, 1]: each 8-bit
  channel divided by 255. The model's TFLite metadata says so
  (`NormalizationOptions` mean 0, std 255; input stats min 0, max 1), as does
  the model card ("RGB with values in [0.0, 1.0]"). Resize or crop the camera
  frame to 256×144 first; the model has no notion of aspect ratio.
- **Output:** `segment_back`, 144×256×1, the probability in [0, 1] that each
  pixel shows a person, after the model's own sigmoid. The bundled
  `labels.txt` names the one class `selfie` and the model card calls the
  output "a mask of person"; the tensor name notwithstanding, 1 means person.

```rust,no_run
use mediapipe_native::Model;
use std::path::Path;

fn main() -> Result<(), mediapipe_native::Error> {
    let mut model = Model::load(Path::new(
        "plans/selfie_segmenter_landscape/selfie_segmenter_landscape.mpplan",
    ))?;
    assert_eq!(model.input_shape(), (144, 256, 3));
    let rgb8 = vec![0u8; 144 * 256 * 3]; // your frame, scaled to 256x144, RGB
    for (dst, &src) in model.input_mut().iter_mut().zip(&rgb8) {
        *dst = f32::from(src) / 255.0;
    }
    model.run();
    let mask = model.output(0); // 144 x 256, row-major, person probability
    assert_eq!(mask.len(), 144 * 256);
    Ok(())
}
```

## Provenance and licence

Version 1 in Google's MediaPipe model storage; on 2026-10-07 the `latest`
URL served the same file:

| File | URL | SHA-256 |
|---|---|---|
| `selfie_segmenter_landscape.tflite` | `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite` | `490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e` |

The model card linked from the Image Segmenter page, "Model Card MediaPipe
Selfie Segmentation" (Tingbo Hou, Siargey Pisarchyk, Karthik Raveendran,
Google, May 2021), states "Apache License, Version 2.0". The model stays under
the ignored `plans/`; nothing here commits model data.

## How the model becomes a plan

`tools/export_tflite.py` writes plan format 4 (`mediapipe-native-plan/4`, see
[architecture](../architecture.md#plan-formats)), because the model needs
operations format 3 lacks. Weights are float16 behind `DEQUANTIZE` and are
exported as the exact float32 values. Each TFLite operation becomes:

| TFLite | Plan |
|---|---|
| `CONV_2D`, `DEPTHWISE_CONV_2D` (3×3, 5×5, stride 1 and 2, SAME) | `conv`, `dwconv`, as for the other models |
| `HARD_SWISH`, `RELU`, `LOGISTIC` after a convolution or addition whose output only they read | `fused` on that operation (`hardswish`, `relu`, `sigmoid`), applied to each finished run of rows |
| `MEAN` over height and width | `avgpool` of the whole image |
| `MUL` of a feature map by a computed `[1,1,1,c]` vector (squeeze-and-excite) | `channelmul`, the vector as second input |
| `ADD` of equal shapes | `add` |
| `RESIZE_BILINEAR` (2× up, `half_pixel_centers`, no `align_corners`) | `resize` |
| MediaPipe `Convolution2DTransposeBias` | 1×1 `conv` to 4 channels, then `depthtospace` with block 2 |

The transposed convolution reads 16 channels at 72×128 with a 2×2 kernel,
stride 2 and one output channel. Its custom options are a raw
`TfLiteTransposeConvParams` (padding SAME, stride 2×2). Because the kernel
equals the stride and the stride divides the input, MediaPipe's SAME padding is
zero and every input pixel writes its own 2×2 block:
`out[2y+i, 2x+j] = bias + Σc in[y, x, c] · w[0, i, j, c]`. The exporter turns
that into a 1×1 convolution whose output channel `i·2 + j` holds `w[0, i, j, :]`
(bias repeated), which is exactly `DEPTH_TO_SPACE`'s channel order. It refuses
any other kernel, stride, padding or option size. The final sigmoid moves ahead
of the depth-to-space, which only moves values, onto the convolution.

The folded sigmoid computes as XNNPACK's `f32-vsigmoid` kernels do (TFLite's
default CPU path), within 1e-7 of the exact value; hard-swish is
`x · min(max(x/6 + 1/2, 0), 1)`, also as XNNPACK computes it. The standalone
`sigmoid` of the older formats keeps libm's `exp`, so their outputs are
unchanged.

## Checking the plan

```bash
mkdir -p plans/_downloads && cd plans/_downloads
curl -fLO https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite
sha256sum selfie_segmenter_landscape.tflite   # compare with the table above
cd ../..
python3 tools/export_tflite.py plans/_downloads/selfie_segmenter_landscape.tflite \
  plans/selfie_segmenter_landscape --source "MediaPipe image_segmenter selfie_segmenter_landscape.tflite (float16/1)"
cargo build --release --locked --example probe
for tier in 0 1 2 3; do
  MEDIAPIPE_NATIVE_TIER=$tier python3 tools/check_tflite_parity.py \
    plans/_downloads/selfie_segmenter_landscape.tflite \
    plans/selfie_segmenter_landscape/selfie_segmenter_landscape.mpplan --work artifacts/parity/selfie_segmenter_landscape/tier$tier
done
```

TensorFlow's interpreter has no `Convolution2DTransposeBias`, so the exporter
and the parity check run a copy of the model in which it is TFLite's builtin
`TRANSPOSE_CONV` with the same weights, bias, padding and stride
(`export_tflite.runnable`). On a random image that copy's result matches
MediaPipe's formula applied to TFLite's own input tensor within 4.5e-5 on
logits spanning ±229, float32 rounding.

On the probe's 24 images on 2026-10-07 (i5-13400), the largest error of the
mask against a float64 run of the plan was 4e-6 at tiers 0-2 and 5e-6 at
tier 3; TFLite's own XNNPACK and reference kernels reach 3e-6, and native and
XNNPACK differ by at most 5e-6. The float64 run itself agrees with TFLite's
reference kernels within 1.2e-8.

`cargo test --test reference` compares the mask with a tier-0 golden, also by
its logarithm because the noise image is almost all background (see
`tests/reference.rs`), and `--test runtime` proves zero allocations.

## Limits

- Parity is measured on synthetic images, where the mask stays below 0.42: it
  shows the native arithmetic reproduces the model, not how well the model
  separates people from backgrounds.
- The square `selfie_segmenter` (256×256) and the multiclass model are other
  models and are not covered here.
- Smoothing the mask over time, upscaling it to the frame and blurring the
  background belong to the application.
