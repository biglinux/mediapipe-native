# Performance

The engine is tuned for total CPU per frame on one thread, on old and new x86
CPUs alike, without changing a single output bit within an instruction-set tier.
This page gives the current numbers, the method behind them, and what has
already been tried, so the next optimization starts from evidence.

## Current cost per frame

CPU time per frame from `mpbench`, median of interleaved blocks (see
[method](#method)). The i3 column is the build of 2026-10-04, measured idle;
the i5 columns are the build of 2026-10-02, measured with unrelated background
load, and were not repeated because the host stayed loaded.

| Plan | i5-13400, AVX2 (tier 3) | i5-13400, AVX (tier 2) | i3-2375M, AVX (tier 2) |
|---|---:|---:|---:|
| face_detector | 0.83 ms | 0.94 ms | 5.1 ms |
| face_landmarks | 2.62 ms | 3.20 ms | 17.2 ms |
| hand_detector | 6.27 ms | 8.09 ms | 43.2 ms |
| hand_landmarks | 6.51 ms | 8.22 ms | 43.8 ms |
| hand_roi_refinement | 0.38 ms | 0.43 ms | 2.5 ms |
| pose_detector | 8.36 ms | 10.60 ms | 56.1 ms |
| pose_landmarks | 4.94 ms | 5.87 ms | 33.9 ms |
| holistic_face_landmarks | 0.80 ms | 0.99 ms | 5.4 ms |
| face_blendshapes | 0.64 ms | 0.83 ms | 4.7 ms |

The i3-2375M (Sandy Bridge, 2011) is the reason tier 2 exists: it has AVX but
no AVX2 or FMA, and running the AVX path on a newer CPU is not a substitute for
measuring it there.

For reference, on 2026-09-29 — before later kernel work —
the seven hand, pose and holistic plans took 0.80–1.12× the CPU of
single-threaded TensorFlow Lite with XNNPACK on the same i5. A controlled
comparison with OpenVINO has not been done; [below](#comparing-with-another-engine)
is how to do it fairly.

## How the kernels spend their time

Two thirds of a MediaPipe graph is 1×1 convolution, and on these CPUs it is
bound by load ports, not multipliers: every input value must be broadcast into
a register before it can be multiplied, and on x86 that broadcast is a load.
The pointwise kernels therefore hold several output blocks per pixel tile so
each weight load feeds more multiply-adds — six pixels by sixteen channels on
AVX2, which retires about 1.5 FMAs per cycle over a whole frame on the i5,
roughly 88% of the two-port peak. Most remaining work is around that tile:

- depthwise windows load each input column once per row of output pixels
  instead of once per tap;
- activations (PReLU, ReLU, RELU6) are applied while a convolution's output is
  still in registers or L1, not in a second pass;
- a depthwise layer feeding a row-local convolution computes its rows just
  ahead of them, so its output is read from cache;
- pointwise weights are repacked into 16-column panels at load time, because a
  `co`-float stride mapped each panel onto a few L1 sets (−10 to −14% frame CPU
  on the i3 for the hand and pose models).

The AVX tier has two kernels of its own because the i3's Sandy Bridge core
behaves differently from anything newer. Its L1 cache cannot serve two reads in
one cycle from the same 16-byte bank of different lines, and a pixel stride
that is a multiple of 128 bytes (any multiple of 32 input channels) puts all
four broadcasts of a pointwise tile in one bank: `perf stat` counted 8–9% of
cycles lost to `l1d_blocks.bank_conflict_cycles`. When a tile feeds at least
four 16-column panels, its four input pixels are first copied to a stack buffer
at a stride of `ci + 4` floats. And widths of 16k + 8 channels (24, 40, 56)
finish with one 24-column block of three pixels instead of a 16- and an
8-column block that reloaded the same broadcasts. Neither changes a value or a
summation order, and both measured slower on CPUs without the conflict, so
they stay on the AVX tier only.

Four later changes, each bit-identical and measured on whole frames:

- On AVX, a pointwise layer whose width is a multiple of 32 and whose weights
  fit the i3's 256 KB L2 uses a two-pixel by 32-channel tile, so each broadcast
  input feeds four registers instead of two: 2–14% faster per layer. With
  larger weights (112→672, 192→1152) it measured 8–11% slower.
- Layers with exactly 16 outputs get their own copy of the 16-column tiles on
  both vector tiers. The constant width drops the panel loop and the strided
  store addressing: 8→16 at 128×128 is 16% faster on the i3 and 12% on the i5.
- When nothing but its paired residual add reads a pointwise convolution's
  output (`dead_after` proves it), the convolution writes each row group to a
  small scratch buffer instead of the tensor pool, so a tensor nobody observes
  is never written back.
- Rows of a 5×5, stride-1 depthwise layer whose width is a multiple of 7 or 8
  are computed whole, borders included: each input column is loaded once per
  kernel row, and a column outside the image is skipped. Before, a border pixel
  was a one-pixel tile with one weight load per multiply-add, which dominated
  the 7×7 and 14×14 maps of the hand and pose models.

`mpbench PLAN 300 --profile` breaks a frame down by operation shape. Use it to
choose the next experiment; a kernel's share of a profile is not a whole-frame
speedup.

## Method

A change is adopted only when it is bit-identical within each tier and faster
over whole frames on the target CPUs.

1. Before editing, build and keep the baseline `mpbench` and `examples/probe`;
   record their hashes and the plan hashes.
2. Run correctness first: `cargo test` and the bit-exact corpus comparison in
   [getting started](getting-started.md), every supported tier in its own
   process.
3. Measure with `tools/bench_abba.py`. Each block runs baseline, candidate,
   candidate, baseline (ABBA), so drift in clock speed or background load hits
   both sides equally. Use at least six blocks, warmup, one CPU pinned by
   affinity, and nothing else compiling or decoding video:

   ```bash
   python3 tools/bench_abba.py --baseline /path/to/baseline-mpbench \
     --candidate target/release/mpbench --root . --cpu 4 \
     --blocks 6 --frames 300 --tiers 2 3 --output artifacts/abba/result.json
   ```

4. Report the median and the range of per-block differences, how many blocks
   were slower, and p95 — not the best run. Keep negative and interrupted runs;
   `--resume` restarts incomplete blocks instead of mixing sessions.

Pick a CPU inside the process's affinity mask and, on hybrid Intel parts, check
in sysfs that it is a performance core: CPU 0 is not necessarily one. The tools
record hostname, kernel and absolute binary paths in their JSON; keep results
under the ignored `artifacts/` and redact them before sharing.

## Tried and rejected

| Idea | Result |
|---|---|
| Binary16 weights for the face-mesh head | The head got faster, the frame not reliably; duplicate weights raised PSS, and the i3 has no F16C |
| Global `-C llvm-args=-align-loops=32` | 1–3% faster on the i3, up to 1% slower on a Ryzen 5 5600H |
| Ten pixels per 5×5 depthwise tile on AVX2 | +0.9% on the hand detector against eight pixels |
| Eight register groups in the one-output-channel kernel | No change on the i3, ±1% on the i5 |
| Unconditional clamps in convolution stores | About +1% on the PReLU face plans; clamps stay conditional |
| `fearless_simd` for portable SIMD | No AVX-without-AVX2 level, so tier 2 cannot be expressed; also a new dependency |
| Six-pixel 16-column pointwise tile on AVX, as on AVX2 | 0.1–6.5% slower on the i3 on every plan |
| The 24-column block on AVX2 (four pixels) | 1.9–3.1% slower on the i5 at tier 3 |
| Bank-skewed copy repeated for every panel when weights outweigh pixels | 8.6% slower on the i3 (pose detector) |
| Bank-skewed copy for tiles feeding two panels | 0.4–0.9% slower on the i3 (face meshes) |
| Panel-outer loop order for irregular widths (`384->97`) | No change on the i3, +1% on the i5 at tier 2 |
| Depthwise rows of narrow maps computed column by column | Up to 8% slower on the i3: a branch per row and tap |
| Private buffers so face-mesh blocks compute depthwise rows ahead of their pair | 4.7% slower on the i3; the 128×128 layers are not bandwidth-bound (5% of cycles wait on L2 misses) |
| Pointwise input loop unrolled by four, or a constant `ci` | Up to 15% slower per layer on the i3 for `ci` ≤ 16; the rolled loop runs from the loop stream detector |
| Two pixels per register with 128-bit broadcasts (`vbroadcastss xmm`, `vinsertf128`) for 8 and 16 outputs on AVX | Same speed for 8→16, 5–20% slower for other shapes on the i3: the 256-bit broadcast's port-5 uop is not what limits these tiles |
| Constant width (32 or 64) in the two-pixel by 32-channel tile | 2–6% slower per layer on the i3 |
| Two-pixel RGB stems on AVX | Face detector (5×5 stem) 4.1% faster, hand landmarks and pose detector (3×3) 0.8–1.2% slower on the i3. Three pixels for 24 channels, which removed the four-pixel tile's register spills, won on all four plans |

Lossy quantization, FP16 arithmetic, pruning, low-rank factorization, global
fast-math and multithreading are out of scope by design: each changes results or
spends more total CPU.

## Ideas not yet tried

| Shape | Keep exact | Decisive measurement |
|---|---|---|
| Residual add and PReLU in the pointwise register tile, so a private convolution row never leaves registers | Bias and reduction order, residual order, activation NaN behaviour | Whole-frame ABBA with a per-layer profile, including the i3 at tier 2 |
| 16→8 at 128×128 on AVX (about 5.8 G/s on the i3, against 7.4 for 8→16) | Bias-first, input-ordered sums | Per-layer profile, then frame ABBA on the i3 |
| Whole-row windows for stride-2 and 3×3 depthwise borders, as for 5×5 stride 1 | Live taps in row and column order | Frame ABBA on the i3 and i5 |

Widening a tile is not free: AVX has sixteen architectural registers, and more
accumulators can force spills. Check the disassembly for the compiler and CPU
actually used before measuring.

## Comparing with another engine

Use the same physical host, the same model and the exact same RGB tensors. Run
FP32 explicitly (OpenVINO may pick a lower internal precision; query the
compiled model), one inference thread, one stream, one request in flight, the
same affinity and warmup. Account for layout conversion in an end-to-end test.
Record engine version, plugin, graph optimizations and thread policy, run ABBA
on identical inputs, and report load time, steady CPU per frame, latency tails
and PSS.
