# Contributing

Useful contributions include a reproducible wrong result, a measurement on
hardware we do not have, and a kernel that is faster without changing a bit.
An issue with a measured failure is as welcome as a patch.

## A reviewable change

Keep each change to one purpose, with a focused regression test, and run
`tools/verify.sh` before opening a pull request. Explain the observed problem,
the smallest change that fixes it, how correctness was shown and on which CPUs,
and which checks passed, failed or were not run.

## Invariants

The test suite enforces most of these.

- Outputs are bit-identical within an instruction-set tier (`tests/reference.rs`
  and the corpus comparison in [getting started](docs/getting-started.md)).
  Equality across tiers is not expected. Do not reorder bias adds, taps,
  reductions, signed-zero or NaN comparisons, or change FMA contraction.
- The reference tolerance is 0.005 in output units.
- No lossy quantization, FP16 arithmetic, pruning or global fast-math.
- `Model::run()` allocates nothing, including the first frame and after the
  model moves to another thread (`tests/runtime.rs`).
- Kernels are chosen by run-time CPU detection and preserve the caller's MXCSR.
  Tier 2 must run on AVX without AVX2, FMA or F16C (`tools/check_isa.py`).
  Release builds never use a global `target-cpu=native` or forced target
  features.
- No new runtime dependency.
- Every `unsafe` block states its bounds, aliasing and instruction-set proof.

## Measuring performance

Keep the baseline binaries, prove the outputs unchanged first, then run
interleaved ABBA blocks on a pinned CPU and report every block, including the
slower ones. [Performance](docs/performance.md) has the commands and what has
already been tried.

## Where files belong

| Path | Contents |
|---|---|
| `src/` | Plan validation, execution and kernels |
| `examples/`, `src/bin/` | `probe` and `mpbench`, developer tools built on the crate |
| `tests/` | Reference outputs and runtime gates; a test that needs a missing plan fails, never skips |
| `tools/` | Python exporters, parity checks and benchmarks |
| `docs/` | Guides |
| `pkgbuild/` | The BigLinux package: `mpbench` and the plans, exported at build time |

Plans stay under the ignored `plans/` and model files are never committed.
Code, comments and documentation are written in English.
