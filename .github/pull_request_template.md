## Problem

What fails, how to reproduce it, and which part of the crate owns it.

## Change

What changed and why this is the smallest fix. For kernels: summation order,
aliasing and instruction-set assumptions.

## Evidence

Passed, failed or not run, with the CPUs used: focused test, `tools/verify.sh`,
bit-exact corpus per tier. For performance: binary and plan hashes and the
complete ABBA blocks.
