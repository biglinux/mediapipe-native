#!/usr/bin/env bash
# Formatting, Clippy, the tests once per instruction-set tier this CPU
# supports, doctests, the tier-2 instruction check and the Python tool tests.
# Needs the exported plans (docs/getting-started.md). Logs go to EVIDENCE_DIR.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ ${RUSTFLAGS:-} == *target-cpu* || ${RUSTFLAGS:-} == *target-feature* || -n ${CARGO_ENCODED_RUSTFLAGS:-} ]]; then
    echo 'Unset global target CPU/feature flags: the tiers must be chosen at run time.' >&2
    exit 2
fi
EVIDENCE=${EVIDENCE_DIR:-"$ROOT/target/verify"}
TARGET=${CARGO_TARGET_DIR:-"$ROOT/target"}
mkdir -p "$EVIDENCE"
run_gate() {
    local name=$1; shift
    printf 'RUN %s\n' "$name"
    if timeout --kill-after=5s "${GATE_TIMEOUT:-900}" "$@" >"$EVIDENCE/$name.log" 2>&1; then
        printf 'PASS %s\n' "$name"
    else
        local status=$?
        cat "$EVIDENCE/$name.log" >&2
        printf 'FAIL %s (%s)\n' "$name" "$status" >&2
        exit "$status"
    fi
}
{ rustc -Vv; cargo --version; uname -a; } > "$EVIDENCE/environment.log"
run_gate fmt cargo fmt --all -- --check
run_gate clippy cargo clippy --release --locked --all-targets -- -D warnings
run_gate build cargo build --release --locked --bin mpbench --example probe
run_gate isa python3 tools/check_isa.py "$TARGET/release/mpbench" --output "$EVIDENCE/isa.json"
# Capping at 3 still reports the tier the CPU and OS actually support.
run_gate tier-detection env MEDIAPIPE_NATIVE_TIER=3 "$TARGET/release/mpbench" \
    plans/face_landmarks/face_landmarks_detector.mpplan 1 --json
TIER=$(python3 - "$EVIDENCE/tier-detection.log" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.startswith('{')]
assert len(rows) == 1 and rows[0]['allocations'] == rows[0]['first_frame_allocations'] == 0
print(rows[0]['tier'])
PY
)
for ((tier=0; tier<=TIER; tier++)); do
    run_gate "release-tier$tier" env MEDIAPIPE_NATIVE_TIER="$tier" \
        cargo test --release --locked --all-targets
    if [[ ${RUN_DEBUG:-0} == 1 ]]; then
        run_gate "debug-tier$tier" env MEDIAPIPE_NATIVE_TIER="$tier" \
            cargo test --locked --all-targets
    fi
done
run_gate doctests cargo test --release --locked --doc
# The tool tests run small fake binaries from temporary directories, which a
# noexec /tmp refuses; keep them under target/ instead.
mkdir -p "$TARGET/tool-tests"
run_gate python-tools env TMPDIR="$TARGET/tool-tests" \
    python3 -m unittest discover -s tools -p 'test_*.py'
printf 'Verified tiers 0..%s. Logs: %s\n' "$TIER" "$EVIDENCE"
