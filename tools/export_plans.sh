#!/usr/bin/env bash
# Export every plan into DEST (default plans/), laid out like
# /usr/share/mediapipe-native, from the MediaPipe models pkgbuild/PKGBUILD
# names. Without DOWNLOADS, the models are downloaded into DEST/_downloads and
# checked against the recipe's SHA-256; makepkg passes its srcdir, where it has
# checked them already. Needs bsdtar, curl without DOWNLOADS and the exporter's
# Python modules (tensorflow, onnx, flatbuffers, numpy).
# Usage: tools/export_plans.sh [DEST [DOWNLOADS]]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEST=$(realpath -m -- "${1:-$ROOT/plans}")
DOWNLOADS=$(realpath -m -- "${2:-$DEST/_downloads}")

if (($# < 2)); then
    mkdir -p "$DOWNLOADS"
    (
        # The recipe's first source is this repository; the rest are the models.
        # shellcheck source-path=SCRIPTDIR/.. source=pkgbuild/PKGBUILD
        source "$ROOT/pkgbuild/PKGBUILD"
        cd "$DOWNLOADS"
        for i in "${!source[@]}"; do
            [[ ${source[i]} == git+* ]] && continue
            curl -fsSLO "${source[i]}"
            printf '%s  %s\n' "${sha256sums[i]}" "${source[i]##*/}"
        done | sha256sum --check --quiet
    )
fi

# A .task bundle is a zip; model names repeat between bundles.
for bundle in face_landmarker hand_landmarker holistic_landmarker; do
    mkdir -p "$DOWNLOADS/$bundle"
    bsdtar -xf "$DOWNLOADS/$bundle.task" -C "$DOWNLOADS/$bundle"
done
cd "$ROOT"

# export BUNDLE/MODEL PLAN_DIR [exporter options]
export_plan() {
    python3 tools/export_tflite.py "$DOWNLOADS/$1.tflite" "$DEST/$2" \
        --source "MediaPipe ${1%%/*}.task ${1#*/}.tflite (float16/1)" "${@:3}"
}
export_plan face_landmarker/face_detector face_detector --format 1
export_plan face_landmarker/face_landmarks_detector face_landmarks --format 1
export_plan hand_landmarker/hand_detector hand_detector
export_plan hand_landmarker/hand_landmarks_detector hand_landmarks
export_plan holistic_landmarker/hand_roi_refinement hand_roi_refinement
export_plan holistic_landmarker/pose_detector pose_detector
export_plan holistic_landmarker/pose_landmarks_detector pose_landmarks
export_plan holistic_landmarker/face_landmarks_detector holistic_face_landmarks
export_plan holistic_landmarker/face_blendshapes face_blendshapes
for model in selfie_segmenter selfie_segmenter_landscape; do
    python3 tools/export_tflite.py "$DOWNLOADS/$model.tflite" "$DEST/$model" \
        --source "MediaPipe image_segmenter $model.tflite (float16/1)"
done
