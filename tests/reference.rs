//! Does a change still produce the right numbers?
//!
//! This is the gate to run after touching a kernel. It needs no camera, no
//! network and no inference library: the input is generated from a fixed seed,
//! and the expected outputs are committed next to this file. They came from
//! the executor on 2026-09-19, and separately agree with ONNX Runtime to
//! within 0.0006 of a pixel and with the numpy reference in
//! `tools/run_numpy.py` to five decimal places.
//!
//! The tolerance is 0.005 in output units. On the face mesh those units are
//! pixels of a 256-pixel image, so that is one two-hundredth of a pixel — tighter
//! than anything that could move a pointer, and loose enough that reordering
//! an accumulation is allowed. A change that trips this has changed the
//! arithmetic, not the rounding.
//!
//! Plans are not in Git (see docs/getting-started.md). A missing plan or
//! reference file fails the test, so a green gate never means that inference
//! was silently skipped. The auxiliary face-mesh outputs were captured at tier
//! 0 on 2026-09-20.
//!
//! The hand, pose and holistic goldens (2026-09-29) are the executor's tier-0
//! outputs for plans exported by `tools/export_tflite.py`; that tool's partner
//! `tools/check_tflite_parity.py` separately holds every tier against
//! TensorFlow Lite and a float64 run of the same graph. The pose segmentation
//! output is raw logits that reach +-650 on this noise image, where tiers
//! legitimately differ by 0.17; it is compared after the sigmoid its consumer
//! applies, with the same 0.005 tolerance, as a mask probability.

use std::path::Path;

use mediapipe_native::Model;

/// The same image `mpbench` uses: a 32-bit xorshift from a fixed seed.
fn fixed_image(len: usize) -> Vec<f32> {
    let mut state: u32 = 0x2545_f491;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn expected(name: &str) -> Vec<f32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference")
        .join(name);
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(raw.len().is_multiple_of(4), "{name}: truncated float file");
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// How an output is compared: as produced, or as the probability a logit
/// output becomes after its consumer's sigmoid.
#[derive(Clone, Copy)]
enum Scale {
    Raw,
    Logit,
}

/// Run one plan on the fixed image and compare every declared output.
fn check(relative: &str, expectations: &[(usize, &str, Scale)]) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    assert!(
        path.exists(),
        "required plan is missing: {}",
        path.display()
    );
    let mut model = Model::load(&path).expect("the plan should load");
    let (h, w, c) = model.input_shape();
    model.input_mut().copy_from_slice(&fixed_image(h * w * c));
    model.run();

    assert_eq!(
        expectations.len(),
        model.output_count(),
        "all declared outputs must be checked"
    );
    for &(index, name, scale) in expectations {
        let want = expected(name);
        let got = model.output(index);
        assert_eq!(got.len(), want.len(), "{name}: wrong number of values");
        let seen = |v: f32| match scale {
            Scale::Raw => v,
            Scale::Logit => 1.0 / (1.0 + (-v).exp()),
        };
        let mut worst = (0usize, 0.0f32);
        for (i, (&g, &e)) in got.iter().zip(&want).enumerate() {
            assert!(e.is_finite(), "{name}: expected value {i} is {e}");
            assert!(g.is_finite(), "{name}: value {i} is {g}");
            let gap = (seen(g) - seen(e)).abs();
            if gap > worst.1 {
                worst = (i, gap);
            }
        }
        assert!(
            worst.1 <= 0.005,
            "{name}: value {} is off by {} (tier {})",
            worst.0,
            worst.1,
            mediapipe_native::ops::simd_tier()
        );
    }
}

#[test]
fn the_face_mesh_still_produces_the_recorded_landmarks() {
    check(
        "plans/face_landmarks/face_landmarks_detector.mpplan",
        &[
            (0, "face_landmarks.bin", Scale::Raw),
            (1, "face_landmarks_aux1.bin", Scale::Raw),
            (2, "face_landmarks_aux2.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_detector_still_produces_the_recorded_boxes_and_scores() {
    check(
        "plans/face_detector/face_detector.mpplan",
        &[
            (0, "face_detector_regressors.bin", Scale::Raw),
            (1, "face_detector_classificators.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_palm_detector_still_produces_the_recorded_boxes_and_scores() {
    check(
        "plans/hand_detector/hand_detector.mpplan",
        &[
            (0, "hand_detector_regressors.bin", Scale::Raw),
            (1, "hand_detector_scores.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_hand_landmarks_still_match_the_recording() {
    check(
        "plans/hand_landmarks/hand_landmarks_detector.mpplan",
        &[
            (0, "hand_landmarks.bin", Scale::Raw),
            (1, "hand_landmarks_presence.bin", Scale::Raw),
            (2, "hand_landmarks_handedness.bin", Scale::Raw),
            (3, "hand_landmarks_world.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_hand_roi_refinement_still_matches_the_recording() {
    check(
        "plans/hand_roi_refinement/hand_roi_refinement.mpplan",
        &[(0, "hand_roi_refinement.bin", Scale::Raw)],
    );
}

#[test]
fn the_pose_detector_still_produces_the_recorded_boxes_and_scores() {
    check(
        "plans/pose_detector/pose_detector.mpplan",
        &[
            (0, "pose_detector_regressors.bin", Scale::Raw),
            (1, "pose_detector_scores.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_pose_landmarks_still_match_the_recording() {
    check(
        "plans/pose_landmarks/pose_landmarks_detector.mpplan",
        &[
            (0, "pose_landmarks.bin", Scale::Raw),
            (1, "pose_landmarks_presence.bin", Scale::Raw),
            (2, "pose_landmarks_segmentation.bin", Scale::Logit),
            (3, "pose_landmarks_heatmap.bin", Scale::Raw),
            (4, "pose_landmarks_world.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_holistic_face_mesh_still_matches_the_recording() {
    check(
        "plans/holistic_face_landmarks/face_landmarks_detector.mpplan",
        &[
            (0, "holistic_face_landmarks.bin", Scale::Raw),
            (1, "holistic_face_landmarks_score.bin", Scale::Raw),
        ],
    );
}

#[test]
fn the_blendshapes_still_match_the_recording() {
    check(
        "plans/face_blendshapes/face_blendshapes.mpplan",
        &[(0, "face_blendshapes.bin", Scale::Raw)],
    );
}
