//! Allocation gate from the first frame and ownership transfer to a worker.
use mediapipe_native::Model;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    path::Path,
};
thread_local! {
    static ARMED:Cell<bool>=const{Cell::new(false)};
    static COUNT:Cell<usize>=const{Cell::new(0)};
}
struct Counting;
fn observe() {
    if ARMED.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|n| n.set(n.get() + 1));
    }
}
// SAFETY: all requests are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        observe();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        observe();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        observe();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[test]
fn models_allocate_nothing_on_first_frame_or_after_thread_transfer() {
    for name in [
        "plans/face_landmarks/face_landmarks_detector.mpplan",
        "plans/face_detector/face_detector.mpplan",
        "plans/hand_detector/hand_detector.mpplan",
        "plans/hand_landmarks/hand_landmarks_detector.mpplan",
        "plans/hand_roi_refinement/hand_roi_refinement.mpplan",
        "plans/pose_detector/pose_detector.mpplan",
        "plans/pose_landmarks/pose_landmarks_detector.mpplan",
        "plans/holistic_face_landmarks/face_landmarks_detector.mpplan",
        "plans/face_blendshapes/face_blendshapes.mpplan",
    ] {
        let mut model = Model::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap();
        model.input_mut().fill(0.5);
        COUNT.set(0);
        ARMED.set(true);
        model.run();
        ARMED.set(false);
        assert_eq!(COUNT.get(), 0, "first frame allocated");
        let want: Vec<Vec<u32>> = (0..model.output_count())
            .map(|o| model.output(o).iter().map(|x| x.to_bits()).collect())
            .collect();
        std::thread::spawn(move || {
            for _ in 0..3 {
                model.input_mut().fill(0.5);
                COUNT.set(0);
                ARMED.set(true);
                model.run();
                ARMED.set(false);
                assert_eq!(COUNT.get(), 0, "worker frame allocated");
                for (i, want) in want.iter().enumerate() {
                    assert!(model
                        .output(i)
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(want.iter().copied()));
                }
            }
        })
        .join()
        .unwrap();
    }
}
