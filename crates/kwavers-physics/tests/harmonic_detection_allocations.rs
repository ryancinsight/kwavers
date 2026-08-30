#![cfg(feature = "clinical-imaging")]

use kwavers_alloc_probe::{ThreadScopedAllocator, Window};
use kwavers_physics::acoustics::imaging::modalities::elastography::harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector,
};
use leto::Array4;

#[global_allocator]
static GLOBAL: ThreadScopedAllocator = ThreadScopedAllocator;

#[test]
fn harmonic_workspace_allocation_count_is_independent_of_spatial_extent() {
    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
    let single_point = Array4::from_elem((1, 1, 1, 130), 1.0);
    let many_points = Array4::from_elem((4, 4, 4, 130), 1.0);

    // Warm Apollo's retained plan before measuring caller-owned workspaces.
    std::hint::black_box(
        detector
            .analyze_harmonics(&single_point, 1_000.0)
            .expect("valid warmup input"),
    );

    let single_change = measure(&detector, &single_point);
    let many_change = measure(&detector, &many_points);
    assert_eq!(single_change, many_change);
    // 14 result-field allocations plus the Hann, FFT-input, and FFT-output
    // workspaces; none depend on the number of spatial points.
    assert_eq!(single_change.allocations, 17);
    assert_eq!(single_change.reallocations, 0);
}

fn measure(detector: &HarmonicDetector, samples: &Array4<f64>) -> kwavers_alloc_probe::Change {
    let window = Window::open();
    let result = detector
        .analyze_harmonics(samples, 1_000.0)
        .expect("valid measured input");
    let change = window.change();
    std::hint::black_box(result);
    change
}
