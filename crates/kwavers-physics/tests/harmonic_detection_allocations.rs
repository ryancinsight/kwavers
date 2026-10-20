#![cfg(feature = "clinical-imaging")]

use kwavers_physics::acoustics::imaging::modalities::elastography::harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector,
};
use leto::Array4;
use mnemosyne::counting::{measure, AllocationDelta, CountingAllocator};
use mnemosyne::Mnemosyne;

#[global_allocator]
static GLOBAL: CountingAllocator<Mnemosyne> = CountingAllocator::new(Mnemosyne);

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

    let single_change = measure_harmonics(&detector, &single_point);
    let many_change = measure_harmonics(&detector, &many_points);
    // The counts are the oracle: 14 result-field allocations plus the Hann,
    // FFT-input, and FFT-output workspaces, none of which depend on the number
    // of spatial points. The bytes do — every result field holds one value per
    // point — so the whole delta cannot be compared once the instrument
    // reports byte volumes.
    assert_eq!(single_change.allocations, many_change.allocations);
    assert_eq!(single_change.reallocations, many_change.reallocations);
    assert_eq!(single_change.allocations, 17);
    assert_eq!(single_change.reallocations, 0);
    assert!(
        many_change.bytes_allocated > single_change.bytes_allocated,
        "64 points must request more result bytes than one: {} against {}",
        many_change.bytes_allocated,
        single_change.bytes_allocated
    );
}

fn measure_harmonics(detector: &HarmonicDetector, samples: &Array4<f64>) -> AllocationDelta {
    let (result, change) = measure(|| {
        detector
            .analyze_harmonics(samples, 1_000.0)
            .expect("valid measured input")
    });
    std::hint::black_box(result);
    change
}
