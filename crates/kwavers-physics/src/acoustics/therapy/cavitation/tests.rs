use super::*;
use kwavers_core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use leto::Array3;

fn detector() -> TherapyCavitationDetector {
    TherapyCavitationDetector::new(MHZ_TO_HZ, 0.0)
}

#[test]
fn test_blake_threshold_1um_value() {
    let det = detector();
    // The detector must use the rigorous SSOT Blake threshold (Blake 1949), not a
    // local approximation.
    let expected = crate::acoustics::mechanics::cavitation::core::thresholds::blake_threshold(
        SURFACE_TENSION_WATER,
        1e-6,
        ATMOSPHERIC_PRESSURE,
        VAPOR_PRESSURE_WATER,
    );
    assert!(
        (det.blake_threshold - expected).abs() < 1.0,
        "Blake threshold {:.1} Pa ≠ rigorous SSOT value {expected:.1} Pa",
        det.blake_threshold
    );
}

#[test]
fn test_blake_threshold_positive() {
    let det = detector();
    assert!(
        det.blake_threshold > 0.0,
        "Blake threshold must be positive; got {}",
        det.blake_threshold
    );
}

#[test]
fn test_blake_threshold_smaller_nucleus_higher_pressure() {
    // Smaller nuclei are more strongly surface-tension-stabilised (2σ/R larger),
    // so the Blake acoustic-amplitude threshold DECREASES monotonically with R₀
    // (Blake 1949). A 1 µm nucleus therefore has a higher threshold than 10 µm.
    let det_1um = TherapyCavitationDetector::new_with_radius(MHZ_TO_HZ, 1e-6);
    let det_10um = TherapyCavitationDetector::new_with_radius(MHZ_TO_HZ, 10e-6);
    assert!(
        det_1um.blake_threshold > det_10um.blake_threshold,
        "1 µm nucleus should have higher Blake threshold than 10 µm \
         (more surface-tension-stabilised): got 1µm={:.1} Pa, 10µm={:.1} Pa",
        det_1um.blake_threshold,
        det_10um.blake_threshold
    );
}

#[test]
fn test_minnaert_frequency_1um() {
    let det = detector();
    let f0 = det.minnaert_frequency(1e-6);
    assert!(
        (f0 - 3.26e6).abs() / 3.26e6 < 0.05,
        "Minnaert f₀(1µm) = {:.3e} Hz, expected ~3.26 MHz",
        f0
    );
}

#[test]
fn test_minnaert_frequency_scales_inversely_with_radius() {
    let det = detector();
    let f1 = det.minnaert_frequency(1e-6);
    let f2 = det.minnaert_frequency(2e-6);
    let ratio = f1 / f2;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "f₀(R) should scale as 1/R: f(1µm)/f(2µm) = {ratio:.6}, expected 2.0"
    );
}

#[test]
fn test_threshold_detection_no_cavitation_below_threshold() {
    let det = detector();
    let p = Array3::from_elem((4, 4, 4), -0.5 * det.blake_threshold);
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| !c),
        "pressure below Blake threshold should give no cavitation"
    );
}

#[test]
fn test_threshold_detection_cavitation_above_threshold() {
    let det = detector();
    let p = Array3::from_elem((4, 4, 4), -2.0 * det.blake_threshold);
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| c),
        "pressure above Blake threshold should give cavitation everywhere"
    );
}

/// Builds an owned F-contiguous `Array3<f64>` directly through leto's public
/// `Layout`/`VecStorage`/`Array` constructors: `as_slice()` (the C-contiguity
/// check) returns `None` while `as_slice_memory_order()` would have returned
/// `Some` — the exact layout a caller's `.transpose([2, 1, 0])` could hand
/// [`TherapyCavitationDetector::detect`].
fn f_ordered_from_fn(shape: [usize; 3], f: impl Fn(usize, usize, usize) -> f64) -> Array3<f64> {
    let [nx, ny, nz] = shape;
    let layout = leto::Layout::f_contiguous(shape).expect("invariant: nonzero shape");
    // `VecStorage::generate` calls its `FnMut` sequentially for positions
    // 0..len, so a captured counter reconstructs the F-order flat index.
    let mut p = 0usize;
    let storage = leto::VecStorage::generate(nx * ny * nz, || {
        let i = p % nx;
        let j = (p / nx) % ny;
        let k = p / (nx * ny);
        p += 1;
        f(i, j, k)
    });
    leto::Array::new(layout, storage).expect("invariant: layout fits storage")
}

/// A pressure field supplied with F-contiguous storage must mark cavitation
/// at the same *logical* voxels as a C-contiguous field carrying identical
/// values — pairing must not depend on `pressure`'s raw memory-order flat
/// index, which for an F-contiguous input does not correspond to
/// `cavitation`'s (always C-contiguous) row-major position.
#[test]
fn test_threshold_detection_pairs_transposed_input_by_logical_index() {
    let det = detector();
    let shape = [2usize, 3, 4];
    let p_high = -2.0 * det.blake_threshold;
    let p_low = -0.1 * det.blake_threshold;
    // (1, 0, 0) is chosen so its C-order flat index (12, since C strides are
    // [12, 4, 1]) differs from its F-order flat index (1, since F strides are
    // [1, 2, 6]): a raw-memory-order pairing would place the hot voxel at
    // logical (0, 0, 1) — the F-order decode of flat position 12 — instead.
    let hot = |i: usize, j: usize, k: usize| i == 1 && j == 0 && k == 0;
    let p = f_ordered_from_fn(shape, |i, j, k| if hot(i, j, k) { p_high } else { p_low });
    assert!(
        p.as_slice().is_none() && p.as_slice_memory_order().is_some(),
        "pressure must be dense in F order for this case to mean anything"
    );

    let cav = det.detect(&p);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                assert_eq!(
                    cav[[i, j, k]],
                    hot(i, j, k),
                    "voxel [{i}, {j}, {k}] must match the logical-index threshold check"
                );
            }
        }
    }
}

#[test]
fn test_threshold_detection_spatial_heterogeneity() {
    let det = detector();
    let p_high = -2.0 * det.blake_threshold;
    let p_low = -0.1 * det.blake_threshold;
    let mut p = Array3::from_elem((2, 2, 2), p_low);
    p[[0, 0, 0]] = p_high;
    p[[1, 1, 1]] = p_high;
    let cav = det.detect(&p);
    assert!(cav[[0, 0, 0]], "voxel 0,0,0 should cavitate");
    assert!(cav[[1, 1, 1]], "voxel 1,1,1 should cavitate");
    assert!(!cav[[0, 0, 1]], "voxel 0,0,1 should not cavitate");
}

#[test]
fn test_spectral_detection_zero_pressure_no_cavitation() {
    let mut det = TherapyCavitationDetector::new(3.26 * MHZ_TO_HZ, 0.0);
    det.method = CavitationDetectionMethod::Spectral;
    let p = Array3::zeros([4, 4, 4]);
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| !c),
        "zero pressure must never trigger cavitation"
    );
}

#[test]
fn test_spectral_detection_resonance_lowers_threshold() {
    let det_far = TherapyCavitationDetector::new(100e3, 0.0);
    let mut det_near = TherapyCavitationDetector::new(3.26 * MHZ_TO_HZ, 0.0);
    det_near.method = CavitationDetectionMethod::Spectral;

    let p_test = -0.80 * det_near.blake_threshold;
    let p = Array3::from_elem((4, 4, 4), p_test);

    let cav_far = det_far.detect(&p);
    let cav_near = det_near.detect(&p);

    assert!(
        cav_far.iter().all(|&c| !c),
        "off-resonance: 80% of threshold should not cavitate"
    );
    assert!(
        cav_near.iter().all(|&c| c),
        "at resonance: 80% of threshold should cavitate"
    );
}

#[test]
fn test_cavitation_index_at_threshold_is_one() {
    let det = detector();
    let ci = det.cavitation_index(det.blake_threshold);
    assert!(
        (ci - 1.0).abs() < 1e-12,
        "CI at P_Blake must equal 1.0; got {ci:.6e}"
    );
}

#[test]
fn test_cavitation_index_zero_pressure_is_zero() {
    let det = detector();
    assert_eq!(det.cavitation_index(0.0), 0.0);
}

#[test]
fn test_stable_cavitation_in_correct_range() {
    let det = detector();
    assert!(det.is_stable_cavitation(0.7 * det.blake_threshold));
    assert!(!det.is_stable_cavitation(0.4 * det.blake_threshold));
    assert!(!det.is_stable_cavitation(1.1 * det.blake_threshold));
}

#[test]
fn test_inertial_cavitation_above_threshold() {
    let det = detector();
    assert!(det.is_inertial_cavitation(det.blake_threshold));
    assert!(det.is_inertial_cavitation(2.0 * det.blake_threshold));
    assert!(!det.is_inertial_cavitation(0.9 * det.blake_threshold));
}

#[test]
fn test_cavitation_probability_at_threshold_is_half() {
    let det = detector();
    let p = det.cavitation_probability(det.blake_threshold);
    assert!(
        (p - 0.5).abs() < 1e-10,
        "probability at CI=1 must be 0.5; got {p:.6e}"
    );
}

#[test]
fn test_cavitation_probability_monotone() {
    let det = detector();
    let p1 = det.cavitation_probability(0.5 * det.blake_threshold);
    let p2 = det.cavitation_probability(1.0 * det.blake_threshold);
    let p3 = det.cavitation_probability(2.0 * det.blake_threshold);
    assert!(
        p1 < p2 && p2 < p3,
        "probability must increase with pressure"
    );
}
