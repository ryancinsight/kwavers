use super::bbb::bbb_opening_dose;
use super::benchmark::{
    evaluate_pressure_field, run_skull_adaptive_transcranial_benchmark,
    SkullAdaptiveBenchmarkConfig,
};
use super::geometry::focused_cap_positions;
use super::observables::acoustic_fus_observables;
use super::skull_ray::acoustic_properties_from_hu;
use super::subspot::{gbm_subspot_covered_fraction, gbm_subspot_raster};
use super::types::TranscranialFusPlanConfig;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::tissue_acoustics::DENSITY_BRAIN;
use ndarray::{Array2, Array3};

#[test]
fn focused_cap_count_and_radius() {
    let positions = focused_cap_positions(64, 0.15, 0.22, 1.18).unwrap();
    assert_eq!(positions.dim(), (64, 3));
    for row in positions.rows() {
        let r = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        assert!((r - 0.15).abs() < 1.0e-12, "element not on sphere: r={r}");
        assert!(row[2] < 0.0, "transcranial cap source must stay on -z side");
    }
}

#[test]
fn focused_cap_rejects_invalid_polar_span() {
    let result = focused_cap_positions(64, 0.15, 1.18, 0.22);
    assert!(result.is_err(), "theta_min >= theta_max must be rejected");
}

#[test]
fn acoustic_properties_brain_branch() {
    let (c, rho, alpha) = acoustic_properties_from_hu(0.0, 650_000.0, SOUND_SPEED_TISSUE, 2800.0);
    assert_eq!(c, SOUND_SPEED_TISSUE);
    assert_eq!(rho, DENSITY_BRAIN);
    assert!(alpha > 0.0);
}

#[test]
fn acoustic_properties_bone_branch() {
    let (c, rho, _alpha) =
        acoustic_properties_from_hu(2000.0, 650_000.0, SOUND_SPEED_TISSUE, 2800.0);
    assert!(
        c > SOUND_SPEED_TISSUE && c <= 2800.0,
        "sound speed out of range: {c}"
    );
    assert!(
        rho > DENSITY_BRAIN && rho <= 1900.0,
        "density out of range: {rho}"
    );
}

#[test]
fn observables_energy_positive() {
    let pressure = Array3::from_elem((4, 4, 4), 1.0e5_f32);
    let (intensity, mi, cavitation) =
        acoustic_fus_observables(&pressure, 650_000.0, DENSITY_BRAIN, SOUND_SPEED_TISSUE, 1.9);
    assert!(intensity.iter().all(|&v| v > 0.0));
    assert!(mi.iter().all(|&v| v > 0.0));
    assert!(cavitation.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn subspot_raster_nonempty_returns_centroid() {
    let mut mask = Array3::from_elem((10, 10, 10), false);
    for ix in 3..7 {
        for iy in 3..7 {
            for iz in 3..7 {
                mask[[ix, iy, iz]] = true;
            }
        }
    }
    let spots = gbm_subspot_raster(&mask, [1.0e-3; 3], 3.0e-3).unwrap();
    assert!(spots.nrows() >= 1);
    assert_eq!(spots.ncols(), 3);
}

#[test]
fn subspot_covered_fraction_counts_radius_supported_tumor_voxels() {
    let mut mask = Array3::from_elem((15, 15, 15), false);
    for ix in 5..10 {
        for iy in 5..10 {
            for iz in 5..10 {
                mask[[ix, iy, iz]] = true;
            }
        }
    }
    let spots = gbm_subspot_raster(&mask, [1.0e-3; 3], 2.0e-3).unwrap();
    let covered = gbm_subspot_covered_fraction(&mask, &spots, [1.0e-3; 3], 2.0e-3);
    assert!(
        covered > 0.50,
        "expected focal support to cover most tumour voxels, got {covered}"
    );
}

#[test]
fn subspot_raster_empty_returns_error() {
    let mask = Array3::from_elem((8, 8, 8), false);
    let result = gbm_subspot_raster(&mask, [1.0e-3; 3], 3.0e-3);
    assert!(result.is_err());
}

#[test]
fn bbb_dose_nonnegative() {
    let mask = Array3::from_elem((8, 8, 8), true);
    let spots = Array2::from_shape_fn((1, 3), |(_, col)| [4_usize, 4, 4][col]);
    let (dose, perm, stable, inertial) = bbb_opening_dose(
        &mask,
        &spots,
        [1.0e-3; 3],
        0.45,
        60.0,
        0.02,
        2.0e-3,
        0.40,
        2.5,
    );
    assert!(dose.iter().all(|&v: &f32| v >= 0.0));
    assert!(perm.iter().all(|&v: &f32| (0.0..=1.0).contains(&v)));
    assert!(stable.iter().all(|&v: &f32| v >= 0.0));
    assert!(inertial.iter().all(|&v: &f32| v >= 0.0));
}

#[test]
fn pressure_metrics_match_known_peak_shift_and_amplitude_error() {
    let mut reference = Array3::<f32>::zeros((3, 3, 3));
    let mut candidate = Array3::<f32>::zeros((3, 3, 3));
    reference[[1, 1, 1]] = 2.0;
    candidate[[2, 1, 1]] = 1.0;

    let metrics = evaluate_pressure_field(&reference, &candidate, [0.5, 1.0, 1.0]).unwrap();

    assert!((metrics.focal_position_error_m - 0.5).abs() < 1.0e-12);
    assert!((metrics.max_pressure_error_percent - 50.0).abs() < 1.0e-12);
    assert!((metrics.relative_l2 - 1.118_033_988_749_895).abs() < 1.0e-12);
    assert_eq!(metrics.reference_focus_index, [1, 1, 1]);
    assert_eq!(metrics.candidate_focus_index, [2, 1, 1]);
}

#[test]
fn skull_adaptive_benchmark_selects_ct_conditioned_aperture_and_evaluates() {
    let shape = (25, 25, 25);
    let mut ct = Array3::<f64>::zeros(shape);
    for ix in 0..shape.0 {
        for iy in 0..shape.1 {
            for iz in 6..=8 {
                ct[[ix, iy, iz]] = 900.0 + 30.0 * ix as f64 + 10.0 * iy as f64;
            }
        }
    }
    let skull = ct.mapv(|hu| hu >= 300.0);
    let brain = Array3::<bool>::from_elem(shape, true);
    let config = SkullAdaptiveBenchmarkConfig {
        fus: TranscranialFusPlanConfig {
            element_count: 128,
            frequency_hz: 650_000.0,
            radius_m: 0.012,
            target_peak_pa: 100_000.0,
            samples_per_ray: 80,
            chunk_size: 2048,
            ..TranscranialFusPlanConfig::default()
        },
        aperture_diameter_m: 0.018,
        minimum_active_elements: 4,
    };

    let result = run_skull_adaptive_transcranial_benchmark(
        &ct,
        &skull,
        &brain,
        [1.0e-3; 3],
        [12, 12, 12],
        &config,
    )
    .unwrap();

    assert!(result.placement.active_element_count >= 4);
    assert!(result.placement.mean_skull_length_m > 0.0);
    assert!(result.placement.mean_amplitude_weight > 0.0);
    assert!((result.metrics.reference_peak_pa - 100_000.0).abs() < 2.0);
    assert!(result.metrics.relative_l2 > 1.0e-6);
    assert!(result.metrics.max_pressure_error_percent.is_finite());
    assert_eq!(result.focus_index, [12, 12, 12]);
    assert_eq!(result.reference_pressure_pa.dim(), shape);
    assert_eq!(result.baseline_pressure_pa.dim(), shape);
}
