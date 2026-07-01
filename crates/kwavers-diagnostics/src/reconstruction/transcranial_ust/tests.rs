use ndarray::{Array2, Array3};

use kwavers_solver::inverse::linear_born_inversion::LinearBornInversionConfig;
use kwavers_transducer::transducers::focused::BowlAngularBounds;
use kwavers_transducer::transducers::TransducerGeometry;

use super::{
    reconstruct_brain_slice, reconstruct_brain_volume, transducer::TranscranialBowlGeometry,
    AcousticSlice, AcousticVolume, CtResampledVolume, TranscranialUstBornInversionConfig,
};

#[test]
fn transcranial_bowl_geometry_uses_distinct_element_positions() {
    let geometry =
        TranscranialBowlGeometry::from_aperture(64, 0.11, BowlAngularBounds::hemisphere()).unwrap();
    assert_eq!(geometry.len(), 64);

    let min_z = geometry
        .elements
        .iter()
        .map(|element| element.z_m)
        .fold(f64::INFINITY, f64::min);
    let max_z = geometry
        .elements
        .iter()
        .map(|element| element.z_m)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(min_z >= 0.0);
    assert!(max_z > 0.9 * 0.11);
    for element in &geometry.elements {
        let radius = (element.x_m.powi(2) + element.y_m.powi(2) + element.z_m.powi(2)).sqrt();
        assert!((radius - 0.11).abs() < 1.0e-12);
        assert!(element.z_m >= 0.0);
    }

    let receivers = geometry.receiver_indices(&[32, 16, 48]);
    assert_eq!(receivers.len(), 64 * 3);
    for source_idx in 0..64 {
        for offset_idx in 0..3 {
            let receiver_idx = receivers[source_idx * 3 + offset_idx];
            assert!(receiver_idx < 64);
            assert_ne!(receiver_idx, source_idx);
        }
    }
}

#[test]
fn transcranial_bowl_geometry_uses_configured_source_aperture() {
    let aperture = BowlAngularBounds::from_axis_projection_bounds(-0.25, 0.95).unwrap();
    let geometry = TranscranialBowlGeometry::from_aperture(80, 0.11, aperture).unwrap();

    let min_projection = geometry
        .elements
        .iter()
        .map(|element| element.z_m / 0.11)
        .fold(f64::INFINITY, f64::min);
    let max_projection = geometry
        .elements
        .iter()
        .map(|element| element.z_m / 0.11)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(geometry.len(), 80);
    assert!((min_projection + 0.25).abs() < 0.04);
    assert!((max_projection - 0.95).abs() < 0.04);
}

#[test]
fn transcranial_ust_volume_inversion_reconstructs_coupled_three_dimensional_array() {
    let mut hu = Array3::<f64>::from_elem((12, 12, 12), -1000.0);
    let center = 5.5;
    for ix in 0..12 {
        for iy in 0..12 {
            for iz in 0..12 {
                let dx = ix as f64 - center;
                let dy = iy as f64 - center;
                let dz = iz as f64 - center;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                hu[[ix, iy, iz]] = if (4.5..=5.4).contains(&r) {
                    760.0
                } else if r < 4.5 {
                    20.0 + 45.0 * (-(dx * dx + 0.7 * dy * dy + 0.5 * dz * dz) / 20.0).exp()
                } else {
                    -1000.0
                };
            }
        }
    }

    let volume = AcousticVolume::from_ct_hu(CtResampledVolume {
        hu,
        spacing_m: 4.0e-3,
        source_slice_index: 6,
        source_volume_index: 6,
    })
    .unwrap();
    let config = TranscranialUstBornInversionConfig {
        element_count: 24,
        radius_m: 0.07,
        linear: LinearBornInversionConfig {
            frequencies_hz: vec![180_000.0, 320_000.0],
            receiver_offsets: vec![12, 6, 18],
            iterations: 6,
            relaxation: 0.8,
            regularization: 5.0e-5,
            ..LinearBornInversionConfig::default()
        },
        ..TranscranialUstBornInversionConfig::default()
    };

    let result = reconstruct_brain_volume(&volume, &config).unwrap();

    assert_eq!(result.reconstruction_sound_speed_m_s.dim(), (12, 12, 12));
    assert_eq!(result.synthetic_data.len(), config.measurement_count());
    assert!(result.metrics.active_voxels > 40);
    assert_eq!(
        result.metrics.continuation_stages,
        config.linear.frequencies_hz.len()
    );
    assert!(
        result.metrics.final_objective < result.metrics.initial_objective,
        "3-D inversion objective did not decrease"
    );
    assert!(
        result.metrics.migration_dynamic_range_m_s > 0.05 * result.metrics.target_dynamic_range_m_s,
        "3-D migration did not recover a finite target dynamic range"
    );
    assert!(
        result.metrics.reconstruction_dynamic_range_m_s
            > 0.10 * result.metrics.target_dynamic_range_m_s,
        "3-D reconstruction did not recover a finite target dynamic range"
    );
    assert_ne!(
        result.reconstruction_sound_speed_m_s[[6, 6, 5]],
        result.reconstruction_sound_speed_m_s[[6, 6, 7]],
        "sliced 3-D reconstruction must retain axial variation"
    );
}

/// Build a 2-D CT phantom: a circular skull ring of bone HU enclosing brain
/// tissue, air outside. `skull_hu == None` yields a skull-free uniform-tissue
/// head (used to isolate the travel-time reduction).
fn ct_head_phantom(n: usize, skull_hu: Option<f64>) -> Array2<f64> {
    let center = (n - 1) as f64 / 2.0;
    let r_outer = 0.42 * n as f64;
    let r_inner = 0.34 * n as f64;
    let mut hu = Array2::<f64>::from_elem((n, n), -1000.0);
    for ix in 0..n {
        for iy in 0..n {
            let r = ((ix as f64 - center).powi(2) + (iy as f64 - center).powi(2)).sqrt();
            hu[[ix, iy]] = match skull_hu {
                Some(bone) if (r_inner..=r_outer).contains(&r) => bone,
                _ if r < r_outer => 40.0, // brain / soft tissue
                _ => -1000.0,             // air
            };
        }
    }
    hu
}

// The CT-derived skull travel-time τ = ∫dl/c through a bony ring is SHORTER than
// the same straight ray through a skull-free soft-tissue head, because the skull
// sound speed (~2.6 km/s) far exceeds brain tissue (~1.55 km/s). This phase
// advance is the dominant transcranial aberration the kernel must model.
#[test]
fn transcranial_skull_traveltime_is_shorter_than_soft_tissue_path() {
    let n = 40usize;
    let spacing = 1.0e-3;
    let skull = AcousticSlice::from_ct_hu(ct_head_phantom(n, Some(820.0)), spacing).unwrap();
    let soft = AcousticSlice::from_ct_hu(ct_head_phantom(n, None), spacing).unwrap();

    // A diametric ray crossing the ring on both sides (well inside the grid).
    let half = 0.018;
    let tau_skull =
        super::sensitivity::slowness_line_integral(&skull, -half, 0.0, 0.0, half, 0.0, 0.0);
    let tau_soft =
        super::sensitivity::slowness_line_integral(&soft, -half, 0.0, 0.0, half, 0.0, 0.0);

    assert!(tau_skull > 0.0 && tau_soft > 0.0);
    assert!(
        tau_skull < 0.97 * tau_soft,
        "skull travel time {tau_skull:.3e}s must be meaningfully shorter than soft-tissue {tau_soft:.3e}s"
    );
}

// For a homogeneous-speed head the travel-time integral must reduce to the
// constant-speed phase: τ = L / c. This is what makes the aberration kernel a
// strict generalization that collapses to the original constant-tissue kernel.
#[test]
fn transcranial_traveltime_reduces_to_distance_over_speed_for_uniform_medium() {
    let n = 40usize;
    let spacing = 1.0e-3;
    // Fully uniform soft-tissue head (HU = 40 everywhere) so every sampled point
    // along any interior ray has the same speed and τ = L/c is exact.
    let soft = AcousticSlice::from_ct_hu(Array2::from_elem((n, n), 40.0), spacing).unwrap();
    let c = soft.sound_speed_m_s[[n / 2, n / 2]];
    assert!(soft
        .sound_speed_m_s
        .iter()
        .all(|&v| (v - c).abs() < 1.0e-12));

    let (ax, bx) = (-0.012_f64, 0.013_f64);
    let (ay, by) = (-0.009_f64, 0.011_f64);
    let tau = super::sensitivity::slowness_line_integral(&soft, ax, ay, 0.0, bx, by, 0.0);
    let length = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    let expected = length / c;
    assert!(
        (tau - expected).abs() <= 1.0e-9 * expected,
        "uniform-medium τ {tau:.6e} must equal L/c {expected:.6e}"
    );
}

// The CT-derived skull sound speed must change the encoded measurement: with the
// aberration model on, the skull phase advance shifts the finite-frequency data
// relative to the constant-tissue-speed kernel. Mirrors the attenuation-model
// data-sensitivity check below.
#[test]
fn transcranial_ct_aberration_changes_encoded_data() {
    let n = 32usize;
    let medium = AcousticSlice::from_ct_hu(ct_head_phantom(n, Some(800.0)), 3.0e-3).unwrap();
    let config = TranscranialUstBornInversionConfig {
        element_count: 48,
        radius_m: 0.07,
        aberration_model: true,
        linear: LinearBornInversionConfig {
            frequencies_hz: vec![220_000.0, 360_000.0],
            receiver_offsets: vec![24, 12, 36],
            iterations: 4,
            ..LinearBornInversionConfig::default()
        },
        ..TranscranialUstBornInversionConfig::default()
    };
    let with_aberration = reconstruct_brain_slice(&medium, &config).unwrap();

    let mut without = config.clone();
    without.aberration_model = false;
    let no_aberration = reconstruct_brain_slice(&medium, &without).unwrap();

    let data_difference: f64 = with_aberration
        .synthetic_data
        .iter()
        .zip(no_aberration.synthetic_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        data_difference > 1.0e-6,
        "CT-derived skull travel time must change the encoded data (got {data_difference:.3e})"
    );
}

#[test]
fn transcranial_ust_inversion_reduces_data_objective_and_recovers_contrast() {
    let mut hu = Array2::<f64>::from_elem((32, 32), -1000.0);
    let center = 15.5;
    for ix in 0..32 {
        for iy in 0..32 {
            let dx = ix as f64 - center;
            let dy = iy as f64 - center;
            let r = (dx * dx + dy * dy).sqrt();
            hu[[ix, iy]] = if (11.5..=13.8).contains(&r) {
                780.0
            } else if r < 11.5 {
                25.0 + 55.0 * (-(dx * dx + 0.6 * dy * dy) / 70.0).exp()
            } else if r < 15.0 {
                40.0
            } else {
                -1000.0
            };
        }
    }

    let medium = AcousticSlice::from_ct_hu(hu, 3.0e-3).unwrap();
    assert!(
        medium.attenuation_np_per_m_mhz[[16, 28]] > medium.attenuation_np_per_m_mhz[[16, 16]],
        "skull attenuation must exceed soft-tissue attenuation"
    );
    let config = TranscranialUstBornInversionConfig {
        element_count: 64,
        radius_m: 0.07,
        linear: LinearBornInversionConfig {
            frequencies_hz: vec![180_000.0, 260_000.0, 340_000.0],
            receiver_offsets: vec![32, 24, 40, 16, 48],
            iterations: 18,
            relaxation: 0.9,
            regularization: 1.0e-5,
            ..LinearBornInversionConfig::default()
        },
        ..TranscranialUstBornInversionConfig::default()
    };

    let result = reconstruct_brain_slice(&medium, &config).unwrap();

    assert!(result.metrics.active_voxels > 100);
    assert_eq!(result.metrics.measurements, config.measurement_count());
    assert_eq!(
        result.metrics.continuation_stages,
        config.linear.frequencies_hz.len()
    );
    assert!(config.linear.attenuation_model);
    assert_eq!(config.harmonic_count(), 2);
    assert!(
        result.metrics.final_objective < 0.35 * result.metrics.initial_objective,
        "final objective {} did not reduce from {}",
        result.metrics.final_objective,
        result.metrics.initial_objective
    );
    assert!(
        result.metrics.migration_pearson_correlation > 0.25,
        "migration correlation too low: {}",
        result.metrics.migration_pearson_correlation
    );
    assert!(
        result.metrics.migration_dynamic_range_m_s > 0.10 * result.metrics.target_dynamic_range_m_s,
        "migration dynamic range {} did not recover target range {}",
        result.metrics.migration_dynamic_range_m_s,
        result.metrics.target_dynamic_range_m_s
    );
    assert!(
        result.metrics.pearson_correlation > 0.35,
        "correlation too low: {}",
        result.metrics.pearson_correlation
    );
    assert!(
        result.metrics.normalized_rmse <= result.metrics.migration_normalized_rmse,
        "Born inversion normalized RMSE {} did not improve over migration {}",
        result.metrics.normalized_rmse,
        result.metrics.migration_normalized_rmse
    );
    assert!(
        result.metrics.reconstruction_dynamic_range_m_s
            > 0.25 * result.metrics.target_dynamic_range_m_s,
        "reconstruction dynamic range {} did not recover target range {}",
        result.metrics.reconstruction_dynamic_range_m_s,
        result.metrics.target_dynamic_range_m_s
    );
    assert!(
        result.metrics.enhanced_dynamic_range_m_s
            >= result.metrics.reconstruction_dynamic_range_m_s,
        "enhanced dynamic range {} did not preserve reconstruction range {}",
        result.metrics.enhanced_dynamic_range_m_s,
        result.metrics.reconstruction_dynamic_range_m_s
    );
    let enhanced_center = result.enhanced_reconstruction_sound_speed_m_s[[16, 16]];
    let reconstruction_center = result.reconstruction_sound_speed_m_s[[16, 16]];
    assert_ne!(enhanced_center, reconstruction_center);
    let migration_center = result.migration_sound_speed_m_s[[16, 16]];
    let initial_center = result.initial_sound_speed_m_s[[16, 16]];
    assert_ne!(migration_center, initial_center);

    let mut no_attenuation = config.clone();
    no_attenuation.linear.attenuation_model = false;
    let no_attenuation_result = reconstruct_brain_slice(&medium, &no_attenuation).unwrap();
    let data_difference: f64 = result
        .synthetic_data
        .iter()
        .zip(no_attenuation_result.synthetic_data.iter())
        .map(|(with_attenuation, without_attenuation)| {
            (with_attenuation - without_attenuation).abs()
        })
        .sum();
    assert!(
        data_difference > 1.0e-6,
        "CT-derived attenuation must change encoded data"
    );

    let mut linear = config.clone();
    linear.linear.nonlinear_harmonic_model = false;
    let linear_result = reconstruct_brain_slice(&medium, &linear).unwrap();
    assert_eq!(linear.harmonic_count(), 1);
    assert_eq!(
        result.synthetic_data.len(),
        2 * linear_result.synthetic_data.len()
    );
    let harmonic_energy = result
        .synthetic_data
        .iter()
        .skip(1)
        .step_by(config.harmonic_count())
        .map(|value| value * value)
        .sum::<f64>();
    assert!(
        harmonic_energy > 0.0,
        "nonlinear harmonic rows must carry finite encoded data"
    );
}
