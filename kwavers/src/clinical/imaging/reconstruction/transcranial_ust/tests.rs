use ndarray::{Array2, Array3};

use super::{
    reconstruct_brain_slice, reconstruct_brain_volume, transducer::TranscranialBowlGeometry,
    AcousticSlice, AcousticVolume, CtResampledVolume, TranscranialUstBornInversionConfig,
};

#[test]
fn transcranial_bowl_geometry_uses_distinct_element_positions() {
    let geometry = TranscranialBowlGeometry::uniform(64, 0.11).unwrap();
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
        frequencies_hz: vec![180_000.0, 320_000.0],
        receiver_offsets: vec![12, 6, 18],
        iterations: 6,
        relaxation: 0.8,
        regularization: 5.0e-5,
        ..TranscranialUstBornInversionConfig::default()
    };

    let result = reconstruct_brain_volume(&volume, &config).unwrap();

    assert_eq!(result.reconstruction_sound_speed_m_s.dim(), (12, 12, 12));
    assert_eq!(result.synthetic_data.len(), config.measurement_count());
    assert!(result.metrics.active_voxels > 40);
    assert_eq!(
        result.metrics.continuation_stages,
        config.frequencies_hz.len()
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
        frequencies_hz: vec![180_000.0, 260_000.0, 340_000.0],
        receiver_offsets: vec![32, 24, 40, 16, 48],
        iterations: 18,
        relaxation: 0.9,
        regularization: 1.0e-5,
        ..TranscranialUstBornInversionConfig::default()
    };

    let result = reconstruct_brain_slice(&medium, &config).unwrap();

    assert!(result.metrics.active_voxels > 100);
    assert_eq!(result.metrics.measurements, config.measurement_count());
    assert_eq!(
        result.metrics.continuation_stages,
        config.frequencies_hz.len()
    );
    assert!(config.attenuation_model);
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
    no_attenuation.attenuation_model = false;
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
    linear.nonlinear_harmonic_model = false;
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
