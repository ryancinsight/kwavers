use super::*;
use kwavers_core::constants::fundamental::{
    DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_transducer::transducers::ElementPosition;
use ndarray::{Array3, ArrayView1};

#[test]
fn pstd_dataset_preserves_shape_and_is_input_sensitive() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 1.0e-3,
        time_step_s: 1.0e-7,
        cycles_per_frequency: 1,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };
    let baseline = Array3::from_elem((12, 12, 3), SOUND_SPEED_WATER_SIM);
    let mut perturbed = baseline.clone();
    perturbed[[6, 6, 1]] = SOUND_SPEED_TISSUE;

    let first = generate_breast_ust_pstd_frequency_dataset(&baseline, &array, &[200_000.0], config)
        .expect("baseline dataset");
    let second =
        generate_breast_ust_pstd_frequency_dataset(&perturbed, &array, &[200_000.0], config)
            .expect("perturbed dataset");

    assert_eq!(first.observed_pressure.dim(), (1, 4, 4));
    assert_eq!(first.transmissions, 4);
    assert_eq!(first.receivers, 4);
    assert_eq!(first.time_steps_per_frequency[0], 50);
    assert_eq!(first.frequency_bin_start_steps_per_frequency[0], 0);
    assert_eq!(first.observations()[0].observed_pressure.shape(), [4, 4]);
    let delta = first
        .observed_pressure
        .iter()
        .zip(second.observed_pressure.iter())
        .map(|(&a, &b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        delta > 0.0,
        "PSTD dataset must respond to sound-speed changes"
    );
}

#[test]
fn pstd_dataset_rejects_unstable_cfl() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");
    let config = BreastUstPstdDatasetConfig {
        spacing_m: 1.0e-3,
        time_step_s: 1.0e-6,
        cycles_per_frequency: 1,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        density_kg_m3: DENSITY_WATER_NOMINAL,
        cpml_thickness_cells: 0,
    };
    let model = Array3::from_elem((12, 12, 3), SOUND_SPEED_WATER_SIM);

    let err = generate_breast_ust_pstd_frequency_dataset(&model, &array, &[200_000.0], config)
        .expect_err("CFL violation must reject");

    assert!(err.to_string().contains("CFL"));
}

#[test]
fn frequency_bin_uses_trailing_steady_state_window() {
    let mut samples = Vec::new();
    samples.extend((0..10).map(|n| 100.0 * (TWO_PI * n as f64 / 10.0).sin()));
    samples.extend((10..20).map(|n| 3.0 * (TWO_PI * n as f64 / 10.0).sin()));
    let trace = ArrayView1::from(&samples);

    let bin = frequency_bin(trace, 1.0, 0.1, 10);

    assert!(bin.re.abs() <= 1.0e-12, "bin={bin}");
    assert!((bin.im + 3.0).abs() <= 1.0e-12, "bin={bin}");
}

#[test]
fn snap_multi_row_ring_array_matches_pstd_grid_centers() {
    let array = MultiRowRingArray::new(4, 1, 0.006, 0.0).expect("array");

    let snapped =
        snap_multi_row_ring_array_to_grid(&array, (8, 8, 3), 1.0e-3).expect("snapped array");

    assert_eq!(snapped.element_count(), array.element_count());
    assert_eq!(snapped.cylindrical_source(0)[0], snapped.elements()[0]);
    assert_eq!(
        snapped.elements()[0],
        ElementPosition {
            x_m: 0.0035,
            y_m: 0.0005,
            z_m: 0.0,
        }
    );
}
