#[cfg(test)]
mod simulation_contract_tests {
    use super::super::Simulation;
    use crate::grid_py::Grid;
    use crate::misc_bindings::time_reversal_reconstruction_impl;
    use kwavers_domain::sensor::recorder::fields::SensorRecordField;
    use ndarray::{array, Array2};

    #[test]
    fn trim_initial_recorder_sample_aligns_with_kwave_time_axis() {
        // Default record_start_index=1: keeps cols [0..Nt) → drops the LAST
        // column (post-final-step value), aligning with k-Wave's Nt-sample
        // window indexed by physical times [0, dt, …, (Nt−1)·dt].
        let nt_plus_one = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed = Simulation::trim_initial_recorder_sample(nt_plus_one, 3, 1);
        assert_eq!(trimmed.shape(), &[2, 3]);
        assert_eq!(trimmed[[0, 0]], 0.0); // p(0) preserved
        assert_eq!(trimmed[[0, 1]], 1.0);
        assert_eq!(trimmed[[0, 2]], 2.0); // p(2dt); p(3dt) dropped
        assert_eq!(trimmed[[1, 0]], 10.0);
        assert_eq!(trimmed[[1, 2]], 12.0);

        // Buffer already Nt cols (e.g. velocity buffers populated only inside
        // step_forward): pass through unchanged at start=1.
        let exact_nt = array![[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]];
        let untouched = Simulation::trim_initial_recorder_sample(exact_nt.clone(), 3, 1);
        assert_eq!(untouched, exact_nt);

        // record_start_index=2: skip col 0, keep cols [1..Nt) → Nt-1 cols.
        let nt_plus_one2 = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed2 = Simulation::trim_initial_recorder_sample(nt_plus_one2, 3, 2);
        assert_eq!(trimmed2.shape(), &[2, 2]);
        assert_eq!(trimmed2[[0, 0]], 1.0);
        assert_eq!(trimmed2[[0, 1]], 2.0);
        assert_eq!(trimmed2[[1, 1]], 12.0);
    }

    #[test]
    fn trim_initial_recorder_view_aligns_with_kwave_time_axis() {
        let nt_plus_one = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed = Simulation::trim_initial_recorder_view(nt_plus_one.view(), 3, 1);
        assert_eq!(trimmed.shape(), &[2, 3]);
        assert_eq!(trimmed[[0, 0]], 0.0);
        assert_eq!(trimmed[[0, 2]], 2.0);
        assert_eq!(trimmed[[1, 2]], 12.0);

        let exact_nt = array![[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]];
        let untouched = Simulation::trim_initial_recorder_view(exact_nt.view(), 3, 1);
        assert_eq!(untouched, exact_nt);
    }

    #[test]
    fn record_modes_to_spec_maps_acoustic_intensity_fields() {
        let modes = vec!["Ix".to_string(), "I_avg_x".to_string()];
        let spec = Simulation::record_modes_to_spec(&modes);

        assert!(spec.contains(SensorRecordField::IntensityX));
        assert!(spec.contains(SensorRecordField::IntensityAvgX));
        assert!(spec.records_pressure());
        assert!(!spec.records_ux());
        assert!(spec.needs_any_velocity());
        assert!(spec.records_intensity_x());
        assert!(!spec.records_intensity_y());
    }

    #[test]
    fn time_reversal_reconstruction_impl_preserves_zero_field_with_pml_crop() {
        let grid = Grid::new(6, 6, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let sensor_data = Array2::zeros((3, 8));
        let sensor_positions = array![[0.0, 0.0, 0.0], [0.0, 0.1e-3, 0.0], [0.0, 0.2e-3, 0.0],];

        let reconstruction = time_reversal_reconstruction_impl(
            sensor_data,
            sensor_positions,
            &grid.inner,
            1500.0,
            1.0e8,
            Some(2),
        )
        .unwrap();

        assert_eq!(reconstruction.dim(), (6, 6, 1));
        assert!(reconstruction.iter().all(|&value| value == 0.0));
    }
}
