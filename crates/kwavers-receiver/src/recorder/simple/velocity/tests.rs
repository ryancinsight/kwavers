use crate::recorder::fields::{SensorRecordField, SensorRecordSpec};
use crate::recorder::simple::SensorRecorder;
use crate::recorder::velocity_statistics::SampledVelocityStats;
use kwavers_core::error::KwaversError;
use leto::{Array1, Array3};

#[test]
fn non_staggered_velocity_sampling_matches_half_cell_shift() {
    let mut mask = Array3::from_elem([4, 3, 2], false);
    mask[[0, 0, 0]] = true;
    mask[[2, 1, 1]] = true;

    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::Pressure,
        SensorRecordField::VelocityNonStaggeredX,
        SensorRecordField::VelocityNonStaggeredY,
        SensorRecordField::VelocityNonStaggeredZ,
    ]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (4, 3, 2), 1, spec).unwrap();

    let pressure = Array3::zeros([4, 3, 2]);
    recorder.record_step(&pressure).unwrap();

    // ux[i,j,k] = i  →  collocated at (0,0,0): (ghost + 0) / 2 = 0.0
    //                    collocated at (2,1,1): (u[1,1,1] + u[2,1,1]) / 2 = (1+2)/2 = 1.5
    let ux = Array3::from_shape_fn([4, 3, 2], |[i, _, _]| i as f64);
    // uy[i,j,k] = 10 + j  →  collocated at (0,0,0): (ghost + 10) / 2 = 5.0
    //                         collocated at (2,1,1): (10 + 11) / 2 = 10.5
    let uy = Array3::from_shape_fn([4, 3, 2], |[_, j, _]| 10.0 + j as f64);
    // uz[i,j,k] = 20 + k  →  collocated at (0,0,0): (ghost + 20) / 2 = 10.0
    //                         collocated at (2,1,1): (20 + 21) / 2 = 20.5
    let uz = Array3::from_shape_fn([4, 3, 2], |[_, _, k]| 20.0 + k as f64);
    recorder.record_velocity_step(&ux, &uy, &uz).unwrap();

    let ux_data = recorder.extract_ux_data().unwrap();
    let uy_data = recorder.extract_uy_data().unwrap();
    let uz_data = recorder.extract_uz_data().unwrap();

    // Sensor 0: (i=0, j=0, k=0)
    assert_eq!(ux_data[[0, 0]], 0.0);
    assert_eq!(uy_data[[0, 0]], 5.0);
    assert_eq!(uz_data[[0, 0]], 10.0);

    // Sensor 1: (i=2, j=1, k=1)
    assert_eq!(ux_data[[1, 0]], 1.5);
    assert_eq!(uy_data[[1, 0]], 10.5);
    assert_eq!(uz_data[[1, 0]], 20.5);
}

#[test]
fn velocity_stats_allocate_only_requested_components() {
    let mut mask = Array3::from_elem([3, 1, 1], false);
    mask[[1, 0, 0]] = true;

    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::Pressure,
        SensorRecordField::VelocityMaxX,
        SensorRecordField::VelocityRmsX,
    ]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (3, 1, 1), 1, spec).unwrap();

    assert_eq!(
        recorder.ux_stats.as_ref().map(|stats| stats.u_max.shape()),
        Some([3, 1, 1])
    );
    assert_eq!(
        recorder.uy_stats.as_ref().map(|stats| stats.u_max.shape()),
        None
    );
    assert_eq!(
        recorder.uz_stats.as_ref().map(|stats| stats.u_max.shape()),
        None
    );

    let pressure = Array3::zeros([3, 1, 1]);
    recorder.record_step(&pressure).unwrap();
    // ux values at (0,0,0)=1.0, (1,0,0)=3.0, (2,0,0)=5.0 → sensor at (1,0,0): max=3, rms=3
    let ux = Array3::from_vec([3, 1, 1], vec![1.0, 3.0, 5.0]).unwrap();
    let uy = Array3::from_elem([3, 1, 1], 7.0);
    let uz = Array3::from_elem([3, 1, 1], 11.0);
    recorder.record_velocity_step(&ux, &uy, &uz).unwrap();

    assert_eq!(recorder.extract_ux_max().unwrap()[0], 3.0);
    assert_eq!(recorder.extract_ux_rms().unwrap()[0], 3.0);
    let mut reusable = Array1::zeros([1]);
    recorder.fill_ux_max(&mut reusable).unwrap();
    assert_eq!(reusable[0], 3.0);
    recorder.fill_ux_min(&mut reusable).unwrap();
    assert_eq!(reusable[0], 3.0);
    recorder.fill_ux_rms(&mut reusable).unwrap();
    assert_eq!(reusable[0], 3.0);

    let mut wrong_len = Array1::zeros([2]);
    let err = recorder.fill_ux_max(&mut wrong_len).unwrap_err();
    assert!(
        matches!(err, KwaversError::DimensionMismatch(message) if message.contains("2 != sensor count 1"))
    );

    let err = recorder.fill_uy_max(&mut reusable).unwrap_err();
    assert!(matches!(err, KwaversError::InvalidInput(message) if message.contains("uy_max")));
    assert_eq!(recorder.extract_uy_max(), None);
    assert_eq!(recorder.extract_uz_max(), None);
    // All-three-component aggregate requires all three stats.
    assert_eq!(
        recorder
            .extract_sampled_velocity_stats()
            .as_ref()
            .map(SampledVelocityStats::num_sensors),
        None
    );
}

#[test]
fn velocity_only_spec_records_after_pressure_step_without_pressure_buffer() {
    let mut mask = Array3::from_elem([3, 1, 1], false);
    mask[[1, 0, 0]] = true;

    let spec = SensorRecordSpec::from_fields(&[SensorRecordField::VelocityX]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (3, 1, 1), 1, spec).unwrap();

    assert_eq!(recorder.extract_pressure_data(), None);

    let pressure = Array3::zeros([3, 1, 1]);
    recorder.record_step(&pressure).unwrap();

    let ux = Array3::from_vec([3, 1, 1], vec![1.0, 3.0, 5.0]).unwrap();
    let uy = Array3::from_elem([3, 1, 1], 7.0);
    let uz = Array3::from_elem([3, 1, 1], 11.0);
    recorder.record_velocity_step(&ux, &uy, &uz).unwrap();

    let ux_data = recorder.extract_ux_data().unwrap();
    assert_eq!(ux_data.shape(), &[1, 1]);
    // Sensor is at (1,0,0) → ux[1,0,0] = 3.0
    assert_eq!(ux_data[[0, 0]], 3.0);
    assert_eq!(recorder.extract_uy_data(), None);
    assert_eq!(recorder.extract_uz_data(), None);
}

#[test]
fn velocity_views_expose_recorded_prefix_without_clone() {
    let mut mask = Array3::from_elem([2, 1, 1], false);
    mask[[1, 0, 0]] = true;

    let spec =
        SensorRecordSpec::from_fields(&[SensorRecordField::Pressure, SensorRecordField::VelocityX]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (2, 1, 1), 2, spec).unwrap();

    recorder.record_step(&Array3::zeros([2, 1, 1])).unwrap();
    // ux[1,0,0] = 9.0
    let ux = Array3::from_vec([2, 1, 1], vec![2.0, 9.0]).unwrap();
    let u0 = Array3::zeros([2, 1, 1]);
    recorder.record_velocity_step(&ux, &u0, &u0).unwrap();

    // Full buffer view: (1 sensor, 2 expected_steps) — col 0 populated, col 1 zero.
    let full = recorder.ux_data_view().unwrap();
    assert_eq!(full.shape(), (1, 2));
    assert_eq!(full[[0, 0]], 9.0);
    assert_eq!(full[[0, 1]], 0.0);

    // uy / uz not allocated.
    assert_eq!(recorder.uy_data_view().map(|view| view.shape()), None);
    assert_eq!(recorder.uz_data_view().map(|view| view.shape()), None);

    // Recorded-prefix view: only the populated column.
    let recorded = recorder.recorded_ux_view().unwrap();
    assert_eq!(recorded.shape(), (1, 1));
    assert_eq!(recorded[[0, 0]], 9.0);
}

#[test]
fn intensity_records_pressure_velocity_product_and_time_average() {
    let mut mask = Array3::from_elem([3, 1, 1], false);
    mask[[1, 0, 0]] = true;

    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::IntensityX,
        SensorRecordField::IntensityAvgX,
    ]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (3, 1, 1), 2, spec).unwrap();

    // IntensityX requires pressure and ix_data, but NOT a separate ux time-series
    // buffer; ux is sampled transiently inside record_velocity_step.
    assert_eq!(
        recorder
            .extract_pressure_data()
            .as_ref()
            .map(|data| data.shape()),
        Some((1, 2))
    );
    assert_eq!(
        recorder.extract_ix_data().as_ref().map(|data| data.shape()),
        Some((1, 2))
    );
    assert_eq!(recorder.extract_ux_data(), None);

    // Step 0: p=4, ux=2  →  I_x = 8
    let mut p0 = Array3::zeros([3, 1, 1]);
    p0[[1, 0, 0]] = 4.0;
    recorder.record_step(&p0).unwrap();
    let ux0 = Array3::from_vec([3, 1, 1], vec![0.0, 2.0, 0.0]).unwrap();
    let u0 = Array3::zeros([3, 1, 1]);
    recorder.record_velocity_step(&ux0, &u0, &u0).unwrap();

    // Step 1: p=10, ux=-1  →  I_x = -10
    let mut p1 = Array3::zeros([3, 1, 1]);
    p1[[1, 0, 0]] = 10.0;
    recorder.record_step(&p1).unwrap();
    let ux1 = Array3::from_vec([3, 1, 1], vec![0.0, -1.0, 0.0]).unwrap();
    recorder.record_velocity_step(&ux1, &u0, &u0).unwrap();

    let ix = recorder.extract_ix_data().unwrap();
    assert_eq!(ix[[0, 0]], 8.0);
    assert_eq!(ix[[0, 1]], -10.0);

    // Time average: (8 + (−10)) / 2 = −1.0
    let i_avg_x = recorder.extract_i_avg_x().unwrap();
    assert_eq!(i_avg_x[0], -1.0);
    let mut reusable = Array1::zeros([1]);
    recorder.fill_i_avg_x(&mut reusable).unwrap();
    assert_eq!(reusable[0], -1.0);

    // y/z intensity not requested.
    assert_eq!(recorder.extract_i_avg_y(), None);
    assert_eq!(recorder.extract_iy_data(), None);
    assert!(recorder.fill_i_avg_y(&mut reusable).is_err());
}

#[test]
fn intensity_average_does_not_allocate_velocity_or_intensity_time_series() {
    let mut mask = Array3::from_elem([3, 1, 1], false);
    mask[[1, 0, 0]] = true;

    // IntensityAvgX alone: running sum yes, time-series buffer no, ux buffer no.
    let spec = SensorRecordSpec::from_fields(&[SensorRecordField::IntensityAvgX]);
    let mut recorder = SensorRecorder::with_spec(Some(&mask), (3, 1, 1), 1, spec).unwrap();

    assert_eq!(
        recorder
            .extract_pressure_data()
            .as_ref()
            .map(|data| data.shape()),
        Some((1, 1))
    );
    assert_eq!(recorder.extract_ux_data(), None);
    assert_eq!(recorder.extract_ix_data(), None);
    assert_eq!(
        recorder.extract_i_avg_x(),
        Some(Array1::from_vec(vec![0.0]))
    );

    let mut pressure = Array3::zeros([3, 1, 1]);
    pressure[[1, 0, 0]] = 4.0;
    recorder.record_step(&pressure).unwrap();

    // ux at sensor (1,0,0) = 2.0  →  I_x = 4.0 * 2.0 = 8.0
    let ux = Array3::from_vec([3, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let u0 = Array3::zeros([3, 1, 1]);
    recorder.record_velocity_step(&ux, &u0, &u0).unwrap();

    assert_eq!(recorder.extract_i_avg_x().unwrap()[0], 8.0);
    let mut reusable = Array1::zeros([1]);
    recorder.fill_i_avg_x(&mut reusable).unwrap();
    assert_eq!(reusable[0], 8.0);

    let mut wrong_len = Array1::zeros([2]);
    assert!(recorder.fill_i_avg_x(&mut wrong_len).is_err());
    // Confirm no transient buffers were populated.
    assert_eq!(recorder.extract_ux_data(), None);
    assert_eq!(recorder.extract_ix_data(), None);
}
