use super::*;

#[test]
fn pstd_receiver_projection_uses_exact_grid_cells_and_adjoint() {
    let grid = GridSpec::new((2, 1, 1), 1.0e-3).unwrap();
    let array = MultiRowRingArray::from_ordered_elements(
        2,
        1,
        1.0e-3,
        0.0,
        vec![grid.center_at(0, 0, 0), grid.center_at(1, 0, 0)],
    )
    .unwrap();
    let operator = GreenOperatorKind::SpectralPstdPeriodic {
        time_step_s: 1.0e-7,
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        temporal_transfer: None,
        absorbing_boundary: AbsorbingBoundary::disabled(),
    };
    let field = [Complex64::new(3.0, -1.0), Complex64::new(-2.0, 0.5)];
    let receiver_values = [Complex64::new(0.25, -0.5), Complex64::new(-1.5, 2.0)];

    let sampled = sample_array_for_operator(grid, &field, &array, operator).unwrap();
    let adjoint = receiver_adjoint_for_operator(grid, &array, &receiver_values, operator).unwrap();

    assert_eq!(sampled, field.iter().cloned().collect::<Vec<_>>());
    assert_eq!(adjoint, receiver_values.iter().cloned().collect::<Vec<_>>());
    assert!(
        (inner_product(&sampled, &receiver_values) - inner_product(&field, &adjoint)).norm()
            <= 1.0e-14
    );
}

#[test]
fn pstd_receiver_projection_rejects_off_grid_receivers() {
    let grid = GridSpec::new((2, 1, 1), 1.0e-3).unwrap();
    let array = MultiRowRingArray::from_ordered_elements(
        2,
        1,
        1.0e-3,
        0.0,
        vec![
            ElementPosition {
                x_m: -0.25e-3,
                y_m: 0.0,
                z_m: 0.0,
            },
            grid.center_at(1, 0, 0),
        ],
    )
    .unwrap();
    let operator = GreenOperatorKind::SpectralPstdPeriodic {
        time_step_s: 1.0e-7,
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        temporal_transfer: None,
        absorbing_boundary: AbsorbingBoundary::disabled(),
    };
    let field = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

    let err = sample_array_for_operator(grid, &field, &array, operator)
        .expect_err("off-grid PSTD receiver must reject");

    assert!(err.to_string().contains("receiver point coordinate"));
}
