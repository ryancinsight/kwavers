use super::*;
use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;

#[test]
fn pstd_receiver_projection_uses_exact_grid_cells_and_adjoint() {
    let grid = GridSpec::new((2, 1, 1), 1.0e-3).unwrap();
    let array = MultiRowRingArray::from_ordered_elements(
        2,
        1,
        Length::from_unit::<Meter>(1.0e-3),
        Length::from_unit::<Meter>(0.0),
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

    let sampled = sample_array_for_operator(grid, &field, array.elements(), operator).unwrap();
    let adjoint =
        receiver_adjoint_for_operator(grid, array.elements(), &receiver_values, operator).unwrap();

    assert_eq!(sampled, field.to_vec());
    assert_eq!(adjoint, receiver_values.to_vec());
    assert!(
        (inner_product(&sampled, &receiver_values) - inner_product(&field, &adjoint)).norm()
            <= 1.0e-14
    );
}

#[test]
fn pstd_receiver_projection_interpolates_off_grid_receivers() {
    // Domain widening (ATLAS-FWI-PSTD-BLI-106): the PSTD projections accept
    // off-node elements through the same BLI stencil as the continuous
    // operators. On-node behavior is covered by
    // `pstd_receiver_projection_uses_exact_grid_cells_and_adjoint`; this test
    // pins the value semantics at an off-node point against the direct BLI
    // definition.
    let grid = GridSpec::new((2, 1, 1), 1.0e-3).unwrap();
    let off_node = ElementPosition {
        x: Length::from_unit::<Meter>(0.25e-3),
        y: Length::from_unit::<Meter>(0.0),
        z: Length::from_unit::<Meter>(0.0),
    };
    let array = MultiRowRingArray::from_ordered_elements(
        2,
        1,
        Length::from_unit::<Meter>(1.0e-3),
        Length::from_unit::<Meter>(0.0),
        vec![off_node, grid.center_at(1, 0, 0)],
    )
    .unwrap();
    let operator = GreenOperatorKind::SpectralPstdPeriodic {
        time_step_s: 1.0e-7,
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        temporal_transfer: None,
        absorbing_boundary: AbsorbingBoundary::disabled(),
    };
    let field = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];

    let sampled = sample_array_for_operator(grid, &field, array.elements(), operator).unwrap();

    // Analytical oracles. Off-node: the BLI definition itself, evaluated by
    // hand against the centered-grid node centers (x = -h/2 and +h/2):
    // u(x) = sum_j sinc(pi (x - c_j) / h) u_j. On-node: exact cell value.
    fn sinc(t: f64) -> f64 {
        if t == 0.0 {
            1.0
        } else {
            t.sin() / t
        }
    }
    let h = 1.0e-3;
    let x = 0.25e-3;
    let w0 = sinc(std::f64::consts::PI * (x - (-0.5 * h)) / h);
    let w1 = sinc(std::f64::consts::PI * (x - 0.5 * h) / h);
    let expected_off_node = field[0] * w0 + field[1] * w1;
    assert!((sampled[0] - expected_off_node).norm() <= 1.0e-14);
    assert_eq!(sampled[1], field[1]);

    // The adjoint remains the transpose: <R u, r> == <u, R^H r>.
    let residual = [Complex64::new(0.5, -0.25), Complex64::new(-1.0, 2.0)];
    let adjoint =
        receiver_adjoint_for_operator(grid, array.elements(), &residual, operator).unwrap();
    let lhs: Complex64 = sampled
        .iter()
        .zip(residual.iter())
        .map(|(&u, &r)| u.conj() * r)
        .sum();
    let rhs: Complex64 = field.iter().zip(adjoint.iter()).map(|(&u, &w)| u * w).sum();
    assert!((lhs - rhs).norm() <= 1.0e-14);
}
