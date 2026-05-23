use super::absorbing::absorbing_weights;
use super::green::{apply_shifted_green, apply_shifted_green_adjoint, shifted_outgoing_green};
use super::grid::{bli_weights, BliConfig, GridSpec};
use super::spectral::{
    apply_shifted_green_pstd_spectral_adjoint_with_boundary,
    apply_shifted_green_pstd_spectral_with_boundary, apply_shifted_green_spectral,
    apply_shifted_green_spectral_adjoint, apply_shifted_green_spectral_adjoint_with_boundary,
    apply_shifted_green_spectral_with_boundary,
};
use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use crate::solver::inverse::linear_born_inversion::ElementPosition;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn scattering_potential_matches_slowness_square_contract() {
    let slowness = Array3::from_shape_vec((2, 1, 1), vec![0.0005, 0.00025]).unwrap();
    let potential = real_scattering_potential(4.0, &slowness, 0.00025).unwrap();

    assert_eq!(potential.len(), 2);
    assert_eq!(
        potential[0],
        16.0 * (0.0005_f64.powi(2) - 0.00025_f64.powi(2))
    );
    assert_eq!(potential[1], 0.0);
}

#[test]
fn convergence_epsilon_bounds_potential_norm() {
    let potential = [-3.0, 2.0, 0.25];
    let epsilon = convergence_epsilon(&potential).unwrap();

    assert_eq!(epsilon, 3.0);
    assert!(potential.iter().all(|value| value.abs() <= epsilon));
}

#[test]
fn shifted_potential_has_negative_imaginary_part() {
    let shifted = shifted_potential(&[2.0, -1.0], 3.0).unwrap();

    assert_eq!(shifted[0], Complex64::new(2.0, -3.0));
    assert_eq!(shifted[1], Complex64::new(-1.0, -3.0));
}

#[test]
fn pointwise_preconditioner_is_finite_for_valid_shift() {
    let shifted = shifted_potential(&[0.0, 2.0], 2.0).unwrap();
    let gamma = pointwise_preconditioner(&shifted, 2.0).unwrap();

    assert_eq!(gamma[0], Complex64::new(-1.0, 0.0));
    assert!(gamma[1].re.is_finite());
    assert!(gamma[1].im.is_finite());
}

#[test]
fn bli_weights_collapse_to_single_on_grid_voxel() {
    let grid = GridSpec::new((3, 3, 3), 0.01).unwrap();
    let point = grid.center_at(1, 1, 1);
    let weights = bli_weights(grid, point, BliConfig::default()).unwrap();

    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].linear_index, grid.linear_index(1, 1, 1));
    assert_eq!(weights[0].weight, 1.0);
}

#[test]
fn dense_green_matches_shifted_outgoing_green_for_unit_point_source() {
    let grid = GridSpec::new((2, 1, 1), 0.01).unwrap();
    let source_index = grid.linear_index(0, 0, 0);
    let receiver_index = grid.linear_index(1, 0, 0);
    let source_density = [
        Complex64::new(1.0 / grid.cell_volume_m3(), 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let potential = [0.0, 0.0];
    let solution = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 4,
            relative_tolerance: 1.0e-12,
        },
    )
    .unwrap();
    let shifted = super::green::shifted_wavenumber(3.0, solution.epsilon);
    let expected = shifted_outgoing_green(
        grid.center_at(0, 0, 0),
        grid.center_at(1, 0, 0),
        shifted,
        grid.min_distance_m(),
    );

    assert_eq!(source_index, 0);
    assert_eq!(receiver_index, 1);
    assert!((solution.field[receiver_index] - expected).norm() <= 1.0e-10);
}

#[test]
fn shifted_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((2, 2, 1), 0.01).unwrap();
    let x = [
        Complex64::new(0.5, -0.25),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.125, 0.5),
        Complex64::new(0.25, -0.875),
    ];
    let y = [
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, 0.125),
        Complex64::new(1.0, -0.5),
        Complex64::new(-0.75, -0.25),
    ];
    let gx = apply_shifted_green(grid, 3.0, 0.2, &x);
    let ghy = apply_shifted_green_adjoint(grid, 3.0, 0.2, &y);
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}

#[test]
fn spectral_green_constant_source_matches_zero_mode_symbol() {
    let grid = GridSpec::new((4, 4, 2), 0.01).unwrap();
    let source = vec![Complex64::new(2.0, -0.5); grid.len()];
    let reference_wavenumber = 11.0;
    let epsilon = 0.25;
    let field = apply_shifted_green_spectral(grid, reference_wavenumber, epsilon, &source);
    let expected = source[0] / Complex64::new(reference_wavenumber * reference_wavenumber, epsilon);

    for value in field {
        assert!(
            (value - expected).norm() <= 1.0e-12 * expected.norm().max(1.0),
            "value={value}, expected={expected}"
        );
    }
}

#[test]
fn pstd_source_density_uses_grid_mask_and_source_kappa_symbol() {
    let grid = GridSpec::new((2, 1, 1), 1.0e-3).unwrap();
    let source = [ElementPosition {
        x_m: -0.5e-3,
        y_m: 0.0,
        z_m: 0.0,
    }];
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let time_step = 1.0e-7;
    let reference_wavenumber = 2.0 * PI * 200_000.0 / sound_speed;
    let density = source_density_for_operator(
        grid,
        &source,
        reference_wavenumber,
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s: time_step,
            reference_sound_speed_m_s: sound_speed,
            temporal_transfer: None,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        },
    )
    .unwrap();
    let q = (0.5 * sound_speed * time_step * PI / grid.spacing_m).cos();
    let scale = 1.0 / grid.cell_volume_m3();

    assert!((density[0].re - 0.5 * (1.0 + q) * scale).abs() <= 1.0e-6);
    assert!((density[1].re - 0.5 * (1.0 - q) * scale).abs() <= 1.0e-6);
    assert!(density.iter().all(|value| value.im.abs() <= 1.0e-12));
}

#[test]
fn pstd_temporal_modal_bin_matches_quarter_cycle_recurrence() {
    let response = pstd_modal_frequency_bin_response(
        0.0,
        PstdTemporalBinConfig {
            frequency_hz: 1.0,
            time_step_s: 0.25,
            total_steps: 4,
            bin_start_step: 0,
            source_gain: 1.0,
        },
    )
    .unwrap();

    assert!((response - Complex64::new(-0.5, -0.5)).norm() <= 1.0e-15);
}

#[test]
fn pstd_temporal_source_transfer_matches_modal_bin_response() {
    let grid = GridSpec::new((2, 1, 1), 1.0).unwrap();
    let source = [grid.center_at(0, 0, 0)];
    let sound_speed = 1.0;
    let time_step = 0.1;
    let frequency_hz = 1.0;
    let reference_wavenumber = 2.0 * PI * frequency_hz / sound_speed;
    let transfer = PstdTemporalTransferConfig {
        source_amplitude_pa: 2.0,
        cycles_per_frequency: 2,
        frequency_bin_cycles: 1,
    }
    .bin_config(frequency_hz, time_step, grid.spacing_m, sound_speed)
    .unwrap();
    let density = source_density_for_operator(
        grid,
        &source,
        reference_wavenumber,
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s: time_step,
            reference_sound_speed_m_s: sound_speed,
            temporal_transfer: Some(transfer),
            absorbing_boundary: AbsorbingBoundary::disabled(),
        },
    )
    .unwrap();
    let field = apply_shifted_green_pstd_spectral_with_boundary(
        grid,
        reference_wavenumber,
        0.0,
        &density,
        time_step,
        sound_speed,
        AbsorbingBoundary::disabled(),
    );

    let mut expected = Array3::<Complex64>::zeros(grid.dimensions);
    expected[[0, 0, 0]] = Complex64::new(1.0, 0.0);
    let mut spectrum = Array3::<Complex64>::zeros(grid.dimensions);
    fft_3d_complex_into(&expected, &mut spectrum);
    let (nx, ny, nz) = grid.dimensions;
    for ix in 0..nx {
        let kx = angular_mode_for_test(ix, nx, grid.spacing_m);
        for iy in 0..ny {
            let ky = angular_mode_for_test(iy, ny, grid.spacing_m);
            for iz in 0..nz {
                let kz = angular_mode_for_test(iz, nz, grid.spacing_m);
                let k = kx.mul_add(kx, ky.mul_add(ky, kz * kz)).sqrt();
                let kappa = pstd_source_kappa_symbol(k, time_step, sound_speed);
                let theta_squared = pstd_modal_theta_squared(k, time_step, sound_speed);
                let response = pstd_modal_frequency_bin_response(theta_squared, transfer).unwrap();
                spectrum[[ix, iy, iz]] *= Complex64::new(kappa, 0.0) * response;
            }
        }
    }
    ifft_3d_complex_inplace(&mut spectrum);
    let expected = spectrum.iter().copied().collect::<Vec<_>>();

    for (&actual, &expected_value) in field.iter().zip(expected.iter()) {
        assert!(
            (actual - expected_value).norm() <= 1.0e-10 * expected_value.norm().max(1.0),
            "actual={actual}, expected={expected_value}"
        );
    }
}

#[test]
fn pstd_temporal_symbols_match_leapfrog_identities() {
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let time_step = 1.0e-7;
    let grid_wavenumber = PI / 1.0e-3;
    let reference_wavenumber = 2.0 * PI * 200_000.0 / sound_speed;
    let half_grid_phase = 0.5 * sound_speed * time_step * grid_wavenumber;
    let half_reference_phase = 0.5 * sound_speed * time_step * reference_wavenumber;

    assert!(
        (pstd_source_kappa_symbol(grid_wavenumber, time_step, sound_speed) - half_grid_phase.cos())
            .abs()
            <= 1.0e-15
    );
    assert!(
        (pstd_modal_theta_squared(grid_wavenumber, time_step, sound_speed)
            - 4.0 * half_grid_phase.sin().powi(2))
        .abs()
            <= 1.0e-15
    );
    let expected_denominator = (4.0 * half_reference_phase.sin().powi(2)
        - 4.0 * half_grid_phase.sin().powi(2))
        / (sound_speed * time_step).powi(2);
    assert!(
        (pstd_leapfrog_symbol(
            reference_wavenumber,
            grid_wavenumber,
            time_step,
            sound_speed
        ) - expected_denominator)
            .abs()
            <= 1.0e-8
    );
}

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

    assert_eq!(sampled, field.to_vec());
    assert_eq!(adjoint, receiver_values.to_vec());
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

#[test]
fn pstd_spectral_green_constant_source_matches_leapfrog_zero_mode_symbol() {
    let grid = GridSpec::new((4, 4, 2), 0.01).unwrap();
    let source = vec![Complex64::new(2.0, -0.5); grid.len()];
    let reference_sound_speed = SOUND_SPEED_WATER_SIM;
    let time_step = 1.0e-7;
    let reference_wavenumber = 11.0;
    let epsilon = 0.25;
    let field = apply_shifted_green_pstd_spectral_with_boundary(
        grid,
        reference_wavenumber,
        epsilon,
        &source,
        time_step,
        reference_sound_speed,
        AbsorbingBoundary::disabled(),
    );
    let scale = reference_sound_speed * time_step;
    let temporal_symbol =
        4.0 * (0.5 * reference_wavenumber * scale).sin().powi(2) / (scale * scale);
    let expected = source[0] / Complex64::new(temporal_symbol, epsilon);

    for value in field {
        assert!(
            (value - expected).norm() <= 1.0e-12 * expected.norm().max(1.0),
            "value={value}, expected={expected}"
        );
    }
}

#[test]
fn spectral_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((4, 3, 2), 0.01).unwrap();
    let x = (0..grid.len())
        .map(|index| Complex64::new(index as f64 * 0.25, -0.125 * index as f64))
        .collect::<Vec<_>>();
    let y = (0..grid.len())
        .map(|index| Complex64::new(0.5 - index as f64 * 0.125, 0.25 * index as f64))
        .collect::<Vec<_>>();
    let gx = apply_shifted_green_spectral(grid, 11.0, 0.25, &x);
    let ghy = apply_shifted_green_spectral_adjoint(grid, 11.0, 0.25, &y);
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}

#[test]
fn pstd_spectral_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((4, 3, 2), 0.01).unwrap();
    let x = (0..grid.len())
        .map(|index| Complex64::new(index as f64 * 0.25, -0.125 * index as f64))
        .collect::<Vec<_>>();
    let y = (0..grid.len())
        .map(|index| Complex64::new(0.5 - index as f64 * 0.125, 0.25 * index as f64))
        .collect::<Vec<_>>();
    let boundary = AbsorbingBoundary::disabled();
    let gx = apply_shifted_green_pstd_spectral_with_boundary(
        grid,
        11.0,
        0.25,
        &x,
        1.0e-7,
        SOUND_SPEED_WATER_SIM,
        boundary,
    );
    let ghy = apply_shifted_green_pstd_spectral_adjoint_with_boundary(
        grid,
        11.0,
        0.25,
        &y,
        1.0e-7,
        SOUND_SPEED_WATER_SIM,
        boundary,
    );
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}

#[test]
fn polynomial_absorbing_boundary_has_unit_interior_and_edge_decay() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let boundary = AbsorbingBoundary::polynomial(1, 2.0, 2).unwrap();
    let weights = absorbing_weights(grid, boundary).unwrap();

    assert_eq!(weights[grid.linear_index(2, 2, 2)], 1.0);
    assert!((weights[grid.linear_index(0, 2, 2)] - (-2.0_f64).exp()).abs() <= f64::EPSILON);
    assert!((weights[grid.linear_index(0, 0, 0)] - (-6.0_f64).exp()).abs() <= f64::EPSILON);
}

#[test]
fn spectral_absorbing_boundary_damps_edge_source_response() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let mut source = vec![Complex64::new(0.0, 0.0); grid.len()];
    source[grid.linear_index(0, 2, 2)] = Complex64::new(1.0, 0.0);
    let periodic = apply_shifted_green_spectral(grid, 11.0, 0.25, &source);
    let absorbed = apply_shifted_green_spectral_with_boundary(
        grid,
        11.0,
        0.25,
        &source,
        AbsorbingBoundary::polynomial(1, 2.0, 2).unwrap(),
    );

    assert!(
        norm(&absorbed) < norm(&periodic),
        "absorbed_norm={}, periodic_norm={}",
        norm(&absorbed),
        norm(&periodic)
    );
}

#[test]
fn spectral_absorbing_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let boundary = AbsorbingBoundary::polynomial(1, 1.5, 2).unwrap();
    let x = (0..grid.len())
        .map(|index| Complex64::new(index as f64 * 0.125, -0.03125 * index as f64))
        .collect::<Vec<_>>();
    let y = (0..grid.len())
        .map(|index| Complex64::new(0.25 - index as f64 * 0.0625, 0.125 * index as f64))
        .collect::<Vec<_>>();
    let gx = apply_shifted_green_spectral_with_boundary(grid, 11.0, 0.25, &x, boundary);
    let ghy = apply_shifted_green_spectral_adjoint_with_boundary(grid, 11.0, 0.25, &y, boundary);
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}

#[test]
fn cbs_solver_reports_decreasing_fixed_point_residual() {
    let grid = GridSpec::new((2, 1, 1), 0.01).unwrap();
    let source_density = [
        Complex64::new(1.0 / grid.cell_volume_m3(), 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let potential = [0.05, -0.02];
    let loose = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 1,
            relative_tolerance: 1.0e-14,
        },
    )
    .unwrap();
    let refined = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 8,
            relative_tolerance: 1.0e-14,
        },
    )
    .unwrap();

    assert!(refined.relative_residual < loose.relative_residual);
    assert!(refined.iterations >= loose.iterations);
}

fn inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&left, &right)| left.conj() * right)
        .sum()
}

fn norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn angular_mode_for_test(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    2.0 * PI * signed_index / (count as f64 * spacing_m)
}
