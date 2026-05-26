use super::*;
use crate::core::constants::numerical::{TWO_PI};

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
    let reference_wavenumber = TWO_PI * 200_000.0 / sound_speed;
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
    let reference_wavenumber = TWO_PI * frequency_hz / sound_speed;
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
    let reference_wavenumber = TWO_PI * 200_000.0 / sound_speed;
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
