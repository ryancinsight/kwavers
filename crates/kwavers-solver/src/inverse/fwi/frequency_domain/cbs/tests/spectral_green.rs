use super::*;

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
