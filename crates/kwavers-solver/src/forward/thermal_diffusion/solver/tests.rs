use super::*;
use aequitas::systems::si::quantities::Time;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use kwavers_medium::{HomogeneousMedium, ThermalProperties};

fn config(spatial_order: usize) -> ThermalDiffusionConfig {
    ThermalDiffusionConfig {
        enable_bioheat: false,
        enable_hyperbolic: false,
        track_thermal_dose: false,
        spatial_order,
        ..Default::default()
    }
}

#[test]
fn second_order_laplacian_keeps_singleton_axis_active_for_other_axes() {
    let grid = Grid::new(5, 5, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    let field = Array3::from_shape_fn((5, 5, 1), |[i, j, _]| (i * i) as f64 + 2.0 * (j * j) as f64);
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    assert_eq!(solver.laplacian_workspace[[2, 2, 0]], 6.0);
    assert_eq!(solver.laplacian_workspace[[0, 2, 0]], 0.0);
}

#[test]
fn fourth_order_laplacian_falls_back_per_axis_on_narrow_dimensions() {
    let grid = Grid::new(7, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(4), &grid);
    let field = Array3::from_shape_fn((7, 3, 1), |[i, j, _]| (i * i) as f64 + 3.0 * (j * j) as f64);
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    let error = (solver.laplacian_workspace[[3, 1, 0]] - 8.0).abs();
    assert!(error <= 8.0 * f64::EPSILON);
}

#[test]
fn fourth_order_laplacian_is_exact_for_quadratic_field_in_3d() {
    let grid = Grid::new(7, 7, 7, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(4), &grid);
    let field = Array3::from_shape_fn((7, 7, 7), |[i, j, k]| {
        let x = i as f64 - 3.0;
        let y = j as f64 - 3.0;
        let z = k as f64 - 3.0;
        x * x + 2.0 * y * y + 3.0 * z * z
    });
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    let error = (solver.laplacian_workspace[[3, 3, 3]] - 12.0).abs();
    assert!(error <= 12.0 * f64::EPSILON);
}

#[test]
fn unsupported_spatial_order_returns_validation_error_without_mutation() {
    let grid = Grid::new(5, 5, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(6), &grid);

    let err = solver.calculate_laplacian(&grid).unwrap_err();

    assert!(
        matches!(err, KwaversError::Validation(_)),
        "unsupported spatial order must return a validation error, got {err:?}"
    );
    assert_eq!(solver.config.spatial_order, 6);
    assert_eq!(solver.laplacian_workspace[[2, 2, 0]], 0.0);
}

#[test]
fn standard_update_consumes_borrowed_source_view_without_source_clone() {
    let grid = Grid::new(3, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    solver.set_temperature(Array3::from_elem([3, 3, 1], 310.0));
    let mut source = Array3::zeros((3, 3, 1));
    source[[1, 1, 0]] = 5.0;

    solver
        .update(&medium, &grid, Time::from_base(2.0), Some(source.view()))
        .unwrap();

    assert_eq!(solver.temperature()[[1, 1, 0]], 320.0);
    assert_eq!(solver.temperature()[[0, 0, 0]], 310.0);
}

#[test]
fn standard_update_consumes_noncontiguous_source_view() {
    let grid = Grid::new(3, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    solver.set_temperature(Array3::from_elem([3, 3, 1], 310.0));
    let mut source = Array3::zeros((3, 3, 1));
    source[[1, 1, 0]] = 5.0;
    let source_view = source
        .slice_with(&s![.., ..;-1, ..])
        .expect("invariant: reversed slice within bounds");
    assert_eq!(source_view.as_slice(), None);

    solver
        .update(&medium, &grid, Time::from_base(2.0), Some(source_view))
        .unwrap();

    assert_eq!(solver.temperature()[[1, 1, 0]], 320.0);
    assert_eq!(solver.temperature()[[0, 0, 0]], 310.0);
}

#[test]
fn standard_update_rejects_mismatched_source_shape_without_mutation() {
    let grid = Grid::new(3, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    solver.set_temperature(Array3::from_elem([3, 3, 1], 310.0));
    let source = Array3::zeros((2, 3, 1));

    let err = solver
        .update(&medium, &grid, Time::from_base(2.0), Some(source.view()))
        .unwrap_err();

    match err {
        KwaversError::DimensionMismatch(message) => assert_eq!(
            message,
            "thermal diffusion source shape [2, 3, 1] does not match temperature shape [3, 3, 1]"
        ),
        other => panic!("expected source dimension mismatch, got {other:?}"),
    }
    assert_eq!(solver.temperature()[[1, 1, 0]], 310.0);
}

/// The oracle for the sampled-diffusivity hoist.
///
/// `update_standard_diffusion` no longer calls `medium.thermal_diffusivity`
/// per cell; it reads `ThermalMaterialFields::alpha`. This recomputes the
/// update with the *original* per-cell accessor and asserts the results are
/// bit-identical over several steps. Bit-equality, not a tolerance: the whole
/// claim of the hoist is that it is the same arithmetic with fewer virtual
/// calls, so any drift here is a real behaviour change.
#[test]
fn standard_update_matches_per_cell_diffusivity_exactly() {
    let grid = Grid::new(5, 4, 3, 1.0e-4, 1.0e-4, 1.0e-4).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let dt = Time::from_base(1.0e-6);

    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    // A curved field, not a linear one: a centered stencil annihilates any
    // linear field, so a ramp would give a zero Laplacian everywhere and both
    // sides of this comparison would agree no matter what the solver did. The
    // `x^2 + 2y^2 + 3z^2` shape is the same quadratic the module's own
    // finite-difference theorem uses.
    let mut initial = Array3::zeros((5, 4, 3));
    for k in 0..3 {
        for j in 0..4 {
            for i in 0..5 {
                let (ii, jj, kk) = (i as f64, j as f64, k as f64);
                initial[[i, j, k]] = 300.0 + ii * ii + 2.0 * jj * jj + 3.0 * kk * kk;
            }
        }
    }
    solver.set_temperature(initial.clone());

    // Reference: the pre-hoist arithmetic, computed independently.
    let mut reference = initial;
    let mut laplacian = Array3::zeros((5, 4, 3));
    let mut saw_nonzero_change = false;
    for _ in 0..3 {
        solver.calculate_laplacian(&grid).unwrap();
        laplacian.assign(&solver.laplacian_workspace);
        // Guard against a vacuous comparison: if the reference never moves,
        // the equality below would hold for any implementation at all.
        assert!(
            laplacian.iter().any(|&value| value != 0.0),
            "the reference Laplacian must be non-zero, or this test proves nothing"
        );
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let alpha = medium.thermal_diffusivity(x, y, z, &grid);
                    let step = dt.into_base() * alpha.mul_add(laplacian[[i, j, k]], 0.0);
                    if step != 0.0 {
                        saw_nonzero_change = true;
                    }
                    reference[[i, j, k]] += step;
                }
            }
        }
        solver.update(&medium, &grid, dt, None).unwrap();
    }

    assert!(
        saw_nonzero_change,
        "the reference update must actually change the field"
    );
    assert_eq!(solver.temperature().as_slice(), reference.as_slice());
}

/// The cache is keyed on the medium's discriminant, so presenting a different
/// medium must re-sample rather than reuse the first medium's coefficients.
/// Without this, a caller that swaps media between steps would silently keep
/// integrating the previous medium's diffusivity.
///
/// The assertions read the cache directly rather than inferring invalidation
/// from a temperature difference: with water's diffusivity of ~1.4e-7 m^2/s
/// the per-step increment is far below the resolution of a ~300 K field, so an
/// indirect numeric check is not a reliable witness for this behaviour.
#[test]
fn swapping_media_resamples_instead_of_reusing_cached_coefficients() {
    let grid = Grid::new(4, 3, 2, 1.0, 1.0, 1.0).unwrap();
    let water =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    // Same grid, different density, so its diffusivity differs.
    let denser =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL * 2.0, SOUND_SPEED_WATER_SIM, &grid);
    let dt = Time::from_base(1.0e7);

    let (x, y, z) = grid.indices_to_coordinates(0, 0, 0);
    assert_ne!(
        water.thermal_diffusivity(x, y, z, &grid),
        denser.thermal_diffusivity(x, y, z, &grid),
        "the two media must differ for this test to mean anything"
    );

    // A curved field: a centered stencil annihilates a linear one, so a ramp
    // would give a zero Laplacian and neither medium's coefficient would
    // influence the update.
    let mut initial = Array3::zeros((4, 3, 2));
    for k in 0..2 {
        for j in 0..3 {
            for i in 0..4 {
                let (ii, jj, kk) = (i as f64, j as f64, k as f64);
                initial[[i, j, k]] = 300.0 + ii * ii + 2.0 * jj * jj + 3.0 * kk * kk;
            }
        }
    }

    // One solver driven twice with different media. The cache is primed with
    // `water` on the first call; the second call must re-sample.
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    assert!(
        solver.thermal_cache.is_none(),
        "a fresh solver must not carry sampled coefficients"
    );

    solver.set_temperature(initial.clone());
    solver.update(&water, &grid, dt, None).unwrap();
    let cached = solver
        .thermal_cache
        .as_ref()
        .expect("update must populate the cache")
        .0;
    assert_eq!(
        cached,
        medium_discriminant(&water),
        "the cache must record the medium it sampled from"
    );
    let water_alpha = solver
        .thermal_cache
        .as_ref()
        .expect("cache present")
        .1
        .alpha[[0, 0, 0]];

    solver.set_temperature(initial);
    solver.update(&denser, &grid, dt, None).unwrap();
    let cached = solver
        .thermal_cache
        .as_ref()
        .expect("update must populate the cache")
        .0;
    assert_eq!(
        cached,
        medium_discriminant(&denser),
        "presenting a different medium must replace the cache key"
    );
    let denser_alpha = solver
        .thermal_cache
        .as_ref()
        .expect("cache present")
        .1
        .alpha[[0, 0, 0]];

    assert_ne!(
        water_alpha, denser_alpha,
        "the re-sampled field must carry the new medium's diffusivity, not \
         the cached one"
    );
}

/// A second step with the *same* medium must reuse the cache rather than
/// re-sampling. This is the half of the contract that produces the win: if
/// every step re-sampled, the hoist would cost an allocation per step instead
/// of removing N virtual calls.
#[test]
fn repeated_steps_with_one_medium_reuse_the_sampled_fields() {
    let grid = Grid::new(4, 3, 2, 1.0, 1.0, 1.0).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let dt = Time::from_base(1.0e7);
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);

    solver.set_temperature(Array3::from_elem((4, 3, 2), 310.0));
    solver.update(&medium, &grid, dt, None).unwrap();

    let alpha_after_first = solver
        .thermal_cache
        .as_ref()
        .expect("cache populated by the first step")
        .1
        .alpha[[1, 1, 1]];
    let key_after_first = solver.thermal_cache.as_ref().expect("cache").0;

    solver.update(&medium, &grid, dt, None).unwrap();
    let alpha_after_second = solver
        .thermal_cache
        .as_ref()
        .expect("cache retained by the second step")
        .1
        .alpha[[1, 1, 1]];
    let key_after_second = solver.thermal_cache.as_ref().expect("cache").0;

    assert_eq!(
        key_after_first, key_after_second,
        "the same medium must map to the same cache key"
    );
    assert_eq!(
        alpha_after_first.to_bits(),
        alpha_after_second.to_bits(),
        "reusing the medium must reproduce the same coefficients bit-for-bit"
    );
}
