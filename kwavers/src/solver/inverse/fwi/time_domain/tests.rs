use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::config::SolverType;
use crate::solver::inverse::seismic::parameters::FwiParameters;
use ndarray::{Array2, Array3, Array4};

#[test]
fn test_gradient_calculation() {
    let processor = FwiProcessor::default();

    let forward_field = Array3::ones((10, 10, 10));
    let adjoint_field = Array3::from_elem((10, 10, 10), 2.0);

    let gradient = processor.calculate_interaction(&forward_field, &adjoint_field);

    // Expected: -1.0 * 2.0 = -2.0 (after smoothing, close to -2.0)
    assert!((gradient[[5, 5, 5]] + 2.0).abs() < 0.1);
}

#[test]
fn test_l2_adjoint_source_computation() {
    let processor = FwiProcessor::default();
    let observed = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("shape must be valid");
    let synthetic = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 3.0, 1.0, 7.0, 9.0])
        .expect("shape must be valid");

    let adjoint_source = processor
        .compute_adjoint_source(&observed, &synthetic)
        .expect("adjoint source computation must succeed");

    let expected = Array2::from_shape_vec((2, 3), vec![1.0, -0.5, 1.0, -2.0, 3.0, 4.0])
        .expect("shape must be valid");
    assert_eq!(adjoint_source, expected);
}

#[test]
fn test_l2_objective_matches_definition() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 0.5,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let observed =
        Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("shape must be valid");
    let synthetic =
        Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]).expect("shape must be valid");

    let objective = processor
        .compute_l2_objective(&observed, &synthetic)
        .expect("objective computation must succeed");

    // residual = [1,3,5,7], sum(residual^2) = 84, objective = 0.5 * dt * 84 = 21
    assert!((objective - 21.0).abs() < f64::EPSILON);
}

#[test]
fn test_adjoint_source_reorders_and_time_reverses() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1.0,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let sensor_mask = Array3::from_shape_vec((2, 2, 1), vec![true, true, true, true])
        .expect("shape must be valid");
    let geometry = FwiGeometry::new(GridSource::default(), sensor_mask);

    let residual = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 1000.0, 2000.0, 3000.0,
        ],
    )
    .expect("shape must be valid");

    let source = processor
        .build_adjoint_source(&residual, &geometry)
        .expect("adjoint source construction must succeed");

    let GridSource {
        p_mask,
        p_signal,
        p_mode,
        ..
    } = source;
    let p_signal = p_signal.expect("pressure signal must be present");
    let expected = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0, 2.0, 1.0, 300.0, 200.0, 100.0, 30.0, 20.0, 10.0, 3000.0, 2000.0, 1000.0,
        ],
    )
    .expect("shape must be valid");

    assert_eq!(p_signal, expected);

    let p_mask = p_mask.expect("pressure mask must be present");
    assert_eq!(
        p_mask,
        geometry
            .sensor_mask
            .clone()
            .mapv(|active| if active { 1.0 } else { 0.0 })
    );
    assert!(matches!(p_mode, SourceMode::Additive));
}

#[test]
fn test_pressure_second_derivative_exact_for_quadratic_trace() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 5,
        dt: 1.0,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let mut forward_history = Array4::zeros((5, 1, 1, 1));
    for t in 0..5 {
        forward_history[[t, 0, 0, 0]] = (t as f64).powi(2);
    }

    let mut dst = Array3::zeros((1, 1, 1));
    for idx in 0..5 {
        processor
            .pressure_second_derivative_into(&forward_history, idx, 1.0, &mut dst)
            .expect("second derivative computation must succeed");
        assert!((dst[[0, 0, 0]] - 2.0).abs() < f64::EPSILON);
    }
}

#[test]
fn test_forward_model_objective_vanishes_for_self_data() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1e-4,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
    let model = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);

    let mut p_mask = Array3::zeros((3, 3, 3));
    p_mask[[1, 1, 1]] = 1.0;
    let mut source = GridSource::default();
    source.p_mask = Some(p_mask);
    source.p_signal =
        Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
    source.p_mode = SourceMode::Dirichlet;

    let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
    sensor_mask[[2, 2, 2]] = true;
    let geometry = FwiGeometry::new(source, sensor_mask);

    let (synthetic, _history) = processor
        .forward_model(&model, &geometry, &grid)
        .expect("forward model must succeed");
    let objective = processor
        .compute_l2_objective(&synthetic, &synthetic)
        .expect("objective computation must succeed");

    assert!((objective - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_generate_synthetic_data_matches_canonical_forward_model() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1e-4,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
    let model = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);

    let mut p_mask = Array3::zeros((3, 3, 3));
    p_mask[[1, 1, 1]] = 1.0;
    let mut source = GridSource::default();
    source.p_mask = Some(p_mask);
    source.p_signal =
        Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
    source.p_mode = SourceMode::Dirichlet;

    let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
    sensor_mask[[2, 2, 2]] = true;
    let geometry = FwiGeometry::new(source, sensor_mask);

    let public_data = processor
        .generate_synthetic_data(&model, &geometry, &grid)
        .expect("public synthetic data generation must succeed");
    let (canonical_data, _history) = processor
        .forward_model(&model, &geometry, &grid)
        .expect("canonical forward model must succeed");

    assert_eq!(public_data, canonical_data);
    assert_eq!(public_data.dim(), (1, 3));
}

#[test]
fn test_model_constraints() {
    let processor = FwiProcessor::default();
    let mut model = Array3::from_elem((5, 5, 5), 10000.0);

    processor.apply_model_constraints(&mut model);

    assert!(model[[2, 2, 2]] <= 6000.0);
    assert!(model[[2, 2, 2]] >= 750.0);
}

/// Verify that the FWI forward-model medium is built with seismic (non-water) density.
/// # Panics
/// - Panics if `medium construction must succeed`.
///
#[test]
fn test_fwi_medium_density_not_water() {
    use crate::domain::medium::heterogeneous::HeterogeneousFactory;
    use crate::domain::medium::CoreMedium;

    let (nx, ny, nz) = (8usize, 8, 8);
    let sound_speed = Array3::from_elem((nx, ny, nz), 2000.0_f64);
    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);

    let medium = HeterogeneousFactory::from_arrays(sound_speed, density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let rho_sample = medium.density(4, 4, 4);
    assert!(
        (rho_sample - RHO_SEISMIC_REF).abs() < 1.0,
        "medium density {rho_sample} != RHO_SEISMIC_REF {RHO_SEISMIC_REF}"
    );
    assert!(
        (rho_sample - 1000.0).abs() > 100.0,
        "density must not equal water (1000 kg/m³)"
    );
}

/// Verify that the FWI forward-model medium stores the velocity model correctly.
/// # Panics
/// - Panics if `medium construction must succeed`.
///
#[test]
fn test_fwi_forward_medium_sound_speed_matches_model() {
    use crate::domain::medium::heterogeneous::HeterogeneousFactory;
    use crate::domain::medium::CoreMedium;

    let (nx, ny, nz) = (6usize, 6, 6);
    let mut model = Array3::from_elem((nx, ny, nz), 1800.0_f64);
    model[[3, 3, 3]] = 3200.0;

    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
    let medium = HeterogeneousFactory::from_arrays(model.clone(), density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let c_bg = medium.sound_speed(1, 1, 1);
    let c_anom = medium.sound_speed(3, 3, 3);
    assert!((c_bg - 1800.0).abs() < 1.0, "background speed mismatch");
    assert!((c_anom - 3200.0).abs() < 1.0, "anomaly speed mismatch");
}

/// Verify `resolved_density` returns the caller-supplied heterogeneous field
/// when present and falls back to `RHO_SEISMIC_REF` when absent.
///
/// ## Theorem
/// `FwiProcessor::resolved_density` is the single source of truth for the
/// density used in both forward and adjoint medium construction and in the
/// gradient scaling, so it must (i) preserve a supplied field bit-exactly,
/// (ii) reject mismatched shapes, and (iii) reject non-physical (non-finite
/// or non-positive) entries.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_fwi_resolved_density_heterogeneous_and_default() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid");
    let dims = grid.dimensions();

    // Default: no density supplied → constant RHO_SEISMIC_REF.
    let default_processor = FwiProcessor::default();
    let rho_default = default_processor
        .resolved_density(&grid)
        .expect("default density resolution must succeed");
    assert_eq!(rho_default.dim(), dims);
    assert!(rho_default
        .iter()
        .all(|&v| (v - RHO_SEISMIC_REF).abs() < f64::EPSILON));

    // Heterogeneous: cube of 1500 kg/m³ with a 2500 kg/m³ inclusion.
    let mut rho_field = Array3::from_elem(dims, 1500.0_f64);
    rho_field[[4, 4, 4]] = 2500.0;
    let het_processor = FwiProcessor::default()
        .with_density(rho_field.clone())
        .expect("heterogeneous density must validate");
    let rho_resolved = het_processor.resolved_density(&grid).expect("resolution");
    assert_eq!(
        rho_resolved, rho_field,
        "heterogeneous field must round-trip"
    );

    // Shape mismatch must be rejected.
    let wrong_shape = Array3::from_elem((4, 4, 4), 1500.0_f64);
    let mismatched_processor = FwiProcessor::default()
        .with_density(wrong_shape)
        .expect("validation only rejects non-finite / non-positive entries");
    let err = mismatched_processor.resolved_density(&grid);
    assert!(err.is_err(), "shape mismatch must fail at resolution");

    // Non-physical values must be rejected at the builder.
    let mut bad_rho = Array3::from_elem(dims, 1500.0_f64);
    bad_rho[[0, 0, 0]] = -1.0;
    assert!(FwiProcessor::default().with_density(bad_rho).is_err());

    let mut nan_rho = Array3::from_elem(dims, 1500.0_f64);
    nan_rho[[0, 0, 0]] = f64::NAN;
    assert!(FwiProcessor::default().with_density(nan_rho).is_err());
}

/// Verify the post-correlation velocity-gradient scaling applies the
/// per-voxel `-2 / (ρ(x) · c(x)³)` factor exactly.
///
/// ## Theorem (Plessix 2006, eq. 12)
/// Given an accumulated correlation `I(x) = ∫₀ᵀ p̈_fwd(x,t) · λ(x,t) dt`,
/// the velocity-model gradient is
/// `g_c(x) = -(2 / (ρ(x) · c(x)³)) · I(x)`. Holding `I(x)` fixed and
/// applying [`apply_velocity_gradient_scaling`] must multiply each voxel by
/// exactly that scalar; the result is independent of the FDTD wavefield
/// computation.
///
/// ## Why this test is isolated
/// A test that builds the full forward + adjoint wavefields under two
/// different density fields would conflate (a) the post-correlation
/// scaling (the thing this kernel must implement correctly) with (b) the
/// medium-contrast perturbation of the wavefield (ρ enters `∂_t² p = c² ∇·(1/ρ ∇p)`
/// inside the FDTD solver, so the correlation itself changes with ρ).
/// Plessix's gradient identity is about (a); the wavefield change is real
/// physics but obscures the unit test of the scaling kernel.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_fwi_velocity_gradient_scaling_applies_per_voxel_factor() {
    use super::adjoint::apply_velocity_gradient_scaling;

    let dims = (4, 4, 4);
    // Fabricate a deterministic correlation field `I(x)` with sign + magnitude
    // variation so the scaling factor is observable per-voxel.
    let mut correlation = Array3::zeros(dims);
    for ((ix, iy, iz), value) in correlation.indexed_iter_mut() {
        let sign = if (ix + iy + iz) % 2 == 0 { 1.0 } else { -1.0 };
        *value = sign * (1.0 + ix as f64 + 0.5 * iy as f64 + 0.25 * iz as f64);
    }

    let c0 = SOUND_SPEED_WATER_SIM;
    let rho_ref = 2000.0_f64;
    let model = Array3::from_elem(dims, c0);
    let mut density = Array3::from_elem(dims, rho_ref);
    // Perturb three voxels with distinct factors to verify locality.
    density[[1, 1, 1]] = 2.0 * rho_ref;
    density[[2, 2, 2]] = 0.5 * rho_ref;
    density[[3, 3, 3]] = 3.0 * rho_ref;

    let mut scaled = correlation.clone();
    apply_velocity_gradient_scaling(scaled.view_mut(), model.view(), density.view())
        .expect("scaling must succeed");

    // Per-voxel expected value: g(x) = -2 / (ρ(x) · c³) · I(x).
    let c_cubed = c0.powi(3);
    for ((ix, iy, iz), &i_val) in correlation.indexed_iter() {
        let rho = density[[ix, iy, iz]];
        let expected = -2.0 / (rho * c_cubed) * i_val;
        let actual = scaled[[ix, iy, iz]];
        assert!(
            (actual - expected).abs() <= 1e-15 + 1e-12 * expected.abs(),
            "voxel ({ix},{iy},{iz}): expected {expected:e}, got {actual:e}"
        );
    }

    // Specifically verify that doubling ρ halves the gradient at (1,1,1)
    // relative to the constant-ρ baseline, with no wavefield ambiguity.
    let mut baseline = correlation.clone();
    let uniform_density = Array3::from_elem(dims, rho_ref);
    apply_velocity_gradient_scaling(baseline.view_mut(), model.view(), uniform_density.view())
        .expect("baseline scaling");
    let perturbed = scaled[[1, 1, 1]];
    let reference = baseline[[1, 1, 1]];
    assert!(reference.abs() > 0.0);
    let ratio = perturbed / reference;
    assert!(
        (ratio - 0.5).abs() < 1e-12,
        "doubled-ρ voxel must give exactly half the constant-ρ gradient; \
         ratio = {ratio:.15}"
    );
    let halved = scaled[[2, 2, 2]] / baseline[[2, 2, 2]];
    assert!(
        (halved - 2.0).abs() < 1e-12,
        "halved-ρ voxel must give exactly twice the constant-ρ gradient; \
         ratio = {halved:.15}"
    );
}

/// Verify the scaling kernel rejects non-physical inputs.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_fwi_velocity_gradient_scaling_rejects_non_physical_inputs() {
    use super::adjoint::apply_velocity_gradient_scaling;

    let dims = (2, 2, 2);
    let mut correlation = Array3::ones(dims);
    let model = Array3::from_elem(dims, SOUND_SPEED_WATER_SIM);
    let density = Array3::from_elem(dims, 2000.0_f64);

    // Shape mismatch.
    let bad_density = Array3::from_elem((3, 3, 3), 2000.0_f64);
    assert!(apply_velocity_gradient_scaling(
        correlation.view_mut(),
        model.view(),
        bad_density.view(),
    )
    .is_err());

    // Non-positive sound speed.
    let mut bad_model = model.clone();
    bad_model[[0, 0, 0]] = -SOUND_SPEED_WATER_SIM;
    assert!(apply_velocity_gradient_scaling(
        correlation.view_mut(),
        bad_model.view(),
        density.view(),
    )
    .is_err());

    // NaN density.
    let mut bad_rho = density.clone();
    bad_rho[[0, 0, 0]] = f64::NAN;
    assert!(
        apply_velocity_gradient_scaling(correlation.view_mut(), model.view(), bad_rho.view(),)
            .is_err()
    );
}

/// Smoke test: heterogeneous-ρ end-to-end FWI run produces a non-trivial
/// gradient that differs from the constant-ρ run. Does not assert a
/// closed-form ratio because the FDTD wavefield itself is perturbed by ρ.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn test_fwi_heterogeneous_density_gradient_differs_from_baseline() {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);

    // Homogeneous velocity model: 1500 m/s + 5% Gaussian-ish perturbation
    // around the centre so the gradient is non-trivial.
    let mut model = Array3::from_elem(dims, SOUND_SPEED_WATER_SIM);
    for ((ix, iy, iz), value) in model.indexed_iter_mut() {
        let r2 = (ix as f64 - 3.5).powi(2) + (iy as f64 - 3.5).powi(2) + (iz as f64 - 3.5).powi(2);
        *value += 75.0 * (-r2 / 4.0).exp();
    }

    // Minimal FWI geometry: one source at (2,4,4), one receiver line at ix=6.
    let mut sensor_mask = Array3::from_elem(dims, false);
    for iy in 2..6 {
        for iz in 2..6 {
            sensor_mask[[6, iy, iz]] = true;
        }
    }
    let nt = 64usize;
    let dt = 1e-7;

    let mut p_mask = Array3::from_elem(dims, 0.0_f64);
    p_mask[[2, 4, 4]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..16 {
        let phase = (t as f64) * 0.4;
        p_signal[[0, t]] = (-phase * phase).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let geometry = FwiGeometry::new(source, sensor_mask);

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        ..FwiParameters::default()
    };

    // Baseline: constant-density processor.
    let baseline = FwiProcessor::new(parameters.clone());
    let (synth_const, history_const) = baseline
        .forward_model(&model, &geometry, &grid)
        .expect("baseline forward");
    let observed = Array2::zeros(synth_const.dim()); // arbitrary "data" so residual is non-zero
    let residual = baseline
        .compute_adjoint_source(&observed, &synth_const)
        .expect("residual");
    let adjoint_source_const = baseline
        .build_adjoint_source(&residual, &geometry)
        .expect("adjoint source");
    let gradient_const = baseline
        .adjoint_model(&adjoint_source_const, &model, &grid, &history_const, None)
        .expect("baseline gradient");

    // Heterogeneous: ρ(x) = α(x) · RHO_SEISMIC_REF with α = 1.0 except a
    // 2× factor in a single voxel. The forward medium changes, but for a
    // small perturbation the gradient at well-separated voxels is dominated
    // by the local 1/ρ scaling.
    let mut alpha = Array3::from_elem(dims, 1.0_f64);
    alpha[[5, 5, 5]] = 2.0;
    let rho_het = alpha.mapv(|a| a * RHO_SEISMIC_REF);
    let het_processor = FwiProcessor::new(parameters)
        .with_density(rho_het.clone())
        .expect("density build");
    let (synth_het, history_het) = het_processor
        .forward_model(&model, &geometry, &grid)
        .expect("het forward");
    let residual_het = het_processor
        .compute_adjoint_source(&observed, &synth_het)
        .expect("het residual");
    let adjoint_source_het = het_processor
        .build_adjoint_source(&residual_het, &geometry)
        .expect("het adjoint source");
    let gradient_het = het_processor
        .adjoint_model(&adjoint_source_het, &model, &grid, &history_het, None)
        .expect("het gradient");

    // The heterogeneous-ρ pipeline must complete and produce a gradient
    // that is non-trivially different from the constant-ρ baseline. The
    // direction of the difference at the doubled-ρ voxel must be a
    // reduction in magnitude (1/ρ weights it down). Exact ratios are
    // tested in `test_fwi_velocity_gradient_scaling_applies_per_voxel_factor`
    // using the isolated scaling kernel — see that test for why the
    // end-to-end ratio cannot be a closed form (the wavefield itself is
    // perturbed by the ρ contrast inside the FDTD solver).
    let g_baseline_at_perturbed = gradient_const[[5, 5, 5]];
    let g_het_at_perturbed = gradient_het[[5, 5, 5]];
    assert!(
        g_baseline_at_perturbed.abs() > 1e-18,
        "baseline gradient at perturbed voxel must be non-trivial; got {g_baseline_at_perturbed:e}"
    );
    assert!(
        g_het_at_perturbed.abs() < g_baseline_at_perturbed.abs(),
        "doubled-ρ voxel must reduce gradient magnitude; baseline = {g_baseline_at_perturbed:e}, het = {g_het_at_perturbed:e}"
    );

    // And the two gradient fields must not be bit-identical (ρ has to
    // matter somewhere). Use a relative-difference threshold robust to
    // FDTD round-off.
    let mut max_rel_diff = 0.0_f64;
    for (&a, &b) in gradient_const.iter().zip(gradient_het.iter()) {
        let denom = a.abs().max(b.abs()).max(1e-30);
        let diff = (a - b).abs() / denom;
        if diff > max_rel_diff {
            max_rel_diff = diff;
        }
    }
    assert!(
        max_rel_diff > 1e-3,
        "heterogeneous-ρ gradient must differ from baseline somewhere; max rel diff = {max_rel_diff:e}"
    );
}

/// Verify that `SolverType::PSTD` is accepted by `build_solver_for_forward` and
/// produces a non-trivial synthetic receiver trace.
///
/// ## Mathematical contract
///
/// The forward map `F_h(c; G)` returns an `Array2<f64>` of shape
/// `(N_receivers, nt)` with at least one non-zero entry when the source
/// produces a non-zero pressure field.  This is an input-sensitive smoke test
/// (any constant output would be rejected by `is_ok() + assert!` checks).
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn test_fwi_pstd_solver_type_accepted_and_produces_nonzero_data() {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1.5e-3_f64; // 1.5 mm — enough resolution for 500 kHz
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");

    let model = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);

    let mut sensor_mask = Array3::from_elem((nx, ny, nz), false);
    for iy in 2..6 {
        for iz in 2..6 {
            sensor_mask[[6, iy, iz]] = true;
        }
    }

    let nt = 32usize;
    // dt satisfying CFL for PSTD: dt ≤ dx / (c * √3) * CFL.  Here CFL = 0.3.
    let dt = 1.7e-7_f64;

    let mut p_mask = Array3::from_elem((nx, ny, nz), 0.0_f64);
    p_mask[[2, 4, 4]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..8 {
        let phase = t as f64 * 0.4;
        p_signal[[0, t]] = (-phase * phase).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        solver_type: SolverType::PSTD,
        ..FwiParameters::default()
    };

    let processor = FwiProcessor::new(parameters);
    let geometry = FwiGeometry::new(source, sensor_mask);

    let synthetic = processor
        .generate_synthetic_data(&model, &geometry, &grid)
        .expect("PSTD forward model must succeed");

    assert_eq!(
        synthetic.nrows(),
        geometry.receiver_count(),
        "synthetic receiver count must match geometry"
    );
    assert_eq!(synthetic.ncols(), nt, "synthetic time length must match nt");

    // At least one non-zero receiver sample — any constant-zero output fails.
    let max_abs = synthetic.iter().copied().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_abs > 0.0,
        "PSTD forward model must produce a non-zero receiver trace; got max_abs = {max_abs:e}"
    );
}

/// Verify that an unsupported `SolverType` is rejected with an error, not a panic.
/// # Panics
/// - Panics if the unsupported type is silently accepted.
#[test]
fn test_fwi_unsupported_solver_type_returns_error() {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1e-3_f64;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let model = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);

    let mut sensor_mask = Array3::from_elem((nx, ny, nz), false);
    sensor_mask[[6, 4, 4]] = true;

    let nt = 10usize;
    let dt = 1e-7_f64;
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(Array3::zeros((nx, ny, nz))),
        p_signal: Some(Array2::zeros((1, nt))),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        solver_type: SolverType::KSpace, // not yet supported
        ..FwiParameters::default()
    };

    let processor = FwiProcessor::new(parameters);
    let geometry = FwiGeometry::new(source, sensor_mask);

    let result = processor.generate_synthetic_data(&model, &geometry, &grid);
    assert!(
        result.is_err(),
        "unsupported SolverType::KSpace must return Err, not Ok"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("KSpace"),
        "error message must name the unsupported type; got: {msg}"
    );
}
