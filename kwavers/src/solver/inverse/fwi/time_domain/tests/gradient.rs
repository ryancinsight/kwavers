use super::super::{FwiGeometry, FwiProcessor, RHO_SEISMIC_REF};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::seismic::parameters::FwiParameters;
use ndarray::{Array2, Array3};

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
    use super::super::adjoint::apply_velocity_gradient_scaling;

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
    use super::super::adjoint::apply_velocity_gradient_scaling;

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
        let r2 = (ix as f64 - 3.5).powi(2)
            + (iy as f64 - 3.5).powi(2)
            + (iz as f64 - 3.5).powi(2);
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
