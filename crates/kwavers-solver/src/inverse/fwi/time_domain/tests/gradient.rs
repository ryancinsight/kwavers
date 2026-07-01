use super::super::{FwiGeometry, FwiProcessor, RHO_SEISMIC_REF};
use crate::inverse::seismic::parameters::FwiParameters;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3, Zip};

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

/// Finite-difference verification that the adjoint-state gradient is a valid
/// descent direction (the FWI "gradient/adjoint test", Fichtner 2010 §A;
/// Plessix 2006 §3; Virieux & Operto 2009 §3.3).
///
/// ## What is verified
/// Let `J(m)` be the data misfit, `g = ∇J(m)` the adjoint-state gradient, and
/// `dJ/ds|_{δm}` the true directional derivative along a perturbation `δm`,
/// measured by a Richardson-extrapolated central finite difference of the
/// *actual* objective the inversion minimises. For two spatially independent
/// directions the test asserts:
/// 1. **Sign / descent correctness** — `g·δm` and `dJ/ds` share sign, so `−g`
///    is a genuine descent direction. This exercises the entire chain end to
///    end (forward solve → residual → time-reversed adjoint injection →
///    `p̈`-cross-correlation imaging condition → `−2/(ρc³)` velocity scaling); a
///    sign error anywhere flips it.
/// 2. **Approximate shape proportionality** — the per-direction ratio
///    `κ = (g·δm)/(dJ/ds)` stays comparable across directions, guarding against
///    gross corruption of the gradient's spatial structure.
///
/// ## What is NOT verified, and why (a real, root-caused defect)
/// An *exact* discrete adjoint would give `κ ≡ 1`. This implementation is an
/// **approximate** adjoint: measured `κ_a ≈ 238`, `κ_b ≈ 191` (both stable under
/// step refinement). Causes — a ~200× global scale offset because the adjoint
/// re-injects the residual through the scaled additive-source path
/// (`2·dt·c₀/(N·dx)`, forward/fdtd/source_handler/scaling.rs) while the forward
/// receiver operator samples pressure directly (not exact transposes); and a
/// ~20% direction-dependent shape error from PML non-self-adjointness and the
/// leapfrog/staggered-grid time-stepping. Both are absorbed by the Armijo line
/// search (so FWI converges), but the gradient is **not** the exact `∇J`. The
/// exact-gradient path (`κ ≈ 1` to <1e-4) is the self-adjoint engine
/// (`FwiEngine::SecondOrderSelfAdjoint`, ADR 016), verified by
/// `self_adjoint::tests::self_adjoint_gradient_matches_finite_difference_kappa_one`;
/// this test characterises the retained approximate-adjoint `FwiEngine::Solver`
/// default.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn test_fwi_adjoint_gradient_is_valid_descent_direction() {
    let (nx, ny, nz) = (12usize, 12, 12);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);
    let c0 = SOUND_SPEED_WATER_SIM;

    // Acquisition: one pressure source on the left, a receiver plane on the
    // right. Both are kept clear of the perturbation support below.
    let mut p_mask = Array3::from_elem(dims, 0.0_f64);
    p_mask[[2, 6, 6]] = 1.0;
    let nt = 80usize;
    let dt = 1e-7; // CFL = c0·dt/dx ≈ 0.15, well inside the 3-D FDTD limit.
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..20 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase * 0.25).exp() * (2.0 * phase).sin();
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
    let mut sensor_mask = Array3::from_elem(dims, false);
    for iy in 3..9 {
        for iz in 3..9 {
            sensor_mask[[9, iy, iz]] = true;
        }
    }
    let geometry = FwiGeometry::new(source, sensor_mask);

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        ..FwiParameters::default()
    };
    let processor = FwiProcessor::new(parameters);

    // Boundary-clear perturbation direction `δm` (max magnitude 1): a smooth
    // slab spanning the full cross-section between the source (ix=2) and the
    // receiver plane (ix=9). A transmission perturbation of this kind changes
    // the arrival time of the through-going wave at every receiver, producing a
    // data misfit and gradient well above the round-off floor (a localised
    // point scatterer in transmission does not). Hard-zeroed within 2 cells of
    // every face so it never overlaps the source, the receiver plane, or the PML.
    let mut delta_m = Array3::zeros(dims);
    for ((ix, iy, iz), value) in delta_m.indexed_iter_mut() {
        let near_boundary =
            ix < 2 || iy < 2 || iz < 2 || ix >= nx - 2 || iy >= ny - 2 || iz >= nz - 2;
        if near_boundary {
            continue;
        }
        // Smooth slab centred at ix=6.5 (between source ix=2 and receivers
        // ix=9), flat in y,z. Kept clear of the source's near field: a
        // perturbation within ~2 cells of the source overlaps the region where
        // the gradient is dominated by the source-wavelet imprint (huge p̈) and
        // is not a faithful ∂J/∂c — exactly what the near-source mute removes.
        let axial = (-((ix as f64 - 6.5).powi(2)) / 0.98).exp();
        *value = axial;
    }
    // Source-voxel mute applied to the adjoint gradient (Sun & Symes 1991;
    // Virieux & Operto 2009). δm is already ~0 in this region, so it does not
    // bias g·δm, but muting keeps the gradient consistent with production runs.
    let mut source_mute = Array3::zeros(dims);
    source_mute[[2, 6, 6]] = 1.0;

    // "True" model carries the slab; the starting model is homogeneous, so the
    // data misfit — and therefore the gradient — is non-trivial at `m`.
    let mut true_model = Array3::from_elem(dims, c0);
    Zip::from(&mut true_model)
        .and(&delta_m)
        .for_each(|c, &d| *c += 100.0 * d);
    let model = Array3::from_elem(dims, c0);

    let observed = processor
        .generate_synthetic_data(&true_model, &geometry, &grid)
        .expect("observed data");

    // Adjoint-state gradient at `m`.
    let (synth0, history0) = processor
        .forward_model(&model, &geometry, &grid)
        .expect("forward at m");
    let j0 = processor
        .compute_misfit_objective(&observed, &synth0)
        .expect("J(m)");
    let residual = processor
        .compute_adjoint_source(&observed, &synth0)
        .expect("adjoint source");
    let adjoint_source = processor
        .build_adjoint_source(&residual, &geometry)
        .expect("adjoint grid source");
    let gradient = processor
        .adjoint_model(
            &adjoint_source,
            &model,
            &grid,
            &history0,
            Some(&source_mute),
        )
        .expect("adjoint gradient");

    assert!(
        j0 > 0.0,
        "data misfit at the starting model must be non-zero; J(m) = {j0:e}"
    );

    // Second, spatially distinct perturbation direction: the same slab tapered
    // by a transverse ramp in y. It overlaps a different sub-volume, so its
    // directional derivative is independent of δm_a's.
    let mut delta_m_b = delta_m.clone();
    for ((_, iy, _), value) in delta_m_b.indexed_iter_mut() {
        *value *= (iy as f64) / (ny as f64);
    }

    // Directional derivative g · δm for a direction.
    let directional = |dir: &Array3<f64>| -> f64 {
        Zip::from(&gradient)
            .and(dir)
            .fold(0.0, |acc, &g, &d| acc + g * d)
    };

    // Richardson-extrapolated central finite-difference slope dJ/ds along `dir`,
    // s the step amplitude. Central differences carry an O(ε²) curvature error;
    // `(4·D(ε/2) − D(ε))/3` cancels it, leaving O(ε⁴), so the result is the
    // true ε→0 directional derivative to high accuracy.
    let fd_slope = |dir: &Array3<f64>| -> f64 {
        let central = |eps: f64| -> f64 {
            let objective_at = |scale: f64| -> f64 {
                let mut perturbed = model.clone();
                Zip::from(&mut perturbed)
                    .and(dir)
                    .for_each(|c, &d| *c += scale * d);
                let synth = processor
                    .generate_synthetic_data(&perturbed, &geometry, &grid)
                    .expect("forward at perturbed model");
                processor
                    .compute_misfit_objective(&observed, &synth)
                    .expect("perturbed objective")
            };
            (objective_at(eps) - objective_at(-eps)) / (2.0 * eps)
        };
        let eps = 4.0_f64; // [m/s]; differences stay well above the round-off floor.
        (4.0 * central(eps / 2.0) - central(eps)) / 3.0
    };

    let g_a = directional(&delta_m);
    let g_b = directional(&delta_m_b);
    let fd_a = fd_slope(&delta_m);
    let fd_b = fd_slope(&delta_m_b);

    assert!(
        g_a.abs() > 0.0 && g_b.abs() > 0.0,
        "directional derivatives must be non-trivial; g·δm_a = {g_a:e}, g·δm_b = {g_b:e}"
    );

    // (1) Descent-direction correctness: the adjoint directional derivative must
    // agree in SIGN with the finite difference for both directions. A sign error
    // anywhere in the residual → time-reversed injection → p̈-correlation →
    // velocity-scaling chain flips this and fails here.
    assert!(
        g_a * fd_a > 0.0,
        "direction A: adjoint and finite difference must share sign; g·δm_a = {g_a:e}, FD = {fd_a:e}"
    );
    assert!(
        g_b * fd_b > 0.0,
        "direction B: adjoint and finite difference must share sign; g·δm_b = {g_b:e}, FD = {fd_b:e}"
    );

    // (2) Approximate-proportionality (characterisation guard, NOT exactness).
    //
    // Define κ = (g·δm) / (dJ/ds). For an EXACT discrete adjoint, κ ≡ 1 for every
    // direction. This implementation is an APPROXIMATE adjoint, so κ ≠ 1 and is
    // mildly direction-dependent; the measured values are κ_a ≈ 238, κ_b ≈ 191
    // (both stable under step-size refinement — a real effect, not FD error). Two
    // distinct, root-caused causes:
    //   • Global scale offset (~200×): the adjoint re-injects the residual through
    //     the SCALED additive-source path (`p_scale = 2·dt·c₀/(N·dx)`, see
    //     forward/fdtd/source_handler/scaling.rs) while the forward receiver
    //     operator samples pressure directly — the two are not exact discrete
    //     transposes.
    //   • Direction-dependent shape error (~20%): PML non-self-adjointness and the
    //     leapfrog/staggered-grid time-stepping whose exact transpose is not the
    //     plain time-reversal used here.
    // Both are absorbed by the Armijo line search (κ>0 ⇒ valid descent), so FWI
    // converges; neither is acceptable for any consumer of the ABSOLUTE gradient
    // magnitude. The exact discrete-adjoint fix is tracked as a [major] item in
    // docs/book/backlog.md; when it lands, replace this guard with κ ≈ 1.
    //
    // This bound only guards against gross corruption of the gradient shape (a
    // future regression that scaled one direction's sensitivity by >~1.6× would
    // fail); it deliberately does not assert the defective magnitudes as a spec.
    let kappa_a = g_a / fd_a;
    let kappa_b = g_b / fd_b;
    assert!(
        kappa_a > 0.0 && kappa_b > 0.0,
        "κ must be positive for a descent direction; κ_a = {kappa_a:.3}, κ_b = {kappa_b:.3}"
    );
    let kappa_ratio = kappa_b / kappa_a;
    assert!(
        (0.6..=1.6).contains(&kappa_ratio),
        "adjoint gradient shape must remain approximately proportional to the true \
         gradient across independent directions (current approximate-adjoint spread \
         ~20%): κ_a = {kappa_a:.3} (g·δm_a={g_a:e}, FD_a={fd_a:e}), \
         κ_b = {kappa_b:.3} (g·δm_b={g_b:e}, FD_b={fd_b:e}), ratio = {kappa_ratio:.4}"
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

/// Deterministic non-trivial test model with directional structure along every
/// axis and diagonal (so all TV difference directions are exercised).
fn directional_tv_test_model(dims: (usize, usize, usize)) -> Array3<f64> {
    let mut m = Array3::<f64>::zeros(dims);
    for ((i, j, k), v) in m.indexed_iter_mut() {
        let (a, b, c) = (i as f64 * 0.3, j as f64 * 0.4, k as f64 * 0.25);
        // Smooth, non-separable pattern: gradients are O(0.1) everywhere, keeping
        // the Huber weight W well above the ε² floor so the functional is smooth.
        *v = a.sin() + (b + 0.2 * a).cos() + 0.5 * (c - 0.1 * b).sin();
    }
    m
}

/// A constant field has zero total-variation gradient for both the axis-only
/// and the four-direction (FDTV) stencils.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_directional_tv_gradient_is_zero_for_constant_field() {
    use super::super::gradient::{directional_tv_gradient, AXIS_TV_DIRECTIONS, FDTV_DIRECTIONS};

    let model = Array3::from_elem((6, 5, 4), 1537.0_f64);
    for dirs in [&AXIS_TV_DIRECTIONS[..], &FDTV_DIRECTIONS[..]] {
        let g = directional_tv_gradient(&model, dirs);
        let max_abs = g.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        // W = ε so every difference is exactly 0/ε = 0.
        assert!(
            max_abs < 1e-12,
            "constant field must give a zero TV gradient; max |g| = {max_abs:e}"
        );
    }
}

/// The analytically derived FDTV gradient is the exact functional derivative of
/// the discrete FDTV functional — verified by a central finite-difference check
/// at every voxel (the gradient test, strongest correctness tier).
///
/// `∂J/∂m[q] ≈ (J(m + h e_q) − J(m − h e_q)) / 2h` to O(h²); with h = 1e-6 and
/// O(1) curvature the central difference matches the analytic gradient to ~1e-8.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_directional_tv_gradient_matches_finite_difference_of_functional() {
    use super::super::gradient::{
        directional_tv_functional, directional_tv_gradient, FDTV_DIRECTIONS,
    };

    let dims = (5, 5, 5);
    let model = directional_tv_test_model(dims);
    let analytic = directional_tv_gradient(&model, &FDTV_DIRECTIONS);

    let h = 1e-6_f64;
    let mut max_err = 0.0_f64;
    for (idx, &g) in analytic.indexed_iter() {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus[idx] += h;
        minus[idx] -= h;
        let fd = (directional_tv_functional(&plus, &FDTV_DIRECTIONS)
            - directional_tv_functional(&minus, &FDTV_DIRECTIONS))
            / (2.0 * h);
        let err = (g - fd).abs();
        max_err = max_err.max(err);
        assert!(
            err <= 1e-6 + 1e-5 * g.abs(),
            "FDTV analytic gradient must equal the finite difference at {idx:?}: \
             analytic = {g:e}, FD = {fd:e}, err = {err:e}"
        );
    }
    // Gradient must be genuinely non-trivial (guards against a zero-returning mock).
    let max_g = analytic.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    assert!(
        max_g > 1e-2,
        "FDTV gradient must be non-trivial on a structured model; max |g| = {max_g:e}"
    );
    assert!(max_err < 1e-5, "worst-case FD error too large: {max_err:e}");
}

/// The FDTV operator genuinely incorporates the diagonal difference directions:
/// for a field with diagonal structure its per-voxel Huber weight — and hence
/// the functional — strictly exceeds the axis-only stencil's, because the
/// non-negative diagonal terms are added inside the same square root.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_fdtv_functional_exceeds_axis_only_for_diagonal_structure() {
    use super::super::gradient::{directional_tv_functional, AXIS_TV_DIRECTIONS, FDTV_DIRECTIONS};

    // A ramp along the (1,1,0) face diagonal: m = i + j. The (1,1,0) difference
    // is 2 per step while (1,−1,0) is 0, so the diagonal directions carry real
    // signal that the axis-only stencil ignores.
    let dims = (6, 6, 3);
    let mut model = Array3::<f64>::zeros(dims);
    for ((i, j, _), v) in model.indexed_iter_mut() {
        *v = i as f64 + j as f64;
    }

    let axis = directional_tv_functional(&model, &AXIS_TV_DIRECTIONS);
    let fdtv = directional_tv_functional(&model, &FDTV_DIRECTIONS);
    assert!(
        fdtv > axis * (1.0 + 1e-6),
        "FDTV must penalize diagonal structure more than the axis-only TV; \
         axis = {axis:e}, fdtv = {fdtv:e}"
    );
}

/// Value-semantic verification of the adaptive FDTV weight schedule:
/// the scale equals `rel_change / max_change`, clamped to `[min_scale, 1]`,
/// with `1.0` as the no-history fallback.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_adaptive_dtv_scale_tracks_relative_change() {
    use super::super::gradient::adaptive_dtv_scale;

    let min = 0.1;

    // No history (or non-positive max) ⇒ full weight.
    assert_eq!(adaptive_dtv_scale(0.05, 0.0, min), 1.0);
    assert_eq!(adaptive_dtv_scale(f64::NAN, 0.2, min), 1.0);

    // Fastest-moving iteration (rel == max) ⇒ full weight.
    assert!((adaptive_dtv_scale(0.2, 0.2, min) - 1.0).abs() < 1e-15);

    // Mid-convergence: exact ratio while above the floor.
    let s = adaptive_dtv_scale(0.1, 0.4, min); // 0.25
    assert!((s - 0.25).abs() < 1e-15, "expected 0.25, got {s}");

    // Near convergence: ratio below the floor ⇒ clamped to min_scale (prior
    // never fully disabled).
    assert!((adaptive_dtv_scale(0.001, 1.0, min) - min).abs() < 1e-15);

    // Monotone: a larger relative change yields a larger (or equal) scale.
    let lo = adaptive_dtv_scale(0.05, 1.0, min);
    let hi = adaptive_dtv_scale(0.5, 1.0, min);
    assert!(
        hi > lo,
        "scale must increase with relative change: {lo} !< {hi}"
    );

    // Result is always within the documented bounds.
    for &(rel, max) in &[(0.0, 1.0), (3.0, 1.0), (0.7, 0.9), (1e-9, 2.0)] {
        let v = adaptive_dtv_scale(rel, max, min);
        assert!(
            (min..=1.0).contains(&v),
            "scale {v} out of [{min}, 1] for rel={rel}, max={max}"
        );
    }
}
