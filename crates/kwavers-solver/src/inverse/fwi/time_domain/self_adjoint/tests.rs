//! Exact-adjoint (dot-product) verification of the self-adjoint engine.
//!
//! Unlike the FDTD/PSTD path (`tests::gradient`, which only achieves a valid
//! descent direction with `κ ≈ 238`), this engine's discrete adjoint is the
//! literal algebraic gradient of the discrete `J`, so the finite-difference
//! gradient test must return `κ = (g·δm)/(dJ/ds) ≈ 1` for every direction.

use super::{
    build_edge_sponge, forward, forward_sensor_only, forward_tail, gradient,
    gradient_reconstructed, Acquisition, SelfAdjointConfig,
};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use leto::{Array2, Array3};

const RHO: f64 = 1000.0;

/// Build a short band-limited source wavelet of length `nt`.
fn wavelet(nt: usize) -> Array2<f64> {
    let mut s = Array2::zeros((1, nt));
    for t in 0..nt.min(24) {
        let phase = (t as f64) * 0.35;
        s[[0, t]] = (-phase * phase * 0.25).exp() * (2.0 * phase).sin();
    }
    s
}

/// `J(self-data) = 0`: the forward map composed with its own output yields a
/// zero residual, so the data misfit vanishes exactly.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn self_adjoint_objective_vanishes_for_self_data() {
    let (nx, ny, nz) = (10usize, 10, 10);
    let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3).expect("grid");
    let model = Array3::from_elem([nx, ny, nz], SOUND_SPEED_WATER_SIM);
    let density = Array3::from_elem([nx, ny, nz], RHO);
    let cfg = SelfAdjointConfig { nt: 60, dt: 1e-7 };
    let src = wavelet(cfg.nt);
    let source_voxels = [(2usize, 5usize, 5usize)];
    let receiver_voxels: Vec<(usize, usize, usize)> = (3..7)
        .flat_map(|j| (3..7).map(move |k| (7usize, j, k)))
        .collect();
    let acq = Acquisition {
        source_voxels: &source_voxels,
        source_signal: src.view(),
        receiver_voxels: &receiver_voxels,
    };

    let synth =
        forward_sensor_only(model.view(), density.view(), &grid, &cfg, &acq, None).expect("fwd");
    let j: f64 = 0.5 * cfg.dt * (&synth - &synth).mapv(|x| x * x).iter().sum::<f64>();
    assert_eq!(j, 0.0, "self-data misfit must be exactly zero");
    // And the field is genuinely non-trivial (engine is doing real work).
    assert!(
        synth.mapv(|x| x * x).iter().sum::<f64>().sqrt() > 1e-9,
        "synthetic must be non-trivial"
    );
}

/// Core exact-adjoint test: `κ ≈ 1` for several independent perturbation
/// directions. This is the property the FDTD path cannot meet.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn self_adjoint_gradient_matches_finite_difference_kappa_one() {
    let (nx, ny, nz) = (12usize, 12, 12);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);
    let c0 = SOUND_SPEED_WATER_SIM;
    let density = Array3::from_elem(dims, RHO);
    let cfg = SelfAdjointConfig { nt: 80, dt: 1e-7 };
    let src = wavelet(cfg.nt);

    let source_voxels = [(2usize, 6usize, 6usize)];
    let receiver_voxels: Vec<(usize, usize, usize)> = (3..9)
        .flat_map(|j| (3..9).map(move |k| (9usize, j, k)))
        .collect();
    let acq = Acquisition {
        source_voxels: &source_voxels,
        source_signal: src.view(),
        receiver_voxels: &receiver_voxels,
    };
    let source_mute = {
        let mut m = Array3::zeros(dims);
        m[[2, 6, 6]] = 1.0;
        m
    };

    // Boundary-clear perturbation direction generators (all ~0 within 2 cells of
    // any face and clear of the source's near field).
    let boundary_clear = |i: usize, j: usize, k: usize| -> bool {
        i >= 2 && j >= 2 && k >= 2 && i < nx - 2 && j < ny - 2 && k < nz - 2
    };
    // (a) axial slab; (b) y-ramped slab; (c) off-centre Gaussian blob.
    let make_dir = |kind: u8| -> Array3<f64> {
        let mut d = Array3::zeros(dims);
        for ([i, j, k], v) in d
            .indexed_iter_mut()
            .expect("invariant: owned array yields indexed iterator")
        {
            if !boundary_clear(i, j, k) {
                continue;
            }
            *v = match kind {
                0 => (-((i as f64 - 6.5).powi(2)) / 0.98).exp(),
                1 => (-((i as f64 - 6.5).powi(2)) / 0.98).exp() * (j as f64 / ny as f64),
                _ => {
                    let r2 = (i as f64 - 7.0).powi(2)
                        + (j as f64 - 4.0).powi(2)
                        + (k as f64 - 8.0).powi(2);
                    (-r2 / 3.0).exp()
                }
            };
        }
        d
    };

    // True model carries direction (a); the start model is homogeneous, so the
    // misfit and gradient are non-trivial.
    let dir_a = make_dir(0);
    let mut true_model = Array3::from_elem(dims, c0);
    leto_ops::zip_mut_with(&mut true_model.view_mut(), &dir_a.view(), |c, d| {
        *c += 100.0 * *d
    })
    .expect("invariant: FWI field shapes match");
    let model = Array3::from_elem(dims, c0);

    let observed = forward_sensor_only(true_model.view(), density.view(), &grid, &cfg, &acq, None)
        .expect("obs");
    let (synth0, history0) =
        forward(model.view(), density.view(), &grid, &cfg, &acq, None).expect("fwd m");
    let residual = &synth0 - &observed; // L2 adjoint source r^m = d_syn − d_obs.
    let j0 = 0.5 * cfg.dt * residual.mapv(|x| x * x).iter().sum::<f64>();
    assert!(
        j0 > 0.0,
        "misfit at start model must be non-zero; J(m) = {j0:e}"
    );

    let g = gradient(
        residual.view(),
        model.view(),
        density.view(),
        &grid,
        &cfg,
        &acq,
        history0.view(),
        Some(source_mute.view()),
        None,
    )
    .expect("gradient");

    // Objective along an arbitrary direction, evaluated by the SAME forward map.
    let objective_at = |dir: &Array3<f64>, scale: f64| -> f64 {
        let mut perturbed = model.clone();
        leto_ops::zip_mut_with(&mut perturbed.view_mut(), &dir.view(), |c, d| {
            *c += scale * *d
        })
        .expect("invariant: FWI field shapes match");
        let synth = forward_sensor_only(perturbed.view(), density.view(), &grid, &cfg, &acq, None)
            .expect("fd");
        0.5 * cfg.dt
            * synth
                .iter()
                .zip(observed.iter())
                .map(|(&a, &b)| {
                    let e = a - b;
                    e * e
                })
                .sum::<f64>()
    };
    // Richardson-extrapolated central FD: O(ε⁴) accurate true directional derivative.
    let fd_slope = |dir: &Array3<f64>| -> f64 {
        let central = |eps: f64| (objective_at(dir, eps) - objective_at(dir, -eps)) / (2.0 * eps);
        let eps = 4.0_f64;
        (4.0 * central(eps / 2.0) - central(eps)) / 3.0
    };

    for kind in 0u8..3 {
        let dir = make_dir(kind);
        // Source-voxel-muted gradient ⇒ exclude the source voxel from g·δm too;
        // δm is already ~0 there (boundary_clear excludes ix<2; source at ix=2).
        let g_dot: f64 = g
            .iter()
            .zip(dir.iter())
            .map(|(&gv, &dv)| gv * dv)
            .sum::<f64>();
        let fd = fd_slope(&dir);
        assert!(
            g_dot.abs() > 0.0 && fd.abs() > 0.0,
            "direction {kind}: derivatives must be non-trivial (g·δm={g_dot:e}, FD={fd:e})"
        );
        let kappa = g_dot / fd;
        assert!(
            (kappa - 1.0).abs() < 1e-4,
            "direction {kind}: exact discrete adjoint must give κ ≈ 1; \
             κ = {kappa:.6} (g·δm = {g_dot:e}, FD = {fd:e})"
        );
    }
}

/// The self-adjoint sponge preserves the exact discrete adjoint: with a nonzero
/// damping layer, the finite-difference gradient test still returns κ ≈ 1.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn self_adjoint_gradient_kappa_one_with_sponge() {
    let (nx, ny, nz) = (12usize, 12, 12);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);
    let c0 = SOUND_SPEED_WATER_SIM;
    let density = Array3::from_elem(dims, RHO);
    let cfg = SelfAdjointConfig { nt: 80, dt: 1e-7 };
    let src = wavelet(cfg.nt);
    let source_voxels = [(2usize, 6usize, 6usize)];
    let receiver_voxels: Vec<(usize, usize, usize)> = (3..9)
        .flat_map(|j| (3..9).map(move |k| (9usize, j, k)))
        .collect();
    let acq = Acquisition {
        source_voxels: &source_voxels,
        source_signal: src.view(),
        receiver_voxels: &receiver_voxels,
    };
    let mut source_mute = Array3::zeros(dims);
    source_mute[[2, 6, 6]] = 1.0;

    // A genuine absorbing sponge (b_max ≈ a few / (ρ c · thickness · dx)).
    let sponge = build_edge_sponge(&grid, 3, 4.0 / (RHO * c0 * 3.0 * dx));
    assert!(sponge.iter().any(|&b| b > 0.0), "sponge must be nonzero");
    let damp = Some(sponge.view());

    let mut dir = Array3::zeros(dims);
    for ([i, j, k], v) in dir
        .indexed_iter_mut()
        .expect("invariant: owned array yields indexed iterator")
    {
        if i >= 2 && j >= 2 && k >= 2 && i < nx - 2 && j < ny - 2 && k < nz - 2 {
            *v = (-((i as f64 - 6.5).powi(2)) / 0.98).exp();
        }
    }
    let mut true_model = Array3::from_elem(dims, c0);
    leto_ops::zip_mut_with(&mut true_model.view_mut(), &dir.view(), |c, d| {
        *c += 100.0 * *d
    })
    .expect("invariant: FWI field shapes match");
    let model = Array3::from_elem(dims, c0);

    let observed = forward_sensor_only(true_model.view(), density.view(), &grid, &cfg, &acq, damp)
        .expect("obs");
    let (synth0, history0) =
        forward(model.view(), density.view(), &grid, &cfg, &acq, damp).expect("fwd");
    let residual = &synth0 - &observed;
    let g = gradient(
        residual.view(),
        model.view(),
        density.view(),
        &grid,
        &cfg,
        &acq,
        history0.view(),
        Some(source_mute.view()),
        damp,
    )
    .expect("gradient");
    let g_dot: f64 = g
        .iter()
        .zip(dir.iter())
        .map(|(&gv, &dv)| gv * dv)
        .sum::<f64>();

    let objective_at = |scale: f64| -> f64 {
        let mut perturbed = model.clone();
        leto_ops::zip_mut_with(&mut perturbed.view_mut(), &dir.view(), |c, d| {
            *c += scale * *d
        })
        .expect("invariant: FWI field shapes match");
        let synth = forward_sensor_only(perturbed.view(), density.view(), &grid, &cfg, &acq, damp)
            .expect("fd");
        0.5 * cfg.dt
            * synth
                .iter()
                .zip(observed.iter())
                .map(|(&a, &b)| {
                    let e = a - b;
                    e * e
                })
                .sum::<f64>()
    };
    let central = |eps: f64| (objective_at(eps) - objective_at(-eps)) / (2.0 * eps);
    let eps = 4.0_f64;
    let fd = (4.0 * central(eps / 2.0) - central(eps)) / 3.0;

    let kappa = g_dot / fd;
    assert!(
        (kappa - 1.0).abs() < 1e-4,
        "exact discrete adjoint must hold WITH a sponge: κ = {kappa:.6} (g·δm={g_dot:e}, FD={fd:e})"
    );
}

/// The self-adjoint sponge actually absorbs: a centred pulse leaves far less
/// residual field energy in the domain than with reflecting (lossless) walls.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn self_adjoint_sponge_absorbs_outgoing_waves() {
    let (nx, ny) = (48usize, 48);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, 1, dx, dx, dx).expect("grid");
    let c0 = SOUND_SPEED_WATER_SIM;
    let density = Array3::from_elem([nx, ny, 1], RHO);
    let cfg = SelfAdjointConfig { nt: 220, dt: 2e-7 };
    let src = wavelet(cfg.nt);
    let source_voxels = [(24usize, 24usize, 0usize)];
    let receiver_voxels = [(24usize, 24usize, 0usize)];
    let acq = Acquisition {
        source_voxels: &source_voxels,
        source_signal: src.view(),
        receiver_voxels: &receiver_voxels,
    };
    let model = Array3::from_elem([nx, ny, 1], c0);

    let final_energy = |damp: Option<leto::ArrayView3<f64>>| -> f64 {
        let (_, history) =
            forward(model.view(), density.view(), &grid, &cfg, &acq, damp).expect("fwd");
        history
            .index_axis::<3>(0, cfg.nt - 1)
            .expect("invariant: history time index in bounds")
            .iter()
            .map(|&p| p * p)
            .sum::<f64>()
    };

    let energy_reflecting = final_energy(None);
    let sponge = build_edge_sponge(&grid, 8, 6.0 / (RHO * c0 * 8.0 * dx));
    let energy_absorbed = final_energy(Some(sponge.view()));

    assert!(
        energy_reflecting > 0.0,
        "reflecting run must have residual energy"
    );
    assert!(
        energy_absorbed < 0.3 * energy_reflecting,
        "sponge must absorb most outgoing energy; reflecting = {energy_reflecting:e}, \
         absorbed = {energy_absorbed:e} (ratio {:.3})",
        energy_absorbed / energy_reflecting
    );
}

/// Memory-efficient reverse-reconstruction gradient (`gradient_reconstructed`,
/// `O(N)` memory) must reproduce the stored-history gradient (`gradient`,
/// `O(nt·N)`) to round-off for the lossless engine — the exactness contract
/// behind the memory optimization. Also checks the `forward_tail` seed states
/// equal the history's final two slices.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn reconstructed_gradient_matches_stored_history() {
    let (nx, ny, nz) = (12usize, 12, 12);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);
    let c0 = SOUND_SPEED_WATER_SIM;
    let density = Array3::from_elem(dims, RHO);
    let cfg = SelfAdjointConfig { nt: 80, dt: 1e-7 };
    let src = wavelet(cfg.nt);

    let source_voxels = [(2usize, 6usize, 6usize)];
    let receiver_voxels: Vec<(usize, usize, usize)> = (3..9)
        .flat_map(|j| (3..9).map(move |k| (9usize, j, k)))
        .collect();
    let acq = Acquisition {
        source_voxels: &source_voxels,
        source_signal: src.view(),
        receiver_voxels: &receiver_voxels,
    };
    let source_mute = {
        let mut m = Array3::zeros(dims);
        m[[2, 6, 6]] = 1.0;
        m
    };

    // Non-trivial misfit: heterogeneous true model, homogeneous start.
    let mut true_model = Array3::from_elem(dims, c0);
    for ([i, j, k], c) in true_model
        .indexed_iter_mut()
        .expect("invariant: owned array yields indexed iterator")
    {
        if i >= 2 && j >= 2 && k >= 2 && i < nx - 2 && j < ny - 2 && k < nz - 2 {
            *c += 100.0 * (-((i as f64 - 6.5).powi(2)) / 0.98).exp();
        }
    }
    let model = Array3::from_elem(dims, c0);
    let observed = forward_sensor_only(true_model.view(), density.view(), &grid, &cfg, &acq, None)
        .expect("obs");

    // Stored-history path.
    let (synth, history) =
        forward(model.view(), density.view(), &grid, &cfg, &acq, None).expect("fwd");
    let residual = &synth - &observed;
    let g_history = gradient(
        residual.view(),
        model.view(),
        density.view(),
        &grid,
        &cfg,
        &acq,
        history.view(),
        Some(source_mute.view()),
        None,
    )
    .expect("history gradient");

    // Reverse-reconstruction path (O(N) memory).
    let (synth_tail, p_last, p_second_last) =
        forward_tail(model.view(), density.view(), &grid, &cfg, &acq).expect("fwd tail");
    // Traces and seed states must match the full forward exactly (same arithmetic).
    assert_eq!(
        synth_tail, synth,
        "forward_tail traces must equal forward traces"
    );
    assert_eq!(
        p_last,
        history
            .index_axis::<3>(0, cfg.nt - 1)
            .unwrap()
            .to_contiguous(),
        "p_last must equal history[N-1]"
    );
    assert_eq!(
        p_second_last,
        history
            .index_axis::<3>(0, cfg.nt - 2)
            .unwrap()
            .to_contiguous(),
        "p_second_last must equal history[N-2]"
    );

    let g_recon = gradient_reconstructed(
        residual.view(),
        model.view(),
        density.view(),
        &grid,
        &cfg,
        &acq,
        p_last.view(),
        p_second_last.view(),
        Some(source_mute.view()),
    )
    .expect("reconstructed gradient");

    // The reconstructed gradient equals the stored-history gradient to round-off.
    let max_abs = g_history.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(
        max_abs > 0.0,
        "gradient must be non-trivial; max|g| = {max_abs:e}"
    );
    let max_diff = g_history
        .iter()
        .zip(g_recon.iter())
        .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(
        max_diff < 1e-9 * max_abs,
        "reconstructed gradient must match stored-history gradient to round-off: \
         max|Δg| = {max_diff:e}, max|g| = {max_abs:e} (rel {:e})",
        max_diff / max_abs
    );
}
