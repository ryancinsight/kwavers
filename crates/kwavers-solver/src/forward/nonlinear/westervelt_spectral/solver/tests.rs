use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::traits::AcousticWaveModel;
use kwavers_source::NullSource;
use ndarray::Array4;

#[test]
fn pressure_buffer_permutation_returns_disjoint_step_roles() {
    let shape = (1, 1, 1);
    let cases = [
        ([0, 1, 2], (0.0, 1.0, 2.0)),
        ([0, 2, 1], (0.0, 2.0, 1.0)),
        ([1, 0, 2], (1.0, 0.0, 2.0)),
        ([1, 2, 0], (1.0, 2.0, 0.0)),
        ([2, 0, 1], (2.0, 0.0, 1.0)),
        ([2, 1, 0], (2.0, 1.0, 0.0)),
    ];

    for (indices, (expected_next, expected_current, expected_previous)) in cases {
        let mut buffers = [
            Array3::from_elem(shape, 0.0),
            Array3::from_elem(shape, 1.0),
            Array3::from_elem(shape, 2.0),
        ];

        {
            let (next, current, previous) =
                WesterveltWave::pressure_buffers_for_step(&mut buffers, indices);
            assert_eq!(next[[0, 0, 0]], expected_next);
            assert_eq!(current[[0, 0, 0]], expected_current);
            assert_eq!(previous[[0, 0, 0]], expected_previous);
            next[[0, 0, 0]] = 10.0;
        }

        assert_eq!(buffers[indices[0]][[0, 0, 0]], 10.0);
        assert_eq!(buffers[indices[1]][[0, 0, 0]], expected_current);
        assert_eq!(buffers[indices[2]][[0, 0, 0]], expected_previous);
    }
}

/// **Theorem (∇²[const] = 0 for FD Laplacian)**:
///
/// The 2nd-order centered stencil:
/// ```text
/// ∂²f/∂x² ≈ (f[i+1] - 2f[i] + f[i-1]) / Δx²
/// ```
/// maps a constant field to zero exactly (no truncation error).
#[test]
fn fd_laplacian_of_constant_is_exactly_zero() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let field = ndarray::Array3::from_elem((8, 8, 8), 42.0_f64);
    let lap = compute_laplacian_fd(&field, &grid);
    // Interior points must be exactly 0.0; boundary stencil is not computed (left 0)
    for i in 1..7usize {
        for j in 1..7usize {
            for k in 1..7usize {
                assert_eq!(
                    lap[[i, j, k]],
                    0.0,
                    "FD Laplacian of constant must be exactly 0 at interior [{i},{j},{k}]"
                );
            }
        }
    }
}

/// **Theorem (FD Laplacian of quadratic = analytical)**:
///
/// For `f(x,y,z) = x²` (discrete: `f[i,j,k] = (i·Δx)²`),
/// the 2nd-order FD stencil gives:
/// ```text
/// ∂²f/∂x² = ((i+1)Δx)² - 2(iΔx)² + ((i-1)Δx)² / Δx² = 2
/// ```
/// exactly, with no Δx² truncation error (the polynomial is of degree 2,
/// within the stencil's null space). ∂²f/∂y² = ∂²f/∂z² = 0, so ∇²f = 2.
#[test]
fn fd_laplacian_of_x_squared_equals_two_interior() {
    let n = 8usize;
    let dx = 1.0e-3_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let mut field = ndarray::Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                field[[i, j, k]] = (i as f64 * dx).powi(2);
            }
        }
    }
    let lap = compute_laplacian_fd(&field, &grid);
    // Interior: ∇²(x²) = 2 exactly (FD stencil is exact for degree-2 polynomials)
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for k in 1..n - 1 {
                assert!(
                    (lap[[i, j, k]] - 2.0).abs() < 1e-10,
                    "FD Laplacian of x² at [{i},{j},{k}]: got {} expected 2.0",
                    lap[[i, j, k]]
                );
            }
        }
    }
}

/// **Theorem (WesterveltWave CFL stability criterion)**:
///
/// `check_stability` implements:
/// ```text
/// CFL = c₀ · Δt / Δx < 0.5
/// ```
/// The spectral leapfrog update `p^{n+1} = 2p^n - p^{n-1} + (c₀Δt)² ∇²p^n`
/// is Von Neumann stable iff `c₀·Δt·|k|_max ≤ 2` (Trefethen 2000 §10.6).
/// For a uniform grid `|k|_max = π/Δx`, this yields `c₀·Δt·π/Δx ≤ 2`,
/// i.e., `c₀·Δt/Δx ≤ 2/π ≈ 0.637`. The implementation conservatively
/// uses 0.5 as the stability threshold.
///
/// For c₀=1500 m/s, Δx=1 mm:
/// - dt = 3.0e-7 → CFL = 1500·3e-7/1e-3 = 0.45 < 0.5 → `true`
/// - dt = 4.0e-7 → CFL = 1500·4e-7/1e-3 = 0.60 ≥ 0.5 → `false`
#[test]
fn check_stability_correctly_classifies_subcritical_and_supercritical_dt() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let solver = WesterveltWave::new(&grid);

    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1.0e-3_f64;

    // Subcritical: CFL = c₀·dt/dx = 1500·3e-7/1e-3 = 0.45 < 0.5
    let dt_stable = 3.0e-7_f64;
    assert!(
        solver.check_stability(dt_stable, &grid, &medium),
        "dt=3e-7 (CFL={:.3}) must be classified as stable (< 0.5)",
        c0 * dt_stable / dx
    );

    // Supercritical: CFL = 1500·4e-7/1e-3 = 0.60 ≥ 0.5
    let dt_unstable = 4.0e-7_f64;
    assert!(
        !solver.check_stability(dt_unstable, &grid, &medium),
        "dt=4e-7 (CFL={:.3}) must be classified as unstable (≥ 0.5)",
        c0 * dt_unstable / dx
    );

    // Boundary: CFL = c₀·dt_boundary/dx = 0.5 exactly → not < 0.5 → false
    let dt_boundary = 0.5 * dx / c0; // = 3.333e-7 → CFL = 0.5 exactly
    assert!(
        !solver.check_stability(dt_boundary, &grid, &medium),
        "dt at CFL=0.5 exactly must be classified as unstable (boundary is exclusive)"
    );
}

#[test]
fn update_wave_preserves_pressure_buffer_storage_for_zero_state() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let source = NullSource::new();
    let mut solver = WesterveltWave::new(&grid);
    let mut fields = Array4::<f64>::zeros((UnifiedFieldType::COUNT, grid.nx, grid.ny, grid.nz));
    let prev_pressure = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut before = solver
        .pressure_buffers
        .each_ref()
        .map(|buffer| buffer.as_ptr() as usize);
    before.sort_unstable();
    let nonlinear_before = solver.nonlinear_scratch.as_ptr();
    let damping_before = solver.damping_scratch.as_ptr();
    let source_mask_before = solver.source_mask_scratch.as_ptr();

    solver
        .update_wave(
            &mut fields,
            &prev_pressure,
            &source,
            &grid,
            &medium,
            1.0e-7,
            0.0,
        )
        .unwrap();

    let mut after = solver
        .pressure_buffers
        .each_ref()
        .map(|buffer| buffer.as_ptr() as usize);
    after.sort_unstable();
    let pressure = fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());

    assert_eq!(after, before);
    assert_eq!(solver.nonlinear_scratch.as_ptr(), nonlinear_before);
    assert_eq!(solver.damping_scratch.as_ptr(), damping_before);
    assert_eq!(solver.source_mask_scratch.as_ptr(), source_mask_before);
    assert!(pressure.iter().all(|value| *value == 0.0));
    assert_eq!(solver.current_step, 1);
}

/// **Regression (Stokes damping stability)**: a *lossy* linear Westervelt run on
/// a band-limited Gaussian must stay finite and bounded.
///
/// Before the coefficient fix the viscoelastic term carried a spurious `c²`
/// (≈2.25e6), so the explicit damping amplified high-k modes (~89× at Nyquist)
/// and lossy runs blew up to Inf/NaN (observed 1.8e26 → NaN). With the correct
/// `δ∇²(∂p/∂t)` coefficient the damping is dissipative and the run is stable.
#[test]
fn lossy_westervelt_stays_finite_and_bounded() {
    let n = 16usize;
    let dx = 0.5e-3;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let c = SOUND_SPEED_WATER_SIM;
    let dt = 0.245 * dx / c; // CFL-stable lossless leapfrog

    // Band-limited Gaussian initial pressure (σ = 3Δx).
    let sigma = 3.0 * dx;
    let two_sigma2 = 2.0 * sigma * sigma;
    let c0 = n as f64 / 2.0;
    let mut ic = Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as f64 - c0) * dx).powi(2)
                    + ((j as f64 - c0) * dx).powi(2)
                    + ((k as f64 - c0) * dx).powi(2);
                ic[[i, j, k]] = 1.0e6 * (-r2 / two_sigma2).exp();
            }
        }
    }

    let mut solver = WesterveltWave::new(&grid);
    solver.set_nonlinearity_scaling(0.0); // linear; damping_scaling stays 1.0 → LOSSY
    let mut fields = Array4::<f64>::zeros((1, n, n, n));
    fields.index_axis_mut(ndarray::Axis(0), 0).assign(&ic);
    let prev = Array3::<f64>::zeros((n, n, n));
    let source = NullSource::new();

    for step in 0..80 {
        solver
            .update_wave(&mut fields, &prev, &source, &grid, &medium, dt, step as f64 * dt)
            .unwrap();
    }

    let p = fields.index_axis(ndarray::Axis(0), 0);
    assert!(
        p.iter().all(|v| v.is_finite()),
        "lossy Westervelt must stay finite (was Inf/NaN before the c² coefficient fix)"
    );
    let max_abs = p.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1.0e6,
        "absorption must not amplify: max|p|={max_abs:.3e} must stay ≤ the initial 1e6 Pa"
    );
}

/// **Theorem (Stokes absorption coefficient)**: for a sinusoidal particle-velocity
/// field `∂ₜp = cos(k₀x)`, the viscoelastic term equals `δ·∇²(∂ₜp) = −δ·k²·∂ₜp`.
///
/// This is the Stokes damping coefficient that drives the leapfrog envelope decay
/// rate `Γ = δ·k²/2` and the classical absorption `α(ω) = δω²/2c³` (`Γ = α·c`). It
/// is verified per cell — exactly, at machine precision — rather than via a
/// multi-step decay run: the FD damping leaves the domain boundaries undamped,
/// and the periodic spectral Laplacian feeds that undamped boundary energy back
/// into the interior, which biases any long-run envelope measurement. The per-cell
/// coefficient (checked at a point interior in all three axes) is the
/// boundary-independent, machine-exact validation of the absorption rate.
#[test]
fn lossy_westervelt_stokes_coefficient_matches_analytical() {
    use kwavers_medium::ViscousProperties;

    let (nx, ny, nz) = (32usize, 4usize, 4usize);
    let dx = 2.0e-5;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let c = SOUND_SPEED_WATER_SIM;
    let dt = 0.245 * dx / c;

    // Diffusivity of sound δ = (4η_s/3 + η_b)/ρ from the actual medium.
    let eta_s = medium.shear_viscosity(0.0, 0.0, 0.0, &grid);
    let eta_b = medium.bulk_viscosity(0.0, 0.0, 0.0, &grid);
    let rho = kwavers_medium::density_at(&medium, 0.0, 0.0, 0.0, &grid);
    let delta = (4.0 * eta_s / 3.0 + eta_b) / rho;

    // Resolved mode m=2 (k·Δx = π/8). FD Laplacian wavenumber vs the continuous k₀.
    let m = 2.0_f64;
    let k0 = std::f64::consts::TAU * m / (nx as f64 * dx);
    let k_fd2 = (2.0 - 2.0 * (k0 * dx).cos()) / (dx * dx);

    // Set p, p_prev so the velocity ∂ₜp = (p − p_prev)/dt = cos(k₀x): a sinusoidal
    // velocity field with zero spatial variation in y, z.
    let mut p = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz));
    for i in 0..nx {
        let cos_kx = (k0 * i as f64 * dx).cos();
        for j in 0..ny {
            for k in 0..nz {
                p[[i, j, k]] = cos_kx; // p_prev = 0 ⟹ ∂ₜp = cos(k₀x)/dt
            }
        }
    }
    let mut damp = Array3::<f64>::zeros((nx, ny, nz));
    super::super::nonlinear::compute_viscoelastic_term_into(
        &mut damp, &p, &p_prev, &medium, &grid, dt,
    );
    p_prev.fill(0.0); // (already zero; keeps the ∂ₜp = p/dt relation explicit)

    // At a point interior in all three axes the term must equal δ·∇²(∂ₜp) exactly.
    let (ai, aj, ak) = (nx / 2, ny / 2, nz / 2);
    let dpdt = p[[ai, aj, ak]] / dt; // (p − p_prev)/dt, p_prev = 0
    let expected = delta * (-k_fd2) * dpdt; // δ·∇²(∂ₜp) = −δ·k_fd²·∂ₜp
    let rel_err = (damp[[ai, aj, ak]] - expected).abs() / expected.abs();
    eprintln!(
        "Stokes coeff: damp={:.6e}  δ·(−k_fd²)·∂ₜp={expected:.6e}  rel_err={rel_err:.2e}",
        damp[[ai, aj, ak]]
    );
    assert!(
        rel_err < 1e-6,
        "viscoelastic term must equal δ·∇²(∂ₜp) = −δ·k²·∂ₜp at machine precision (rel_err={rel_err:.2e})"
    );

    // The FD damping wavenumber matches the continuous k₀ for this resolved mode,
    // so the discrete envelope-decay rate δ·k_fd²/2 reproduces the physical Stokes
    // rate Γ = δ·k₀²/2 = α(ω)·c, α(ω) = δω²/2c³ (ω = c·k₀).
    let k_fd_rel = (k_fd2 - k0 * k0).abs() / (k0 * k0);
    assert!(
        k_fd_rel < 0.02,
        "FD damping wavenumber must match k₀² within 2% so Γ=δk²/2 matches the physical Stokes rate (rel={k_fd_rel:.3})"
    );
}

/// **Theorem (Stokes absorption rate, end-to-end)**: a single resolved standing
/// mode decays through the real leapfrog at the analytical rate `Γ = δ·k²/2`
/// (i.e. `α(f)=δω²/2c³`, `Γ = α·c`).
///
/// Now that the FD damping is periodic (no undamped boundary planes feeding the
/// interior), the multi-step envelope decay is a clean exponential and can be
/// compared end-to-end against `δ·k_fd²/2` (k_fd² = (2−2cos(k₀Δx))/Δx² ≈ k₀²).
#[test]
fn lossy_westervelt_absorption_rate_matches_stokes_end_to_end() {
    use kwavers_medium::ViscousProperties;

    let (nx, ny, nz) = (32usize, 4usize, 4usize);
    let dx = 2.0e-5;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let c = SOUND_SPEED_WATER_SIM;
    let dt = 0.245 * dx / c;

    let eta_s = medium.shear_viscosity(0.0, 0.0, 0.0, &grid);
    let eta_b = medium.bulk_viscosity(0.0, 0.0, 0.0, &grid);
    let rho = kwavers_medium::density_at(&medium, 0.0, 0.0, 0.0, &grid);
    let delta = (4.0 * eta_s / 3.0 + eta_b) / rho;

    let m = 2.0_f64;
    let k0 = std::f64::consts::TAU * m / (nx as f64 * dx);
    let k_fd2 = (2.0 - 2.0 * (k0 * dx).cos()) / (dx * dx);
    let gamma_analytic = delta * k_fd2 / 2.0;

    let (k_squared, _, _, _) = super::super::spectral::initialize_kspace_grids(&grid);
    let mut p = Array3::<f64>::zeros((nx, ny, nz));
    for i in 0..nx {
        let v = (k0 * i as f64 * dx).cos();
        for j in 0..ny {
            for k in 0..nz {
                p[[i, j, k]] = v;
            }
        }
    }
    let mut p_prev = p.clone(); // zero initial velocity
    let mut damp = Array3::<f64>::zeros((nx, ny, nz));

    let n_steps = 40_000usize;
    let window = (std::f64::consts::TAU / (c * k0) / dt).ceil() as usize + 2;
    let (ai, aj, ak) = (nx / 2, ny / 2, nz / 2);
    let amp = |f: &Array3<f64>| f[[ai, aj, ak]].abs();
    let mut max_early: f64 = amp(&p);
    let mut max_late = 0.0_f64;
    let dt2 = dt * dt;
    for step in 0..n_steps {
        let lap = super::super::spectral::compute_laplacian_spectral(&p, &k_squared);
        super::super::nonlinear::compute_viscoelastic_term_into(
            &mut damp, &p, &p_prev, &medium, &grid, dt,
        );
        let mut p_next = Array3::<f64>::zeros((nx, ny, nz));
        ndarray::Zip::from(&mut p_next)
            .and(&p)
            .and(&p_prev)
            .and(&lap)
            .and(&damp)
            .for_each(|pn, &pc, &pp, &l, &d| *pn = 2.0 * pc - pp + dt2 * (c * c * l + d));
        p_prev = std::mem::replace(&mut p, p_next);
        if step < window {
            max_early = max_early.max(amp(&p));
        }
        if step >= n_steps - window {
            max_late = max_late.max(amp(&p));
        }
    }

    assert!(p.iter().all(|v| v.is_finite()), "lossy single-mode run must stay finite");
    let t_span = (n_steps - window) as f64 * dt;
    let gamma_measured = -(max_late / max_early).ln() / t_span;
    let rel_err = (gamma_measured - gamma_analytic).abs() / gamma_analytic;
    eprintln!(
        "Stokes e2e: Γ_meas={gamma_measured:.4e}/s Γ_analytic(δk_fd²/2)={gamma_analytic:.4e}/s rel_err={rel_err:.3}"
    );
    assert!(
        rel_err < 0.10,
        "end-to-end Stokes decay {gamma_measured:.4e}/s must match δk²/2={gamma_analytic:.4e}/s within 10% (rel_err={rel_err:.3})"
    );
}
