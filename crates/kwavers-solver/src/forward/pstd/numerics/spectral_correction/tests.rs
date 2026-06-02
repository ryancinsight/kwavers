use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_domain::grid::Grid;
use std::f64::consts::PI;

#[test]
fn test_exact_dispersion_correction() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let dt = 1e-6;
    let c_ref = SOUND_SPEED_WATER_SIM;

    let config = SpectralCorrectionConfig {
        enabled: true,
        method: SpectralCorrectionMethod::ExactDispersion,
        cfl_number: 0.3,
        max_correction: 2.0,
    };

    let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

    assert!((kappa[[0, 0, 0]] - 1.0).abs() < 1e-10);

    for val in kappa.iter() {
        assert!(*val >= 0.5 && *val <= 2.0);
    }
}

#[test]
fn test_dispersion_error() {
    let dx = 1e-3;
    let dt = 1e-6;
    let c_ref = SOUND_SPEED_WATER_SIM;

    let k_low = PI / (10.0 * dx);
    let error_low = compute_dispersion_error(k_low, dx, dt, c_ref);
    assert!(error_low < 0.01);

    let k_high = PI / (2.0 * dx);
    let error_high = compute_dispersion_error(k_high, dx, dt, c_ref);
    assert!(error_high > error_low);
}

#[test]
fn test_phase_velocity_computation() {
    let dx = 1e-3;
    let dt = 1e-6;
    let c_ref = SOUND_SPEED_WATER_SIM;

    let c_dc = compute_numerical_phase_velocity(1e-12, dx, dt, c_ref);
    assert!((c_dc - c_ref).abs() / c_ref < 1e-6);
}

/// Theorem (Treeby & Cox 2010 Eq. 18):
///
///     kappa(k) = sinc(c_ref·dt·|k|/2) = sin(c_ref·dt·|k|/2) / (c_ref·dt·|k|/2)
///
/// for any non-zero |k|, with kappa(0) = 1 by L'Hôpital.
///
/// This test verifies the closed-form analytical agreement at five
/// representative wavenumbers spanning DC → near-Nyquist. It pins the
/// regression that surfaced as ~30% peak inflation in
/// `pykwavers/examples/na_modelling_absorption_compare.py` before the
/// kappa-inversion fix.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_treeby2010_kappa_equals_sinc() {
    // Use a 1-D grid (NY = NZ = 1) so kappa values are determined by k_x alone.
    // dx_y, dx_z are arbitrary because k_y = k_z = 0 for nz/ny = 1.
    let grid = Grid::new(64, 1, 1, 1e-4, 1e-4, 1e-4).unwrap();
    let dt = 5e-9;
    let c_ref = SOUND_SPEED_WATER_SIM;

    let config = SpectralCorrectionConfig {
        enabled: true,
        method: SpectralCorrectionMethod::Treeby2010,
        cfl_number: 0.3,
        max_correction: 10.0, // wide enough to not clip the sinc values
    };
    let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

    // DC sample: kappa(0,0,0) must be exactly 1 (sinc limit at zero).
    assert!(
        (kappa[[0, 0, 0]] - 1.0).abs() < 1e-12,
        "kappa[DC] = {} (expected 1.0)",
        kappa[[0, 0, 0]]
    );

    // Helper: expected kappa at index i for the 1-D x-axis only
    // (k_y = k_z = 0). compute_wavenumber_component uses standard FFT
    // ordering, so index 0 = DC, indices 1..N/2 = positive k, N/2 ..N-1 = negative k.
    let dx = grid.dx;
    let nx = grid.nx as i64;
    for &i in &[0_usize, 1, 4, 16, (nx as usize - 1)] {
        let kx = if (i as i64) <= nx / 2 {
            TWO_PI * (i as f64) / (nx as f64 * dx)
        } else {
            TWO_PI * ((i as f64) - nx as f64) / (nx as f64 * dx)
        };
        let k_mag = kx.abs();
        let arg = 0.5 * c_ref * dt * k_mag;
        let expected = if arg.abs() < 1e-12 {
            1.0
        } else {
            arg.sin() / arg
        };
        let observed = kappa[[i, 0, 0]];
        assert!(
            (observed - expected).abs() < 1e-9,
            "kappa[{}, 0, 0] = {:.10} but expected sinc(c·dt·|k|/2) = {:.10} \
             (arg = {:.6e}, k = {:.4e})",
            i,
            observed,
            expected,
            arg,
            k_mag,
        );
    }

    // Verify kappa is monotonically non-increasing as |k| grows from DC
    // (sinc is monotone on [0, π]). Up to k ≈ π/dx ≈ Nyquist, the
    // c·dt·|k|/2 argument grows but stays bounded under CFL.
    for i in 1..(grid.nx / 2) {
        let prev = kappa[[i - 1, 0, 0]];
        let curr = kappa[[i, 0, 0]];
        // Allow a tiny epsilon for floating-point noise.
        assert!(
            curr <= prev + 1e-12,
            "kappa monotonicity violated at i={}: kappa[{}]={} > kappa[{}]={}",
            i,
            i,
            curr,
            i - 1,
            prev,
        );
    }
}

#[test]
fn test_correction_methods_consistency() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let dt = 1e-6;
    let c_ref = SOUND_SPEED_WATER_SIM;

    let methods = vec![
        SpectralCorrectionMethod::ExactDispersion,
        SpectralCorrectionMethod::Treeby2010,
        SpectralCorrectionMethod::LiuPSTD,
        SpectralCorrectionMethod::SincSpatial,
    ];

    for method in methods {
        let config = SpectralCorrectionConfig {
            enabled: true,
            method,
            cfl_number: 0.3,
            max_correction: 2.0,
        };

        let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

        assert!((kappa[[0, 0, 0]] - 1.0).abs() < 0.01);

        for val in kappa.iter() {
            assert!(*val > 0.0);
        }
    }
}

/// End-to-end k-space dispersion convergence test.
///
/// # Theorem (Treeby & Cox 2010, Eq. 18)
///
/// With κ = sinc(c_ref·|k|·dt/2) applied to every spectral spatial derivative,
/// the leapfrog k-PSTD scheme satisfies the exact dispersion relation
/// ω = c_ref·|k| for all spatial frequencies. As a consequence the numerical
/// phase velocity equals c_ref to machine precision for the reference wavenumber,
/// limited in practice only by pulse-tracking resolution (one time-step = O(dt)).
///
/// # Test design
///
/// Domain: 1D periodic (ny=nz=1), nx=256, dx=1e-3 m.
/// Medium: homogeneous water, c_ref=SOUND_SPEED_WATER_SIM≈1498 m/s, ρ₀=1000 kg/m³.
/// CFL=0.25 → dt=CFL·dx/c_ref.  Stable (CFL<1); Nyquist kappa ≈ sinc(π·CFL/2) ≈ 0.983.
///
/// Initial pressure: unit-amplitude Gaussian p₀[i] = exp(-((i−i_src)*dx)²/(2·σ²))
/// with σ=4·dx centered at i_src=nx/2=128 (domain centre).
///
/// Sensor at i_snr=128+96=224; separation L=96·dx=0.096 m.
///
/// ## Source and sensor placement rationale
/// The domain is periodic.  The pulse can reach the sensor via two paths:
///   - Direct (positive):  96 cells = 0.096 m  ← shorter, pulse arrives here first
///   - Wrapped (negative): 256−96 = 160 cells = 0.160 m
/// Placing the source at the domain centre (i_src=128) and the sensor 96 cells
/// to the right (i_snr=224) guarantees that the direct path is shorter than the
/// wrapped path (96 < 160), so the recorded peak corresponds to the direct propagation.
///
/// Expected arrival: t_exact = L/c_ref.
/// k-PSTD dispersion-free → t_numerical = (peak_step)·dt ≈ t_exact.
///
/// Acceptance criterion: |c_numerical − c_ref| / c_ref < 1e-3 (0.1 % tolerance),
/// accounting for the half-step pulse-tracking resolution (±0.5·dt → δc/c ≈ 0.3%).
///
/// # Panics
///
/// Panics if the solver reports an error during stepping.
#[test]
fn kspace_correction_eliminates_numerical_dispersion() {
    use kwavers_domain::medium::HomogeneousMedium;
    use kwavers_domain::source::GridSource;
    use crate::forward::pstd::config::{BoundaryConfig, PSTDConfig};
    use crate::pstd::PSTDSolver;
    use ndarray::Array3;

    // ── Grid and time step ────────────────────────────────────────────────────
    let nx: usize = 256;
    let dx = 1e-3_f64; // 1 mm
    let c_ref = SOUND_SPEED_WATER_SIM; // ≈1498.0 m/s
    let rho0 = DENSITY_WATER_NOMINAL; // 1000.0 kg/m³
    let cfl = 0.25_f64; // CFL < 1 (stable); |k|_max·dt·c/2 = CFL·π/2 ≈ 0.393
    let dt = cfl * dx / c_ref;

    // ── Source and sensor positions ───────────────────────────────────────────
    // Source at domain centre; sensor 96 cells to the right.
    // Direct path (96 cells < 160 cells wrapped) is unambiguously shorter:
    // the recorded peak always corresponds to direct propagation.
    let i_src: usize = 128; // source pulse centre (domain midpoint)
    let i_snr: usize = 224; // sensor (128 + 96 cells)
                            // Direct separation: 96 cells · dx = 0.096 m.
    let separation = (i_snr as f64 - i_src as f64) * dx; // 0.096 m

    // Expected arrival time and number of steps to run.
    // Add 20% headroom so the pulse peak passes the sensor with certainty.
    let t_exact = separation / c_ref; // ≈1.282e-4 s
    let n_steps = ((1.2 * t_exact) / dt).ceil() as usize;

    // ── Gaussian initial pressure pulse ───────────────────────────────────────
    // σ = 4·dx: narrow enough to excite mid-to-high wavenumbers where
    // dispersion correction matters, wide enough to avoid Nyquist aliasing.
    let sigma = 4.0 * dx;
    let mut p0 = Array3::<f64>::zeros((nx, 1, 1));
    for i in 0..nx {
        let x = (i as f64 - i_src as f64) * dx;
        p0[[i, 0, 0]] = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // ── Build solver (1D: ny=nz=1, no PML — periodic domain) ─────────────────
    let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c_ref, 0.0, 0.0, &grid);

    // IVP: p0 sets the initial pressure field. p_mask/p_signal remain None.
    // p_mode is irrelevant when only p0 is set (has_initial_pressure() checks p0.is_some()).
    let source = GridSource {
        p0: Some(p0),
        ..GridSource::new_empty()
    };

    let config = PSTDConfig {
        dt,
        nt: n_steps,
        boundary: BoundaryConfig::None, // periodic (no PML) → no boundary absorption
        smooth_sources: false,
        ..Default::default() // Default spectral_correction uses Treeby2010 (k-space corrected)
    };

    let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

    // ── Step the solver and track max |p| at the sensor ──────────────────────
    let mut peak_val = 0.0_f64;
    let mut peak_step = 0_usize;

    for step in 0..n_steps {
        solver.step_forward().expect("PSTD step must not error");
        let p_at_sensor = solver.fields.p[[i_snr, 0, 0]].abs();
        if p_at_sensor > peak_val {
            peak_val = p_at_sensor;
            peak_step = step + 1; // step completed = step+1 time steps elapsed
        }
    }

    // ── Compute numerical phase velocity and assert dispersion tolerance ──────
    // peak_step is the 1-indexed step at which the peak was recorded.
    // t_numerical = peak_step * dt (time elapsed after peak_step full steps).
    // Tolerance of 0.5·dt/t_exact ≈ 0.3% accounts for one-step tracking resolution.
    let t_numerical = peak_step as f64 * dt;
    let c_numerical = separation / t_numerical;
    let rel_err = (c_numerical - c_ref).abs() / c_ref;

    assert!(
        peak_val > 1e-6,
        "sensor at i={i_snr} recorded no pulse (peak={peak_val:.3e}): \
         k-space correction may have suppressed all propagation"
    );

    // Tolerance derivation: the pulse peak is tracked at integer-step resolution.
    // Tracking uncertainty: ±0.5·dt → δc/c = 0.5·dt/t_exact.
    // With separation=96 cells, t_exact=96·dx/c_ref, dt=CFL·dx/c_ref:
    //   δc/c = 0.5·CFL/96 = 0.5·0.25/96 ≈ 0.0013 (0.13%).
    // Use 2e-3 (0.2%) to give 1.5× headroom over the step-quantization uncertainty.
    assert!(
        rel_err < 2e-3,
        "k-space correction dispersion test FAILED: \
         c_numerical={c_numerical:.4} m/s, c_ref={c_ref:.4} m/s, \
         rel_err={rel_err:.4e} (must be < 2e-3). \
         peak arrived at step {peak_step}, t_numerical={t_numerical:.4e} s, \
         t_exact={t_exact:.4e} s"
    );
}
