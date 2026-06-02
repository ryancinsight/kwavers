use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use kwavers_domain::source::GridSource;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::forward::pstd::config::PSTDConfig;
use crate::pstd::PSTDSolver;

#[test]
fn test_lossless_mode_no_pressure_correction() {
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );

    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Lossless,
        dt: 1e-7,
        ..PSTDConfig::default()
    };

    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    // Populate dpx/dpy, div_u (rho_total surrogate), and p with arbitrary
    // non-zero state. Lossless absorption must leave p untouched.
    // (dpz eliminated in Opt-12: dpx is reused for all three gradient axes)
    solver.dpx.fill(2.0);
    solver.dpy.fill(3.0);
    solver.div_u.fill(7.0);
    solver.fields.p.fill(11.0);
    let p_before = solver.fields.p.clone();

    solver.apply_absorption_to_pressure().unwrap();

    for (a, b) in solver.fields.p.iter().zip(p_before.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "p changed in lossless mode: {} -> {}",
            b,
            a
        );
    }
}

/// Quantitative pressure-correction formula test.
///
/// **Theorem (Treeby & Cox 2010 Eq. 19–21):** the pressure-side absorption
/// correction satisfies
/// ```text
///   Δp(x) = c₀² · ( τ · L1(x)  −  η · L2(x) )
/// ```
/// with `L1 = IFFT( |k|^(y−2) · FFT(ρ₀·∇·u) )` and
/// `L2 = IFFT( |k|^(y−1) · FFT(ρ_total) )`. The DC bin of nabla1/nabla2 is
/// set to zero per Treeby & Cox Eq. 10 to remove the |k|=0 singularity.
///
/// This test pins the formula by setting up a uniform field where the FFT
/// concentrates all energy at the DC bin: with nabla1[DC]=nabla2[DC]=0 the
/// L1 and L2 fields must be exactly zero everywhere, so the correction must
/// also be exactly zero. Any drift here detects an off-by-one in the
/// nabla-multiply, an FFT scaling error, or a missing DC nullification.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_pressure_correction_formula_dc_bin_nullification() {
    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    medium.set_acoustic_properties(0.5, 1.5, 0.0).unwrap();

    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.5,
            alpha_power: 1.5,
        },
        dt: 1e-7,
        ..PSTDConfig::default()
    };

    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    // Uniform divergence → FFT concentrates at DC where nabla1=nabla2=0.
    // apply_absorption_to_pressure reads div_ux/div_uy/div_uz (the divergence cache),
    // not dpx/dpy/dpz (which are overwritten by Step 3 and 4 before being read).
    solver.div_ux.fill(1.0);
    solver.div_uy.fill(0.0);
    solver.div_uz.fill(0.0);
    solver.div_u.fill(1.0);
    solver.fields.p.fill(0.0);

    solver.apply_absorption_to_pressure().unwrap();

    let max_abs: f64 = solver
        .fields
        .p
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    // Expect numerically exact zero (up to FFT round-off).
    assert!(
        max_abs < 1e-9,
        "DC nullification failed: max|p| = {} (expected ~0); confirms either \
         nabla DC bin not zeroed, FFT scaling drift, or scratch buffer leak",
        max_abs
    );
}

/// Pointwise verification of the dispersion (η) term in the Treeby & Cox
/// pressure correction.
///
/// **Setup:** velocity is zero so L1 vanishes and only `−η·L2` contributes.
/// A single Fourier mode at `(kx_idx, 0, 0)` is excited in `self.div_u` (the
/// ρ_total scratch). The pressure correction must equal
/// ```text
///   Δp(x) = − c₀² · η · ρ_total_amp · |k|^(y−1) · cos(kx·x)
/// ```
/// pointwise. The absorbing (τ·L1) term is exercised end-to-end by the
/// pykwavers parity scripts in `examples/na_modelling_absorption_compare.py`.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_pressure_correction_dispersion_term_matches_analytical() {
    const NX: usize = 32;
    const NY: usize = 1;
    const NZ: usize = 1;
    const DX: f64 = 1e-4;
    const C0: f64 = SOUND_SPEED_WATER_SIM;
    const RHO0: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    const ALPHA: f64 = 0.75;
    const Y: f64 = 1.5;

    let grid = Grid::new(NX, NY, NZ, DX, DX, DX).unwrap();
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    medium.set_acoustic_properties(ALPHA, Y, 0.0).unwrap();
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: ALPHA,
            alpha_power: Y,
        },
        dt: 1e-7,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    // Velocity = 0 → ∇·u = 0 → L1 = 0; only the dispersion term −η·L2
    // contributes. div_u carries the ρ_total scratch as in update_pressure.
    solver.fields.ux.fill(0.0);
    solver.fields.uy.fill(0.0);
    solver.fields.uz.fill(0.0);
    let kx_idx = 1usize;
    let rho_total_amp = 1.0_f64;
    for i in 0..NX {
        let phase = (kx_idx as f64) * TWO_PI * (i as f64) / (NX as f64);
        solver.div_u[[i, 0, 0]] = rho_total_amp * phase.cos();
    }
    solver.fields.p.fill(0.0);

    // Read the kernel's stored η and |k|^(y−1) directly so the test pins the
    // formula without duplicating the grid's wavenumber convention.
    let n2_kernel = solver.absorption.as_ref().unwrap().nabla2[[kx_idx, 0, 0]];
    let eta = solver.absorption.as_ref().unwrap().eta[[0, 0, 0]];
    let _ = RHO0; // unused with L1=0; kept for symmetry of physical constants

    solver.apply_absorption_to_pressure().unwrap();

    // Analytical reference (L1=0 because velocity is zero):
    //   Δp(x) = − c₀² · η · rho_total_amp · |k|^(y−1) · cos(kx·x).
    let amp = -C0 * C0 * eta * rho_total_amp * n2_kernel;

    // Compare every grid point against the analytical reference. Use the peak
    // analytical amplitude as the relative-error denominator so the tolerance
    // stays meaningful at zero-crossings.
    let denom = amp.abs().max(1e-30);
    let mut worst_rel: f64 = 0.0;
    for i in 0..NX {
        let phase = (kx_idx as f64) * TWO_PI * (i as f64) / (NX as f64);
        let expected = amp * phase.cos();
        let observed = solver.fields.p[[i, 0, 0]];
        let rel = (observed - expected).abs() / denom;
        worst_rel = worst_rel.max(rel);
    }
    assert!(
        worst_rel < 1e-9,
        "dispersion-term mismatch vs analytical reference: \
         worst peak-relative error = {:e} (denom = peak amp = {:e})",
        worst_rel,
        denom
    );
    // Sanity: peak amp must be physically meaningful (catches the pre-fix
    // no-op regression where the correction collapsed to ~10⁻¹¹).
    assert!(
        amp.abs() > 1e-3,
        "Reference amp magnitude {} is too small to discriminate the pre-fix \
         no-op behavior; this test is unsound",
        amp
    );
}

#[test]
fn test_fft_absorption_energy_dissipation() {
    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.5,
            alpha_power: 1.5,
        },
        dt: 1e-7,
        ..PSTDConfig::default()
    };

    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    for k in 0..16_usize {
        for j in 0..16_usize {
            for i in 0..16_usize {
                let val = (i as f64 * std::f64::consts::PI / 8.0).sin();
                solver.rhox[[i, j, k]] = val / 3.0;
                solver.rhoy[[i, j, k]] = val / 3.0;
                solver.rhoz[[i, j, k]] = val / 3.0;
                // apply_absorption_to_pressure reads div_u* (not dpx/dpy/dpz).
                solver.div_ux[[i, j, k]] = val;
                solver.div_uy[[i, j, k]] = val;
                solver.div_uz[[i, j, k]] = val;
            }
        }
    }

    // The previous "energy must change" assertion was cosmetic — a 10⁻¹¹
    // correction satisfied it just as well as a physically correct one.
    // The pressure-side correction is now exercised quantitatively in
    // `test_pressure_correction_matches_analytical_single_mode`. Here we
    // simply assert that with a non-trivial input the L1/L2 buffers are
    // populated and the correction is non-negligible.
    solver.fields.p.fill(0.0);
    solver.div_u.assign(&solver.rhox);
    solver.div_u.scaled_add(1.0, &solver.rhoy);
    solver.div_u.scaled_add(1.0, &solver.rhoz);
    solver.apply_absorption_to_pressure().unwrap();

    let max_correction: f64 = solver
        .fields
        .p
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_correction > 1e-3,
        "Pressure-side absorption correction collapsed to {} (no-op regression);\
         the bug closed in this fix made this < 1e-9. Threshold 1e-3 catches \
         off-by-orders regressions while remaining insensitive to FFT noise.",
        max_correction
    );
}

/// Verify that `update_pressure` (linear, lossless) correctly populates both:
///   - `self.div_u = rhox + rhoy + rhoz` (ρ_total, needed by the L2 absorption term)
///   - `self.fields.p = c₀² · (rhox + rhoy + rhoz)` (linear EOS)
///
/// Mathematical invariants:
///   - `∀ i: div_u[i] = rhox[i] + rhoy[i] + rhoz[i]`
///   - `∀ i: p[i] = c0[i]² · div_u[i]`
///
/// These invariants cover the fused single-pass EOS path (Opt-4) — both outputs
/// are written simultaneously by the same Zip pass.  Regression against the
/// pre-fusion 2-pass implementation: both implementations must produce bit-for-bit
/// identical results (same arithmetic operations on f64).
/// # Panics
/// - Panics if `PSTDSolver::new` fails.
///
#[test]
fn test_update_pressure_linear_eos_populates_div_u_and_p() {
    let nx = 16_usize;
    let grid = Grid::new(nx, nx, nx, 1e-4, 1e-4, 1e-4).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Lossless,
        dt: 1e-7,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

    // Plant non-trivial analytically derived density state.
    // rhox[i,j,k] = (i+1)*0.001, rhoy = (j+1)*0.002, rhoz = (k+1)*0.003
    // Expected: div_u[i,j,k] = (i+1)*0.001 + (j+1)*0.002 + (k+1)*0.003
    //           p[i,j,k]     = c0[i,j,k]² · div_u[i,j,k]
    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                solver.rhox[[i, j, k]] = (i + 1) as f64 * 0.001;
                solver.rhoy[[i, j, k]] = (j + 1) as f64 * 0.002;
                solver.rhoz[[i, j, k]] = (k + 1) as f64 * 0.003;
            }
        }
    }

    solver.update_pressure(1e-7).unwrap();

    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                let rx = solver.rhox[[i, j, k]];
                let ry = solver.rhoy[[i, j, k]];
                let rz = solver.rhoz[[i, j, k]];
                let expected_div = rx + ry + rz;
                let c = solver.materials.c0[[i, j, k]];
                let expected_p = c * c * expected_div;

                let actual_div = solver.div_u[[i, j, k]];
                let actual_p = solver.fields.p[[i, j, k]];

                assert!(
                    (actual_div - expected_div).abs() < 1e-15,
                    "div_u mismatch at [{i},{j},{k}]: expected {expected_div:.15e}, got {actual_div:.15e}"
                );
                assert!(
                    (actual_p - expected_p).abs() < 1e-8 * expected_p.abs().max(1e-20),
                    "p mismatch at [{i},{j},{k}]: expected {expected_p:.15e}, got {actual_p:.15e}"
                );
            }
        }
    }
}
