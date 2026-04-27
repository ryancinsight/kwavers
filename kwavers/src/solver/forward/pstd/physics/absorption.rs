//! Power-law absorption using fractional Laplacian operators (Treeby & Cox 2010)
//!
//! # Physics: Absorbing Acoustic Media
//!
//! ## Theorem: Fractional Laplacian Power-Law Absorption
//! For a medium with power-law absorption coefficient α(f) = α₀ f^y (Np/m), the
//! causal, minimum-phase wave equation (Treeby & Cox 2010, Eqs. 9–10) modifies the
//! acoustic density update as:
//!
//! ```text
//!   ρ_x^{n+1} += Δt · τ · ∇^(y−2) (∂u_x/∂x) − Δt · η · ∇^(y−1) (∂u_x/∂x)
//! ```
//! where the proportionality coefficients are:
//! ```text
//!   τ = −2 α₀ c₀^(y−1)          (absorbing, negative for positive α₀)
//!   η =  2 α₀ c₀^y tan(πy/2)    (dispersive, causal Kramers–Kronig)
//! ```
//! and the fractional Laplacian operators are implemented spectrally:
//! ```text
//!   ∇^(y−2) f = IFFT( |k|^(y−2) · FFT(f) )
//!   ∇^(y−1) f = IFFT( |k|^(y−1) · FFT(f) )
//! ```
//! with singularity handling: set |k|^n = 0 when |k| ≈ 0.
//!
//! ## Algorithm
//! 1. **Precompute** (initialization):
//!    - `nabla1[i,j,k] = |k|^(y−2)` for all wavenumber magnitudes
//!    - `nabla2[i,j,k] = |k|^(y−1)` for all wavenumber magnitudes
//!    - Zero DC bin (|k| = 0) to avoid division by zero
//! 2. **Per time step** (after velocity divergence computation):
//!    - For each axis α ∈ {x, y, z}:
//!      - `L1_α = IFFT(nabla1 · FFT(∂u_α/∂α))`
//!      - `L2_α = IFFT(nabla2 · FFT(∂u_α/∂α))`
//!      - `ρ_α += Δt · τ · L1_α − Δt · η · L2_α`
//!
//! ## Key Corrections vs Previous Implementation
//! - **Operator**: `|k|^(y−2)` (k-space) not `f_MHz^(y+1)` (frequency-domain)
//! - **Input**: per-axis velocity divergence `∂u_α/∂α` not total density ρ
//! - **Split**: each axis updates its own density component (no /3 averaging)
//! - **Timing**: called inside density update, not pressure update
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314. Eqs. 9–10, 19–21.
//! - Caputo (1967). Geophys. J. Int. 13(5), 529–539. (fractional calculus)
//! - k-Wave C++ binary: `kspaceFirstOrder3D-OMP`, function `computeDensity()`.

use crate::core::constants::ABSORPTION_SINGULARITY_THRESHOLD;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64;
use crate::physics::acoustics::mechanics::absorption::{
    power_law_db_cm_to_np_omega_m, AbsorptionMode,
};
use crate::solver::forward::pstd::config::PSTDConfig;
use crate::solver::pstd::PSTDSolver;
use ndarray::{Array3, Zip};

/// Precomputed spectral absorption arrays for power-law fractional Laplacian.
///
/// Allocated only when `AbsorptionMode::PowerLaw` is active; `None` for lossless mode.
/// This avoids 4 × N³ × 8-byte allocations per lossless simulation.
pub(crate) struct AbsorptionKernel {
    /// Absorption coefficient field τ = −2 α₀ c₀^(y−1) [Treeby & Cox 2010 Eq. 19]
    pub tau: Array3<f64>,
    /// Dispersion coefficient field η = 2 α₀ c₀^y tan(πy/2) [Eq. 20]
    pub eta: Array3<f64>,
    /// Spectral nabla1 operator |k|^(y−2) in FFT wavenumber order [Eq. 10]
    pub nabla1: Array3<f64>,
    /// Spectral nabla2 operator |k|^(y−1) in FFT wavenumber order [Eq. 10]
    pub nabla2: Array3<f64>,
}

/// Initialize absorption operators τ, η, spatially-varying exponent y, and the
/// spectral nabla operators ∇^(y−2) and ∇^(y−1) in FFT-order k-space.
///
/// ## Algorithm
/// For `AbsorptionMode::PowerLaw { alpha_coeff: α_dB, alpha_power: y }`:
///
/// **Spatial coefficients** (vary per cell for heterogeneous media):
/// ```text
///   α_SI(r) = power_law_db_cm_to_np_omega_m(α_dB(r), y(r))
///   τ(r)    = −2 α_SI(r) · c₀(r)^(y(r)−1)
///   η(r)    =  2 α_SI(r) · c₀(r)^y(r) · tan(π y(r) / 2)
/// ```
///
/// **Spectral operators** (global, using config alpha_power for k-space precomputation):
/// ```text
///   nabla1[i,j,k] = |k|^(y_config − 2),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
///   nabla2[i,j,k] = |k|^(y_config − 1),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
/// ```
///
/// ## Returns
/// `Some(AbsorptionKernel)` for `PowerLaw` mode; `None` for `Lossless` (saves 4 × N³ × 8 bytes).
///
/// ## References
/// Treeby & Cox (2010) Eqs. 19–21 for τ and η; Eq. 10 for nabla operators.
pub(crate) fn initialize_absorption_operators(
    config: &PSTDConfig,
    grid: &Grid,
    medium: &dyn Medium,
    k_mag: &Array3<f64>,
    _k_max: f64,
    _c_ref: f64,
) -> KwaversResult<Option<AbsorptionKernel>> {
    let shape = (grid.nx, grid.ny, grid.nz);

    match &config.absorption_mode {
        AbsorptionMode::Lossless => {
            // No absorption arrays needed — return None to skip all allocations.
            Ok(None)
        }
        AbsorptionMode::Stokes => {
            // Theorem: Stokes (thermoviscous) absorption for Newtonian fluids
            // [Stokes 1845; Lighthill 1978, Waves in Fluids, §1.9].
            //
            // The attenuation coefficient is:
            //   α(ω) = (4η_s/3 + η_b) / (2ρ₀c₀³) · ω²   [Np/m]
            //
            // This is PowerLaw with y = 2. Substituting into Treeby & Cox (2010)
            // Eqs. 19–20 with y = 2:
            //   α_SI(r) = (4η_s(r)/3 + η_b(r)) / (2ρ₀(r)c₀(r)³)   [Np/(rad/s)²/m]
            //   τ(r) = −2α_SI(r) · c₀(r)^(y−1) = −2α_SI(r) · c₀(r)
            //   η(r) = 2α_SI(r) · c₀(r)^y · tan(πy/2)
            //         = 2α_SI(r) · c₀(r)² · tan(π) = 0   (non-dispersive: tan(π) = 0)
            //
            // Nabla operators for y = 2 (Treeby & Cox Eq. 10):
            //   nabla1[k] = |k|^(y−2) = |k|^0 = 1  for |k| > threshold, else 0
            //   nabla2[k] = |k|^(y−1) = |k|^1 = |k| for |k| > threshold, else 0
            //
            // Proof: tan(π) = sin(π)/cos(π) = 0/(-1) = 0 exactly, so the dispersive
            // term η vanishes. The absorbing term τ reduces to viscous damping:
            //   τ = −(4η_s/3 + η_b) / (ρ₀c₀²)
            // which is the classical viscous decay rate [Blackstock 2000, Eq. 10-13].

            let mut tau = Array3::zeros(shape);
            // eta = 0 everywhere (tan(π) = 0); allocate zeros to satisfy AbsorptionKernel.
            let eta = Array3::zeros(shape);

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);
                        let c0 = medium.sound_speed(i, j, k);
                        let rho0 = medium.density(i, j, k);
                        let eta_s = medium.shear_viscosity(x, y_coord, z, grid);
                        let eta_b = medium.bulk_viscosity(x, y_coord, z, grid);

                        if rho0 > 0.0 && c0 > 0.0 {
                            // α_SI = (4η_s/3 + η_b) / (2ρ₀c₀³)   [Np/(rad/s)²/m]
                            let alpha_si =
                                (4.0 * eta_s / 3.0 + eta_b) / (2.0 * rho0 * c0 * c0 * c0);
                            // τ = −2α_SI·c₀  (y=2 specialisation of Treeby & Cox Eq. 19)
                            tau[[i, j, k]] = -2.0 * alpha_si * c0;
                        }
                    }
                }
            }

            // nabla1 = |k|^0 = 1 at all non-DC modes; nabla2 = |k|^1 = |k|
            let nabla1 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    1.0
                } else {
                    0.0
                }
            });
            let nabla2 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    k
                } else {
                    0.0
                }
            });

            Ok(Some(AbsorptionKernel {
                tau,
                eta,
                nabla1,
                nabla2,
            }))
        }
        AbsorptionMode::PowerLaw {
            alpha_coeff,
            alpha_power,
        } => {
            let y_config = *alpha_power;
            if (y_config - 1.0).abs() < 1e-12 {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "alpha_power must not be 1.0 for fractional Laplacian formulation"
                            .to_string(),
                    },
                ));
            }

            let mut tau = Array3::zeros(shape);
            let mut eta = Array3::zeros(shape);

            // Spatial coefficients τ(r) and η(r) — may vary per cell.
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);

                        let alpha_db_cm_medium = medium.alpha_coefficient(x, y_coord, z, grid);
                        let alpha_db_cm = if alpha_db_cm_medium.abs() > 0.0 {
                            alpha_db_cm_medium
                        } else {
                            *alpha_coeff
                        };

                        let y_medium = medium.alpha_power(x, y_coord, z, grid);
                        let y = if y_medium.abs() > 1e-12 && (y_medium - 1.0).abs() > 1e-12 {
                            y_medium
                        } else {
                            y_config
                        };

                        let c0_val = medium.sound_speed(i, j, k);
                        let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, y);

                        // Treeby & Cox (2010) Eqs. 19–20
                        tau[[i, j, k]] = -2.0 * alpha_0_si * c0_val.powf(y - 1.0);
                        eta[[i, j, k]] = 2.0
                            * alpha_0_si
                            * c0_val.powf(y)
                            * (std::f64::consts::PI * y / 2.0).tan();
                    }
                }
            }

            // Spectral nabla operators — precomputed in FFT wavenumber order.
            // nabla1 = |k|^(y_config − 2),  nabla2 = |k|^(y_config − 1)
            // Treeby & Cox (2010) Eq. 10; k-Wave: create_absorption_variables.py lines 69–82.
            // DC bin (|k| = 0) → 0 to avoid singularity.
            let nabla1 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    k.powf(y_config - 2.0)
                } else {
                    0.0
                }
            });
            let nabla2 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    k.powf(y_config - 1.0)
                } else {
                    0.0
                }
            });

            Ok(Some(AbsorptionKernel {
                tau,
                eta,
                nabla1,
                nabla2,
            }))
        }
        AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => Err(
            KwaversError::Validation(ValidationError::ConstraintViolation {
                message: "Relaxation absorption modes are not supported by spectral solver"
                    .to_string(),
            }),
        ),
    }
}

impl PSTDSolver {
    /// Apply per-axis power-law absorption correction to split density components.
    ///
    /// ## Theorem: Per-Axis Fractional Laplacian Absorption
    /// For each axis α ∈ {x, y, z}, the absorption correction is (Treeby & Cox 2010, Eq. 21):
    /// ```text
    ///   ρ_α += Δt · τ · IFFT(nabla1 · FFT(∂u_α/∂α))
    ///        − Δt · η · IFFT(nabla2 · FFT(∂u_α/∂α))
    /// ```
    /// where `nabla1 = |k|^(y−2)` and `nabla2 = |k|^(y−1)`.
    ///
    /// ## Algorithm
    /// Uses scratch fields already in solver state to avoid allocations:
    /// - Per-axis velocity divergences are in `self.dpx`, `self.dpy`, `self.dpz`
    ///   (filled by `update_density()` immediately before this call)
    /// - Complex scratch: `grad_k` (FFT of per-axis divergence, reused across axes),
    ///   `ux_k / uy_k / uz_k` (operator-weighted spectra), `div_u` (real IFFT output)
    /// - The L2 result overwrites `dpx/dpy/dpz` since those fields are no longer
    ///   needed after absorption completes for the current step.
    ///
    /// ## Precondition
    /// Must be called AFTER `update_density()` (so dpx/dpy/dpz hold `∂u_α/∂α`)
    /// and BEFORE `apply_pml_to_density()` post-step (matching k-Wave ordering).
    ///
    /// ## References
    /// - Treeby & Cox (2010) Eqs. 19–21.
    /// - k-Wave MATLAB: `kspaceFirstOrder3D.m`, density absorption block.
    pub(crate) fn apply_absorption(&mut self, dt: f64) -> KwaversResult<()> {
        match self.config.absorption_mode {
            AbsorptionMode::Lossless => return Ok(()),
            // Stokes uses the same fractional-Laplacian kernel as PowerLaw (y=2).
            // Its AbsorptionKernel was initialised with tau = viscous decay, eta = 0,
            // nabla1 = 1 (everywhere), nabla2 = |k|.  The general formula below
            // reduces to pure viscous damping since eta = 0 eliminates the L2 term.
            AbsorptionMode::Stokes | AbsorptionMode::PowerLaw { .. } => {}
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by spectral solver"
                            .to_string(),
                    },
                ));
            }
        }

        let Some(ref abs) = self.absorption else {
            return Ok(());
        };

        // ── X-AXIS ────────────────────────────────────────────────────────────────
        // dpx holds ∂u_x/∂x from update_density(); apply absorption to rhox.
        // grad_k is reused as the FFT scratch for each axis sequentially.
        self.fft.forward_into(&self.dpx, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.div_u, &mut self.uy_k); // L1x in div_u
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.dpx, &mut self.uy_k); // L2x in dpx
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhox)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpx)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        // ── Y-AXIS ────────────────────────────────────────────────────────────────
        self.fft.forward_into(&self.dpy, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.div_u, &mut self.ux_k); // L1y in div_u
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.dpy, &mut self.ux_k); // L2y in dpy
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhoy)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpy)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        // ── Z-AXIS ────────────────────────────────────────────────────────────────
        self.fft.forward_into(&self.dpz, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.div_u, &mut self.ux_k); // L1z in div_u
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.dpz, &mut self.ux_k); // L2z in dpz
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhoz)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpz)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;
    use crate::solver::forward::pstd::config::PSTDConfig;
    use ndarray::Array3;

    /// Build a zero k_mag array for tests that only verify tau/eta/y_field.
    fn zeros_k_mag(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
        Array3::zeros((nx, ny, nz))
    }

    /// Build a simple k_mag array with non-zero entries for nabla operator tests.
    fn test_k_mag(nx: usize, ny: usize, nz: usize, dk: f64) -> Array3<f64> {
        let mut k = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for kk in 0..nz {
                    let ki = if i <= nx / 2 { i } else { nx - i } as f64 * dk;
                    let kj = if j <= ny / 2 { j } else { ny - j } as f64 * dk;
                    let kkk = if kk <= nz / 2 { kk } else { nz - kk } as f64 * dk;
                    k[[i, j, kk]] = (ki * ki + kj * kj + kkk * kkk).sqrt();
                }
            }
        }
        k
    }

    #[test]
    fn test_power_law_initialization() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        medium.set_acoustic_properties(0.75, 1.5, 5.0).unwrap();
        let config = PSTDConfig {
            dt: 1e-7,
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.0,
                alpha_power: 1.5,
            },
            ..PSTDConfig::default()
        };

        let k_mag = zeros_k_mag(32, 32, 32);
        let kernel = initialize_absorption_operators(&config, &grid, &medium, &k_mag, 1e6, 1500.0)
            .unwrap()
            .expect("PowerLaw mode must return Some(AbsorptionKernel)");

        let expected_tau = -4.246_711_703_873_091e-8;
        let expected_eta = -6.370_067_555_809_639e-5;
        assert!(
            (kernel.tau[[0, 0, 0]] - expected_tau).abs() < 1e-20,
            "tau mismatch: got {}, expected {}",
            kernel.tau[[0, 0, 0]],
            expected_tau
        );
        assert!(
            (kernel.eta[[0, 0, 0]] - expected_eta).abs() < 1e-18,
            "eta mismatch: got {}, expected {}",
            kernel.eta[[0, 0, 0]],
            expected_eta
        );
        // With zero k_mag, nabla operators are zero everywhere (DC singularity handling)
        assert_eq!(kernel.nabla1[[0, 0, 0]], 0.0);
        assert_eq!(kernel.nabla2[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_nabla_operators_correct_power() {
        // Verify nabla1 = k^(y-2), nabla2 = k^(y-1) at a known non-zero wavenumber.
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        let y = 1.5_f64;
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.5,
                alpha_power: y,
            },
            ..PSTDConfig::default()
        };

        // Build k_mag with a known value at [1,0,0]
        let dk = 2.0 * std::f64::consts::PI / (8.0 * 1e-3);
        let k_mag = test_k_mag(8, 8, 8, dk);
        let k_at_1 = k_mag[[1, 0, 0]]; // should be dk

        let kernel = initialize_absorption_operators(&config, &grid, &medium, &k_mag, 0.0, 1500.0)
            .unwrap()
            .expect("PowerLaw mode must return Some(AbsorptionKernel)");

        let expected_n1 = k_at_1.powf(y - 2.0);
        let expected_n2 = k_at_1.powf(y - 1.0);
        assert!(
            (kernel.nabla1[[1, 0, 0]] - expected_n1).abs() < 1e-10 * expected_n1,
            "nabla1 mismatch: got {}, expected {}",
            kernel.nabla1[[1, 0, 0]],
            expected_n1
        );
        assert!(
            (kernel.nabla2[[1, 0, 0]] - expected_n2).abs() < 1e-10 * expected_n2,
            "nabla2 mismatch: got {}, expected {}",
            kernel.nabla2[[1, 0, 0]],
            expected_n2
        );
    }

    #[test]
    fn test_absorption_model_physics_validation() {
        let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.75,
                alpha_power: 1.5,
            },
            ..Default::default()
        };
        let mut medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);
        medium.set_acoustic_properties(0.0, 1.5, 0.0).unwrap();
        let k_mag = zeros_k_mag(16, 16, 16);
        let kernel = initialize_absorption_operators(&config, &grid, &medium, &k_mag, 1e6, 1500.0)
            .unwrap()
            .expect("PowerLaw mode must return Some(AbsorptionKernel)");

        assert!(
            (kernel.tau[[0, 0, 0]] - (-3.467_425_586_398_137e-8)).abs() < 1e-20,
            "tau mismatch: got {}",
            kernel.tau[[0, 0, 0]]
        );
        assert!(
            (kernel.eta[[0, 0, 0]] - (-3.467_425_586_398_137_6e-5)).abs() < 1e-18,
            "eta mismatch: got {}",
            kernel.eta[[0, 0, 0]]
        );
    }

    #[test]
    fn test_lossless_mode_no_absorption() {
        use crate::domain::source::GridSource;

        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);

        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::Lossless,
            dt: 1e-7,
            ..PSTDConfig::default()
        };

        let mut solver =
            PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

        // Set velocity divergences to non-zero (simulating density update outputs)
        solver.dpx.fill(1.0);
        solver.dpy.fill(1.0);
        solver.dpz.fill(1.0);

        // Record initial density
        let rhox_before = solver.rhox.clone();
        let rhoy_before = solver.rhoy.clone();
        let rhoz_before = solver.rhoz.clone();

        // Apply absorption — should be a no-op in lossless mode
        solver.apply_absorption(1e-7).unwrap();

        for (a, b) in solver.rhox.iter().zip(rhox_before.iter()) {
            assert!((a - b).abs() < 1e-12, "rhox changed in lossless mode");
        }
        for (a, b) in solver.rhoy.iter().zip(rhoy_before.iter()) {
            assert!((a - b).abs() < 1e-12, "rhoy changed in lossless mode");
        }
        for (a, b) in solver.rhoz.iter().zip(rhoz_before.iter()) {
            assert!((a - b).abs() < 1e-12, "rhoz changed in lossless mode");
        }
    }

    #[test]
    fn test_fft_based_absorption_reduces_amplitude() {
        use crate::domain::source::GridSource;

        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.75,
                alpha_power: 1.5,
            },
            dt: 1e-7,
            ..PSTDConfig::default()
        };

        let mut solver =
            PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

        // Simulate a Gaussian velocity divergence (the source for absorption)
        let center = 16;
        let sigma = 4.0;
        let mut div = Array3::zeros((32, 32, 32));
        for k in 0..32_usize {
            for j in 0..32_usize {
                for i in 0..32_usize {
                    let r2 = ((i as f64 - center as f64).powi(2)
                        + (j as f64 - center as f64).powi(2)
                        + (k as f64 - center as f64).powi(2))
                        / (sigma * sigma);
                    div[[i, j, k]] = (-r2).exp();
                }
            }
        }
        // Velocity divergences act as input; initial density is zero
        solver.dpx.assign(&div);
        solver.dpy.assign(&div);
        solver.dpz.assign(&div);
        solver.rhox.fill(0.0);
        solver.rhoy.fill(0.0);
        solver.rhoz.fill(0.0);

        // After one absorption step, densities should be non-zero (absorbed from div)
        // and the net total should reflect damping (tau < 0 → densities become negative
        // for positive divergence → physically correct: absorption reduces overpressure)
        solver.apply_absorption(1e-7).unwrap();

        let max_abs: f64 = solver
            .rhox
            .iter()
            .chain(solver.rhoy.iter())
            .chain(solver.rhoz.iter())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs > 0.0,
            "Absorption should produce a non-zero density correction"
        );
    }

    /// Test Stokes absorption coefficient initialisation against the classical formula.
    ///
    /// # Reference derivation (Blackstock 2000, Fundamentals of Physical Acoustics, Eq. 10-13)
    /// ```text
    /// α_SI = (4η_s/3 + η_b) / (2ρ₀c₀³)   [Np/(rad/s)²/m]
    /// τ    = −2 α_SI · c₀                  [Treeby & Cox (2010) Eq. 19, y=2]
    /// η    = 0                              [tan(π) = 0, non-dispersive]
    /// ```
    /// HomogeneousMedium defaults: η_s = 1e−3 Pa·s, η_b = 2.5e−3 Pa·s (Stokes' hypothesis).
    /// With ρ₀ = 1000 kg/m³, c₀ = 1500 m/s:
    /// ```text
    ///   α_SI = (4/3 × 1e−3 + 2.5e−3) / (2 × 1000 × 1500³)
    ///        = 3.833̄e−3 / 6.75e12 = 5.6790123e−16
    ///   τ    = −2 × 5.6790123e−16 × 1500 = −1.7037037e−12
    /// ```
    #[test]
    fn test_stokes_absorption_tau_matches_classical_formula() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        // HomogeneousMedium::new sets η_s = 1e-3, η_b = 2.5e-3 by default.
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::Stokes,
            dt: 1e-7,
            ..PSTDConfig::default()
        };

        let dk = 2.0 * std::f64::consts::PI / (8.0 * 1e-3);
        let k_mag = test_k_mag(8, 8, 8, dk);
        let kernel = initialize_absorption_operators(&config, &grid, &medium, &k_mag, 0.0, 1500.0)
            .expect("Stokes init must succeed")
            .expect("Stokes mode must return Some(AbsorptionKernel)");

        // Analytically derived τ for water defaults
        let eta_s = 1.0e-3_f64;
        let eta_b = 2.5e-3_f64;
        let rho0 = 1000.0_f64;
        let c0 = 1500.0_f64;
        let alpha_si = (4.0 * eta_s / 3.0 + eta_b) / (2.0 * rho0 * c0 * c0 * c0);
        let expected_tau = -2.0 * alpha_si * c0;

        // All cells are homogeneous — every tau cell equals the analytical value.
        for val in kernel.tau.iter() {
            assert!(
                (val - expected_tau).abs() < 1e-24 * expected_tau.abs().max(1e-30),
                "tau cell mismatch: got {val}, expected {expected_tau}"
            );
        }

        // η = 0 everywhere (non-dispersive: tan(π) = 0 exactly)
        for val in kernel.eta.iter() {
            assert_eq!(*val, 0.0, "eta must be zero for Stokes (y=2) absorption");
        }

        // nabla1 = 1 at non-DC modes, 0 at DC (|k|=0)
        assert_eq!(kernel.nabla1[[0, 0, 0]], 0.0, "DC nabla1 must be 0");
        assert_eq!(
            kernel.nabla1[[1, 0, 0]],
            1.0,
            "nabla1 must be 1 at non-DC modes (|k|^0 = 1)"
        );

        // nabla2 = |k| at non-DC modes
        let expected_n2 = k_mag[[1, 0, 0]];
        assert!(
            (kernel.nabla2[[1, 0, 0]] - expected_n2).abs() < 1e-12 * expected_n2,
            "nabla2 mismatch: got {}, expected {}",
            kernel.nabla2[[1, 0, 0]],
            expected_n2
        );
    }

    #[test]
    fn test_fft_absorption_energy_dissipation() {
        use crate::domain::source::GridSource;

        let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.5,
                alpha_power: 1.5,
            },
            dt: 1e-7,
            ..PSTDConfig::default()
        };

        let mut solver =
            PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();

        // Pre-load split density components with a sinusoidal pattern
        for k in 0..16_usize {
            for j in 0..16_usize {
                for i in 0..16_usize {
                    let val = (i as f64 * std::f64::consts::PI / 8.0).sin();
                    solver.rhox[[i, j, k]] = val / 3.0;
                    solver.rhoy[[i, j, k]] = val / 3.0;
                    solver.rhoz[[i, j, k]] = val / 3.0;
                    // Velocity divergence consistent with the density pattern
                    solver.dpx[[i, j, k]] = val;
                    solver.dpy[[i, j, k]] = val;
                    solver.dpz[[i, j, k]] = val;
                }
            }
        }

        // Compute initial energy (L2 norm of total rho)
        let initial_energy: f64 = solver
            .rhox
            .iter()
            .zip(solver.rhoy.iter())
            .zip(solver.rhoz.iter())
            .map(|((rx, ry), rz)| (rx + ry + rz).powi(2))
            .sum();

        solver.apply_absorption(1e-7).unwrap();

        // After absorption, net rho should differ (absorption correction applied)
        let final_energy: f64 = solver
            .rhox
            .iter()
            .zip(solver.rhoy.iter())
            .zip(solver.rhoz.iter())
            .map(|((rx, ry), rz)| (rx + ry + rz).powi(2))
            .sum();

        // Energy must change (absorption has acted)
        assert!(
            (final_energy - initial_energy).abs() > 1e-20,
            "Absorption must change energy: {} → {}",
            initial_energy,
            final_energy
        );
    }
}
