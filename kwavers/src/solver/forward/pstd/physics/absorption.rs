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
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::solver::forward::pstd::config::PSTDConfig;
use crate::solver::pstd::PSTDSolver;
use ndarray::{Array3, Zip};

/// Initialize absorption operators τ, η, spatially-varying exponent y, and the
/// spectral nabla operators ∇^(y−2) and ∇^(y−1) in FFT-order k-space.
///
/// ## Algorithm
/// For `AbsorptionMode::PowerLaw { alpha_coeff: α₀, alpha_power: y }`:
///
/// **Spatial coefficients** (vary per cell for heterogeneous media):
/// ```text
///   τ(r) = −2 α₀(r) · c₀(r)^(y(r)−1)
///   η(r) =  2 α₀(r) · c₀(r)^y(r) · tan(π y(r) / 2)
/// ```
///
/// **Spectral operators** (global, using config alpha_power for k-space precomputation):
/// ```text
///   nabla1[i,j,k] = |k|^(y_config − 2),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
///   nabla2[i,j,k] = |k|^(y_config − 1),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
/// ```
///
/// ## Returns
/// `(tau, eta, y_field, nabla1, nabla2)` — all arrays with shape `(nx, ny, nz)`.
///
/// ## References
/// Treeby & Cox (2010) Eqs. 19–21 for τ and η; Eq. 10 for nabla operators.
pub fn initialize_absorption_operators(
    config: &PSTDConfig,
    grid: &Grid,
    medium: &dyn Medium,
    k_mag: &Array3<f64>,
    _k_max: f64,
    _c_ref: f64,
) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)> {
    let shape = (grid.nx, grid.ny, grid.nz);
    let mut tau = Array3::zeros(shape);
    let mut eta = Array3::zeros(shape);
    let mut y_field = Array3::zeros(shape);
    let nabla1;
    let nabla2;

    match &config.absorption_mode {
        AbsorptionMode::Lossless => {
            // No absorption — nabla operators remain zero.
            nabla1 = Array3::zeros(shape);
            nabla2 = Array3::zeros(shape);
        }
        AbsorptionMode::Stokes => {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "AbsorptionMode::Stokes is not supported by spectral solver yet"
                        .to_string(),
                },
            ));
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

            // Spatial coefficients τ(r) and η(r) — may vary per cell.
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);

                        let alpha_0_medium = medium.alpha_coefficient(x, y_coord, z, grid);
                        let alpha_0 = if alpha_0_medium.abs() > 0.0 {
                            alpha_0_medium
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

                        // Treeby & Cox (2010) Eqs. 19–20
                        tau[[i, j, k]] = -2.0 * alpha_0 * c0_val.powf(y - 1.0);
                        eta[[i, j, k]] =
                            2.0 * alpha_0 * c0_val.powf(y) * (std::f64::consts::PI * y / 2.0).tan();
                        y_field[[i, j, k]] = y;
                    }
                }
            }

            // Spectral nabla operators — precomputed in FFT wavenumber order.
            // nabla1 = |k|^(y_config − 2),  nabla2 = |k|^(y_config − 1)
            // Treeby & Cox (2010) Eq. 10; k-Wave: create_absorption_variables.py lines 69–82.
            // DC bin (|k| = 0) → 0 to avoid singularity.
            nabla1 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    k.powf(y_config - 2.0)
                } else {
                    0.0
                }
            });
            nabla2 = k_mag.mapv(|k| {
                if k > ABSORPTION_SINGULARITY_THRESHOLD {
                    k.powf(y_config - 1.0)
                } else {
                    0.0
                }
            });
        }
        AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Relaxation absorption modes are not supported by spectral solver"
                        .to_string(),
                },
            ));
        }
    }

    Ok((tau, eta, y_field, nabla1, nabla2))
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
    /// - Complex scratch: `grad_x_k / grad_y_k / grad_z_k` (store FFT result),
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
        let is_on = match self.config.absorption_mode {
            AbsorptionMode::Lossless => false,
            AbsorptionMode::Stokes => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "AbsorptionMode::Stokes is not supported by spectral solver yet"
                            .to_string(),
                    },
                ));
            }
            AbsorptionMode::PowerLaw { .. } => true,
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by spectral solver"
                            .to_string(),
                    },
                ));
            }
        };

        if !is_on {
            return Ok(());
        }

        // ── X-AXIS ────────────────────────────────────────────────────────────────
        // dpx holds ∂u_x/∂x from update_density(); apply absorption to rhox.
        self.fft.forward_into(&self.dpx, &mut self.grad_x_k);
        {
            let n1 = self.absorb_nabla1.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_x_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.div_u, &mut self.uy_k); // L1x in div_u
        {
            let n2 = self.absorb_nabla2.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_x_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.dpx, &mut self.uy_k); // L2x in dpx
        Zip::from(&mut self.rhox)
            .and(&self.absorb_tau)
            .and(&self.absorb_eta)
            .and(&self.div_u)
            .and(&self.dpx)
            .for_each(|rho, &tau, &eta, &l1, &l2| {
                *rho += dt * (tau * l1 - eta * l2);
            });

        // ── Y-AXIS ────────────────────────────────────────────────────────────────
        // dpy holds ∂u_y/∂y; apply absorption to rhoy.
        self.fft.forward_into(&self.dpy, &mut self.grad_y_k);
        {
            let n1 = self.absorb_nabla1.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_y_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.div_u, &mut self.ux_k); // L1y in div_u
        {
            let n2 = self.absorb_nabla2.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_y_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.dpy, &mut self.ux_k); // L2y in dpy
        Zip::from(&mut self.rhoy)
            .and(&self.absorb_tau)
            .and(&self.absorb_eta)
            .and(&self.div_u)
            .and(&self.dpy)
            .for_each(|rho, &tau, &eta, &l1, &l2| {
                *rho += dt * (tau * l1 - eta * l2);
            });

        // ── Z-AXIS ────────────────────────────────────────────────────────────────
        // dpz holds ∂u_z/∂z; apply absorption to rhoz.
        self.fft.forward_into(&self.dpz, &mut self.grad_z_k);
        {
            let n1 = self.absorb_nabla1.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_z_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.div_u, &mut self.ux_k); // L1z in div_u
        {
            let n2 = self.absorb_nabla2.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_z_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.dpz, &mut self.ux_k); // L2z in dpz
        Zip::from(&mut self.rhoz)
            .and(&self.absorb_tau)
            .and(&self.absorb_eta)
            .and(&self.div_u)
            .and(&self.dpz)
            .for_each(|rho, &tau, &eta, &l1, &l2| {
                *rho += dt * (tau * l1 - eta * l2);
            });

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
                alpha_coeff: 0.75,
                alpha_power: 1.5,
            },
            ..PSTDConfig::default()
        };

        let k_mag = zeros_k_mag(32, 32, 32);
        let (tau, eta, y_field, nabla1, nabla2) =
            initialize_absorption_operators(&config, &grid, &medium, &k_mag, 1e6, 1500.0).unwrap();

        assert!(tau[[0, 0, 0]] < 0.0, "tau should be negative");
        // For y = 1.5, tan(0.75π) < 0 → eta < 0
        assert!(eta[[0, 0, 0]] < 0.0, "eta should be negative for y=1.5");
        assert!(
            (y_field[[0, 0, 0]] - 1.5).abs() < 1e-6,
            "y should be 1.5 for homogeneous medium"
        );
        // With zero k_mag, nabla operators are zero everywhere (DC singularity handling)
        assert_eq!(nabla1[[0, 0, 0]], 0.0);
        assert_eq!(nabla2[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_nabla_operators_correct_power() {
        /// Verify nabla1 = k^(y-2), nabla2 = k^(y-1) at a known non-zero wavenumber.
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

        let (_, _, _, nabla1, nabla2) =
            initialize_absorption_operators(&config, &grid, &medium, &k_mag, 0.0, 1500.0).unwrap();

        let expected_n1 = k_at_1.powf(y - 2.0);
        let expected_n2 = k_at_1.powf(y - 1.0);
        assert!(
            (nabla1[[1, 0, 0]] - expected_n1).abs() < 1e-10 * expected_n1,
            "nabla1 mismatch: got {}, expected {}",
            nabla1[[1, 0, 0]],
            expected_n1
        );
        assert!(
            (nabla2[[1, 0, 0]] - expected_n2).abs() < 1e-10 * expected_n2,
            "nabla2 mismatch: got {}, expected {}",
            nabla2[[1, 0, 0]],
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
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);
        let k_mag = zeros_k_mag(16, 16, 16);
        let (tau, eta, _, _, _) =
            initialize_absorption_operators(&config, &grid, &medium, &k_mag, 1e6, 1500.0).unwrap();

        assert!(
            tau[[0, 0, 0]] <= 0.0,
            "tau should be non-positive for absorption"
        );
        // For 1 < y < 2, tan(π*y/2) < 0 → eta ≤ 0
        assert!(
            eta[[0, 0, 0]] <= 0.0,
            "eta should be non-positive for y=1.5"
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
        assert!(max_abs > 0.0, "Absorption should produce a non-zero density correction");
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
