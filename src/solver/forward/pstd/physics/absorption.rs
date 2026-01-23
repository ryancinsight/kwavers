//! Power law absorption implementation
//!
//! Implements fractional Laplacian for power law absorption:
//! ∂p/∂t = -τ∇^(y+1)p - η∇^(y+2)p
//!
//! References:
//! - Treeby & Cox (2010), Eq. 9-10
//! - Caputo (1967) for fractional derivatives

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64;
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::solver::forward::pstd::config::PSTDConfig;
use crate::solver::pstd::PSTDSolver;
use ndarray::{Array3, Zip};

/// Initialize absorption operators τ, η, and spatially-varying exponent y
///
/// Returns (tau, eta, y_field) where y_field contains the spatially-varying power law exponent.
/// For homogeneous media, y_field will be constant. For heterogeneous media (e.g., tissue),
/// y can vary spatially for more realistic absorption modeling.
///
/// References:
/// - Treeby & Cox (2010): k-Wave absorption formulation
/// - fullwave25: Spatially-varying absorption exponent implementation
pub fn initialize_absorption_operators(
    config: &PSTDConfig,
    grid: &Grid,
    medium: &dyn Medium,
    _k_max: f64,
    _c_ref: f64,
) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
    let mut tau = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut eta = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut y_field = Array3::zeros((grid.nx, grid.ny, grid.nz));

    match &config.absorption_mode {
        AbsorptionMode::Lossless => {
            // No absorption - arrays remain zero
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
            let y_default = *alpha_power;
            if (y_default - 1.0).abs() < 1e-12 {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "alpha_power must not be 1.0 for fractional Laplacian formulation"
                            .to_string(),
                    },
                ));
            }
            // Calculate absorption terms from medium properties
            // Now supports spatially-varying exponent from heterogeneous media
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);

                        // Use medium properties for spatially varying absorption coefficient
                        let alpha_0_medium = medium.alpha_coefficient(x, y_coord, z, grid);
                        let alpha_0 = if alpha_0_medium.abs() > 0.0 {
                            alpha_0_medium
                        } else {
                            *alpha_coeff
                        };

                        // Use medium properties for spatially varying absorption exponent
                        // This is the key enhancement from fullwave25
                        let y_medium = medium.alpha_power(x, y_coord, z, grid);
                        let y = if y_medium.abs() > 1e-12 && (y_medium - 1.0).abs() > 1e-12 {
                            y_medium
                        } else {
                            y_default
                        };

                        let c0_val = medium.sound_speed(i, j, k);

                        // Calculate tau and eta based on Treeby & Cox (2010)
                        // tau = -2 * alpha_0 * c0^(y-1)
                        // eta = 2 * alpha_0 * c0^y * tan(pi * y / 2)

                        let tau_val = -2.0 * alpha_0 * c0_val.powf(y - 1.0);
                        let eta_val =
                            2.0 * alpha_0 * c0_val.powf(y) * (std::f64::consts::PI * y / 2.0).tan();

                        tau[[i, j, k]] = tau_val;
                        eta[[i, j, k]] = eta_val;
                        y_field[[i, j, k]] = y;
                    }
                }
            }
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

    Ok((tau, eta, y_field))
}

impl PSTDSolver {
    /// Apply power law absorption using fractional Laplacian method
    ///
    /// Now supports spatially-varying absorption exponent for heterogeneous media.
    /// For homogeneous media, the exponent field is constant.
    ///
    /// Algorithm:
    /// 1. FFT density field to frequency domain
    /// 2. Compute fractional Laplacian L^(y+1) and L^(y+2) using average exponent
    /// 3. IFFT back to spatial domain
    /// 4. Apply spatially-varying tau and eta coefficients
    ///
    /// Note: We use a spatial average of y for the frequency-domain operations,
    /// then apply the exact spatially-varying coefficients in the final update.
    /// This is an approximation but maintains computational efficiency while
    /// capturing the primary spatially-varying effects through tau and eta.
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

        // Compute spatial average of exponent for frequency-domain operations
        // For homogeneous media, this will be constant
        // For heterogeneous media, this provides a reasonable approximation
        let y_mean = self.absorb_y.mean().unwrap_or(1.05);

        let c_ref = self.c_ref;
        let two_pi = 2.0 * std::f64::consts::PI;

        // FFT rho -> p_k (using p_k as scratch for rho_k)
        self.fft.forward_into(&self.rho, &mut self.p_k);

        // Compute L1 term: IFFT( f(k)^(y_mean+1) * rho_k )
        // Using spatial average exponent for frequency domain operations
        Zip::from(&mut self.ux_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.k_vec.1)
            .and(&self.k_vec.2)
            .for_each(|out, &rho_k, &kx, &ky, &kz| {
                let k_sq: f64 = kx * kx + ky * ky + kz * kz;
                if k_sq > 1e-14 {
                    let k_mag = k_sq.sqrt();
                    let freq = k_mag * c_ref / two_pi;
                    let f_mhz = freq / 1e6;
                    let term = f_mhz.powf(y_mean + 1.0);
                    *out = rho_k * Complex64::new(term, 0.0);
                } else {
                    *out = Complex64::new(0.0, 0.0);
                }
            });

        self.fft
            .inverse_into(&self.ux_k, &mut self.dpx, &mut self.uy_k);

        // Compute L2 term: IFFT( f(k)^(y_mean+2) * rho_k )
        Zip::from(&mut self.ux_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.k_vec.1)
            .and(&self.k_vec.2)
            .for_each(|out, &rho_k, &kx, &ky, &kz| {
                let k_sq: f64 = kx * kx + ky * ky + kz * kz;
                if k_sq > 1e-14 {
                    let k_mag = k_sq.sqrt();
                    let freq = k_mag * c_ref / two_pi;
                    let f_mhz = freq / 1e6;
                    let term = f_mhz.powf(y_mean + 2.0);
                    *out = rho_k * Complex64::new(term, 0.0);
                } else {
                    *out = Complex64::new(0.0, 0.0);
                }
            });

        self.fft
            .inverse_into(&self.ux_k, &mut self.dpy, &mut self.uy_k);

        // Update rho: rho += dt * (tau * l1 + eta * l2)
        // tau and eta already incorporate the exact spatially-varying exponent
        // This provides the full spatially-varying absorption effect
        Zip::from(&mut self.rho)
            .and(&self.absorb_tau)
            .and(&self.absorb_eta)
            .and(&self.dpx) // L1
            .and(&self.dpy) // L2
            .for_each(|rho, &tau, &eta, &l1, &l2| {
                *rho += dt * (tau * l1 + eta * l2);
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

    #[test]
    fn test_power_law_initialization() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid); // Lossless by default

        let config = PSTDConfig {
            dt: 1e-7,
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: 0.75,
                alpha_power: 1.5,
            },
            ..PSTDConfig::default()
        };

        let (tau, eta, y_field) =
            initialize_absorption_operators(&config, &grid, &medium, 1e6, 1500.0).unwrap();

        // Check signs (tau negative, eta depends on y)
        assert!(tau[[0, 0, 0]] < 0.0, "tau should be negative");
        // For y = 1.5, tan(0.75 * PI) is negative
        assert!(eta[[0, 0, 0]] < 0.0, "eta should be negative for y=1.5");
        // Check that y field is initialized correctly
        assert!(
            (y_field[[0, 0, 0]] - 1.5).abs() < 1e-6,
            "y should be 1.5 for homogeneous medium"
        );
    }

    #[test]
    fn test_absorption_model_physics_validation() {
        let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();

        // Test that all models satisfy physical constraints
        let models = vec![
            (
                "PowerLaw",
                AbsorptionMode::PowerLaw {
                    alpha_coeff: 0.75,
                    alpha_power: 1.5,
                },
            ),
            // MultiRelaxation and Causal modes are currently not supported by initialize_absorption_operators
        ];

        for (name, mode) in models {
            let config = PSTDConfig {
                absorption_mode: mode,
                ..Default::default()
            };

            let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);
            let (tau, eta, _y_field) =
                initialize_absorption_operators(&config, &grid, &medium, 1e6, 1500.0).unwrap();

            // Physical constraints: absorption should reduce amplitude (tau < 0)
            // and dispersion should be positive (eta > 0) for causal absorption
            if !matches!(config.absorption_mode, AbsorptionMode::Lossless) {
                assert!(
                    tau[[0, 0, 0]] <= 0.0,
                    "{} model: tau should be non-positive for absorption",
                    name
                );

                // For y < 1, eta is positive. For 1 < y < 2, eta is negative.
                if let AbsorptionMode::PowerLaw { alpha_power, .. } = config.absorption_mode {
                    if alpha_power < 1.0 {
                        assert!(
                            eta[[0, 0, 0]] >= 0.0,
                            "{} model: eta should be non-negative for physical dispersion",
                            name
                        );
                    } else if alpha_power > 1.0 && alpha_power < 2.0 {
                        assert!(
                            eta[[0, 0, 0]] <= 0.0,
                            "{} model: eta should be non-positive for physical dispersion",
                            name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fft_based_absorption_reduces_amplitude() {
        use crate::domain::medium::HomogeneousMedium;
        use crate::domain::source::GridSource;

        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);

        // Create initial pressure field (Gaussian pulse)
        let mut p = Array3::zeros((32, 32, 32));
        let center = 16;
        let sigma = 4.0;
        for k in 0..32 {
            for j in 0..32 {
                for i in 0..32 {
                    let r2 = ((i as f64 - center as f64).powi(2)
                        + (j as f64 - center as f64).powi(2)
                        + (k as f64 - center as f64).powi(2))
                        / (sigma * sigma);
                    p[[i, j, k]] = (-r2).exp();
                }
            }
        }

        let initial_max = p.iter().cloned().fold(0.0f64, f64::max);

        // Initialize Spectral solver with power law absorption
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

        // Set density to our test field (absorption acts on rho)
        solver.rho.assign(&p);

        let dt = 1e-7;
        solver.apply_absorption(dt).unwrap();

        let final_max = solver.rho.iter().cloned().fold(0.0f64, f64::max);

        // Absorption should reduce amplitude
        assert!(
            final_max < initial_max,
            "Absorption should reduce amplitude: {} → {}",
            initial_max,
            final_max
        );

        // Should reduce by measurable amount
        let reduction = 1.0 - final_max / initial_max;
        assert!(
            reduction > 0.0,
            "Absorption should reduce amplitude, got {}%",
            reduction * 100.0
        );
    }

    #[test]
    fn test_fft_absorption_energy_dissipation() {
        use crate::domain::medium::HomogeneousMedium;
        use crate::domain::source::GridSource;

        let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);

        // Create initial field
        let mut rho = Array3::zeros((16, 16, 16));
        for k in 0..16 {
            for j in 0..16 {
                for i in 0..16 {
                    rho[[i, j, k]] = (i as f64 * std::f64::consts::PI / 8.0).sin();
                }
            }
        }

        // Initialize PSTD solver
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
        solver.rho.assign(&rho);

        // Compute initial energy (L2 norm)
        let initial_energy: f64 = solver.rho.iter().map(|x| x * x).sum();

        // Apply absorption
        solver.apply_absorption(1e-7).unwrap();

        // Compute final energy
        let final_energy: f64 = solver.rho.iter().map(|x| x * x).sum();

        assert!(
            final_energy < initial_energy,
            "Absorption must dissipate energy: {} → {}",
            initial_energy,
            final_energy
        );
    }

    #[test]
    fn test_lossless_mode_no_absorption() {
        use crate::domain::medium::HomogeneousMedium;
        use crate::domain::source::GridSource;

        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1500.0, 1000.0, 0.0, 0.0, &grid);

        // Create initial field
        let rho = Array3::from_elem((32, 32, 32), 1.0);
        let rho_orig = rho.clone();

        // Initialize PSTD solver in lossless mode
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::Lossless,
            dt: 1e-7,
            ..PSTDConfig::default()
        };

        let mut solver =
            PSTDSolver::new(config, grid.clone(), &medium, GridSource::default()).unwrap();
        solver.rho.assign(&rho);

        // Apply absorption (should be no-op)
        solver.apply_absorption(1e-7).unwrap();

        // Field should be unchanged
        for (a, b) in solver.rho.iter().zip(rho_orig.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
