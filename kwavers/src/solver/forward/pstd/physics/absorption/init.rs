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
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314. Eqs. 9–10, 19–21.
//! - Caputo (1967). Geophys. J. Int. 13(5), 529–539. (fractional calculus)

use super::kernel::AbsorptionKernel;
use crate::core::constants::ABSORPTION_SINGULARITY_THRESHOLD;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::acoustics::mechanics::absorption::{
    power_law_db_cm_to_np_omega_m, AbsorptionMode,
};
use crate::solver::forward::pstd::config::PSTDConfig;
use ndarray::Array3;

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
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
///
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
        AbsorptionMode::Lossless => Ok(None),
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
                            let alpha_si =
                                (4.0 * eta_s / 3.0 + eta_b) / (2.0 * rho0 * c0 * c0 * c0);
                            tau[[i, j, k]] = -2.0 * alpha_si * c0;
                        }
                    }
                }
            }

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
                        message: "alpha_power must not be 1.0 for fractional Laplacian formulation".to_owned(),
                    },
                ));
            }

            let mut tau = Array3::zeros(shape);
            let mut eta = Array3::zeros(shape);

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
            // Treeby & Cox (2010) Eq. 10; DC bin → 0 to avoid singularity.
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
                message: "Relaxation absorption modes are not supported by spectral solver".to_owned(),
            }),
        ),
    }
}
