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
use super::strata::{resolve_exponent_spectrum, uniform_spectrum};
use crate::forward::pstd::config::{PSTDConfig, PowerLawDispersion};
use kwavers_core::constants::acoustic_parameters::REFERENCE_FREQUENCY_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_omega_m;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_physics::acoustics::mechanics::RelaxationAbsorption;
use leto::Array3;

/// Distance from a degenerate exponent (`y = 0`, or `y = 1` for the causal
/// form) below which an exponent counts as that value. It sits far above the
/// rounding of a value near 1 (`f64::EPSILON ≈ 2.2e-16`) and far below any
/// physically distinct tissue exponent, which differ in the second decimal.
const DEGENERATE_EXPONENT_TOL: f64 = 1e-12;

/// Initialize absorption operators τ, η, spatially-varying exponent y, and the
/// spectral nabla operators ∇^(y−2) and ∇^(y−1) on the r2c half spectrum.
///
/// ## Algorithm
/// For `AbsorptionMode::PowerLaw { alpha_coeff, alpha_power: y }`:
///
/// **Spatial coefficients** (vary per cell for heterogeneous media):
/// ```text
///   α_SI(r) = power_law_db_cm_to_np_omega_m(α_dB(r), y(r))   (caller dB path)
///   α_SI(r) = α₀(r) · (1/(2π·1 MHz))^y(r)                     (medium SI path)
///   τ(r)    = −2 α_SI(r) · c₀(r)^(y(r)−1)
///   η(r)    =  2 α_SI(r) · c₀(r)^y(r) · tan(π y(r) / 2)   (causal dispersion)
///   η(r)    =  0                                         (dispersion omitted)
/// ```
/// `y(r)` is the explicit exponent when the configuration gives a
/// coefficient, otherwise the medium's.
///
/// **Spectral operators**, from the same resolved exponent: when the range of
/// `y(r)` is within `STRATA_UNIFORM_TOL` and, under causal dispersion, moving
/// to its midpoint keeps `tan(πy/2)` within the same relative bound (it does
/// not for a range near `y = 1`), every voxel takes the midpoint `y` (for τ
/// and η too) and
/// ```text
///   nabla1[i,j,k] = |k|^(y − 2),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
///   nabla2[i,j,k] = |k|^(y − 1),  set 0 when |k| < ABSORPTION_SINGULARITY_THRESHOLD
/// ```
/// and otherwise stratified operators blended per voxel.
///
/// ## Returns
/// `Some(AbsorptionKernel)` for `PowerLaw` mode; `None` for `Lossless` (saves 4 × N³ × 8 bytes).
///
/// ## References
/// Treeby & Cox (2010) Eqs. 19–21 for τ and η; Eq. 10 for nabla operators.
/// # Errors
/// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
///
pub(crate) fn initialize_absorption_operators(
    config: &PSTDConfig,
    grid: &Grid,
    medium: &dyn Medium,
    k_mag: &Array3<f64>,
    _k_max: f64,
    _c_ref: f64,
) -> KwaversResult<Option<AbsorptionKernel>> {
    let shape = [grid.nx, grid.ny, grid.nz];

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
            let mut alpha_si_stokes = Array3::zeros(shape);
            let eta = Array3::zeros(shape);

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);
                        let c0 = medium.sound_speed(i, j, k).into_base();
                        let rho0 = medium.density(i, j, k).into_base();
                        let eta_s = medium.shear_viscosity(x, y_coord, z, grid).into_base();
                        let eta_b = medium.bulk_viscosity(x, y_coord, z, grid).into_base();

                        if rho0 > 0.0 && c0 > 0.0 {
                            let a_si = (4.0 * eta_s / 3.0 + eta_b) / (2.0 * rho0 * c0 * c0 * c0);
                            tau[[i, j, k]] = -2.0 * a_si * c0;
                            alpha_si_stokes[[i, j, k]] = a_si;
                        }
                    }
                }
            }

            Ok(Some(AbsorptionKernel {
                tau,
                eta,
                alpha_si: alpha_si_stokes,
                // Stokes ⇒ y = 2 everywhere: |k|^0 = 1 and |k|^1 = |k| exactly.
                spectrum: uniform_spectrum(2.0, k_mag),
            }))
        }
        AbsorptionMode::PowerLaw {
            alpha_coeff,
            alpha_power,
        } => {
            let y_config = *alpha_power;
            let dispersive = config.power_law_dispersion == PowerLawDispersion::Causal;
            if dispersive && (y_config - 1.0).abs() < DEGENERATE_EXPONENT_TOL {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "alpha_power must not be 1.0 with causal dispersion, where \
                                  tan(pi*y/2) is singular; set PowerLawDispersion::Omitted \
                                  to absorb without dispersion"
                            .to_owned(),
                    },
                ));
            }

            // Pass 1 resolves each voxel's power law; the exponent field then
            // fixes the spectrum, whose exponent pass 2 uses for τ and η so the
            // coefficients and the operators share one y per voxel.
            let mut alpha_db_field = Array3::zeros(shape);
            let mut alpha_np_field = Array3::zeros(shape);
            let mut y_field = Array3::from_elem(shape, y_config);

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);

                        // The power law's two parameters are resolved
                        // together, from one owner. An explicit request applies
                        // uniformly and outranks whatever the medium reports;
                        // absent one the medium owns both, read here per voxel,
                        // which is what preserves a heterogeneous medium's
                        // spatial variation.
                        //
                        // Splitting them is what the old rule did, and it does
                        // not survive contact with `alpha(f) = alpha_0 f^y`:
                        // taking the coefficient from the caller and the
                        // exponent from the medium evaluates neither party's
                        // power law. Both resolutions also keyed on a sentinel
                        // (`!= 0.0`, `!= 1.0`), so a homogeneous medium's
                        // defaulted water values silently outranked a caller's
                        // request on both (ADR 120).
                        // The `Some` arm carries the caller's k-Wave dB prefactor;
                        // the `None` arm carries the medium's SI prefactor in
                        // Np/m at 1 MHz. Each converts to the spectral Np/ω^y/m
                        // coefficient on its own path; the two unit domains
                        // meet only as `alpha_0_si`.
                        let (alpha_db_cm, alpha_np_m, y) = match alpha_coeff {
                            Some(requested) => (*requested, 0.0, y_config),
                            None => {
                                let y_medium = medium.alpha_power(x, y_coord, z, grid);
                                // y = 1 is degenerate only for the causal form.
                                let degenerate = y_medium.abs() <= DEGENERATE_EXPONENT_TOL
                                    || (dispersive
                                        && (y_medium - 1.0).abs() <= DEGENERATE_EXPONENT_TOL);
                                let y = if !degenerate { y_medium } else { y_config };
                                (
                                    0.0,
                                    medium.alpha_coefficient(x, y_coord, z, grid).into_base(),
                                    y,
                                )
                            }
                        };
                        alpha_db_field[[i, j, k]] = alpha_db_cm;
                        alpha_np_field[[i, j, k]] = alpha_np_m;
                        y_field[[i, j, k]] = y;
                    }
                }
            }

            // Uniform (k-Wave-equivalent) operators at the midpoint of y(x)'s
            // range when it collapses to one exponent; stratified operators
            // reproducing each voxel's exponent otherwise.
            let spectrum = resolve_exponent_spectrum(&y_field, k_mag, config.power_law_dispersion);

            let mut tau = Array3::zeros(shape);
            let mut eta = Array3::zeros(shape);
            let mut alpha_si_field = Array3::zeros(shape);
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let y = spectrum.exponent_at([i, j, k]);
                        let c0_val = medium.sound_speed(i, j, k).into_base();
                        let alpha_0_si = if alpha_coeff.is_some() {
                            power_law_db_cm_to_np_omega_m(alpha_db_field[[i, j, k]], y)
                        } else {
                            alpha_np_field[[i, j, k]]
                                * (1.0 / (TWO_PI * REFERENCE_FREQUENCY_HZ)).powf(y)
                        };

                        // Treeby & Cox (2010) Eqs. 19–20
                        tau[[i, j, k]] = -2.0 * alpha_0_si * c0_val.powf(y - 1.0);
                        if dispersive {
                            eta[[i, j, k]] = 2.0
                                * alpha_0_si
                                * c0_val.powf(y)
                                * (std::f64::consts::PI * y / 2.0).tan();
                        }
                        alpha_si_field[[i, j, k]] = alpha_0_si;
                    }
                }
            }

            Ok(Some(AbsorptionKernel {
                tau,
                eta,
                alpha_si: alpha_si_field,
                spectrum,
            }))
        }
        AbsorptionMode::MultiRelaxation { tau, weights } => {
            let relaxation = RelaxationAbsorption::new(tau.clone(), weights.clone())
                .ok_or_else(invalid_relax)?;
            build_relaxation_kernel(grid, medium, k_mag, &relaxation, None)
        }
        AbsorptionMode::Causal {
            relaxation_times,
            alpha_0,
        } => {
            let relaxation = RelaxationAbsorption::unit_weights(relaxation_times.clone())
                .ok_or_else(invalid_relax)?;
            build_relaxation_kernel(grid, medium, k_mag, &relaxation, Some(*alpha_0))
        }
    }
}

/// Error for an ill-formed relaxation spectrum.
fn invalid_relax() -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: "relaxation absorption requires non-empty, positive (τ, weight) pairs".to_owned(),
    })
}

/// Keep a realized power-law exponent away from the `y = 1` singularity of the
/// fractional dispersion term `tan(πy/2)` and inside the model's valid `(0, 2)`
/// range. Relaxation media operating exactly at a loss peak (`y_eff ≈ 1`) are
/// nudged to the nearest side; the `±0.05` band bounds the realization error.
fn clamp_fractional_exponent(y: f64) -> f64 {
    // y = 2 (viscous/Stokes) is valid: tan(π) = 0 ⇒ no dispersion term.
    let y = y.clamp(0.1, 2.0);
    if (y - 1.0).abs() < 0.05 {
        if y >= 1.0 {
            1.05
        } else {
            0.95
        }
    } else {
        y
    }
}

/// Realize a relaxation absorption spectrum through the fractional-Laplacian
/// path at the medium drive frequency. `α(ω_ref)` (or `causal_alpha_0` when set)
/// and the spectrum's local exponent `y_eff = d ln α/d ln ω` define an equivalent
/// power-law prefactor per voxel; the exponent is `c₀`-independent, so a single
/// spectral symbol suffices (no strata). Exact at `ω_ref` (book §4.4.2/§4.4.3).
fn build_relaxation_kernel(
    grid: &Grid,
    medium: &dyn Medium,
    k_mag: &Array3<f64>,
    relaxation: &RelaxationAbsorption,
    causal_alpha_0: Option<f64>,
) -> KwaversResult<Option<AbsorptionKernel>> {
    let shape = [grid.nx, grid.ny, grid.nz];
    let f_ref = medium.reference_frequency().into_base();
    let omega_ref = 2.0 * std::f64::consts::PI * f_ref;
    let y_eff = clamp_fractional_exponent(relaxation.local_exponent(omega_ref));

    let mut tau = Array3::zeros(shape);
    let mut eta = Array3::zeros(shape);
    let mut alpha_si_field = Array3::zeros(shape);

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let c0_val = medium.sound_speed(i, j, k).into_base();
                let alpha_np_m =
                    causal_alpha_0.unwrap_or_else(|| relaxation.attenuation(omega_ref, c0_val));
                // The dB round trip (Np/m → dB/cm → Np/ω^y/m) is exactly
                // α_np_m / ω_ref^y; write it directly in SI.
                let alpha_0_si = alpha_np_m / omega_ref.powf(y_eff);

                tau[[i, j, k]] = -2.0 * alpha_0_si * c0_val.powf(y_eff - 1.0);
                eta[[i, j, k]] = 2.0
                    * alpha_0_si
                    * c0_val.powf(y_eff)
                    * (std::f64::consts::PI * y_eff / 2.0).tan();
                alpha_si_field[[i, j, k]] = alpha_0_si;
            }
        }
    }

    Ok(Some(AbsorptionKernel {
        tau,
        eta,
        alpha_si: alpha_si_field,
        // y_eff is c₀-independent ⇒ spatially uniform exponent
        spectrum: uniform_spectrum(y_eff, k_mag),
    }))
}
