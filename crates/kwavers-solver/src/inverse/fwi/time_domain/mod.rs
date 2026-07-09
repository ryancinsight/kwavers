//! Full Waveform Inversion implementation.
//!
//! # Specification
//!
//! For the acoustic least-squares objective
//!
//! ```text
//! J(c) = (dt / 2) Σ_{r,t} (d_syn(r,t;c) - d_obs(r,t))²
//! ```
//!
//! the reduced gradient is obtained by the adjoint-state identity
//!
//! ```text
//! ∂J/∂m(x) = -∫_0^T λ(x,T-t) ∂²p(x,t)/∂t² dt,     m = c⁻²
//! ∂J/∂c(x) = -2 c(x)⁻³ ∂J/∂m(x)
//! ```
//!
//! The discrete implementation follows the k-Wave time-reversal convention:
//! the residual is reversed in time and injected through the same receiver mask
//! used for data acquisition.
//!
//! # Theorems
//!
//! 1. **L2 residual theorem.** The Fréchet derivative of `J` with respect to
//!    the data is `d_syn - d_obs`. This fixes the sign of the adjoint source.
//! 2. **Time-reversal (approximate adjoint).** Injecting the reversed residual
//!    on the receiver mask produces an *approximate* discrete adjoint wavefield
//!    for the acoustic linearized operator. It is an exact discrete transpose
//!    only when the source injection, receiver sampling, spatial stencil, and
//!    absorbing boundary are each self-adjoint; the present FDTD/PSTD path is
//!    not (the additive source carries the k-Wave `2·dt·c₀/(N·dx)` scaling while
//!    receivers sample pressure directly, and the PML/leapfrog stepping is not
//!    self-adjoint). The resulting gradient is therefore correct in *direction*
//!    (a valid descent direction, sign-verified) but carries a global scale
//!    offset (~200×) and a mild direction-dependent shape error (~20%), both
//!    absorbed by the line search. See
//!    `tests::gradient::test_fwi_adjoint_gradient_is_valid_descent_direction`.
//!    For the **exact** gradient (`κ ≈ 1`), select
//!    [`FwiEngine::SecondOrderSelfAdjoint`] (the dedicated self-adjoint
//!    second-order engine, ADR 016) instead of the default [`FwiEngine::Solver`].
//! 3. **Chain-rule theorem.** The sound-speed gradient follows from
//!    `m = c⁻²` by `dm/dc = -2 c⁻³`.
//!
//! # References
//! - Tarantola (1984): *Inversion of seismic reflection data in the acoustic approximation*
//! - Plessix (2006): *A review of the adjoint-state method for computing the gradient of a functional*
//! - Virieux & Operto (2009): *An overview of full-waveform inversion in exploration geophysics*
//!
//! # Sign convention for model updates
//!
//! `adjoint_model` returns `g = +∂J/∂c`. Descent: `c_new = c − step × g`.
//!
//! # Module layout
//!
//! - `geometry`: `FwiGeometry` struct and index-mapping helpers.
//! - `forward`: FDTD solver construction, forward-model runs, `generate_synthetic_data`.
//! - `adjoint`: adjoint FDTD run, adjoint-source construction, L2 residual/objective.
//! - `gradient`: gradient smoothing, regularization, near-source mute, TV/Laplacian helpers.
//! - `constraints`: CFL validation, model clamping, pressure second-derivative.
//! - `inversion`: `invert`, `invert_multi_source`, `invert_multi_source_masked`, shot-gradient dispatch.
//! - `search`: line-search and joint-objective helpers.

mod adjoint;
pub mod adjoint_state;
mod constraints;
mod encoded_source;
mod field_ops;
mod forward;
mod frequency_continuation;
mod gradient;
mod inversion;
pub mod mofi;
mod search;
mod self_adjoint;

pub mod geometry;

pub use adjoint_state::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
pub use encoded_source::{encode_shots, hadamard_codes};
pub use geometry::FwiGeometry;
pub use mofi::{
    align as mofi_align, align_from as mofi_align_from, align_homotopy as mofi_align_homotopy,
    align_nonrigid as mofi_align_nonrigid, align_pipeline as mofi_align_pipeline,
    align_with_calibration as mofi_align_with_calibration,
    coarse_pose_search as mofi_coarse_pose_search, default_homotopy as mofi_default_homotopy,
    recommended_search_misfit as mofi_recommended_search_misfit,
    sample_displacement as mofi_sample_displacement, transform as mofi_transform,
    CoarseSearchConfig as MofiCoarseSearchConfig, FfdBasis as MofiFfdBasis,
    FfdConfig as MofiFfdConfig, FfdField as MofiFfdField, FfdResult as MofiFfdResult,
    MofiCalibratedResult, MofiConfig, MofiResult, MofiStage, PipelineConfig as MofiPipelineConfig,
    PipelineResult as MofiPipelineResult, RigidTransform,
};

#[cfg(test)]
mod tests;

use crate::inverse::reconstruction::seismic::{DataWeighting, MisfitType};
use crate::inverse::seismic::parameters::FwiParameters;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::Array3;

/// Reference density for seismic-tradition acoustic FWI [kg/m³].
///
/// Gardner et al. (1974) relate seismic velocity to density via ρ = a·Vᵇ
/// (a = 310, b = 0.25 for consolidated sedimentary rock). Uniform value
/// consistent with typical upper-crust consolidated sediments (~2000 kg/m³).
/// Used by the acoustic FWI driver when a heterogeneous-density model is
/// not supplied; replaced via medium construction when one is.
///
/// Reference: Gardner, G.H.F. et al. (1974). Geophysics 39(6), 770–780.
pub(super) const RHO_SEISMIC_REF: f64 = 2000.0; // kg/m³

pub(super) const FWI_FIELD_CHUNK: usize = 16 * 1024;

/// Forward/adjoint engine backing a [`FwiProcessor`].
///
/// The default [`FwiEngine::Solver`] drives the shared FDTD/PSTD
/// `Box<dyn Solver>` and produces a correct *descent direction* but an
/// **approximate** adjoint (global scale offset + ~20% shape error; the Armijo
/// line search absorbs both). [`FwiEngine::SecondOrderSelfAdjoint`] uses the
/// dedicated self-adjoint second-order acoustic engine
/// ([`self_adjoint`]) whose discrete adjoint is the literal algebraic gradient
/// of the discrete objective, so the finite-difference gradient test returns
/// `κ ≈ 1`. Choose it when the absolute gradient magnitude matters
/// (Gauss-Newton, Hessian-vector products, fixed-step updates, gradient-norm
/// stopping). See ADR 016.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FwiEngine {
    /// Shared FDTD/PSTD `Box<dyn Solver>` (approximate adjoint; default).
    #[default]
    Solver,
    /// Self-adjoint second-order acoustic engine (exact gradient, `κ ≈ 1`).
    SecondOrderSelfAdjoint,
}

/// Full Waveform Inversion processor (time-domain, FDTD-driven).
///
/// ## Density handling
///
/// `density_model` is an optional spatially varying density field [kg/m³]
/// matching the velocity-model grid. When supplied, it is used by both the
/// forward FDTD medium and the adjoint medium, and the local value
/// `ρ(x)` enters the gradient scaling per Plessix (2006) eq. (12):
///
/// ```text
/// g_c(x) = -(2 / (ρ(x) · c(x)³)) · ∫₀ᵀ p̈_fwd(x, t) · λ(x, t) dt
/// ```
///
/// When `density_model` is `None`, the processor falls back to the constant
/// [`RHO_SEISMIC_REF`] for both media and the gradient scaling — a global
/// scalar that is absorbed into the line-search step-size and therefore
/// produces a descent direction identical to the heterogeneous formulation
/// up to a uniform rescaling.
///
/// ## Misfit selection and cycle-skipping
///
/// `misfit_type` selects the data-misfit functional `J(d_syn, d_obs)` and the
/// matching adjoint source (Fréchet derivative `∂J/∂d_syn`) injected through the
/// receiver mask. The default [`MisfitType::L2Norm`] is the classical
/// least-squares functional `J = (dt/2)‖d_syn − d_obs‖²`, whose non-convexity in
/// the model is the source of *cycle skipping*: when the initial model
/// mispredicts an arrival by more than half a period, the L2 gradient pushes
/// toward the nearest cycle rather than the true one (Virieux & Operto 2009 §4).
///
/// The alternative misfits trade pointwise amplitude matching for a more convex
/// objective with respect to time shifts:
/// - [`MisfitType::Envelope`] — instantaneous-envelope L2 (Bozdağ et al. 2011),
/// - [`MisfitType::Phase`] — instantaneous-phase L2 (Fichtner et al. 2008),
/// - [`MisfitType::Wasserstein`] — 1-Wasserstein optimal transport
///   (Engquist & Froese 2014; Métivier et al. 2016), convex in the time shift
///   for *positive* distributions; on raw oscillatory traces the
///   non-negativity transform weakens that guarantee, so it is most effective
///   on envelope- or amplitude-like quantities,
/// - [`MisfitType::Correlation`], [`MisfitType::L1Norm`] — auxiliary norms.
///
/// All adjoint sources are computed by the canonical
/// [`MisfitFunction`](crate::inverse::reconstruction::seismic::MisfitFunction)
/// dispatcher (SSOT); the time-domain FWI driver only routes the selected
/// functional into the objective evaluation and the adjoint-source construction.
#[derive(Debug, Clone)]
pub struct FwiProcessor {
    pub(super) parameters: FwiParameters,
    pub(super) density_model: Option<Array3<f64>>,
    pub(super) misfit_type: MisfitType,
    /// Optional zero-phase low-pass corner [Hz] applied to both observed and
    /// synthetic traces before the misfit/adjoint evaluation. `None` (default)
    /// uses the full recorded bandwidth. Set by the frequency-continuation
    /// driver per multiscale stage; see [`Self::with_band_limit`] and
    /// [`Self::invert_multiscale`].
    pub(super) band_limit_hz: Option<f64>,
    /// Forward/adjoint engine. Default [`FwiEngine::Solver`] (approximate
    /// adjoint); [`FwiEngine::SecondOrderSelfAdjoint`] for the exact gradient.
    pub(super) engine: FwiEngine,
    /// Optional self-adjoint absorbing sponge `b(x) ≥ 0` for the
    /// [`FwiEngine::SecondOrderSelfAdjoint`] engine. `None` (default) uses
    /// lossless reflecting boundaries; a symmetric diagonal sponge keeps the
    /// discrete adjoint exact (`κ ≈ 1`) while absorbing outgoing waves.
    pub(super) sa_damping: Option<Array3<f64>>,
    /// Data-fidelity weighting for the L2 misfit (the low-dose-CT PWLS lesson).
    /// Default [`DataWeighting::Uniform`] is plain least squares;
    /// [`DataWeighting::InverseNoiseVariance`] weights each trace by its inverse
    /// noise variance (MBIR/PWLS). Applied only to [`MisfitType::L2Norm`].
    pub(super) data_weighting: DataWeighting,
}

impl FwiProcessor {
    /// Create new FWI processor with specified parameters, constant reference
    /// density, the default [`MisfitType::L2Norm`] data misfit, and full
    /// bandwidth.
    #[must_use]
    pub fn new(parameters: FwiParameters) -> Self {
        Self {
            parameters,
            density_model: None,
            misfit_type: MisfitType::L2Norm,
            band_limit_hz: None,
            engine: FwiEngine::default(),
            sa_damping: None,
            data_weighting: DataWeighting::default(),
        }
    }

    /// Select the data-fidelity weighting for the L2 misfit (the low-dose-CT PWLS
    /// lesson). [`DataWeighting::Uniform`] (default) is plain least squares;
    /// [`DataWeighting::InverseNoiseVariance`] weights each trace by its inverse
    /// noise variance — the maximum-likelihood estimator under heteroscedastic
    /// noise, and the FWI analogue of CT model-based iterative reconstruction
    /// (Sauer & Bouman 1993; Thibault et al. 2007). Applies to
    /// [`MisfitType::L2Norm`]; other misfits ignore it.
    #[must_use]
    pub fn with_data_weighting(mut self, weighting: DataWeighting) -> Self {
        self.data_weighting = weighting;
        self
    }

    /// Supply a self-adjoint absorbing sponge `b(x) ≥ 0` for the
    /// [`FwiEngine::SecondOrderSelfAdjoint`] engine (ignored by the default
    /// solver engine). Must match the grid shape and be finite and non-negative.
    ///
    /// ## Errors
    /// Returns [`KwaversError::Validation`] if any value is negative or non-finite.
    pub fn with_self_adjoint_damping(mut self, damping: Array3<f64>) -> KwaversResult<Self> {
        for &b in damping.iter() {
            if !b.is_finite() || b < 0.0 {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!("self-adjoint damping must be finite and ≥ 0; got {b}"),
                    },
                ));
            }
        }
        self.sa_damping = Some(damping);
        Ok(self)
    }

    /// Select the forward/adjoint engine.
    ///
    /// [`FwiEngine::SecondOrderSelfAdjoint`] yields the exact discrete gradient
    /// (`κ ≈ 1`); [`FwiEngine::Solver`] (default) is the shared FDTD/PSTD path
    /// (approximate adjoint, correct descent direction). See ADR 016.
    #[must_use]
    pub fn with_engine(mut self, engine: FwiEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Select the data-misfit functional used by the inversion driver.
    ///
    /// Replaces the default L2 least-squares misfit with a cycle-skipping-robust
    /// alternative (envelope, phase, or Wasserstein optimal transport). The same
    /// functional is used consistently for objective evaluation, the convergence
    /// test, the Armijo line search, and the adjoint-source construction.
    #[must_use]
    pub fn with_misfit(mut self, misfit_type: MisfitType) -> Self {
        self.misfit_type = misfit_type;
        self
    }

    /// Apply a zero-phase low-pass with the given corner frequency [Hz] to the
    /// data before every misfit and adjoint-source evaluation.
    ///
    /// `None` restores full-bandwidth inversion. This is the primitive the
    /// multiscale [`Self::invert_multiscale`] driver toggles per stage; setting
    /// it directly inverts a single band.
    #[must_use]
    pub fn with_band_limit(mut self, corner_hz: Option<f64>) -> Self {
        self.band_limit_hz = corner_hz;
        self
    }

    /// Supply a heterogeneous density field [kg/m³] used by both the forward
    /// FDTD medium and the adjoint medium, and applied as a local `1/ρ(x)`
    /// factor in the velocity-model gradient.
    ///
    /// ## Errors
    /// Returns [`KwaversError::Validation`] if any density value is non-finite
    /// or non-positive.
    pub fn with_density(mut self, density: Array3<f64>) -> KwaversResult<Self> {
        for &value in density.iter() {
            if !value.is_finite() || value <= 0.0 {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "FWI density model must be finite and positive; got {value}"
                        ),
                    },
                ));
            }
        }
        self.density_model = Some(density);
        Ok(self)
    }

    /// Resolve the density field used for forward / adjoint medium
    /// construction. Returns the caller-supplied heterogeneous field when
    /// present, validated against the grid; otherwise returns a constant
    /// [`RHO_SEISMIC_REF`] field of the requested shape.
    pub(super) fn resolved_density(&self, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = grid.dimensions();
        match self.density_model.as_ref() {
            Some(density) => {
                if density.shape() != [nx, ny, nz] {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: format!(
                                "FWI density model shape {:?} does not match grid {:?}",
                                density.shape(),
                                (nx, ny, nz)
                            ),
                        },
                    ));
                }
                Ok(density.clone())
            }
            None => Ok(Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF)),
        }
    }
}

impl Default for FwiProcessor {
    fn default() -> Self {
        Self::new(FwiParameters::default())
    }
}

#[cfg(test)]
impl FwiProcessor {
    /// Read back the configured misfit type (test introspection of the wiring).
    #[must_use]
    pub(super) fn misfit_type(&self) -> MisfitType {
        self.misfit_type
    }
}
