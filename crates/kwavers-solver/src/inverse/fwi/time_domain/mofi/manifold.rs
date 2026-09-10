//! The manifold update and the homotopy schedules that drive it.

use super::super::{geometry::FwiGeometry, FwiProcessor};
use super::transform::{transform_template, PlaneGeometry};
use super::*;
use crate::inverse::reconstruction::seismic::MisfitType;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::{Array2, Array3};

/// Apply a rigid SE(2) transform to a model field on `grid`.
///
/// `c_φ(x) = model(T_φ⁻¹ x)` by bilinear resampling about the grid centre;
/// out-of-domain points take `background_c`. This is the same reparametrisation
/// MOFI optimises over ([`align`]); exposed for building misaligned synthetics and
/// for applying a recovered [`RigidTransform`].
///
/// # Example
/// ```no_run
/// use kwavers_solver::inverse::fwi::time_domain::{mofi_transform, RigidTransform};
/// use kwavers_grid::Grid;
/// use leto::Array3;
/// let grid = Grid::new(32, 32, 1, 1e-3, 1e-3, 1e-3).unwrap();
/// let template = Array3::from_elem([32, 32, 1], 1500.0);
/// let phi = RigidTransform { theta_rad: 6_f64.to_radians(), delta_x_m: 2e-3, delta_y_m: -1e-3 };
/// let misaligned = mofi_transform(&template, &phi, &grid, 1500.0);
/// assert_eq!(misaligned.shape(), template.shape());
/// ```
#[must_use]
pub fn transform(
    model: &Array3<f64>,
    phi: &RigidTransform,
    grid: &Grid,
    background_c: f64,
) -> Array3<f64> {
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    transform_template(model, phi, &geom, background_c)
}
/// SE(2) manifold update (paper Appendix A) in the balanced scaled space.
///
/// `dir_scaled` is a descent direction in `(L·θ, δ₁, δ₂)` space; `step` is the
/// arc length \[m\]. The rotation increment is `dθ = step·dir₀ / L` (wrapped to
/// `[−π, π]` via the log/exp maps), and the translation increment
/// `(step·dir₁, step·dir₂)` is rotated by the current rotation `R_{θ}`.
pub(super) fn manifold_update(
    phi: &RigidTransform,
    dir_scaled: [f64; 3],
    step: f64,
    length_scale: f64,
) -> RigidTransform {
    let dtheta = step * dir_scaled[0] / length_scale;
    let theta_sum = phi.theta_rad + dtheta;
    // SO(2) log∘exp = wrap to (−π, π], the shortest-geodesic representative.
    let theta_new = theta_sum.sin().atan2(theta_sum.cos());

    let sx = step * dir_scaled[1];
    let sy = step * dir_scaled[2];
    let (s, c) = phi.theta_rad.sin_cos();
    RigidTransform {
        theta_rad: theta_new,
        delta_x_m: phi.delta_x_m + (c * sx - s * sy),
        delta_y_m: phi.delta_y_m + (s * sx + c * sy),
    }
}
/// A robust-to-precise default homotopy: Wasserstein → envelope → L2.
///
/// Wasserstein (optimal transport) is convex in time shifts (good for large
/// pose error), envelope removes carrier oscillation, and L2 sharpens the final
/// fit. `config` is reused for every stage.
#[must_use]
pub fn default_homotopy(config: MofiConfig) -> [MofiStage; 3] {
    [
        MofiStage {
            misfit_type: MisfitType::Wasserstein,
            band_limit_hz: None,
            config,
        },
        MofiStage {
            misfit_type: MisfitType::Envelope,
            band_limit_hz: None,
            config,
        },
        MofiStage {
            misfit_type: MisfitType::L2Norm,
            band_limit_hz: None,
            config,
        },
    ]
}
/// Align by a warm-started misfit homotopy: run each stage in turn, threading the
/// recovered transform forward.
///
/// `processor` selects the engine (must be [`FwiEngine::SecondOrderSelfAdjoint`])
/// and the *base* configuration; per stage it is cloned with the stage's misfit
/// and band-limit. Returns the final stage's result, with `initial_misfit` taken
/// from the first stage (at `φ = 0`).
/// # Errors
/// - As [`align_from`]; also fails if `stages` is empty.
///
/// # Panics
///
/// Panics if a caller-supplied shape or an internal solver state violates
/// the precondition required by this operation.
pub fn align_homotopy(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    stages: &[MofiStage],
) -> KwaversResult<MofiResult> {
    if stages.is_empty() {
        return Err(KwaversError::InvalidInput(
            "MOFI homotopy requires at least one stage".to_owned(),
        ));
    }
    let mut phi = RigidTransform::identity();
    let mut initial_misfit = None;
    let mut last = None;
    for stage in stages {
        let staged = processor
            .clone()
            .with_misfit(stage.misfit_type)
            .with_band_limit(stage.band_limit_hz);
        let result = align_from(
            &staged,
            template,
            observed,
            geometry,
            grid,
            &stage.config,
            phi,
        )?;
        phi = result.transform;
        initial_misfit.get_or_insert(result.initial_misfit);
        last = Some(result);
    }
    let mut result = last.expect("non-empty stages produce a result");
    result.initial_misfit = initial_misfit.expect("first stage sets initial misfit");
    Ok(result)
}
/// Align `template` to `observed` acoustic data by rigid (SE(2)) manifold
/// optimisation of the FWI misfit, starting from the identity transform.
///
/// `processor` must select [`FwiEngine::SecondOrderSelfAdjoint`] so the chained
/// gradient uses the exact `∂f/∂c`. `template` is the reference sound-speed image
/// on the inversion grid; `observed` is the recorded data (recorder/Fortran row
/// order, as produced by the engine's forward model).
/// # Errors
/// - Returns [`crate::KwaversError::InvalidInput`] if the processor is not on the
///   self-adjoint engine, or [`crate::KwaversError::Validation`] on shape/geometry
///   problems; propagates forward/adjoint solve errors.
pub fn align(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &MofiConfig,
) -> KwaversResult<MofiResult> {
    align_from(
        processor,
        template,
        observed,
        geometry,
        grid,
        config,
        RigidTransform::identity(),
    )
}
