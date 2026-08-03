//! Manifold Optimisation for FWI (MOFI): guidance-free rigid alignment of a
//! sound-speed template to acoustic data (Bates et al., *Ultrasound in Medicine
//! & Biology*, 2026, "Automatic Skull-Template Alignment Without a Guidance
//! Image").
//!
//! # Idea
//! Transcranial FWI needs a CT-derived skull template aligned to the patient.
//! Standard FWI updates the model `c` pixel-wise; MOFI instead reparametrises the
//! model as a **rigid-body (SE(2)) transform** of the template,
//! `φ = {θ, δ₁, δ₂}`, and minimises the *acoustic* misfit over just those three
//! parameters — no MRI guidance image. The chained gradient is
//! ```text
//! ∂f/∂φ = (∂c_φ/∂φ)ᵀ ∂f/∂c
//! ```
//! where `∂f/∂c` is the standard FWI adjoint-state gradient and `∂c_φ/∂φ` is the
//! analytic Jacobian of the rigid reparametrisation ([`transform`]).
//!
//! # Manifold update (paper Appendix A)
//! Updates respect the SE(2) Lie-group geometry: the rotation is updated through
//! the SO(2) log/exp maps (which keeps `θ ∈ [−π, π]` and follows the shortest
//! geodesic), and the translation increment is rotated by the current rotation
//! before being added, `δ^{k+1} = δ^k + R_{θ^k} Δ_δ`. Optimisation uses
//! gradient normalisation and an Armijo line search (the paper found explicit
//! line search most stable).
//!
//! # Exact gradient
//! MOFI's accuracy rests on a faithful `∂f/∂c`. This driver uses the self-adjoint
//! engine ([`super::FwiEngine::SecondOrderSelfAdjoint`], ADR 016), whose gradient
//! is the exact discrete `∂f/∂c` (`κ ≈ 1`); the processor passed to [`align`]
//! must select that engine.
//!
//! Parameters are balanced for the line search by optimising in the scaled space
//! `(L·θ, δ₁, δ₂)` where `L` is the domain half-width — a rotation moves the
//! template edge by `≈ L·dθ`, comparable to a translation `dδ`.

mod config;
mod nonrigid;
mod ops;
mod transform;

#[cfg(test)]
mod tests;

pub use config::*;
pub use nonrigid::{align_nonrigid, sample_displacement, FfdBasis, FfdConfig, FfdField, FfdResult};
pub use ops::{
    align, align_from, align_homotopy, align_pipeline, align_with_calibration, coarse_pose_search,
    default_homotopy, recommended_search_misfit, transform,
};
pub use transform::RigidTransform;
