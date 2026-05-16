//! Phase C-2 supervised + physics-residual training loop.
//!
//! Combines two loss terms:
//!
//! * **Data loss** — supervised MSE on `(p_min, p_max, p_rms)` voxel
//!   tuples sampled from the [`KernelCube`](
//!   crate::physics::field_surrogate::KernelCube) ground truth.
//! * **Helmholtz residual loss** — finite-difference
//!   `R = ∇²p + k²p` on the predicted `p_max` channel evaluated at a
//!   set of collocation points (typically the same batch as the data
//!   loss, possibly augmented with samples drawn at random `(f0, pnp)`
//!   for outside-cube generalization).
//!
//! Both losses are computed in the same autodiff graph; one
//! `backward()` produces the combined gradient.
//!
//! ## API surface
//!
//! * [`TrainingConfig`] — hyperparameters (learning rate, loss
//!   weights, FD epsilon, sound speed).
//! * [`TrainingBatch`] — one mini-batch's worth of network inputs +
//!   data targets + per-sample physical `f0` for the Helmholtz term.
//! * [`ParamFieldPINNTrainer::step`] — one forward/backward/update step,
//!   returns the per-step loss components.
//! * [`TrainingMetrics`] — running per-epoch loss aggregates.

mod helmholtz;
mod trainer;
pub mod types;

pub use trainer::ParamFieldPINNTrainer;
pub use types::{StepMetrics, TrainingBatch, TrainingConfig, TrainingMetrics};
