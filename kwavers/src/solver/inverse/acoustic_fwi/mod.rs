//! Shared acoustic FWI adjoint-state core.
//!
//! # Contracts
//!
//! - L2 residual: `r = d_syn - d_obs`
//! - Discrete objective: `J = (dt / 2) ||r||²`
//! - Time reversal: reverse the sample axis without changing amplitudes
//! - Gradient accumulation: `G += scale * forward ⊙ adjoint`
//!
//! # Theorems
//!
//! 1. The Fréchet derivative of the discrete least-squares objective with
//!    respect to synthetic data is `d_syn - d_obs`.
//! 2. Multiplying the sum of squares by a positive timestep preserves
//!    non-negativity and the zero set.
//! 3. The imaging-condition accumulation is linear in each factor and can be
//!    expressed as a weighted Hadamard product over matching time slices.
//!
//! # Proof sketches
//!
//! 1. Differentiate `1/2 ||d_syn - d_obs||²` with respect to `d_syn`.
//! 2. The coefficient `dt / 2` is positive for valid simulations.
//! 3. A zero-lag correlation is the pointwise product integrated over time.

pub mod adjoint_state;

pub use adjoint_state::{
    accumulate_signed_correlation, l2_objective, l2_residual, reverse_time_axis,
};
