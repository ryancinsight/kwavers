//! Dynamic Tanh (DyT) activation — `γ · tanh(α · x) + β` with
//! per-layer learnable `(α, γ, β)`.
//!
//! Replaces the fixed `x.tanh()` activations used in
//! [`super::network::ParamFieldPINNNetwork`] with a learnable
//! activation that lets each hidden layer adjust:
//!
//! * `α` — input scale. `α < 1` keeps activations in tanh's linear
//!   region (preserves amplitude — critical for fitting sharp focal
//!   peaks where a fixed `tanh()` saturates at ±1 and dampens the
//!   network's ability to predict values close to 1.0). `α > 1`
//!   saturates earlier, giving smoother gradients in the rim.
//! * `γ` — output scale. Learnable per-layer amplitude that
//!   complements the next-layer weights without redundancy because
//!   it is shared across the channel dimension.
//! * `β` — output bias. Optional, defaults to 0 (identity bias).
//!
//! ## Reference
//!
//! Zhu, J. et al. (2025). "Transformers without Normalization."
//! Meta AI / NYU.
//!
//! The original DyT replaces LayerNorm; we use the same parametric
//! form in place of activations in a coordinate-input MLP because
//! the underlying issue — fixed-scale tanh saturation degrading the
//! ability to fit peaked output distributions — applies identically
//! here. The technique is orthogonal to LayerNorm; we are not
//! removing normalisation (we have none) but augmenting tanh.

use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

/// Per-layer Dynamic Tanh activation:
/// `y = γ · tanh(α · x) + β`.
///
/// All three parameters are scalars stored as length-1 tensors so
/// they participate in Burn's autodiff + optimiser machinery via the
/// `Module` derive macro. Initialisation follows Zhu 2025: `α = 0.5`,
/// `γ = 1.0`, `β = 0.0`.
#[derive(Module, Debug)]
pub struct DynamicTanh<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> DynamicTanh<B> {
    /// Construct with `α = 1.0` (recovers vanilla `tanh`), `γ = 1.0`,
    /// `β = 0.0`. Starting from the vanilla-tanh fixed point keeps
    /// the early-training landscape identical to a plain `tanh()`
    /// network, then lets Adam learn an adjustment to α (typically
    /// downward to preserve amplitude near sharp peaks).
    ///
    /// Zhu 2025 recommends `α = 0.5` for transformers replacing
    /// LayerNorm, but for PINN regression on coordinate inputs the
    /// `α = 1.0` initialisation converges faster — the smaller-α
    /// init bakes in an extra linear-region scaling that interacts
    /// badly with the cosine-annealed LR's high-LR exploration phase.
    #[must_use]
    pub fn new(device: &B::Device) -> Self {
        Self::with_init(1.0, 1.0, 0.0, device)
    }

    /// Construct with caller-specified scalar initialisation values.
    /// Useful for ablations (e.g. `α = 1.0` recovers vanilla `tanh`
    /// with a learnable γ on top).
    #[must_use]
    pub fn with_init(alpha: f32, gamma: f32, beta: f32, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(
                Tensor::<B, 1>::from_floats([alpha], device),
            ),
            gamma: Param::from_tensor(
                Tensor::<B, 1>::from_floats([gamma], device),
            ),
            beta: Param::from_tensor(
                Tensor::<B, 1>::from_floats([beta], device),
            ),
        }
    }

    /// Apply the DyT activation to a 2-D input tensor `[batch, F]`.
    ///
    /// Broadcasting: `α, γ, β` are scalar `[1]` tensors; Burn's
    /// arithmetic ops broadcast them across the `[batch, F]` shape
    /// without an explicit reshape.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let alpha = self.alpha.val().unsqueeze::<2>(); // [1, 1]
        let gamma = self.gamma.val().unsqueeze::<2>();
        let beta = self.beta.val().unsqueeze::<2>();
        (x * alpha).tanh() * gamma + beta
    }

    /// Read the current scalar values — useful for logging /
    /// inspecting the learned per-layer dynamics after training.
    #[must_use]
    pub fn scalars(&self) -> (f32, f32, f32) {
        let read = |p: &Param<Tensor<B, 1>>| -> f32 {
            let host: Vec<f32> = p
                .val()
                .into_data()
                .convert::<f32>()
                .into_vec()
                .unwrap_or_else(|_| vec![0.0]);
            host[0]
        };
        (read(&self.alpha), read(&self.gamma), read(&self.beta))
    }
}
