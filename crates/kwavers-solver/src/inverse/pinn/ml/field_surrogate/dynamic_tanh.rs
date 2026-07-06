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

use coeus_autograd::Var;

/// Per-layer Dynamic Tanh activation:
/// `y = γ · tanh(α · x) + β`.
///
/// All three parameters are scalars stored as length-1 `Var` leaves so
/// they participate in `coeus_autograd`'s autodiff + `coeus_optim`
/// machinery via `parameters()`/`load_parameters()`. Initialisation
/// follows Zhu 2025: `α = 0.5`, `γ = 1.0`, `β = 0.0`.
#[derive(Clone)]
pub struct DynamicTanh<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    alpha: Var<f32, B>,
    gamma: Var<f32, B>,
    beta: Var<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for DynamicTanh<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTanh").finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> DynamicTanh<B> {
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
    pub fn new() -> Self {
        Self::with_init(1.0, 1.0, 0.0)
    }

    /// Construct with caller-specified scalar initialisation values.
    /// Useful for ablations (e.g. `α = 1.0` recovers vanilla `tanh`
    /// with a learnable γ on top).
    #[must_use]
    pub fn with_init(alpha: f32, gamma: f32, beta: f32) -> Self {
        let backend = B::default();
        let leaf = |v: f32| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![1], &[v], &backend),
                true,
            )
        };
        Self {
            alpha: leaf(alpha),
            gamma: leaf(gamma),
            beta: leaf(beta),
        }
    }

    /// Flatten `(α, γ, β)` in that order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.alpha.clone(), self.gamma.clone(), self.beta.clone()]
    }

    /// Write updated `(α, γ, β)` values back (optimizer round-trip).
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        self.alpha = params[0].clone();
        self.gamma = params[1].clone();
        self.beta = params[2].clone();
    }

    /// Apply the DyT activation to a 2-D input `Var` `[batch, F]`.
    ///
    /// Broadcasting: `α, γ, β` are scalar `[1]` `Var`s; `coeus_autograd`
    /// arithmetic ops broadcast them across the `[batch, F]` shape and
    /// reduce gradients back to `[1]` on the backward pass.
    pub fn forward(&self, x: &Var<f32, B>) -> Var<f32, B> {
        let scaled = coeus_autograd::mul(x, &self.alpha);
        let activated = coeus_autograd::tanh(&scaled);
        coeus_autograd::add(&coeus_autograd::mul(&activated, &self.gamma), &self.beta)
    }

    /// Read the current scalar values — useful for logging /
    /// inspecting the learned per-layer dynamics after training.
    #[must_use]
    pub fn scalars(&self) -> (f32, f32, f32) {
        let read = |v: &Var<f32, B>| -> f32 { v.tensor.as_slice()[0] };
        (read(&self.alpha), read(&self.gamma), read(&self.beta))
    }
}
