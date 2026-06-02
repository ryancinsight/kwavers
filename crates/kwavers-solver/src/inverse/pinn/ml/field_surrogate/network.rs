//! Burn-based MLP for the parameterised field-surrogate PINN.
//!
//! Input: `(x_norm, y_norm, z_norm, f0_norm, pnp_norm)` — shape
//! `[batch, 5]`. Output: `(p_min_norm, p_max_norm, p_rms_norm)` —
//! shape `[batch, 3]`. All inputs and outputs are caller-normalised
//! to `[-1, 1]` to match `tanh` activation dynamics.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

use kwavers_core::error::KwaversResult;

use super::config::{ParamFieldPINNConfig, INPUT_DIM, OUTPUT_DIM};
use super::dynamic_tanh::DynamicTanh;

/// Parameterised field-surrogate PINN network.
///
/// Mirrors the structure of `PINN3DNetwork` from
/// `burn_wave_equation_3d` but with five input dimensions and three
/// output dimensions. The hidden stack uses **Dynamic Tanh (DyT)**
/// activations (`γ · tanh(α · x) + β`, Zhu 2025) — `α`, `γ`, `β` are
/// per-layer learnable scalars that let the network adjust tanh
/// saturation per-layer, closing the focal-peak underprediction that
/// the fixed-`tanh` baseline plateaus on. The output layer is linear
/// (regression head).
#[derive(Module, Debug)]
pub struct ParamFieldPINNNetwork<B: Backend> {
    input_layer: Linear<B>,
    input_act: DynamicTanh<B>,
    hidden_layers: Vec<Linear<B>>,
    hidden_acts: Vec<DynamicTanh<B>>,
    output_layer: Linear<B>,
}

impl<B: Backend> ParamFieldPINNNetwork<B> {
    /// Initialise a fresh network with random weights on `device`.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` via
    /// [`ParamFieldPINNConfig::validate`] when the config is malformed
    /// (empty hidden stack, zero-width layer, non-positive lr, etc.).
    pub fn new(config: &ParamFieldPINNConfig, device: &B::Device) -> KwaversResult<Self> {
        config.validate()?;

        let first_hidden = config.hidden_layers[0];
        let input_layer = LinearConfig::new(INPUT_DIM, first_hidden).init(device);
        let input_act = DynamicTanh::new(device);

        let mut hidden_layers = Vec::with_capacity(config.hidden_layers.len().saturating_sub(1));
        let mut hidden_acts = Vec::with_capacity(config.hidden_layers.len().saturating_sub(1));
        for window in config.hidden_layers.windows(2) {
            let &[in_features, out_features] = window else {
                continue;
            };
            hidden_layers.push(LinearConfig::new(in_features, out_features).init(device));
            hidden_acts.push(DynamicTanh::new(device));
        }

        let last_hidden = *config
            .hidden_layers
            .last()
            .expect("validate() guarantees non-empty hidden_layers");
        let output_layer = LinearConfig::new(last_hidden, OUTPUT_DIM).init(device);

        Ok(Self {
            input_layer,
            input_act,
            hidden_layers,
            hidden_acts,
            output_layer,
        })
    }

    /// Read the learned DyT scalars per layer — `Vec<(α, γ, β)>` in
    /// activation order (input act first, then hidden acts).
    /// Useful for inspecting what the network has converged on
    /// (e.g. layers near the output typically learn `α > 1` to
    /// saturate harder; layers near the input often learn `α < 1`
    /// to preserve amplitude).
    #[must_use]
    pub fn dyt_scalars(&self) -> Vec<(f32, f32, f32)> {
        let mut out = Vec::with_capacity(1 + self.hidden_acts.len());
        out.push(self.input_act.scalars());
        for act in &self.hidden_acts {
            out.push(act.scalars());
        }
        out
    }

    /// Number of intermediate hidden layers (excluding input + output).
    #[must_use]
    pub fn hidden_layer_count(&self) -> usize {
        self.hidden_layers.len()
    }

    /// Forward pass on a batch of pre-concatenated inputs.
    ///
    /// `input` shape `[batch, 5]`, output shape `[batch, 3]`.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = self.input_act.forward(self.input_layer.forward(input));
        for (layer, act) in self.hidden_layers.iter().zip(self.hidden_acts.iter()) {
            h = act.forward(layer.forward(h));
        }
        // Linear output (no activation): regression head; caller maps
        // the [-1, 1]-normalised output back to physical units using
        // the per-channel scale factors stored alongside the model.
        self.output_layer.forward(h)
    }

    /// Convenience overload accepting the five input columns
    /// separately. Concatenates and dispatches to `forward`.
    pub fn forward_xyz_params(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        f0: Tensor<B, 2>,
        pnp: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![x, y, z, f0, pnp], 1);
        self.forward(input)
    }
}
