//! Coeus-based MLP for the parameterised field-surrogate PINN.
//!
//! Input: `(x_norm, y_norm, z_norm, f0_norm, pnp_norm)` — shape
//! `[batch, 5]`. Output: `(p_min_norm, p_max_norm, p_rms_norm)` —
//! shape `[batch, 3]`. All inputs and outputs are caller-normalised
//! to `[-1, 1]` to match `tanh` activation dynamics.

use coeus_autograd::Var;
use coeus_nn::{Linear, Module};

use kwavers_core::error::KwaversResult;

use super::config::{ParamFieldPINNConfig, INPUT_DIM, OUTPUT_DIM};
use super::dynamic_tanh::DynamicTanh;

/// Parameterised field-surrogate PINN network.
///
/// Mirrors the structure of `PINN3DNetwork` from
/// `wave_equation_3d` but with five input dimensions and three
/// output dimensions. The hidden stack uses **Dynamic Tanh (DyT)**
/// activations (`γ · tanh(α · x) + β`, Zhu 2025) — `α`, `γ`, `β` are
/// per-layer learnable scalars that let the network adjust tanh
/// saturation per-layer, closing the focal-peak underprediction that
/// the fixed-`tanh` baseline plateaus on. The output layer is linear
/// (regression head).
pub struct ParamFieldPINNNetwork<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    input_layer: Linear<f32, B>,
    input_act: DynamicTanh<B>,
    hidden_layers: Vec<Linear<f32, B>>,
    hidden_acts: Vec<DynamicTanh<B>>,
    output_layer: Linear<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> Clone
    for ParamFieldPINNNetwork<B>
{
    fn clone(&self) -> Self {
        Self {
            input_layer: self.input_layer.clone(),
            input_act: self.input_act.clone(),
            hidden_layers: self.hidden_layers.clone(),
            hidden_acts: self.hidden_acts.clone(),
            output_layer: self.output_layer.clone(),
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ParamFieldPINNNetwork<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamFieldPINNNetwork")
            .field("hidden_layer_count", &(self.hidden_layers.shape()[0] * self.hidden_layers.shape()[1] * self.hidden_layers.shape()[2]))
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ParamFieldPINNNetwork<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Initialise a fresh network with random weights.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` via
    /// [`ParamFieldPINNConfig::validate`] when the config is malformed
    /// (empty hidden stack, zero-width layer, non-positive lr, etc.).
    pub fn new(config: &ParamFieldPINNConfig) -> KwaversResult<Self> {
        config.validate()?;

        let first_hidden = config.hidden_layers[0];
        let input_layer = Linear::new(INPUT_DIM, first_hidden, true);
        let input_act = DynamicTanh::new();

        let mut hidden_layers = Vec::with_capacity((config.hidden_layers.shape()[0] * config.hidden_layers.shape()[1] * config.hidden_layers.shape()[2]).saturating_sub(1));
        let mut hidden_acts = Vec::with_capacity((config.hidden_layers.shape()[0] * config.hidden_layers.shape()[1] * config.hidden_layers.shape()[2]).saturating_sub(1));
        for window in config.hidden_layers.windows(2) {
            let &[in_features, out_features] = window else {
                continue;
            };
            hidden_layers.push(Linear::new(in_features, out_features, true));
            hidden_acts.push(DynamicTanh::new());
        }

        let last_hidden = *config
            .hidden_layers
            .last()
            .expect("validate() guarantees non-empty hidden_layers");
        let output_layer = Linear::new(last_hidden, OUTPUT_DIM, true);

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
    #[must_use]
    pub fn dyt_scalars(&self) -> Vec<(f32, f32, f32)> {
        let mut out = Vec::with_capacity(1 + (self.hidden_acts.shape()[0] * self.hidden_acts.shape()[1] * self.hidden_acts.shape()[2]));
        out.push(self.input_act.scalars());
        for act in &self.hidden_acts {
            out.push(act.scalars());
        }
        out
    }

    /// Number of intermediate hidden layers (excluding input + output).
    #[must_use]
    pub fn hidden_layer_count(&self) -> usize {
        (self.hidden_layers.shape()[0] * self.hidden_layers.shape()[1] * self.hidden_layers.shape()[2])
    }

    /// Flatten all layer + activation parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.input_layer.parameters();
        params.extend(self.input_act.parameters());
        for (layer, act) in self.hidden_layers.iter().zip(self.hidden_acts.iter()) {
            params.extend(layer.parameters());
            params.extend(act.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    /// Write updated parameter values back into the network's layers.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        let n = self.input_layer.parameters().len();
        self.input_layer
            .load_parameters(&params[offset..offset + n]);
        offset += n;
        let n = self.input_act.parameters().len();
        self.input_act.load_parameters(&params[offset..offset + n]);
        offset += n;
        for (layer, act) in self
            .hidden_layers
            .iter_mut()
            .zip(self.hidden_acts.iter_mut())
        {
            let n = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + n]);
            offset += n;
            let n = act.parameters().len();
            act.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.output_layer.parameters().len();
        self.output_layer
            .load_parameters(&params[offset..offset + n]);
    }

    /// Forward pass on a batch of pre-concatenated inputs.
    ///
    /// `input` shape `[batch, 5]`, output shape `[batch, 3]`.
    pub fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let mut h = self.input_act.forward(&self.input_layer.forward(input));
        for (layer, act) in self.hidden_layers.iter().zip(self.hidden_acts.iter()) {
            h = act.forward(&layer.forward(&h));
        }
        // Linear output (no activation): regression head; caller maps
        // the [-1, 1]-normalised output back to physical units using
        // the per-channel scale factors stored alongside the model.
        self.output_layer.forward(&h)
    }

    /// Convenience overload accepting the five input columns
    /// separately. Concatenates and dispatches to `forward`.
    pub fn forward_xyz_params(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        z: &Var<f32, B>,
        f0: &Var<f32, B>,
        pnp: &Var<f32, B>,
    ) -> Var<f32, B> {
        let input = coeus_autograd::cat(&[x, y, z, f0, pnp], 1);
        self.forward(&input)
    }
}
