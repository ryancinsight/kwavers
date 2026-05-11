//! `ElasticPINN2D` — neural network architecture for 2D elastic wave PINN.
//!
//! ## Mathematical Formulation
//!
//! The network approximates the displacement field:
//! ```text
//! u(x, y, t; θ) : ℝ³ → ℝ²    (uₓ, uᵧ outputs)
//! ```
//!
//! Architecture: Input[3] → Dense → tanh → … → Dense → Output[2]
//!
//! For inverse problems, material parameters λ, μ, ρ are trainable `Param` tensors.
//!
//! ## References
//! - Raissi et al. (2019). "Physics-informed neural networks." *J. Comput. Phys.*, 378, 686–707.

use super::super::config::Config;
use crate::core::error::{KwaversError, KwaversResult};

#[cfg(feature = "pinn")]
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

/// Physics-Informed Neural Network for the 2D elastic wave equation.
///
/// Approximates displacement field u(x, y, t) = [uₓ, uᵧ].
///
/// # Type Parameters
/// - `B`: Burn backend (e.g., `NdArray` for CPU, `Wgpu` for GPU).
#[cfg(feature = "pinn")]
#[derive(Module, Debug)]
pub struct ElasticPINN2D<B: Backend> {
    /// Input layer: 3 → first hidden layer width.
    pub input_layer: Linear<B>,
    /// Hidden layers with tanh activations.
    pub hidden_layers: Vec<Linear<B>>,
    /// Output layer: last hidden → 2 (uₓ, uᵧ).
    pub output_layer: Linear<B>,
    /// Trainable Lamé parameter λ (Pa) — Some only for inverse problems.
    pub lambda: Option<Param<Tensor<B, 1>>>,
    /// Trainable shear modulus μ (Pa) — Some only for inverse problems.
    pub mu: Option<Param<Tensor<B, 1>>>,
    /// Trainable density ρ (kg/m³) — Some only for inverse problems.
    pub rho: Option<Param<Tensor<B, 1>>>,
}

#[cfg(feature = "pinn")]
impl<B: Backend> ElasticPINN2D<B> {
    /// Construct a PINN from `config` and initialize on `device`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `lambda_init required`.
    /// - Panics if `mu_init required`.
    /// - Panics if `rho_init required`.
    ///
    pub fn new(config: &Config, device: &B::Device) -> KwaversResult<Self> {
        config.validate().map_err(KwaversError::InvalidInput)?;

        let first_hidden = config.hidden_layers[0];
        let input_layer = LinearConfig::new(3, first_hidden).init(device);

        let mut hidden_layers = Vec::with_capacity(config.hidden_layers.len().saturating_sub(1));
        for i in 0..config.hidden_layers.len().saturating_sub(1) {
            hidden_layers.push(
                LinearConfig::new(config.hidden_layers[i], config.hidden_layers[i + 1])
                    .init(device),
            );
        }

        let last_hidden = *config.hidden_layers.last().unwrap();
        let output_layer = LinearConfig::new(last_hidden, 2).init(device);

        let lambda = if config.optimize_lambda {
            let v = config.lambda_init.expect("lambda_init required");
            Some(Param::from_tensor(Tensor::from_floats([v as f32], device)))
        } else {
            None
        };
        let mu = if config.optimize_mu {
            let v = config.mu_init.expect("mu_init required");
            Some(Param::from_tensor(Tensor::from_floats([v as f32], device)))
        } else {
            None
        };
        let rho = if config.optimize_rho {
            let v = config.rho_init.expect("rho_init required");
            Some(Param::from_tensor(Tensor::from_floats([v as f32], device)))
        } else {
            None
        };

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
            lambda,
            mu,
            rho,
        })
    }

    /// Forward pass: u(x, y, t) ∈ ℝ^{batch × 2}.
    ///
    /// Concatenates inputs to `[batch, 3]`, applies tanh-activated fully connected layers,
    /// then a linear output projection.
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![x, y, t], 1);
        let mut h = self.input_layer.forward(input).tanh();
        for layer in &self.hidden_layers {
            h = layer.forward(h).tanh();
        }
        self.output_layer.forward(h)
    }

    /// Return λ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_lambda(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.lambda {
            Some(p) => p.val(),
            None => {
                let dev = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &dev)
            }
        }
    }

    /// Return μ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_mu(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.mu {
            Some(p) => p.val(),
            None => {
                let dev = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &dev)
            }
        }
    }

    /// Return ρ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_rho(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.rho {
            Some(p) => p.val(),
            None => {
                let dev = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &dev)
            }
        }
    }

    /// Estimated material parameters for inverse problems.
    ///
    /// Returns `None` for each parameter that is not being optimized.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn estimated_parameters(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let extract = |p: &Option<Param<Tensor<B, 1>>>| {
            p.as_ref().map(|param| {
                let data = param.val().to_data();
                data.as_slice::<f32>().unwrap()[0] as f64
            })
        };
        (extract(&self.lambda), extract(&self.mu), extract(&self.rho))
    }

    /// Device the model parameters reside on.
    pub fn device(&self) -> B::Device {
        self.input_layer.weight.device()
    }

    /// Total trainable parameter count (weights + biases + material scalars).
    ///
    /// Burn `Linear` weight shape: `[in_features, out_features]`,
    /// bias shape: `[out_features]`.
    pub fn num_parameters(&self) -> usize {
        let layer_params = |w: &Tensor<B, 2>| {
            let d = w.dims();
            d[0] * d[1] + d[1] // weights + biases (out_features)
        };

        let mut count = layer_params(&self.input_layer.weight);
        for layer in &self.hidden_layers {
            count += layer_params(&layer.weight);
        }
        count += layer_params(&self.output_layer.weight);

        if self.lambda.is_some() {
            count += 1;
        }
        if self.mu.is_some() {
            count += 1;
        }
        if self.rho.is_some() {
            count += 1;
        }

        count
    }

    /// Serialize model weights to `path` using `BinFileRecorder` (full precision).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> KwaversResult<()> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(path.as_ref().to_path_buf(), &recorder)
            .map_err(|e| KwaversError::InvalidInput(format!("Model checkpoint save failed: {e:?}")))
    }

    /// Placeholder: use `Trainer::load_checkpoint` for loading (requires `Config`).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(
        _path: P,
        _device: &B::Device,
    ) -> KwaversResult<Self> {
        Err(KwaversError::InvalidInput(
            "Use Trainer::load_checkpoint instead — direct model loading requires config"
                .to_string(),
        ))
    }
}

/// Stub for environments where the `pinn` feature is disabled.
#[cfg(not(feature = "pinn"))]
#[derive(Debug)]
pub struct ElasticPINN2D {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "pinn"))]
impl ElasticPINN2D {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(_config: &Config, _device: &()) -> KwaversResult<Self> {
        Err(KwaversError::InvalidInput(
            "ElasticPINN2D requires the 'pinn' feature to be enabled".to_owned(),
        ))
    }
}
