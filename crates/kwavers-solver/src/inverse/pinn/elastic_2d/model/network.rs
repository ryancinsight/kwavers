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
//! For inverse problems, material parameters λ, μ, ρ are trainable
//! leaf `Var` scalars.
//!
//! ## References
//! - Raissi et al. (2019). "Physics-informed neural networks." *J. Comput. Phys.*, 378, 686–707.

use super::super::config::Config;
use coeus_autograd::Var;
use coeus_nn::{Linear, Module};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Physics-Informed Neural Network for the 2D elastic wave equation.
///
/// Approximates displacement field u(x, y, t) = [uₓ, uᵧ].
pub struct ElasticPINN2D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Input layer: 3 → first hidden layer width.
    pub input_layer: Linear<f32, B>,
    /// Hidden layers with tanh activations.
    pub hidden_layers: Vec<Linear<f32, B>>,
    /// Output layer: last hidden → 2 (uₓ, uᵧ).
    pub output_layer: Linear<f32, B>,
    /// Trainable Lamé parameter λ (Pa) — Some only for inverse problems.
    pub lambda: Option<Var<f32, B>>,
    /// Trainable shear modulus μ (Pa) — Some only for inverse problems.
    pub mu: Option<Var<f32, B>>,
    /// Trainable density ρ (kg/m³) — Some only for inverse problems.
    pub rho: Option<Var<f32, B>>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> Clone for ElasticPINN2D<B> {
    fn clone(&self) -> Self {
        Self {
            input_layer: self.input_layer.clone(),
            hidden_layers: self.hidden_layers.clone(),
            output_layer: self.output_layer.clone(),
            lambda: self.lambda.clone(),
            mu: self.mu.clone(),
            rho: self.rho.clone(),
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ElasticPINN2D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElasticPINN2D")
            .field("hidden_layers", &(self.hidden_layers.len()))
            .field("lambda", &self.lambda.is_some())
            .field("mu", &self.mu.is_some())
            .field("rho", &self.rho.is_some())
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ElasticPINN2D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Construct a PINN from `config`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `lambda_init required`.
    /// - Panics if `mu_init required`.
    /// - Panics if `rho_init required`.
    pub fn new(config: &Config) -> KwaversResult<Self> {
        config.validate().map_err(KwaversError::InvalidInput)?;

        let first_hidden = config.hidden_layers[0];
        let input_layer = Linear::new(3, first_hidden, true);

        let mut hidden_layers = Vec::with_capacity((config.hidden_layers.len()).saturating_sub(1));
        for i in 0..(config.hidden_layers.len()).saturating_sub(1) {
            hidden_layers.push(Linear::new(
                config.hidden_layers[i],
                config.hidden_layers[i + 1],
                true,
            ));
        }

        let last_hidden = *config.hidden_layers.last().unwrap();
        let output_layer = Linear::new(last_hidden, 2, true);

        let backend = B::default();
        let leaf_scalar = |v: f32| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![1], &[v], &backend),
                true,
            )
        };

        let lambda = if config.optimize_lambda {
            let v = config.lambda_init.expect("lambda_init required");
            Some(leaf_scalar(v as f32))
        } else {
            None
        };
        let mu = if config.optimize_mu {
            let v = config.mu_init.expect("mu_init required");
            Some(leaf_scalar(v as f32))
        } else {
            None
        };
        let rho = if config.optimize_rho {
            let v = config.rho_init.expect("rho_init required");
            Some(leaf_scalar(v as f32))
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

    /// Collect all trainable parameters, in the order `load_parameters` expects them back.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.input_layer.parameters();
        for layer in &self.hidden_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        if let Some(lambda) = &self.lambda {
            params.push(lambda.clone());
        }
        if let Some(mu) = &self.mu {
            params.push(mu.clone());
        }
        if let Some(rho) = &self.rho {
            params.push(rho.clone());
        }
        params
    }

    /// Write optimizer-updated parameter values back into this network's layers.
    ///
    /// # Panics
    /// - Panics if `(params.len())` does not match `self.parameters().len()`.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        let n_in = self.input_layer.parameters().len();
        self.input_layer
            .load_parameters(&params[offset..offset + n_in]);
        offset += n_in;
        for layer in &mut self.hidden_layers {
            let n = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n_out = self.output_layer.parameters().len();
        self.output_layer
            .load_parameters(&params[offset..offset + n_out]);
        offset += n_out;
        if let Some(lambda) = &mut self.lambda {
            *lambda = params[offset].clone();
            offset += 1;
        }
        if let Some(mu) = &mut self.mu {
            *mu = params[offset].clone();
            offset += 1;
        }
        if let Some(rho) = &mut self.rho {
            *rho = params[offset].clone();
            offset += 1;
        }
        assert_eq!(
            offset,
            (params.len()),
            "load_parameters: parameter count mismatch"
        );
    }

    /// Forward pass: u(x, y, t) ∈ ℝ^{batch × 2}.
    ///
    /// Concatenates inputs to `[batch, 3]`, applies tanh-activated fully connected layers,
    /// then a linear output projection.
    pub fn forward(&self, x: &Var<f32, B>, y: &Var<f32, B>, t: &Var<f32, B>) -> Var<f32, B> {
        let input = coeus_autograd::cat(&[x, y, t], 1);
        let mut h = coeus_autograd::tanh(&self.input_layer.forward(&input));
        for layer in &self.hidden_layers {
            h = coeus_autograd::tanh(&layer.forward(&h));
        }
        self.output_layer.forward(&h)
    }

    /// Return λ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_lambda(&self, fixed_value: f64) -> coeus_tensor::Tensor<f32, B> {
        match &self.lambda {
            Some(p) => p.tensor.clone(),
            None => {
                let backend = B::default();
                coeus_tensor::Tensor::from_slice_on(vec![1], &[fixed_value as f32], &backend)
            }
        }
    }

    /// Return μ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_mu(&self, fixed_value: f64) -> coeus_tensor::Tensor<f32, B> {
        match &self.mu {
            Some(p) => p.tensor.clone(),
            None => {
                let backend = B::default();
                coeus_tensor::Tensor::from_slice_on(vec![1], &[fixed_value as f32], &backend)
            }
        }
    }

    /// Return ρ as a scalar tensor; uses `fixed_value` when not optimizing.
    pub fn get_rho(&self, fixed_value: f64) -> coeus_tensor::Tensor<f32, B> {
        match &self.rho {
            Some(p) => p.tensor.clone(),
            None => {
                let backend = B::default();
                coeus_tensor::Tensor::from_slice_on(vec![1], &[fixed_value as f32], &backend)
            }
        }
    }

    /// Estimated material parameters for inverse problems.
    ///
    /// Returns `None` for each parameter that is not being optimized.
    pub fn estimated_parameters(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let extract =
            |p: &Option<Var<f32, B>>| p.as_ref().map(|param| param.tensor.as_slice()[0] as f64);
        (extract(&self.lambda), extract(&self.mu), extract(&self.rho))
    }

    /// Total trainable parameter count (weights + biases + material scalars).
    ///
    /// `Linear` weight shape: `[out_features, in_features]`, bias shape: `[out_features]`.
    pub fn num_parameters(&self) -> usize {
        let layer_params = |layer: &Linear<f32, B>| {
            let shape = layer.weight.tensor.shape();
            let (out_features, in_features) = (shape[0], shape[1]);
            out_features * in_features + out_features // weights + biases
        };

        let mut count = layer_params(&self.input_layer);
        for layer in &self.hidden_layers {
            count += layer_params(layer);
        }
        count += layer_params(&self.output_layer);

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

    /// Serialize model weights to `path` as a `coeus_tensor::StateDict` — a
    /// native binary format (zero-copy `bytemuck` casts on CPU-addressable
    /// storage), keyed by parameter index in `parameters()` order.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> KwaversResult<()> {
        let mut state = coeus_tensor::StateDict::<f32, B>::new();
        for (i, param) in self.parameters().iter().enumerate() {
            state.insert(format!("param_{i}"), param.tensor.clone());
        }
        let mut file = std::fs::File::create(path.as_ref())
            .map_err(|e| KwaversError::InvalidInput(format!("Checkpoint create failed: {e}")))?;
        state
            .save(&mut file)
            .map_err(|e| KwaversError::InvalidInput(format!("Checkpoint save failed: {e}")))
    }

    /// Placeholder: use `Trainer::load_checkpoint` for loading (requires `Config`).
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(_path: P) -> KwaversResult<Self> {
        Err(KwaversError::InvalidInput(
            "Use Trainer::load_checkpoint instead — direct model loading requires config"
                .to_string(),
        ))
    }
}
