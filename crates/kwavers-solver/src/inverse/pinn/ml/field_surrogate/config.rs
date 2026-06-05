//! Configuration for the parameterised field-surrogate PINN.

use kwavers_core::error::{KwaversError, KwaversResult};

/// Network + training hyperparameters for [`super::ParamFieldPINNNetwork`].
///
/// The fixed input/output dimensionality reflects the physics:
///
/// * Input:  `(x_norm, y_norm, z_norm, f0_norm, pnp_norm)`  — five dims.
///   Spatial coordinates are normalised to `[-1, 1]` over the kernel
///   bounding box, `f0` and `pnp` to `[-1, 1]` over the sweep range.
/// * Output: `(p_min_norm, p_max_norm, p_rms_norm)` — three dims, each
///   normalised to `[-1, 1]` against the per-channel max across the
///   training cube.
///
/// All normalisation factors live alongside the network weights so
/// the trained model is self-contained at inference time.
#[derive(Debug, Clone)]
pub struct ParamFieldPINNConfig {
    /// Hidden layer dimensions; tanh activations between layers.
    /// Default `[128, 128, 128]` matches `BurnPINN3DConfig` for the
    /// transient wave-equation PINN — adequate capacity for the
    /// smooth focal-envelope shapes produced by linear-water PSTD.
    pub hidden_layers: Vec<usize>,
    /// Optimiser learning rate. Default 1e-3 — slightly higher than
    /// the transient PINN (1e-4) since the static surrogate has no
    /// PDE residual loss term and converges cleanly under MSE only.
    pub learning_rate: f64,
    /// Mini-batch size for supervised training. Default 4096; at
    /// 5+3 = 8 floats per sample and ~3 M voxels per cube corner the
    /// full cube fits in CPU RAM, so batch size is tuned for GPU
    /// throughput rather than memory.
    pub batch_size: usize,
    /// Maximum L2 norm of the gradient vector before clipping.
    /// Default 1.0.
    pub max_grad_norm: f64,
    /// Weight on the supervised MSE loss. Default 1.0.
    pub data_weight: f64,
    /// Weight on the optional Helmholtz residual loss. The residual
    /// is computed only when the network co-predicts a linear-pressure
    /// channel; defaults to 0.0 for the pure-supervised configuration.
    pub helmholtz_weight: f64,
}

impl Default for ParamFieldPINNConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![128, 128, 128],
            learning_rate: 1.0e-3,
            batch_size: 4096,
            max_grad_norm: 1.0,
            data_weight: 1.0,
            helmholtz_weight: 0.0,
        }
    }
}

impl ParamFieldPINNConfig {
    /// Reject pathological configurations early.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when:
    /// - `hidden_layers` is empty
    /// - any hidden-layer width is zero
    /// - `learning_rate <= 0`
    /// - `batch_size == 0`
    /// - `max_grad_norm <= 0`
    pub fn validate(&self) -> KwaversResult<()> {
        if self.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "ParamFieldPINNConfig.hidden_layers must be non-empty".into(),
            ));
        }
        if self.hidden_layers.iter().any(|&h| h == 0) {
            return Err(KwaversError::InvalidInput(
                "ParamFieldPINNConfig.hidden_layers entries must be > 0".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "ParamFieldPINNConfig.learning_rate must be > 0".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "ParamFieldPINNConfig.batch_size must be > 0".into(),
            ));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "ParamFieldPINNConfig.max_grad_norm must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Fixed dimensionality of the network input.
pub const INPUT_DIM: usize = 5;
/// Fixed dimensionality of the network output: (p_min, p_max, p_rms).
pub const OUTPUT_DIM: usize = 3;
