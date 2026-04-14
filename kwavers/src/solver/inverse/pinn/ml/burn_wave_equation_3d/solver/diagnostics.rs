use super::core::BurnPINN3DWave;
use crate::core::error::{KwaversError, KwaversResult};
use burn::tensor::{backend::Backend, Tensor};

/// Gradient diagnostics for monitoring training stability
///
/// Tracks parameter update magnitudes as a proxy for gradient norms.
/// This provides insight into gradient flow without requiring direct
/// access to Burn's opaque Gradients type.
///
/// # Mathematical Specification
///
/// Parameter update norm: ||Δθ||₂ = ||θ_new - θ_old||₂
/// Relative update: ||Δθ||₂ / (||θ_old||₂ + ε)
///
/// These metrics help detect:
/// - Gradient explosion (large updates)
/// - Vanishing gradients (tiny updates)
/// - Training stagnation (near-zero updates)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for future Burn API gradient introspection
struct GradientDiagnostics {
    /// L2 norm of parameter updates ||Δθ||₂
    pub update_norm: f64,
    /// Relative update magnitude ||Δθ||₂ / ||θ||₂
    pub relative_update: f64,
    /// Maximum absolute parameter change
    pub max_update: f64,
}

impl GradientDiagnostics {
    /// Compute diagnostics by comparing old and new parameters
    ///
    /// # Arguments
    ///
    /// * `old_params` - Parameters before optimizer step
    /// * `new_params` - Parameters after optimizer step
    ///
    /// # Returns
    ///
    /// Gradient diagnostics including update norms and relative changes
    #[allow(dead_code)] // Reserved for future use when Burn exposes parameter access
    fn compute<B: Backend>(
        old_params: &[Tensor<B, 2>],
        new_params: &[Tensor<B, 2>],
    ) -> KwaversResult<Self> {
        if old_params.len() != new_params.len() {
            return Err(KwaversError::InvalidInput(
                "Parameter count mismatch between old and new".into(),
            ));
        }

        let mut update_norm_sq = 0.0_f64;
        let mut param_norm_sq = 0.0_f64;
        let mut max_update = 0.0_f64;

        for (old, new) in old_params.iter().zip(new_params.iter()) {
            // Compute parameter difference: Δθ = θ_new - θ_old
            let diff = new.clone().sub(old.clone());

            // Extract values for norm computation
            let diff_data = diff.to_data();
            let old_data = old.to_data();

            let diff_values: Vec<f32> = diff_data
                .to_vec()
                .map_err(|_| KwaversError::InvalidInput("Failed to extract diff values".into()))?;

            let old_values: Vec<f32> = old_data
                .to_vec()
                .map_err(|_| KwaversError::InvalidInput("Failed to extract param values".into()))?;

            // Accumulate squared norms
            for &val in &diff_values {
                let val_f64 = val as f64;
                update_norm_sq += val_f64 * val_f64;
                max_update = max_update.max(val_f64.abs());
            }

            for &val in &old_values {
                let val_f64 = val as f64;
                param_norm_sq += val_f64 * val_f64;
            }
        }

        let update_norm = update_norm_sq.sqrt();
        let param_norm = param_norm_sq.sqrt();
        let relative_update = if param_norm > 1e-12 {
            update_norm / param_norm
        } else {
            update_norm // If params are near-zero, just report absolute
        };

        Ok(Self {
            update_norm,
            relative_update,
            max_update,
        })
    }
}

impl<B: Backend> BurnPINN3DWave<B> {
    /// Extract all network parameters as a vector of tensors
    ///
    /// Used for gradient diagnostics by comparing parameters before/after optimizer step.
    /// This provides a workaround for Burn's opaque Gradients type.
    ///
    /// # Returns
    ///
    /// Vector of parameter tensors (weights and biases from all layers)
    #[allow(dead_code)] // Reserved for future gradient diagnostics when Burn API ready
    fn extract_parameters(&self) -> Vec<Tensor<B, 2>> {
        // Note: Burn's Module trait doesn't expose internal parameters directly.
        // As a workaround, we use num_params() to get a rough estimate of total parameters.
        // For now, return an empty vector - gradient diagnostics will be disabled
        // until Burn provides parameter introspection API.
        //
        // KNOWN_LIMITATION: Blocked on Burn parameter introspection API.
        // Alternative: Track loss history and use loss gradient as proxy.
        Vec::new()
    }
}
