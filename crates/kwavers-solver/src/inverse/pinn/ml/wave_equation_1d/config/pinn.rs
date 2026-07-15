use super::LossWeights;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Configuration for Coeus-backed 1D Wave Equation PINN.
///
/// Architecture: 2 inputs (x, t) → hidden layers → 1 output (u).
/// Loss: L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc.
#[derive(Debug, Clone)]
pub struct PinnConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub loss_weights: LossWeights,
    pub num_collocation_points: usize,
}

impl Default for PinnConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![50, 50, 50, 50],
            learning_rate: 1e-3,
            loss_weights: LossWeights::default(),
            num_collocation_points: 10_000,
        }
    }
}

impl PinnConfig {
    pub fn for_gpu() -> Self {
        Self {
            hidden_layers: vec![100, 100, 100, 100, 100],
            learning_rate: 5e-4,
            loss_weights: LossWeights::default(),
            num_collocation_points: 50_000,
        }
    }

    /// For prototyping.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn for_prototyping() -> Self {
        Self {
            hidden_layers: vec![20, 20, 20],
            learning_rate: 1e-3,
            loss_weights: LossWeights::default(),
            num_collocation_points: 1_000,
        }
    }
    /// Validate.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Configuration must have at least one hidden layer".into(),
            ));
        }
        for (i, &size) in self.hidden_layers.iter().enumerate() {
            if size == 0 {
                return Err(KwaversError::InvalidInput(format!(
                    "Hidden layer {} has size 0 (must be positive)",
                    i
                )));
            }
        }
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Learning rate must be positive and finite (got {})",
                self.learning_rate
            )));
        }
        if self.num_collocation_points < 100 {
            return Err(KwaversError::InvalidInput(format!(
                "Number of collocation points must be at least 100 (got {})",
                self.num_collocation_points
            )));
        }
        self.loss_weights.validate()?;
        Ok(())
    }

    /// Total trainable parameters: Σ (in×out + out) across all layer transitions.
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        let first = self.hidden_layers[0];
        total += 2 * first + first;
        for i in 0..(self.hidden_layers.len()) - 1 {
            let (in_size, out_size) = (self.hidden_layers[i], self.hidden_layers[i + 1]);
            total += in_size * out_size + out_size;
        }
        let last = self.hidden_layers[(self.hidden_layers.len()) - 1];
        total += last + 1;
        total
    }
}
