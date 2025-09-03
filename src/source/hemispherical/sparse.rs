//! Sparse array optimization

use crate::error::{ConfigError, KwaversError, KwaversResult};

/// Sparse array optimizer
#[derive(Debug, Clone)]
pub struct SparseArrayOptimizer {
    /// Density factor (0.0-1.0)
    density_factor: f64,
    /// Selection strategy
    strategy: SelectionStrategy,
}

impl SparseArrayOptimizer {
    /// Create new optimizer
    pub fn new(density_factor: f64) -> KwaversResult<Self> {
        if !(0.0..=1.0).contains(&density_factor) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "density_factor".to_string(),
                value: density_factor.to_string(),
                constraint: "must be between 0.0 and 1.0".to_string(),
            }));
        }

        Ok(Self {
            density_factor,
            strategy: SelectionStrategy::PowerOptimal,
        })
    }

    /// Optimize element selection
    pub fn optimize(&self, elements: &mut [ElementSelection]) -> KwaversResult<()> {
        let num_active = (elements.len() as f64 * self.density_factor) as usize;

        // Select elements based on strategy
        match self.strategy {
            SelectionStrategy::PowerOptimal => {
                // Select elements that maximize power delivery
                elements
                    .iter_mut()
                    .take(num_active)
                    .for_each(|e| e.is_selected = true);
            }
            SelectionStrategy::UniformSparse => {
                // Select uniformly distributed subset
                let step = elements.len() / num_active;
                elements
                    .iter_mut()
                    .step_by(step)
                    .for_each(|e| e.is_selected = true);
            }
        }

        Ok(())
    }
}

/// Element selection strategy
#[derive(Debug, Clone, Copy)]
enum SelectionStrategy {
    /// Optimize for power delivery
    PowerOptimal,
    /// Uniform sparse distribution
    UniformSparse,
}

/// Element selection state
#[derive(Debug, Clone)]
pub struct ElementSelection {
    /// Element index
    pub index: usize,
    /// Selection state
    pub is_selected: bool,
    /// Contribution weight
    pub weight: f64,
}

impl ElementSelection {
    /// Create new selection
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            index,
            is_selected: false,
            weight: 1.0,
        }
    }
}
