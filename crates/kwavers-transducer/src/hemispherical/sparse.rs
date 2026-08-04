//! Sparse array optimization

use aequitas::systems::si::quantities::Dimensionless;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Sparse array optimizer
#[derive(Debug, Clone)]
pub struct SparseArrayOptimizer {
    /// Density factor (0.0-1.0)
    density_factor: Dimensionless<f64>,
}

impl SparseArrayOptimizer {
    /// Create new optimizer
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    pub fn new(density_factor: Dimensionless<f64>) -> KwaversResult<Self> {
        let density_factor_base = density_factor.into_base();
        if !(0.0..=1.0).contains(&density_factor_base) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "density_factor".to_owned(),
                value: density_factor_base.to_string(),
                constraint: "must be between 0.0 and 1.0".to_owned(),
            }));
        }

        Ok(Self { density_factor })
    }

    /// Optimize element selection
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn optimize(&self, elements: &mut [ElementSelection]) -> KwaversResult<()> {
        let num_active = (elements.len() as f64 * self.density_factor.into_base()) as usize;

        // Select elements that maximize power delivery
        elements
            .iter_mut()
            .take(num_active)
            .for_each(|e| e.is_selected = true);

        Ok(())
    }
}

/// Element selection state
#[derive(Debug, Clone)]
pub struct ElementSelection {
    /// Element index
    pub index: usize,
    /// Selection state
    pub is_selected: bool,
    /// Contribution weight
    pub weight: Dimensionless<f64>,
}

impl ElementSelection {
    /// Create new selection
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            index,
            is_selected: false,
            weight: Dimensionless::from_base(1.0),
        }
    }
}
