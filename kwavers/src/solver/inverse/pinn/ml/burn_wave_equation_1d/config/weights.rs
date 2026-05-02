use crate::core::error::{KwaversError, KwaversResult};

/// Loss function weights: **L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc**
///
/// Default (1.0, 1.0, 10.0) prioritizes boundary enforcement.
/// Reference: Raissi et al. (2019).
#[derive(Debug, Clone, Copy)]
pub struct BurnLossWeights {
    pub data: f64,
    pub pde: f64,
    pub boundary: f64,
}

impl Default for BurnLossWeights {
    fn default() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 10.0,
        }
    }
}

impl BurnLossWeights {
    pub fn validate(&self) -> KwaversResult<()> {
        if self.data < 0.0 || !self.data.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Data loss weight must be non-negative and finite (got {})",
                self.data
            )));
        }
        if self.pde < 0.0 || !self.pde.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "PDE loss weight must be non-negative and finite (got {})",
                self.pde
            )));
        }
        if self.boundary < 0.0 || !self.boundary.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Boundary loss weight must be non-negative and finite (got {})",
                self.boundary
            )));
        }
        Ok(())
    }

    pub fn data_driven() -> Self {
        Self {
            data: 10.0,
            pde: 1.0,
            boundary: 5.0,
        }
    }

    pub fn physics_driven() -> Self {
        Self {
            data: 0.1,
            pde: 10.0,
            boundary: 10.0,
        }
    }

    pub fn balanced() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 1.0,
        }
    }
}
