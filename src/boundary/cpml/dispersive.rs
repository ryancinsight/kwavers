//! Dispersive media support for CPML

/// Parameters for dispersive media in CPML
#[derive(Debug, Clone))]
pub struct DispersiveParameters {
    /// Number of relaxation mechanisms
    pub num_mechanisms: usize,

    /// Relaxation times for each mechanism
    pub tau: Vec<f64>,

    /// Relaxation strengths
    pub delta: Vec<f64>,

    /// Static permittivity
    pub epsilon_s: f64,

    /// Infinite frequency permittivity
    pub epsilon_inf: f64,
}

impl Default for DispersiveParameters {
    fn default() -> Self {
        Self {
            num_mechanisms: 0,
            tau: Vec::new(),
            delta: Vec::new(),
            epsilon_s: 1.0,
            epsilon_inf: 1.0,
        }
    }
}

impl DispersiveParameters {
    /// Create parameters for a Debye medium
    pub fn debye(tau: f64, epsilon_s: f64, epsilon_inf: f64) -> Self {
        Self {
            num_mechanisms: 1,
            tau: vec![tau],
            delta: vec![epsilon_s - epsilon_inf],
            epsilon_s,
            epsilon_inf,
        }
    }

    /// Create parameters for a Cole-Cole medium
    pub fn cole_cole(tau: f64, alpha: f64, epsilon_s: f64, epsilon_inf: f64) -> Self {
        // Cole-Cole requires special handling, simplified here
        Self {
            num_mechanisms: 1,
            tau: vec![tau * alpha], // Approximation
            delta: vec![epsilon_s - epsilon_inf],
            epsilon_s,
            epsilon_inf,
        }
    }

    /// Validate dispersive parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.tau.len() != self.num_mechanisms {
            return Err("Mismatch between number of mechanisms and tau values".to_string());
        }

        if self.delta.len() != self.num_mechanisms {
            return Err("Mismatch between number of mechanisms and delta values".to_string());
        }

        for &tau in &self.tau {
            if tau <= 0.0 {
                return Err("Relaxation time must be positive".to_string());
            }
        }

        if self.epsilon_s <= 0.0 || self.epsilon_inf <= 0.0 {
            return Err("Permittivity values must be positive".to_string());
        }

        Ok(())
    }
}
