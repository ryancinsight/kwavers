//! Dispersive media support for CPML

/// Parameters for dispersive media in CPML
#[derive(Debug, Clone)]
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
    #[must_use]
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
    ///
    /// **Implementation**: Multi-pole Debye expansion approximating Cole-Cole dispersion
    /// Uses 3 Debye poles to capture the fractional derivative behavior τ^α.
    /// This provides much better accuracy than single-pole approximation.
    ///
    /// **Reference**: Cole & Cole (1941) "Dispersion and Absorption in Dielectrics"
    /// **Reference**: Mainardi (2010) "Fractional Calculus and Waves in Linear Viscoelasticity"
    #[must_use]
    pub fn cole_cole(tau: f64, alpha: f64, epsilon_s: f64, epsilon_inf: f64) -> Self {
        // Multi-pole Debye expansion to approximate Cole-Cole fractional behavior
        // τ^α ≈ Σ w_i / (1 + iωτ_i) where τ_i are chosen to match the fractional response
        let delta_total = epsilon_s - epsilon_inf;

        // Use 3-pole expansion for better Cole-Cole approximation
        // For Cole-Cole: ε(ω) = ε∞ + Σ Δε_i / (1 + (iωτ_i)^α)
        // Choose time constants to span the frequency range of interest
        let weights: [f64; 3] = [0.3, 0.4, 0.3];
        let tau_factors: [f64; 3] = [0.1, 1.0, 10.0]; // Log-spaced time constants

        let mut taus = Vec::new();
        let mut deltas = Vec::new();

        for (&w, &factor) in weights.iter().zip(tau_factors.iter()) {
            // Correct Cole-Cole pole placement: τ_i = τ * factor^(1/α)
            // This ensures proper frequency dependence for fractional derivative
            taus.push(tau * factor.powf(1.0 / alpha));
            deltas.push(delta_total * w);
        }

        Self {
            num_mechanisms: 3,
            tau: taus,
            delta: deltas,
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
