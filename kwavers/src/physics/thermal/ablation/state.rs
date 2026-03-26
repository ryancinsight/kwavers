use super::kinetics::AblationKinetics;

/// Tissue ablation state at a point
#[derive(Debug, Clone, Copy)]
pub struct AblationState {
    /// Temperature [°C]
    pub temperature: f64,
    /// Accumulated thermal damage (Ω)
    pub damage: f64,
    /// Tissue viability (0-1, 1=fully viable)
    pub viability: f64,
    /// Is tissue ablated?
    pub ablated: bool,
}

impl AblationState {
    /// Create new ablation state
    pub fn new(temperature: f64, _kinetics: &AblationKinetics) -> Self {
        Self {
            temperature,
            damage: 0.0,
            viability: 1.0,
            ablated: false,
        }
    }

    /// Update ablation state with temperature change
    pub fn update(&mut self, temperature: f64, kinetics: &AblationKinetics, dt: f64) {
        self.temperature = temperature;

        // Convert to absolute temperature
        let t_abs = temperature + 273.15;

        // Skip if below ablation threshold
        if temperature < kinetics.ablation_threshold {
            return;
        }

        // Calculate damage rate
        let damage_rate = kinetics.damage_rate(t_abs);

        // Accumulate damage
        self.damage = AblationKinetics::accumulated_damage(self.damage, damage_rate, dt);

        // Update viability
        self.viability = kinetics.viability(self.damage);

        // Check ablation
        self.ablated = kinetics.is_ablated(self.damage);
    }
}
