//! Cavitation damage model with Miner's rule fatigue and erosion

use super::material::{DamageParameters, MaterialProperties};
use crate::core::constants::cavitation::{IMPACT_ENERGY_COEFFICIENT, MATERIAL_REMOVAL_EFFICIENCY};
use crate::physics::bubble_dynamics::bubble_field::BubbleStateFields;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Cavitation damage model
#[derive(Debug)]
pub struct CavitationDamage {
    /// Accumulated damage field (dimensionless damage parameter)
    pub damage_field: Array3<f64>,
    /// Erosion rate field [kg/(m²·s)]
    pub erosion_rate: Array3<f64>,
    /// Impact pressure field (Pa)
    pub impact_pressure: Array3<f64>,
    /// Number of impacts field
    pub impact_count: Array3<u32>,
    /// Material properties
    pub material: MaterialProperties,
    /// Damage parameters
    pub params: DamageParameters,
}

impl CavitationDamage {
    /// Create new damage model
    #[must_use]
    pub fn new(
        grid_shape: (usize, usize, usize),
        material: MaterialProperties,
        params: DamageParameters,
    ) -> Self {
        Self {
            damage_field: Array3::zeros(grid_shape),
            erosion_rate: Array3::zeros(grid_shape),
            impact_pressure: Array3::zeros(grid_shape),
            impact_count: Array3::zeros(grid_shape),
            material,
            params,
        }
    }

    /// Update damage from bubble collapse
    pub fn update_damage(
        &mut self,
        bubble_states: &BubbleStateFields,
        liquid_properties: (f64, f64), // (density, sound_speed)
        dt: f64,
    ) {
        let (rho_l, c_l) = liquid_properties;

        self.impact_pressure.fill(0.0);

        for ((i, j, k), &is_collapsing) in bubble_states.is_collapsing.indexed_iter() {
            if is_collapsing > 0.5 {
                let r = bubble_states.radius[[i, j, k]];
                let v = bubble_states.velocity[[i, j, k]].abs();
                let p_internal = bubble_states.pressure[[i, j, k]];

                // Water hammer pressure from bubble collapse
                let p_hammer = rho_l * c_l * v;
                // Jet impact pressure (for asymmetric collapse)
                let p_jet = 0.5 * rho_l * v.powi(2);
                let p_impact = p_hammer.max(p_jet).max(p_internal);

                if p_impact > self.params.threshold_pressure {
                    self.impact_pressure[[i, j, k]] = p_impact;
                    self.impact_count[[i, j, k]] += 1;

                    let damage_increment =
                        self.calculate_damage_increment(p_impact, r, self.impact_count[[i, j, k]]);

                    self.damage_field[[i, j, k]] += damage_increment * dt;
                    self.erosion_rate[[i, j, k]] = self.calculate_erosion_rate(p_impact, r, v);
                }
            }
        }
    }

    /// Calculate damage increment from single impact (Miner's rule)
    pub(crate) fn calculate_damage_increment(
        &self,
        impact_pressure: f64,
        bubble_radius: f64,
        _impact_count: u32,
    ) -> f64 {
        let stress_intensity = impact_pressure * self.params.concentration_factor;

        // Fatigue damage (Miner's rule)
        let fatigue_damage = if stress_intensity > 0.0 {
            let cycles_to_failure = (self.material.ultimate_strength / stress_intensity)
                .powf(self.material.fatigue_exponent);
            1.0 / cycles_to_failure
        } else {
            0.0
        };

        let pit_probability = self.params.pit_efficiency
            * (impact_pressure / self.material.yield_strength - 1.0).max(0.0);

        let impact_area = PI * bubble_radius.powi(2);

        (fatigue_damage + pit_probability) * impact_area * self.params.fatigue_rate
    }

    /// Calculate erosion rate from impact
    pub(crate) fn calculate_erosion_rate(
        &self,
        impact_pressure: f64,
        bubble_radius: f64,
        collapse_velocity: f64,
    ) -> f64 {
        if impact_pressure < self.material.hardness {
            return 0.0;
        }

        let impact_energy = IMPACT_ENERGY_COEFFICIENT * impact_pressure * bubble_radius.powi(3);
        let removal_efficiency = MATERIAL_REMOVAL_EFFICIENCY * self.material.erosion_resistance;
        let volume_removed = removal_efficiency * impact_energy / self.material.hardness;

        self.material.density * volume_removed * collapse_velocity / bubble_radius
    }

    /// Get total accumulated damage
    #[must_use]
    pub fn total_damage(&self) -> f64 {
        self.damage_field.sum()
    }

    /// Get maximum damage location
    #[must_use]
    pub fn max_damage_location(&self) -> (usize, usize, usize) {
        let mut max_damage = 0.0;
        let mut max_loc = (0, 0, 0);

        for ((i, j, k), &damage) in self.damage_field.indexed_iter() {
            if damage > max_damage {
                max_damage = damage;
                max_loc = (i, j, k);
            }
        }

        max_loc
    }

    /// Calculate mean time to failure based on damage accumulation
    pub fn mean_time_to_failure(&self, dt: f64) -> f64 {
        let max_damage = self.damage_field.iter().copied().fold(0.0, f64::max);

        if max_damage > 0.0 {
            let damage_rate = max_damage / dt;
            1.0 / damage_rate
        } else {
            f64::INFINITY
        }
    }

    /// Get erosion depth field (m)
    #[must_use]
    pub fn erosion_depth(&self, time: f64) -> Array3<f64> {
        let density = self.material.density;
        let mut depth = Array3::zeros(self.erosion_rate.dim());

        Zip::from(&mut depth)
            .and(&self.erosion_rate)
            .par_for_each(|out, &rate| {
                if rate > 0.0 {
                    *out = rate * time / density;
                }
            });

        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_damage() -> CavitationDamage {
        CavitationDamage::new((4, 4, 4), MaterialProperties::default(), DamageParameters::default())
    }

    /// A freshly constructed damage model has zero total damage and zero erosion.
    #[test]
    fn new_model_has_zero_total_damage_and_erosion() {
        let d = make_damage();
        assert_eq!(d.total_damage(), 0.0, "initial total_damage must be zero");
        for &v in d.erosion_rate.iter() {
            assert_eq!(v, 0.0, "initial erosion_rate must be zero");
        }
    }

    /// `mean_time_to_failure` returns ∞ when all damage is zero.
    #[test]
    fn mean_time_to_failure_infinity_when_no_damage() {
        let d = make_damage();
        assert_eq!(d.mean_time_to_failure(1e-6), f64::INFINITY);
    }

    /// `max_damage_location` returns (0,0,0) when all values are equal (zero).
    #[test]
    fn max_damage_location_zero_for_zero_field() {
        let d = make_damage();
        assert_eq!(d.max_damage_location(), (0, 0, 0));
    }

    /// `calculate_damage_increment` with zero impact pressure produces zero damage.
    ///
    /// Both fatigue (stress_intensity=0 branch) and pit probability
    /// (max(0, 0/yield−1)=0) are zero.
    #[test]
    fn calculate_damage_increment_zero_for_zero_impact_pressure() {
        let d = make_damage();
        let increment = d.calculate_damage_increment(0.0, 1e-6, 0);
        assert_eq!(increment, 0.0, "zero pressure must give zero damage increment");
    }

    /// `calculate_erosion_rate` returns zero when impact pressure < material hardness.
    ///
    /// Hardness for SS316 = 2 GPa; using p=1e9 Pa (< 2e9) → erosion = 0.
    #[test]
    fn calculate_erosion_rate_zero_below_hardness() {
        let d = make_damage();
        let rate = d.calculate_erosion_rate(1e9, 1e-6, 100.0);
        assert_eq!(rate, 0.0, "erosion must be zero when p < hardness");
    }

    /// `calculate_erosion_rate` is positive when impact pressure exceeds hardness.
    ///
    /// p = 3e9 > 2e9 (hardness) → positive erosion rate.
    #[test]
    fn calculate_erosion_rate_positive_above_hardness() {
        let d = make_damage();
        let rate = d.calculate_erosion_rate(3e9, 1e-6, 100.0);
        assert!(rate > 0.0, "erosion must be positive when p > hardness (got {rate:.3e})");
    }

    /// `erosion_depth` at t=0 is identically zero (no time elapsed).
    #[test]
    fn erosion_depth_zero_at_time_zero() {
        let d = make_damage();
        let depth = d.erosion_depth(0.0);
        for &v in depth.iter() {
            assert_eq!(v, 0.0, "erosion_depth at t=0 must be zero");
        }
    }
}
