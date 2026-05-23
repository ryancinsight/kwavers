//! Blood perfusion models for bioheat transfer
//!
//! References:
//! - Kolios et al. (2003) "Blood flow cooling and ultrasonic lesion formation"
//! - Curra et al. (2000) "Numerical simulations of heating patterns"

use crate::core::constants::acoustic_parameters::BLOOD_VISCOSITY_37C;
use crate::core::constants::fundamental::DENSITY_BLOOD;
use crate::core::constants::medical::BLOOD_SPECIFIC_HEAT;
use crate::core::constants::thermodynamic::{BODY_TEMPERATURE_C, THERMAL_CONDUCTIVITY_BLOOD};
use ndarray::Array3;

/// Temperature-dependent perfusion model
#[derive(Debug)]
pub struct ThermalPerfusionModel {
    /// Baseline perfusion rate (kg/m³/s)
    w_b0: f64,
    /// Temperature threshold for perfusion shutdown (°C)
    t_shutdown: f64,
    /// Temperature for maximum perfusion (°C)
    t_max: f64,
    /// Maximum perfusion multiplier
    max_multiplier: f64,
}

impl ThermalPerfusionModel {
    /// Create new perfusion model
    #[must_use]
    pub fn new(baseline_perfusion: f64) -> Self {
        Self {
            w_b0: baseline_perfusion,
            t_shutdown: 50.0,    // Perfusion stops above 50°C
            t_max: 42.0,         // Maximum perfusion at mild hyperthermia
            max_multiplier: 2.0, // Double perfusion at peak
        }
    }

    /// Calculate perfusion rate based on temperature
    #[must_use]
    pub fn perfusion_rate(&self, temperature: f64) -> f64 {
        if temperature > self.t_shutdown {
            // Perfusion shutdown due to vascular damage
            0.0
        } else if temperature > self.t_max {
            // Linear decrease from max to shutdown
            let fraction = (self.t_shutdown - temperature) / (self.t_shutdown - self.t_max);
            self.w_b0 * self.max_multiplier * fraction
        } else if temperature > BODY_TEMPERATURE_C {
            // Linear increase from baseline to max
            let fraction = (temperature - BODY_TEMPERATURE_C) / (self.t_max - BODY_TEMPERATURE_C);
            self.w_b0 * (self.max_multiplier - 1.0).mul_add(fraction, 1.0)
        } else {
            // Below body temperature
            self.w_b0
        }
    }

    /// Update perfusion field based on temperature field
    #[must_use]
    pub fn update_perfusion_field(&self, temperature: &Array3<f64>) -> Array3<f64> {
        temperature.mapv(|t| self.perfusion_rate(t))
    }

    /// Check if perfusion is shut down
    #[must_use]
    pub fn is_shutdown(&self, temperature: f64) -> bool {
        temperature > self.t_shutdown
    }
}

/// Vessel cooling model for large blood vessels
#[derive(Debug)]
pub struct VesselCooling {
    /// Vessel locations (i, j, k, radius)
    vessels: Vec<(usize, usize, usize, f64)>,
    /// Blood flow velocity (m/s)
    velocity: f64,
    /// Blood temperature (°C)
    blood_temp: f64,
}

impl Default for VesselCooling {
    fn default() -> Self {
        Self::new()
    }
}

impl VesselCooling {
    /// Create new vessel cooling model
    #[must_use]
    pub fn new() -> Self {
        Self {
            vessels: Vec::new(),
            velocity: 0.1, // 10 cm/s typical
            blood_temp: BODY_TEMPERATURE_C,
        }
    }

    /// Add a vessel
    pub fn add_vessel(&mut self, i: usize, j: usize, k: usize, radius: f64) {
        self.vessels.push((i, j, k, radius));
    }

    /// Convective heat-transfer rate (W/m³ per unit volume) at grid point
    /// `(i, j, k)`, summed over all registered vessels.
    ///
    /// Sign convention follows Newton's law of cooling
    ///
    /// ```text
    /// q = h · (T_tissue − T_blood)
    /// ```
    ///
    /// so the return value is **positive when tissue is hotter than blood
    /// (heat leaves tissue, cooling)** and **negative when tissue is colder
    /// than blood (heat enters tissue, warming)**. Callers should
    /// *subtract* this quantity from the tissue energy balance.
    ///
    /// Previously this function applied `.abs()` to `(T - T_blood)`, which
    /// made every vessel always remove heat regardless of the temperature
    /// difference — a violation of the second law in the
    /// `T_tissue < T_blood` regime (e.g. cryotherapy, pre-ablation
    /// recovery, sub-body-temperature initial conditions).
    #[must_use]
    pub fn cooling_rate(&self, i: usize, j: usize, k: usize, dx: f64, temperature: f64) -> f64 {
        let mut total_cooling = 0.0;
        let delta_t = temperature - self.blood_temp;

        for &(vi, vj, vk, radius) in &self.vessels {
            let distance = ((k as f64 - vk as f64) * dx)
                .mul_add(
                    (k as f64 - vk as f64) * dx,
                    ((j as f64 - vj as f64) * dx).mul_add(
                        (j as f64 - vj as f64) * dx,
                        ((i as f64 - vi as f64) * dx).powi(2),
                    ),
                )
                .sqrt();

            if distance < radius {
                // Inside vessel — convective cooling.
                // h [W/(m²·K)] = Nu · k_blood / D,  D = 2·radius (pipe diameter).
                let diameter = 2.0 * radius;
                let h = self.nusselt_number(diameter) * THERMAL_CONDUCTIVITY_BLOOD / diameter;
                total_cooling += h * delta_t;
            } else if distance < 2.0 * radius {
                // Near-vessel boundary layer — taper the vessel-surface h
                // linearly from its surface value (weight=1 at distance=radius)
                // to zero at the outer boundary (weight=0 at distance=2·radius).
                // velocity_factor accounts for flow-speed-dependent boundary
                // layer thinning (∝ √v for fixed geometry).
                let diameter = 2.0 * radius;
                let h_surface =
                    self.nusselt_number(diameter) * THERMAL_CONDUCTIVITY_BLOOD / diameter;
                let proximity = 2.0 - distance / radius; // ∈ (0, 1]
                let velocity_factor = (self.velocity / 0.1).sqrt();
                let h = h_surface * proximity * velocity_factor;
                total_cooling += h * delta_t;
            }
        }

        total_cooling
    }

    /// Reynolds number for blood flow: `Re = ρ · v · D / μ`.
    ///
    /// `D` is the pipe **diameter** (not radius).
    /// Uses [`crate::core::constants::acoustic_parameters::BLOOD_VISCOSITY_37C`] and
    /// [`crate::core::constants::fundamental::DENSITY_BLOOD`] as SSOT values.
    fn calculate_reynolds_number(&self, diameter: f64) -> f64 {
        (DENSITY_BLOOD * self.velocity * diameter) / BLOOD_VISCOSITY_37C
    }

    /// Nusselt number for blood in a circular pipe of given `diameter`.
    ///
    /// Blood Prandtl number derived from SSOT constants:
    ///
    /// ```text
    /// Pr = μ · c_p / k = BLOOD_VISCOSITY_37C · BLOOD_SPECIFIC_HEAT / THERMAL_CONDUCTIVITY_BLOOD
    ///    = 3.5e-3 · 3617 / 0.52 ≈ 24.3
    /// ```
    ///
    /// Regime selection (Incropera & DeWitt, §8):
    /// - Re < 2300 (laminar): Nu = 3.66 — fully developed Graetz solution at
    ///   constant wall temperature (Sieder-Tate 1936, Nu₀ = 3.66).
    /// - Re ≥ 2300 (turbulent/transitional): Dittus-Boelter: Nu = 0.023 · Re^0.8 · Pr^0.4.
    ///
    /// Physiological blood-vessel flow is almost always laminar
    /// (e.g. Re ≈ 60 for a 2 mm-diameter vessel at 0.1 m/s), so the
    /// Dittus-Boelter branch is only reached by unusually large, fast vessels.
    fn nusselt_number(&self, diameter: f64) -> f64 {
        // Pr from SSOT: μ·c_p/k
        let prandtl = BLOOD_VISCOSITY_37C * BLOOD_SPECIFIC_HEAT / THERMAL_CONDUCTIVITY_BLOOD;
        let reynolds = self.calculate_reynolds_number(diameter);
        if reynolds < 2300.0 {
            // Laminar pipe flow, constant wall temperature (Graetz, fully developed)
            3.66_f64
        } else {
            // Turbulent pipe flow, Dittus-Boelter (valid Pr ∈ [0.7, 160], Re > 10000;
            // used here for Re ≥ 2300 as a conservative transitional estimate).
            0.023 * reynolds.powf(0.8) * prandtl.powf(0.4)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfusion_temperature_dependence() {
        let model = ThermalPerfusionModel::new(1.0);

        // Normal temperature
        assert_eq!(model.perfusion_rate(BODY_TEMPERATURE_C), 1.0);

        // Mild hyperthermia - increased perfusion
        let rate_42 = model.perfusion_rate(42.0);
        assert!(rate_42 > 1.5 && rate_42 <= 2.0);

        // High temperature - shutdown
        assert_eq!(model.perfusion_rate(55.0), 0.0);
    }

    #[test]
    fn test_vessel_cooling() {
        let mut vessel_model = VesselCooling::new();
        vessel_model.add_vessel(5, 5, 5, 2.0);

        // At vessel center
        let cooling_center = vessel_model.cooling_rate(5, 5, 5, 1.0, 45.0);
        assert!(cooling_center > 0.0);

        // Far from vessel
        let cooling_far = vessel_model.cooling_rate(20, 20, 20, 1.0, 45.0);
        assert_eq!(cooling_far, 0.0);
    }

    /// Vessel cooling must flip sign when tissue is colder than blood.
    ///
    /// Newton's law of cooling `q = h·(T_tissue − T_blood)` requires that
    /// q < 0 when T_tissue < T_blood — the vessel *heats* the surrounding
    /// tissue. The previous .abs() guard violated this invariant by
    /// always producing positive cooling regardless of the sign of ΔT.
    #[test]
    fn vessel_cooling_sign_flips_with_blood_temperature_difference() {
        let mut vessel_model = VesselCooling::new();
        vessel_model.add_vessel(5, 5, 5, 2.0);

        // Tissue hotter than blood → positive cooling (heat leaves tissue).
        let hot = vessel_model.cooling_rate(5, 5, 5, 1.0, 45.0);
        assert!(hot > 0.0, "tissue at 45°C above blood at {BODY_TEMPERATURE_C}°C must give positive cooling, got {hot}");

        // Tissue colder than blood → negative cooling (heat enters tissue).
        let cold = vessel_model.cooling_rate(5, 5, 5, 1.0, 25.0);
        assert!(cold < 0.0, "tissue at 25°C below blood at {BODY_TEMPERATURE_C}°C must give negative cooling, got {cold}");

        // Symmetric ΔT must give exactly opposite-signed cooling.
        let plus = vessel_model.cooling_rate(5, 5, 5, 1.0, BODY_TEMPERATURE_C + 5.0);
        let minus = vessel_model.cooling_rate(5, 5, 5, 1.0, BODY_TEMPERATURE_C - 5.0);
        assert!(
            (plus + minus).abs() < 1.0e-9 * plus.abs(),
            "symmetric ΔT must give antisymmetric cooling: plus={plus}, minus={minus}"
        );
    }
}
