//! Cavitation cloud dynamics for lithotripsy simulation.
//!
//! This module implements bubble cloud formation, growth, and collapse dynamics
//! relevant to shock wave lithotripsy, where cavitation plays a key role in
//! stone erosion and tissue bioeffects.

use kwavers_core::constants::cavitation::{
    SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
};
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::bubble_dynamics::gilmore::GilmoreSolver;
use kwavers_physics::acoustics::bubble_dynamics::{BubbleParameters, BubbleState};
use ndarray::Array3;
use std::f64::consts::PI;

/// Parameters for cavitation cloud dynamics.
#[derive(Debug, Clone)]
pub struct CloudParameters {
    /// Initial bubble radius (m)
    pub initial_bubble_radius: f64,
    /// Bubble number density (#/m³)
    pub bubble_density: f64,
    /// Ambient pressure (Pa)
    pub ambient_pressure: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
    /// Viscosity (Pa·s)
    pub viscosity: f64,
    /// Erosion efficiency (kg/J)
    pub erosion_efficiency: f64,
    /// Representative cavitation drive frequency [Hz] used to size the inertial
    /// bubble growth (R_max) via the Gilmore single-bubble solver. The
    /// rarefactional half-cycle duration ≈ 1/(2f) sets how large a bubble grows;
    /// set per modality (lithotripsy shock tail ≈ 0.25–0.5 MHz; histotripsy ≈ 1 MHz).
    pub drive_frequency: f64,
}

impl Default for CloudParameters {
    fn default() -> Self {
        Self {
            initial_bubble_radius: 1e-6, // 1 micron
            bubble_density: 1e12,        // 10^12 bubbles/m³
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            surface_tension: SURFACE_TENSION_WATER,
            viscosity: VISCOSITY_WATER,
            erosion_efficiency: 1e-12, // kg/J (empirical, Sapozhnikov et al. 2002)
            drive_frequency: 1.0e6,    // 1 MHz representative cavitation drive
        }
    }
}

/// Cavitation cloud dynamics model.
#[derive(Debug, Clone)]
pub struct CavitationCloudDynamics {
    /// Cloud parameters
    parameters: CloudParameters,
    /// Cloud density field (void fraction or bubble count)
    density_field: Array3<f64>,
    /// Total eroded mass accumulated
    accumulated_eroded_mass: f64,
}

impl CavitationCloudDynamics {
    /// Create new cavitation cloud dynamics model with parameters and grid dimensions.
    #[must_use]
    pub fn new(parameters: CloudParameters, dimensions: (usize, usize, usize)) -> Self {
        Self {
            parameters,
            density_field: Array3::zeros(dimensions),
            accumulated_eroded_mass: 0.0,
        }
    }

    /// Get cloud parameters.
    #[must_use]
    pub fn parameters(&self) -> &CloudParameters {
        &self.parameters
    }

    /// Initialize cloud based on geometry and pressure field.
    pub fn initialize_cloud(&mut self, geometry: &Array3<f64>, pressure: &Array3<f64>) {
        if self.density_field.dim() != pressure.dim() {
            self.density_field = Array3::zeros(pressure.dim());
        }
        // Simple init: nucleate bubbles where pressure < threshold (-1 MPa) and near stone
        let threshold = -MPA_TO_PA;
        for ((i, j, k), p) in pressure.indexed_iter() {
            if *p < threshold && geometry[[i, j, k]] < 0.5 {
                // Near stone but not inside?
                self.density_field[[i, j, k]] = self.parameters.bubble_density;
            } else {
                self.density_field[[i, j, k]] = 0.0;
            }
        }
    }

    /// Evolve cloud dynamics for a time step using a pressure field.
    ///
    /// This implements a simplified pressure-driven model:
    /// - Nucleation/growth when pressure drops below a Blake-like threshold.
    /// - Collapse when pressure exceeds ambient.
    /// - Erosion proportional to collapse energy.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn evolve_cloud(
        &mut self,
        dt: f64,
        _time: f64,
        pressure: &Array3<f64>,
    ) -> KwaversResult<()> {
        if pressure.dim() != self.density_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Pressure field dimensions must match cavitation cloud".to_owned(),
            ));
        }

        // Single-bubble inertial dynamics are now resolved by the real Gilmore
        // (1952) compressible solver (`inertial_collapse_energy`), capturing the
        // growth to R_max under rarefaction and the violent collapse — replacing
        // the former static-R₀ linear erosion proxy. Still absent (collective /
        // research-frontier, tracked in gap_audit CLD-1): multi-bubble acoustic
        // coupling and emission back-reaction; cloud-scale energy focusing
        // (Maeda & Colonius 2018); shock-bubble Richtmyer-Meshkov and
        // Rayleigh-Taylor cloud instabilities; inter-phase mass transfer.

        let ambient = self.parameters.ambient_pressure.max(1.0);
        let r0 = self.parameters.initial_bubble_radius.max(1e-12);
        let max_density = self.parameters.bubble_density.max(0.0);

        // Rayleigh collapse time scale: t_c = R0 * sqrt(rho / Delta_p).
        let t_char = r0 * (DENSITY_WATER_NOMINAL / ambient).sqrt();
        let growth_rate = 1.0 / t_char;
        let collapse_rate = 2.0 / t_char;

        let p_crit = ambient - 2.0 * self.parameters.surface_tension / r0;

        // Physics-based per-bubble inertial collapse energy from the field's peak
        // rarefactional pressure (real Gilmore growth → R_max ≫ R₀). One
        // single-bubble solve per call; the per-cell erosion scales with the
        // local collapsing-bubble count below.
        let p_peak_neg = pressure.iter().copied().fold(0.0_f64, f64::min); // ≤ 0
        let collapse_energy = if p_peak_neg < 0.0 {
            self.inertial_collapse_energy(-p_peak_neg)
        } else {
            0.0
        };

        for ((i, j, k), density) in self.density_field.indexed_iter_mut() {
            let p = pressure[[i, j, k]];
            let drive = if p < p_crit {
                ((p_crit - p) / p_crit.max(1.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let current_density = *density;
            let growth = drive * (max_density - current_density).max(0.0) * growth_rate;
            let collapse = (1.0 - drive) * current_density * collapse_rate;
            let updated = (growth - collapse)
                .mul_add(dt, current_density)
                .clamp(0.0, max_density)
                .max(0.0);

            // Erosion = (bubbles collapsing this step) × (real inertial collapse
            // energy) × erosion efficiency. The collapse energy reflects the
            // physically-correct R_max growth, not the static R₀ volume.
            let erosion_increment =
                collapse * collapse_energy * self.parameters.erosion_efficiency * dt;

            self.accumulated_eroded_mass += erosion_increment.max(0.0);
            *density = updated;
        }

        Ok(())
    }

    /// Maximum radius [m] reached by a representative cloud bubble driven at
    /// pressure amplitude `peak_pressure` [Pa] and the cloud's drive frequency,
    /// from the real Gilmore (1952) compressible single-bubble dynamics.
    ///
    /// This resolves the inertial growth under rarefaction (`R_max ≫ R₀` for
    /// strong tension) that a static-R₀ model cannot. Only the smooth growth
    /// phase is needed for `R_max`; integration stops if the violent collapse
    /// drives the state non-finite (R_max already captured).
    #[must_use]
    pub fn representative_max_radius(&self, peak_pressure: f64) -> f64 {
        let params = BubbleParameters {
            r0: self.parameters.initial_bubble_radius.max(1e-12),
            p0: self.parameters.ambient_pressure.max(1.0),
            sigma: self.parameters.surface_tension,
            mu_liquid: self.parameters.viscosity,
            driving_frequency: self.parameters.drive_frequency.max(1.0),
            ..BubbleParameters::default()
        };
        let solver = GilmoreSolver::new(params.clone());
        let mut state = BubbleState::at_equilibrium(&params);
        let period = 1.0 / params.driving_frequency;
        let n_steps = 2000usize; // one acoustic period, fine enough for the smooth R_max
        let dt = period / n_steps as f64;
        let mut r_max = state.radius;
        let mut t = 0.0;
        for _ in 0..n_steps {
            state = solver.step_rk4(&state, peak_pressure, t, dt);
            if !state.radius.is_finite() || state.radius <= 0.0 {
                break; // violent collapse beyond fixed-step RK4; R_max already captured
            }
            r_max = r_max.max(state.radius);
            t += dt;
        }
        r_max
    }

    /// Inertial (Rayleigh) collapse energy [J] of a representative bubble that
    /// grew to `R_max` under the local rarefaction: `E = (4/3)π R_max³ (p₀ − p_v)`.
    #[must_use]
    pub fn inertial_collapse_energy(&self, peak_pressure: f64) -> f64 {
        let r_max = self.representative_max_radius(peak_pressure);
        let p0 = self.parameters.ambient_pressure.max(1.0);
        let driving_potential = (p0 - VAPOR_PRESSURE_WATER).max(0.0);
        (4.0 / 3.0) * PI * r_max.powi(3) * driving_potential
    }

    /// Get total eroded mass at specific time (time is ignored in this simple stateful model).
    #[must_use]
    pub fn total_eroded_mass(&self, _time: f64) -> f64 {
        self.accumulated_eroded_mass
    }

    /// Get cloud density field.
    #[must_use]
    pub fn cloud_density(&self) -> &Array3<f64> {
        &self.density_field
    }
}

impl Default for CavitationCloudDynamics {
    fn default() -> Self {
        Self::new(CloudParameters::default(), (1, 1, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_growth_under_tension() {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params, (2, 2, 2));
        cloud.density_field.fill(0.0);

        let pressure = Array3::from_elem((2, 2, 2), -2.0 * MPA_TO_PA);
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.density_field.sum() > 0.0);
    }

    #[test]
    fn test_cloud_collapse_under_compression() {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density * 0.5);

        let pressure = Array3::from_elem((2, 2, 2), 2.0 * MPA_TO_PA);
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.density_field.sum() < params.bubble_density * 0.5 * 8.0);
    }

    #[test]
    fn test_erosion_accumulates_on_collapse() {
        // The corrected model sizes the collapse energy from the rarefactional
        // growth (R_max via Gilmore), so a realistic cavitation field has both a
        // rarefaction region (bubbles grow) and a compression region (they
        // collapse and erode). A pure-compression field has no growth and
        // therefore no inertial collapse energy — that only "eroded" under the
        // former static-R₀ proxy.
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density * 0.5);

        // Mixed field: one strong rarefaction cell (sizes R_max), rest compression.
        let mut pressure = Array3::from_elem((2, 2, 2), 2.0 * MPA_TO_PA);
        pressure[[0, 0, 0]] = -10.0 * MPA_TO_PA;
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.total_eroded_mass(0.0) > 0.0, "erosion must accumulate on collapse");
    }

    #[test]
    fn test_gilmore_growth_exceeds_equilibrium_radius() {
        // Real Gilmore dynamics: a strong rarefaction grows the bubble well
        // beyond R₀ (inertial cavitation), unlike the static-R₀ proxy.
        let params = CloudParameters::default();
        let cloud = CavitationCloudDynamics::new(params.clone(), (1, 1, 1));
        let r0 = params.initial_bubble_radius;
        let r_max_strong = cloud.representative_max_radius(12.0 * MPA_TO_PA);
        assert!(
            r_max_strong > 3.0 * r0,
            "strong tension must grow R_max well beyond R0: {r_max_strong} vs R0={r0}"
        );
        // Monotone: stronger tension grows a larger bubble.
        let r_max_weak = cloud.representative_max_radius(3.0 * MPA_TO_PA);
        assert!(r_max_strong > r_max_weak, "R_max must increase with drive amplitude");
    }

    #[test]
    fn test_inertial_collapse_energy_scales_with_drive() {
        // E = (4/3)π R_max³ Δp grows with the rarefactional amplitude (via R_max).
        let cloud = CavitationCloudDynamics::new(CloudParameters::default(), (1, 1, 1));
        let e_weak = cloud.inertial_collapse_energy(3.0 * MPA_TO_PA);
        let e_strong = cloud.inertial_collapse_energy(12.0 * MPA_TO_PA);
        assert!(e_weak > 0.0 && e_strong > e_weak, "collapse energy must rise with drive");
    }

    #[test]
    fn test_deeper_rarefaction_erodes_more() {
        // The accuracy payoff: deeper rarefaction → larger R_max → more erosion.
        let run = |p_neg_mpa: f64| {
            let params = CloudParameters::default();
            let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
            cloud.density_field.fill(params.bubble_density * 0.5);
            let mut pressure = Array3::from_elem((2, 2, 2), 2.0 * MPA_TO_PA);
            pressure[[0, 0, 0]] = p_neg_mpa * MPA_TO_PA;
            cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();
            cloud.total_eroded_mass(0.0)
        };
        assert!(run(-15.0) > run(-5.0), "deeper rarefaction must erode more");
    }
}
