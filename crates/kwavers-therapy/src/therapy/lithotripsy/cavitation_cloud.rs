//! Cavitation cloud dynamics for lithotripsy simulation.
//!
//! This module implements bubble cloud formation, growth, and collapse dynamics
//! relevant to shock wave lithotripsy, where cavitation plays a key role in
//! stone erosion and tissue bioeffects.

use kwavers_core::constants::cavitation::{
    SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER, VISCOSITY_WATER,
};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::bubble_dynamics::adaptive_integration::integrate_bubble_dynamics_adaptive;
use kwavers_physics::acoustics::bubble_dynamics::gilmore::GilmoreSolver;
use kwavers_physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
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
///
/// Each cell carries a **real, time-resolved representative bubble** — its radius
/// `R(t)` and wall velocity `Ṙ(t)` are integrated across [`Self::evolve_cloud`]
/// calls by the canonical adaptive Keller-Miksis solver under the local
/// instantaneous pressure (ADR 027). `density_field` is the seeded bubble *number
/// density* (nuclei per cell); erosion is deposited per genuine inertial collapse.
/// Cells are independent oscillators — collective coupling is not modeled (CLD-1).
#[derive(Debug, Clone)]
pub struct CavitationCloudDynamics {
    /// Cloud parameters
    parameters: CloudParameters,
    /// Seeded bubble number density per cell (#/m³)
    density_field: Array3<f64>,
    /// Per-cell representative bubble radius `R(t)` [m]
    radius_field: Array3<f64>,
    /// Per-cell representative wall velocity `Ṙ(t)` [m/s]
    velocity_field: Array3<f64>,
    /// Local pressure at the previous step [Pa] (for the `dp/dt` finite difference)
    prev_pressure: Array3<f64>,
    /// Whether `prev_pressure` has been seeded by a first call
    pressure_seeded: bool,
    /// Total eroded mass accumulated
    accumulated_eroded_mass: f64,
}

impl CavitationCloudDynamics {
    /// Create new cavitation cloud dynamics model with parameters and grid dimensions.
    #[must_use]
    pub fn new(parameters: CloudParameters, dimensions: (usize, usize, usize)) -> Self {
        let r0 = parameters.initial_bubble_radius.max(1e-12);
        Self {
            density_field: Array3::zeros(dimensions),
            radius_field: Array3::from_elem(dimensions, r0),
            velocity_field: Array3::zeros(dimensions),
            prev_pressure: Array3::zeros(dimensions),
            pressure_seeded: false,
            accumulated_eroded_mass: 0.0,
            parameters,
        }
    }

    /// Keller-Miksis bubble parameters for a representative cloud bubble
    /// (pure-mechanical: thermal/mass-transfer off, so the state is exactly
    /// `(R, Ṙ)`).
    fn bubble_parameters(&self) -> BubbleParameters {
        BubbleParameters {
            r0: self.parameters.initial_bubble_radius.max(1e-12),
            p0: self.parameters.ambient_pressure.max(1.0),
            sigma: self.parameters.surface_tension,
            mu_liquid: self.parameters.viscosity,
            driving_frequency: self.parameters.drive_frequency.max(1.0),
            use_thermal_effects: false,
            use_mass_transfer: false,
            ..BubbleParameters::default()
        }
    }

    /// Get cloud parameters.
    #[must_use]
    pub fn parameters(&self) -> &CloudParameters {
        &self.parameters
    }

    /// Initialize cloud based on geometry and pressure field.
    pub fn initialize_cloud(&mut self, geometry: &Array3<f64>, pressure: &Array3<f64>) {
        let r0 = self.parameters.initial_bubble_radius.max(1e-12);
        if self.density_field.dim() != pressure.dim() {
            self.density_field = Array3::zeros(pressure.dim());
            self.radius_field = Array3::from_elem(pressure.dim(), r0);
            self.velocity_field = Array3::zeros(pressure.dim());
            self.prev_pressure = Array3::zeros(pressure.dim());
            self.pressure_seeded = false;
        } else {
            // Reset the representative bubbles to equilibrium for a fresh run.
            self.radius_field.fill(r0);
            self.velocity_field.fill(0.0);
            self.pressure_seeded = false;
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

    /// Evolve the cloud by one time step under the instantaneous pressure field.
    ///
    /// Each seeded cell (`density > 0`) carries a real representative bubble whose
    /// `(R, Ṙ)` is advanced by `dt` with the canonical **adaptive Keller-Miksis**
    /// integrator under the local instantaneous pressure (and `dp/dt` from the
    /// previous call), resolving violent collapse via sub-stepping. Bubble history
    /// is carried across calls in `radius_field`/`velocity_field`, so calling this
    /// at acoustic-resolution time steps reproduces the true `R(t)` waveform per
    /// cell (ADR 027). Erosion is the compression work `∫p dV` done on each
    /// collapsing bubble (`∫p dV ≈` the Rayleigh collapse energy over a full
    /// collapse), localized per cell.
    ///
    /// Cells are independent oscillators — inter-bubble coupling and cloud-scale
    /// collective collapse are not modeled (CLD-1, research frontier).
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if the pressure field shape mismatches or
    ///   `dt` is non-finite/`≤ 0`; propagates integrator errors.
    pub fn evolve_cloud(
        &mut self,
        dt: f64,
        time: f64,
        pressure: &Array3<f64>,
    ) -> KwaversResult<()> {
        if pressure.dim() != self.density_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Pressure field dimensions must match cavitation cloud".to_owned(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "evolve_cloud requires dt > 0, got {dt}"
            )));
        }

        let params = self.bubble_parameters();
        let solver = KellerMiksisModel::new(params.clone());
        let r0 = params.r0;
        let p_vapor = VAPOR_PRESSURE_WATER;
        let efficiency = self.parameters.erosion_efficiency;
        let (nx, ny, nz) = self.density_field.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = [i, j, k];
                    let p_local = pressure[idx];
                    let p_prev = if self.pressure_seeded {
                        self.prev_pressure[idx]
                    } else {
                        p_local // first call: dp/dt = 0
                    };
                    self.prev_pressure[idx] = p_local;

                    let density = self.density_field[idx];
                    if density <= 0.0 {
                        continue; // no nuclei here — representative radius stays at R₀
                    }

                    let dp_dt = (p_local - p_prev) / dt;
                    let r_before = self.radius_field[idx].max(1e-12);

                    // Reconstruct the (R, Ṙ) state (pure-mechanical bubble) and
                    // advance adaptively by dt under the local instantaneous pressure.
                    let mut state = BubbleState::new(&params);
                    state.radius = r_before;
                    state.wall_velocity = self.velocity_field[idx];
                    integrate_bubble_dynamics_adaptive(
                        &solver, &mut state, p_local, dp_dt, dt, time,
                    )?;

                    if !state.radius.is_finite() || state.radius <= 0.0 {
                        // Destructive collapse beyond the integrator's range —
                        // re-nucleate at R₀ (a fresh bubble) for the next cycle.
                        state.radius = r0;
                        state.wall_velocity = 0.0;
                    }
                    let r_after = state.radius;

                    // Erosion = compression work on the collapsing bubble:
                    // dE = (p∞ − p_v)·(−dV) when the bubble shrinks under net
                    // compression. Integrated over a full collapse this is the
                    // Rayleigh collapse energy; per step it is robust and local.
                    if r_after < r_before {
                        let p_drive = (p_local - p_vapor).max(0.0);
                        let dv = (4.0 / 3.0) * PI * (r_before.powi(3) - r_after.powi(3));
                        let erosion = density * p_drive * dv * efficiency;
                        self.accumulated_eroded_mass += erosion.max(0.0);
                    }

                    self.radius_field[idx] = r_after;
                    self.velocity_field[idx] = state.wall_velocity;
                }
            }
        }
        self.pressure_seeded = true;
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

    /// Get the per-cell representative bubble radius field `R(t)` [m].
    #[must_use]
    pub fn cloud_radius(&self) -> &Array3<f64> {
        &self.radius_field
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

    /// Drive a single-cell cloud through a pressure sequence; return the cloud.
    fn drive_single_cell(amplitudes: &[f64], dt: f64) -> CavitationCloudDynamics {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (1, 1, 1));
        cloud.density_field.fill(params.bubble_density);
        for (n, &amp) in amplitudes.iter().enumerate() {
            let t = n as f64 * dt;
            cloud
                .evolve_cloud(dt, t, &Array3::from_elem((1, 1, 1), amp))
                .unwrap();
        }
        cloud
    }

    #[test]
    fn cell_matches_standalone_keller_miksis() {
        // KEYSTONE (ADR 027): a cloud cell IS a real Keller-Miksis bubble — its
        // R(t) reproduces the standalone adaptive integrator driven by the exact
        // same (p, dp/dt, dt) sequence, bit-for-bit.
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (1, 1, 1));
        cloud.density_field.fill(params.bubble_density);

        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let mut ref_state = BubbleState::new(&bp);

        let f = params.drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let amp = 0.5e6; // moderate drive — gentle oscillation, no destructive reset
        let mut p_prev: Option<f64> = None;
        for n in 0..200 {
            let t = n as f64 * dt;
            let p = amp * (2.0 * PI * f * t).sin();
            cloud
                .evolve_cloud(dt, t, &Array3::from_elem((1, 1, 1), p))
                .unwrap();
            // Mirror the cloud's exact per-cell computation as the reference.
            let dp_dt = (p - p_prev.unwrap_or(p)) / dt;
            let mut s = BubbleState::new(&bp);
            s.radius = ref_state.radius;
            s.wall_velocity = ref_state.wall_velocity;
            integrate_bubble_dynamics_adaptive(&solver, &mut s, p, dp_dt, dt, t).unwrap();
            ref_state.radius = s.radius;
            ref_state.wall_velocity = s.wall_velocity;
            p_prev = Some(p);
        }
        let cloud_r = cloud.cloud_radius()[[0, 0, 0]];
        assert!(
            (cloud_r - ref_state.radius).abs() <= 1e-12 * ref_state.radius.max(1e-9),
            "cloud cell R(t) must equal standalone KM: {cloud_r} vs {}",
            ref_state.radius
        );
    }

    #[test]
    fn test_bubble_grows_under_sustained_tension() {
        // Real inertial growth: under sustained rarefaction the representative
        // bubble expands beyond R₀ (the static-R₀ model could not).
        let r0 = CloudParameters::default().initial_bubble_radius;
        let dt = (1.0 / CloudParameters::default().drive_frequency) / 200.0;
        let tension = vec![-2.0 * MPA_TO_PA; 60];
        let cloud = drive_single_cell(&tension, dt);
        assert!(
            cloud.cloud_radius()[[0, 0, 0]] > r0,
            "bubble must grow under tension: R={} R0={r0}",
            cloud.cloud_radius()[[0, 0, 0]]
        );
    }

    #[test]
    fn test_cells_without_nuclei_stay_at_equilibrium() {
        // density = 0 ⇒ no bubble integrated ⇒ radius stays at R₀, no erosion.
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        // density_field defaults to zeros.
        let pressure = Array3::from_elem((2, 2, 2), -5.0 * MPA_TO_PA);
        cloud.evolve_cloud(1e-8, 0.0, &pressure).unwrap();
        let r0 = params.initial_bubble_radius;
        assert!(cloud.cloud_radius().iter().all(|&r| (r - r0).abs() < 1e-18));
        assert_eq!(cloud.total_eroded_mass(0.0), 0.0);
    }

    #[test]
    fn test_erosion_accumulates_over_growth_then_collapse() {
        // Grow the bubble under tension, then compress: the compression work on
        // the shrinking bubble (∫p dV) deposits erosion.
        let dt = (1.0 / CloudParameters::default().drive_frequency) / 200.0;
        let mut seq = vec![-3.0 * MPA_TO_PA; 50]; // growth
        seq.extend(vec![3.0 * MPA_TO_PA; 50]); // compression/collapse
        let cloud = drive_single_cell(&seq, dt);
        assert!(
            cloud.total_eroded_mass(0.0) > 0.0,
            "erosion must accumulate over a growth+collapse cycle"
        );
    }

    #[test]
    fn test_deeper_rarefaction_grows_larger() {
        // Accuracy payoff (clean, monotone form): deeper rarefaction grows the
        // bubble larger. (Amplitudes kept in the well-resolved regime so the
        // per-call integrator does not overshoot into the destructive-collapse
        // reset, which would non-monotonically cap explosive growth.)
        let dt = (1.0 / CloudParameters::default().drive_frequency) / 400.0;
        let grow_to = |tension_mpa: f64| {
            let seq = vec![tension_mpa * MPA_TO_PA; 30];
            drive_single_cell(&seq, dt).cloud_radius()[[0, 0, 0]]
        };
        let r0 = CloudParameters::default().initial_bubble_radius;
        let r_shallow = grow_to(-2.0);
        let r_deep = grow_to(-4.0);
        assert!(r_deep > r_shallow && r_shallow > r0, "deeper tension must grow larger: {r_deep} > {r_shallow} > {r0}");
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
}
