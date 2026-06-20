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
use kwavers_physics::acoustics::bubble_dynamics::bubbly_medium::commander_prosperetti_attenuation;
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
    /// Grid cell spacing `(dx, dy, dz)` [m] — sets inter-bubble distances for the
    /// acoustic coupling (ADR 028).
    pub cell_spacing: [f64; 3],
    /// Enable inter-bubble acoustic coupling (radiated-pressure perturbation of
    /// neighbours, ADR 028). **Opt-in** (default `false`): the coupling sum is
    /// `O(active²)` per step and amplifies the drive into the stiff
    /// violent-collapse regime, so enabling it on a large cloud is costly — set a
    /// finite `interaction_radius` cutoff there. Off ⇒ the independent-oscillator
    /// model (ADR 027). When enabled, prefer tractable problem sizes / cutoffs.
    pub coupling_enabled: bool,
    /// Cutoff distance [m] beyond which inter-bubble coupling is neglected, bounding
    /// the `O(active²)` coupling sum. `INFINITY` ⇒ all active pairs.
    pub interaction_radius: f64,
    /// Enable cloud-scale acoustic shielding: the incident field is attenuated by
    /// the cloud's void fraction (Commander–Prosperetti) as it penetrates, so the
    /// periphery screens the interior (ADR 029). Opt-in (default `false`); `O(N)`.
    pub shielding_enabled: bool,
    /// Axis (0=x, 1=y, 2=z) along which the incident wave penetrates the cloud
    /// for the shielding screen.
    pub incident_axis: usize,
    /// Whether the incident wave enters from the high-index face (and travels
    /// toward index 0) instead of the low-index face.
    pub incident_from_high: bool,
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
            cell_spacing: [1.0e-3; 3],          // 1 mm
            coupling_enabled: false,            // opt-in (O(active²); see field docs)
            interaction_radius: f64::INFINITY,
            shielding_enabled: false,           // opt-in (ADR 029)
            incident_axis: 2,                   // z by default
            incident_from_high: false,
        }
    }
}

/// Incompressible near-field pressure [Pa] radiated by a pulsating bubble of
/// radius `r`, wall velocity `r_dot`, and wall acceleration `r_ddot` at distance
/// `d > 0`:  `p_rad = (ρ_L/d)·(r²·r̈ + 2·r·ṙ²) = (ρ_L/d)·d/dt(r²ṙ)`.
///
/// This is the coupling pressure a bubble adds to its neighbours' driving field
/// (Mettin et al. 1997; Ida 2002). Returns `0` for non-positive distance.
#[must_use]
pub fn bubble_radiated_pressure(rho: f64, distance: f64, r: f64, r_dot: f64, r_ddot: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    let source_strength = r_ddot.mul_add(r * r, 2.0 * r * r_dot * r_dot); // r²r̈ + 2rṙ²
    rho * source_strength / distance
}

/// Cavitation cloud dynamics model.
///
/// Each cell carries a **real, time-resolved representative bubble** — its radius
/// `R(t)` and wall velocity `Ṙ(t)` are integrated across [`Self::evolve_cloud`]
/// calls by the canonical adaptive Keller-Miksis solver under its **total**
/// driving pressure (local external + inter-bubble acoustic coupling, ADR 028).
/// `density_field` is the seeded bubble *number density* (nuclei per cell); erosion
/// is deposited per genuine inertial collapse. Two cloud-scale collective effects
/// are available (opt-in): inter-bubble acoustic coupling (ADR 028) and void-fraction
/// shielding of the incident field (ADR 029). Cloud-interface instabilities and a
/// self-consistent collective solve remain open (CLD-1).
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
    prev_total_pressure: Array3<f64>,
    /// Whether `prev_total_pressure` has been seeded by a first call
    total_seeded: bool,
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
            prev_total_pressure: Array3::zeros(dimensions),
            total_seeded: false,
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
            self.prev_total_pressure = Array3::zeros(pressure.dim());
            self.total_seeded = false;
        } else {
            // Reset the representative bubbles to equilibrium for a fresh run.
            self.radius_field.fill(r0);
            self.velocity_field.fill(0.0);
            self.total_seeded = false;
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

    /// Cloud-scale acoustic shielding (ADR 029): return the incident `pressure`
    /// field screened by the cloud's void fraction along the incident axis, via
    /// Beer-Lambert with the Commander-Prosperetti attenuation. Bubbles between
    /// the entry face and a cell reduce the field driving that cell, so the
    /// periphery shields the interior. `O(N)` (one prefix sum per column).
    #[must_use]
    pub fn shielded_pressure(&self, pressure: &Array3<f64>) -> Array3<f64> {
        let params = self.bubble_parameters();
        let (f, c, rho, mu, p0, gamma, r0) = (
            params.driving_frequency,
            params.c_liquid,
            params.rho_liquid,
            params.mu_liquid,
            params.p0,
            params.gamma,
            params.r0,
        );
        let axis = self.parameters.incident_axis.min(2);
        let ds = self.parameters.cell_spacing[axis];
        let (nx, ny, nz) = pressure.dim();
        let n_axis = [nx, ny, nz][axis];
        let (n_a, n_b) = match axis {
            0 => (ny, nz),
            1 => (nx, nz),
            _ => (nx, ny),
        };
        let make_idx = |m: usize, a: usize, b: usize| -> [usize; 3] {
            match axis {
                0 => [m, a, b],
                1 => [a, m, b],
                _ => [a, b, m],
            }
        };

        let mut out = pressure.clone();
        for a in 0..n_a {
            for b in 0..n_b {
                let mut optical_depth = 0.0_f64; // Σ α·Δs from the entry face
                for step in 0..n_axis {
                    // Walk the axis in propagation order (from the entry face).
                    let m = if self.parameters.incident_from_high {
                        n_axis - 1 - step
                    } else {
                        step
                    };
                    let idx = make_idx(m, a, b);
                    let n = self.density_field[idx];
                    let r = self.radius_field[idx].max(0.0);
                    let alpha = if n > 0.0 && r > 0.0 {
                        // Void fraction β = n·(4/3)π R³ (clamped for CP validity).
                        let beta = (n * (4.0 / 3.0) * PI * r.powi(3)).clamp(0.0, 1.0 - 1e-9);
                        commander_prosperetti_attenuation(f, beta, r0, c, rho, mu, p0, gamma)
                            .max(0.0)
                    } else {
                        0.0
                    };
                    // Attenuate by all bubbles before this cell + half the local cell.
                    let tau_center = alpha.mul_add(0.5 * ds, optical_depth);
                    out[idx] = pressure[idx] * (-tau_center).exp();
                    optical_depth += alpha * ds;
                }
            }
        }
        out
    }

    /// Position [m] of cell `(i,j,k)` from the configured cell spacing.
    #[inline]
    fn cell_position(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        let [dx, dy, dz] = self.parameters.cell_spacing;
        [i as f64 * dx, j as f64 * dy, k as f64 * dz]
    }

    /// Inter-bubble acoustic source strengths `S_j = R_j² R̈_j + 2 R_j Ṙ_j²` for
    /// every active cell, with positions, evaluated explicitly at the previous
    /// total driving pressure (ADR 028, Pass 1). Empty when coupling is disabled.
    fn coupling_sources(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        time: f64,
    ) -> Vec<([f64; 3], f64)> {
        if !self.parameters.coupling_enabled {
            return Vec::new();
        }
        let (nx, ny, nz) = self.density_field.dim();
        let mut sources = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = [i, j, k];
                    if self.density_field[idx] <= 0.0 {
                        continue;
                    }
                    let r = self.radius_field[idx].max(1e-12);
                    let v = self.velocity_field[idx];
                    let mut state = BubbleState::new(params);
                    state.radius = r;
                    state.wall_velocity = v;
                    // Explicit (lagged) source acceleration from the previous total
                    // driving pressure (no implicit all-bubble solve).
                    let p_src = if self.total_seeded {
                        self.prev_total_pressure[idx]
                    } else {
                        0.0
                    };
                    let accel = solver
                        .calculate_acceleration(&mut state, p_src, 0.0, time)
                        .unwrap_or(0.0);
                    let strength = accel.mul_add(r * r, 2.0 * r * v * v); // R²R̈ + 2RṘ²
                    if strength.is_finite() {
                        sources.push((self.cell_position(i, j, k), strength));
                    }
                }
            }
        }
        sources
    }

    /// Evolve the cloud by one time step under the instantaneous pressure field.
    ///
    /// Each seeded cell (`density > 0`) carries a real representative bubble whose
    /// `(R, Ṙ)` is advanced by `dt` with the canonical **adaptive Keller-Miksis**
    /// integrator under its **total** driving pressure — the local external
    /// pressure plus the **inter-bubble acoustic coupling** `Σ_{j≠i}(ρ/d_ij)·S_j`
    /// from neighbouring bubbles (ADR 028) — resolving collapse via sub-stepping.
    /// Bubble history is carried across calls, so acoustic-resolution stepping
    /// reproduces the coupled per-cell `R(t)`. With one active cell or coupling
    /// disabled this reduces exactly to the independent-oscillator model (ADR 027).
    /// Erosion is the compression work `∫p dV` on each collapsing bubble
    /// (≈ Rayleigh collapse energy over a full collapse), localized per cell.
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
        let rho = params.rho_liquid;
        let p_vapor = VAPOR_PRESSURE_WATER;
        let efficiency = self.parameters.erosion_efficiency;
        let r_cut = self.parameters.interaction_radius;
        let (nx, ny, nz) = self.density_field.dim();

        // Cloud-scale shielding (ADR 029): screen the incident field by the cloud's
        // void fraction before it drives the bubbles (avoids cloning when off).
        let shielded_field;
        let driving: &Array3<f64> = if self.parameters.shielding_enabled {
            shielded_field = self.shielded_pressure(pressure);
            &shielded_field
        } else {
            pressure
        };

        // Pass 1: explicit inter-bubble acoustic source strengths (ADR 028).
        let sources = self.coupling_sources(&solver, &params, time);

        // Pass 2: per-cell total driving pressure (screened external + coupling).
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = [i, j, k];
                    let density = self.density_field[idx];
                    if density <= 0.0 {
                        // No nuclei: radius stays at R₀; still track prev pressure.
                        self.prev_total_pressure[idx] = driving[idx];
                        continue;
                    }

                    // Coupling pressure from all other active bubbles within cutoff.
                    let pos_i = self.cell_position(i, j, k);
                    let mut p_couple = 0.0;
                    for &(pos_j, strength) in &sources {
                        let dx = pos_i[0] - pos_j[0];
                        let dy = pos_i[1] - pos_j[1];
                        let dz = pos_i[2] - pos_j[2];
                        let d = (dx * dx + dy * dy + dz * dz).sqrt();
                        if d > 0.0 && d <= r_cut {
                            p_couple += rho * strength / d; // (ρ/d)·S_j
                        }
                    }

                    let p_total = driving[idx] + p_couple;
                    let p_prev = if self.total_seeded {
                        self.prev_total_pressure[idx]
                    } else {
                        p_total // first call: dp/dt = 0
                    };
                    self.prev_total_pressure[idx] = p_total;
                    let dp_dt = (p_total - p_prev) / dt;

                    let r_before = self.radius_field[idx].max(1e-12);
                    let mut state = BubbleState::new(&params);
                    state.radius = r_before;
                    state.wall_velocity = self.velocity_field[idx];
                    let integration =
                        integrate_bubble_dynamics_adaptive(&solver, &mut state, p_total, dp_dt, dt, time);

                    // A single bubble's adaptive non-convergence or non-finite state
                    // signals a destructive inertial collapse beyond the integrator's
                    // range (more frequent once coupling amplifies the drive). Handle
                    // it gracefully — re-nucleate at R₀ — rather than crashing the whole
                    // cloud; the prior r_before > R₀ still deposits the collapse work below.
                    if integration.is_err() || !state.radius.is_finite() || state.radius <= 0.0 {
                        state.radius = r0;
                        state.wall_velocity = 0.0;
                    }
                    let r_after = state.radius;

                    // Erosion = compression work on the collapsing bubble:
                    // dE = (p∞ − p_v)·(−dV) when the bubble shrinks under net
                    // compression (≈ Rayleigh collapse energy over a full collapse).
                    if r_after < r_before {
                        let p_drive = (p_total - p_vapor).max(0.0);
                        let dv = (4.0 / 3.0) * PI * (r_before.powi(3) - r_after.powi(3));
                        let erosion = density * p_drive * dv * efficiency;
                        self.accumulated_eroded_mass += erosion.max(0.0);
                    }

                    self.radius_field[idx] = r_after;
                    self.velocity_field[idx] = state.wall_velocity;
                }
            }
        }
        self.total_seeded = true;
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

    // ── Inter-bubble acoustic coupling (ADR 028) ──────────────────────────────

    #[test]
    fn radiated_pressure_matches_closed_form() {
        // p_rad = (ρ/d)(r²r̈ + 2rṙ²).
        let (rho, d, r, rdot, rddot) = (1000.0, 2.0e-3, 5.0e-6, 0.3, 1.0e9);
        let expected = rho / d * (r * r * rddot + 2.0 * r * rdot * rdot);
        let got = bubble_radiated_pressure(rho, d, r, rdot, rddot);
        assert!((got - expected).abs() < 1e-9 * expected.abs(), "expected {expected}, got {got}");
        // Zero/negative distance guarded.
        assert_eq!(bubble_radiated_pressure(rho, 0.0, r, rdot, rddot), 0.0);
    }

    #[test]
    fn radiated_pressure_scales_inverse_distance() {
        let p1 = bubble_radiated_pressure(1000.0, 1.0e-3, 5.0e-6, 0.3, 1.0e9);
        let p2 = bubble_radiated_pressure(1000.0, 2.0e-3, 5.0e-6, 0.3, 1.0e9);
        assert!((p1 - 2.0 * p2).abs() < 1e-9 * p1.abs(), "doubling distance must halve p_rad");
    }

    /// Drive a 2-cell cloud (separated by `spacing` along x) and return cell-0 radius.
    fn drive_two_cells(coupling: bool, spacing: f64, amplitudes: &[f64], dt: f64) -> f64 {
        let params = CloudParameters {
            coupling_enabled: coupling,
            cell_spacing: [spacing, 1.0, 1.0],
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 1, 1));
        cloud.density_field.fill(params.bubble_density);
        for (n, &amp) in amplitudes.iter().enumerate() {
            let t = n as f64 * dt;
            cloud
                .evolve_cloud(dt, t, &Array3::from_elem((2, 1, 1), amp))
                .unwrap();
        }
        cloud.cloud_radius()[[0, 0, 0]]
    }

    fn sinusoid(amp: f64, f: f64, dt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|m| amp * (2.0 * PI * f * (m as f64 * dt)).sin()).collect()
    }

    #[test]
    fn coupling_changes_two_bubble_trajectory() {
        let f = CloudParameters::default().drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let seq = sinusoid(0.5e6, f, dt, 200);
        let r_on = drive_two_cells(true, 1.0e-3, &seq, dt);
        let r_off = drive_two_cells(false, 1.0e-3, &seq, dt);
        assert!(
            (r_on - r_off).abs() > 1e-12 * r_off.max(1e-9),
            "coupling must change the two-bubble trajectory: on={r_on}, off={r_off}"
        );
    }

    #[test]
    fn closer_bubbles_couple_more_strongly() {
        let f = CloudParameters::default().drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let seq = sinusoid(0.5e6, f, dt, 200);
        let deviation = |spacing: f64| {
            (drive_two_cells(true, spacing, &seq, dt) - drive_two_cells(false, spacing, &seq, dt))
                .abs()
        };
        assert!(
            deviation(0.5e-3) > deviation(2.0e-3),
            "closer bubbles must couple more strongly (1/d)"
        );
    }

    #[test]
    fn lone_active_cell_is_unaffected_by_coupling() {
        // Multi-cell grid with a single active bubble ⇒ no neighbours ⇒ coupling
        // on/off identical (reduces exactly to ADR 027).
        let f = CloudParameters::default().drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let seq = sinusoid(0.5e6, f, dt, 100);
        let run = |coupling: bool| {
            let params = CloudParameters {
                coupling_enabled: coupling,
                ..CloudParameters::default()
            };
            let mut cloud = CavitationCloudDynamics::new(params.clone(), (3, 1, 1));
            cloud.density_field[[1, 0, 0]] = params.bubble_density; // only the middle cell
            for (n, &amp) in seq.iter().enumerate() {
                cloud
                    .evolve_cloud(dt, n as f64 * dt, &Array3::from_elem((3, 1, 1), amp))
                    .unwrap();
            }
            cloud.cloud_radius()[[1, 0, 0]]
        };
        assert_eq!(run(true), run(false), "a lone bubble must be unaffected by coupling");
    }

    // ── Cloud-scale acoustic shielding (ADR 029) ──────────────────────────────

    #[test]
    fn shielding_is_beer_lambert_exponential_decay() {
        use kwavers_physics::acoustics::bubble_dynamics::bubbly_medium::commander_prosperetti_attenuation;
        // Uniform dense cloud, drive near the bubble resonance ⇒ measurable α.
        let params = CloudParameters {
            shielding_enabled: true,
            incident_axis: 2,
            bubble_density: 1.0e15,
            drive_frequency: 3.0e6,
            ..CloudParameters::default()
        };
        let nz = 6;
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (1, 1, nz));
        cloud.density_field.fill(params.bubble_density); // radius_field stays at R₀
        let p_in = 1.0e6;
        let pressure = Array3::from_elem((1, 1, nz), p_in);
        let screened = cloud.shielded_pressure(&pressure);

        let bp = cloud.bubble_parameters();
        let beta = params.bubble_density * (4.0 / 3.0) * PI * bp.r0.powi(3);
        let alpha = commander_prosperetti_attenuation(
            bp.driving_frequency,
            beta,
            bp.r0,
            bp.c_liquid,
            bp.rho_liquid,
            bp.mu_liquid,
            bp.p0,
            bp.gamma,
        );
        assert!(alpha > 0.0, "expected positive attenuation for this cloud");
        let ds = params.cell_spacing[2];
        for k in 0..nz {
            let expected = p_in * (-(alpha * ds * (k as f64 + 0.5))).exp();
            assert!(
                (screened[[0, 0, k]] - expected).abs() <= 1e-9 * p_in,
                "k={k}: {} vs Beer-Lambert {expected}",
                screened[[0, 0, k]]
            );
        }
        assert!(
            screened[[0, 0, nz - 1]] < screened[[0, 0, 0]],
            "interior must be screened below the entry face"
        );
    }

    #[test]
    fn no_nuclei_means_no_shielding() {
        // β = 0 ⇒ α = 0 ⇒ the field passes through unattenuated.
        let params = CloudParameters {
            shielding_enabled: true,
            ..CloudParameters::default()
        };
        let cloud = CavitationCloudDynamics::new(params, (1, 1, 4)); // density_field = 0
        let pressure = Array3::from_elem((1, 1, 4), 2.0e6);
        let screened = cloud.shielded_pressure(&pressure);
        assert!(screened.iter().all(|&v| (v - 2.0e6).abs() < 1e-9));
    }

    #[test]
    fn denser_cloud_screens_interior_more() {
        let interior = |density: f64| {
            let params = CloudParameters {
                shielding_enabled: true,
                bubble_density: density,
                drive_frequency: 3.0e6,
                ..CloudParameters::default()
            };
            let nz = 6;
            let mut cloud = CavitationCloudDynamics::new(params, (1, 1, nz));
            cloud.density_field.fill(density);
            let pressure = Array3::from_elem((1, 1, nz), 1.0e6);
            cloud.shielded_pressure(&pressure)[[0, 0, nz - 1]]
        };
        assert!(
            interior(1.0e16) < interior(1.0e15),
            "a denser cloud must screen its interior more strongly"
        );
    }
}
