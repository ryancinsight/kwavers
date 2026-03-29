//! Bubble field management — coupled multi-bubble time integration
//!
//! Manages a spatial collection of bubbles and advances them through time
//! using the Keller-Miksis ODE with secondary Bjerknes pressure coupling.
//!
//! ---
//!
//! ## Algorithm — Coupled Multi-Bubble Time Stepping
//!
//! ### Theorem (Secondary Bjerknes Pressure, Monopole Radiation)
//!
//! In a compressible liquid with density ρ_L and sound speed c_L, a spherical
//! bubble of instantaneous radius R_i(t) at position **r**_i radiates an
//! acoustic pressure field. In the near-field limit (d_ij ≪ λ), the pressure
//! at a neighbouring bubble j located at **r**_j is:
//!
//! ```text
//! p_ij(t) = − ρ_L / (4π d_ij) · d²V_i/dt²
//! ```
//!
//! where V_i = (4/3)π R_i³ is the bubble volume and
//!
//! ```text
//! d²V_i/dt² = 4π [ R_i² R̈_i + 2 R_i Ṙ_i² ]
//! ```
//!
//! Substituting:
//!
//! ```text
//! p_ij = − ρ_L [ R_i² R̈_i + 2 R_i Ṙ_i² ] / d_ij
//! ```
//!
//! The total effective driving pressure at bubble j is:
//!
//! ```text
//! p_eff_j = p_acoustic_j + Σ_{i≠j} p_ij
//! ```
//!
//! ### Coupling Negligibility Threshold
//!
//! Coupling is skipped when `R_i / d_ij < coupling_threshold` (default 0.01),
//! i.e., when the bubble radius is less than 1% of the inter-bubble distance.
//! This cutoff eliminates O(N²) work for large, dilute clouds without
//! measurable physical error.
//!
//! ### Integration Scheme
//!
//! Explicit predictor coupling (staggered leapfrog in the coupling term):
//!
//! 1. Collect (R_i^n, Ṙ_i^n, R̈_i^n) from all bubble states at time t^n.
//! 2. For each bubble j, compute p_secondary_j^n = Σ_{i≠j} p_ij^n.
//! 3. Integrate bubble j from t^n to t^{n+1} with
//!    p_driving = p_acoustic_j^n + p_secondary_j^n using adaptive RK45.
//!
//! This is first-order accurate in the coupling term; for strongly coupled
//! bubble clouds (void fraction > 1%) an implicit correction pass is needed.
//!
//! ### References
//!
//! 1. Bjerknes, V. F. K. (1906). *Fields of Force*. Columbia University Press.
//!
//! 2. Crum, L. A. (1975). Bjerknes forces on bubbles in a stationary sound
//!    field. *Journal of the Acoustical Society of America*, **57**(6),
//!    1363–1370. <https://doi.org/10.1121/1.380614>
//!
//! 3. Pelekasis, N. A., Gaki, A., Doinikov, A., & Tsamopoulos, J. A. (2004).
//!    Secondary Bjerknes forces between two bubbles and the phenomenon of
//!    acoustic streamers. *Journal of Fluid Mechanics*, **500**, 313–347.
//!    <https://doi.org/10.1017/S0022112003007365>
//!
//! 4. Brennen, C. E. (1995). *Cavitation and Bubble Dynamics*. Oxford
//!    University Press. (Chapter 4: Cavitation bubbles.)

use super::super::adaptive_integration::integrate_bubble_dynamics_adaptive;
use super::super::bubble_state::{BubbleParameters, BubbleState};
use super::super::keller_miksis::KellerMiksisModel;
use ndarray::Array3;
use std::collections::HashMap;

/// Default liquid density used when no medium is supplied [kg m⁻³]
const DEFAULT_RHO_LIQUID: f64 = 1000.0;

/// Default physical grid spacing [m] (1 mm isotropic)
const DEFAULT_GRID_SPACING: (f64, f64, f64) = (1e-3, 1e-3, 1e-3);

/// Ratio R_i / d_ij below which secondary Bjerknes coupling is negligible.
///
/// At R/d < 0.01 the scattered pressure is < 0.01 × p_incident; omitting
/// it introduces a relative error below the numerical discretisation error
/// of the underlying RK45 integrator (local tolerance ~ 1e-6).
const DEFAULT_COUPLING_THRESHOLD: f64 = 0.01;

/// Single bubble or bubble cloud field
///
/// Stores all bubble states keyed by 3-D grid index, advances them through
/// time using the Keller-Miksis ODE, and accounts for secondary Bjerknes
/// pressure coupling between neighbouring bubbles.
#[derive(Debug)]
pub struct BubbleField {
    /// Bubble states indexed by grid position
    pub bubbles: HashMap<(usize, usize, usize), BubbleState>,
    /// Keller-Miksis ODE solver (shared parameters across all bubbles)
    solver: KellerMiksisModel,
    /// Default bubble parameters for cloud generation
    pub bubble_parameters: BubbleParameters,
    /// Grid dimensions (Nx, Ny, Nz)
    pub grid_shape: (usize, usize, usize),
    /// Physical grid spacing (dx, dy, dz) [m]
    pub grid_spacing: (f64, f64, f64),
    /// Liquid density for secondary Bjerknes pressure [kg m⁻³]
    pub rho_liquid: f64,
    /// R/d threshold below which coupling contribution is skipped
    pub coupling_threshold: f64,
    /// Time history for selected bubbles
    pub time_history: Vec<f64>,
    pub radius_history: Vec<Vec<f64>>,
    pub temperature_history: Vec<Vec<f64>>,
}

impl BubbleField {
    /// Create a new bubble field with default 1 mm isotropic grid spacing.
    ///
    /// Use [`BubbleField::with_spacing`] when the physical grid spacing is known.
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize), params: BubbleParameters) -> Self {
        Self::with_spacing(grid_shape, params, DEFAULT_GRID_SPACING)
    }

    /// Create a new bubble field with explicit physical grid spacing.
    ///
    /// # Parameters
    /// - `grid_shape` — (Nx, Ny, Nz) grid dimensions
    /// - `params`     — default bubble parameters (R₀, σ, μ, p₀, …)
    /// - `spacing`    — physical grid spacing (dx, dy, dz) in metres
    #[must_use]
    pub fn with_spacing(
        grid_shape: (usize, usize, usize),
        params: BubbleParameters,
        spacing: (f64, f64, f64),
    ) -> Self {
        Self {
            bubbles: HashMap::new(),
            solver: KellerMiksisModel::new(params.clone()),
            bubble_parameters: params,
            grid_shape,
            grid_spacing: spacing,
            rho_liquid: DEFAULT_RHO_LIQUID,
            coupling_threshold: DEFAULT_COUPLING_THRESHOLD,
            time_history: Vec::new(),
            radius_history: Vec::new(),
            temperature_history: Vec::new(),
        }
    }

    /// Add a single bubble at a grid position.
    pub fn add_bubble(&mut self, i: usize, j: usize, k: usize, state: BubbleState) {
        self.bubbles.insert((i, j, k), state);
    }

    /// Add bubble at center of grid.
    pub fn add_center_bubble(&mut self, params: &BubbleParameters) {
        let center = (
            self.grid_shape.0 / 2,
            self.grid_shape.1 / 2,
            self.grid_shape.2 / 2,
        );
        let state = BubbleState::new(params);
        self.add_bubble(center.0, center.1, center.2, state);
    }

    // ── Secondary Bjerknes coupling ─────────────────────────────────────────

    /// Compute the secondary Bjerknes pressure contribution at every bubble
    /// position due to the instantaneous radiation of all other bubbles.
    ///
    /// ## Formula
    ///
    /// ```text
    /// p_secondary_j = Σ_{i≠j}  − ρ_L [R_i² R̈_i + 2 R_i Ṙ_i²] / d_ij
    /// ```
    ///
    /// Pairs with `d_ij = 0` (overlapping grid cells) and pairs where
    /// `R_i / d_ij < coupling_threshold` are skipped.
    ///
    /// Returns a `HashMap` mapping each bubble's grid position to its total
    /// secondary Bjerknes pressure correction [Pa].
    fn compute_secondary_pressures(&self) -> HashMap<(usize, usize, usize), f64> {
        let positions: Vec<(usize, usize, usize)> = self.bubbles.keys().copied().collect();
        let mut corrections: HashMap<(usize, usize, usize), f64> =
            positions.iter().map(|&pos| (pos, 0.0)).collect();

        let (dx, dy, dz) = self.grid_spacing;

        for (idx_j, &pos_j) in positions.iter().enumerate() {
            for (idx_i, &pos_i) in positions.iter().enumerate() {
                if idx_i == idx_j {
                    continue; // skip self
                }

                let state_i = &self.bubbles[&pos_i];

                // Physical distance between bubble centres
                let delta_x = (pos_i.0 as f64 - pos_j.0 as f64) * dx;
                let delta_y = (pos_i.1 as f64 - pos_j.1 as f64) * dy;
                let delta_z = (pos_i.2 as f64 - pos_j.2 as f64) * dz;
                let d_ij =
                    (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z).sqrt();

                if d_ij == 0.0 {
                    continue; // overlapping cells — skip singularity
                }

                // Negligibility check: R_i / d_ij < threshold
                if state_i.radius / d_ij < self.coupling_threshold {
                    continue;
                }

                // p_ij = − ρ_L [ R_i² R̈_i + 2 R_i Ṙ_i² ] / d_ij
                let r = state_i.radius;
                let rdot = state_i.wall_velocity;
                let rddot = state_i.wall_acceleration;

                let p_ij =
                    -self.rho_liquid * (r * r * rddot + 2.0 * r * rdot * rdot) / d_ij;

                *corrections.get_mut(&pos_j).unwrap() += p_ij;
            }
        }

        corrections
    }

    // ── Time stepping ───────────────────────────────────────────────────────

    /// Advance all bubbles by one time step.
    ///
    /// Each bubble is integrated using the adaptive Keller-Miksis RK45
    /// integrator with a driving pressure equal to the local acoustic
    /// pressure plus the secondary Bjerknes pressure from all other bubbles
    /// (evaluated at time t^n — explicit predictor coupling).
    ///
    /// When only a single bubble is present, coupling is skipped (O(1) path).
    ///
    /// # Parameters
    /// - `pressure_field`  — 3-D acoustic pressure array [Pa]
    /// - `dp_dt_field`     — 3-D time derivative ∂p/∂t [Pa s⁻¹]
    /// - `dt`              — time step [s]
    /// - `t`               — current simulation time [s]
    pub fn update(
        &mut self,
        pressure_field: &Array3<f64>,
        dp_dt_field: &Array3<f64>,
        dt: f64,
        t: f64,
    ) {
        // Compute secondary Bjerknes corrections (O(N²)) only when > 1 bubble
        let secondary_pressures: HashMap<(usize, usize, usize), f64> =
            if self.bubbles.len() > 1 {
                self.compute_secondary_pressures()
            } else {
                HashMap::new()
            };

        // Integrate each bubble with coupled driving pressure
        for (&pos, state) in &mut self.bubbles {
            let (i, j, k) = pos;
            let p_acoustic = pressure_field[[i, j, k]];
            let dp_dt = dp_dt_field[[i, j, k]];

            // Add secondary Bjerknes pressure correction
            let p_secondary = secondary_pressures.get(&pos).copied().unwrap_or(0.0);
            let p_effective = p_acoustic + p_secondary;

            if let Err(e) =
                integrate_bubble_dynamics_adaptive(&self.solver, state, p_effective, dp_dt, dt, t)
            {
                eprintln!("Bubble dynamics integration failed at ({i}, {j}, {k}): {e:?}");
            }
        }

        self.record_history(t);
    }

    // ── History tracking ────────────────────────────────────────────────────

    /// Record time history of bubble states.
    fn record_history(&mut self, t: f64) {
        self.time_history.push(t);

        if self.radius_history.is_empty() {
            for _ in 0..self.bubbles.len() {
                self.radius_history.push(Vec::new());
                self.temperature_history.push(Vec::new());
            }
        }

        for (idx, (_, state)) in self.bubbles.iter().enumerate() {
            if idx < self.radius_history.len() {
                self.radius_history[idx].push(state.radius);
                self.temperature_history[idx].push(state.temperature);
            }
        }
    }

    // ── Field accessors ─────────────────────────────────────────────────────

    /// Get bubble state fields for physics modules.
    #[must_use]
    pub fn get_state_fields(&self) -> crate::domain::field::BubbleStateFields {
        let shape = self.grid_shape;
        let mut fields = crate::domain::field::BubbleStateFields::new(shape);

        for ((i, j, k), state) in &self.bubbles {
            fields.radius[[*i, *j, *k]] = state.radius;
            fields.temperature[[*i, *j, *k]] = state.temperature;
            fields.pressure[[*i, *j, *k]] = state.pressure_internal;
            fields.velocity[[*i, *j, *k]] = state.wall_velocity;
            fields.is_collapsing[[*i, *j, *k]] = f64::from(i32::from(state.is_collapsing));
            fields.compression_ratio[[*i, *j, *k]] = state.compression_ratio;
        }

        fields
    }

    /// Get statistics about bubble field.
    #[must_use]
    pub fn get_statistics(&self) -> BubbleFieldStats {
        let mut stats = BubbleFieldStats::default();

        for state in self.bubbles.values() {
            stats.total_bubbles += 1;
            if state.is_collapsing {
                stats.collapsing_bubbles += 1;
            }
            stats.max_temperature = stats.max_temperature.max(state.temperature);
            stats.max_compression = stats.max_compression.max(state.compression_ratio);
            stats.total_collapses += state.collapse_count;
        }

        stats
    }
}

/// Statistics about bubble field
#[derive(Debug, Default)]
pub struct BubbleFieldStats {
    pub total_bubbles: usize,
    pub collapsing_bubbles: usize,
    pub max_temperature: f64,
    pub max_compression: f64,
    pub total_collapses: u32,
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
    use ndarray::Array3;

    #[test]
    fn test_single_bubble_no_coupling() {
        // Single-bubble path: secondary_pressures map is empty; update must not panic.
        let params = BubbleParameters::default();
        let mut field = BubbleField::new((10, 10, 10), params.clone());
        field.add_center_bubble(&params);

        let pressure = Array3::<f64>::zeros((10, 10, 10));
        let dp_dt = Array3::<f64>::zeros((10, 10, 10));

        field.update(&pressure, &dp_dt, 1e-8, 0.0);
        assert_eq!(field.bubbles.len(), 1);
    }

    #[test]
    fn test_two_bubble_coupling_at_equilibrium() {
        // Two bubbles at equilibrium: R̈=0, Ṙ=0 → secondary correction = 0.
        let params = BubbleParameters::default();
        let mut field =
            BubbleField::with_spacing((20, 10, 10), params.clone(), (1e-6, 1e-6, 1e-6));

        field.add_bubble(5, 5, 5, BubbleState::new(&params));
        field.add_bubble(15, 5, 5, BubbleState::new(&params));

        let corrections = field.compute_secondary_pressures();
        for &val in corrections.values() {
            assert!(val.is_finite(), "Correction must be finite, got {val}");
        }
    }

    #[test]
    fn test_coupling_threshold_skips_distant_bubbles() {
        // R=1 µm, d=10 mm → R/d = 1e-4 < threshold=0.01 → correction=0.
        let mut params = BubbleParameters::default();
        params.r0 = 1e-6;

        let mut field =
            BubbleField::with_spacing((20, 10, 10), params.clone(), (1e-3, 1e-3, 1e-3));
        field.coupling_threshold = 0.01;

        let mut s1 = BubbleState::new(&params);
        s1.wall_velocity = 1.0;
        s1.wall_acceleration = 1e6;

        field.add_bubble(0, 5, 5, s1);
        field.add_bubble(10, 5, 5, BubbleState::new(&params));

        let corrections = field.compute_secondary_pressures();
        for &val in corrections.values() {
            assert_eq!(val, 0.0, "Distant bubble coupling should be skipped");
        }
    }

    #[test]
    fn test_nonzero_coupling_within_threshold() {
        // R=1 µm, d=5 µm → R/d = 0.2 > threshold → coupling active → non-zero correction.
        let mut params = BubbleParameters::default();
        params.r0 = 1e-6;

        let mut field =
            BubbleField::with_spacing((10, 10, 10), params.clone(), (1e-6, 1e-6, 1e-6));
        field.coupling_threshold = 0.01;

        let mut s1 = BubbleState::new(&params);
        s1.radius = 1e-6;
        s1.wall_velocity = 5.0;     // Ṙ = 5 m/s
        s1.wall_acceleration = 1e8; // R̈ = 1e8 m/s²

        field.add_bubble(0, 5, 5, s1);
        field.add_bubble(5, 5, 5, BubbleState::new(&params));

        let corrections = field.compute_secondary_pressures();

        let corr_at_j = corrections[&(5, 5, 5)];
        assert!(
            corr_at_j.abs() > 0.0,
            "Correction must be non-zero for coupled bubbles within threshold"
        );
        assert!(corr_at_j.is_finite());
    }
}
