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
use kwavers_math::linear_algebra::iterative::lsqr::{
    solve_lsqr_matfree, LsqrConfig, MatFreeOperator,
};
use kwavers_physics::acoustics::bubble_dynamics::adaptive_integration::integrate_bubble_dynamics_adaptive;
use kwavers_physics::acoustics::bubble_dynamics::bubbly_medium::commander_prosperetti_attenuation;
use kwavers_physics::acoustics::bubble_dynamics::gilmore::GilmoreSolver;
use kwavers_physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use kwavers_physics::acoustics::bubble_dynamics::{BubbleParameters, BubbleState};
use leto::{
    Array1,
    Array2,
    Array3,
};
use leto_ops::solve;
use std::f64::consts::PI;

/// How the inter-bubble acoustic coupling is solved each step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingScheme {
    /// Explicit/lagged single pass — source strengths from the previous total
    /// pressure (ADR 028). Cheapest; correct for small `dt`.
    Explicit,
    /// Self-consistent fixed-point iteration with under-relaxation `ω ∈ (0,1]`
    /// (ADR 030/031). `ω = 1` is plain Jacobi; `ω < 1` damps the divergence for
    /// stronger coupling. `O(iterations·active²)`.
    ImplicitFixedPoint {
        /// Under-relaxation factor `ω ∈ (0, 1]`.
        under_relaxation: f64,
    },
    /// Direct linear solve of the affine coupling system `(I − D·G)·S = e`
    /// (ADR 031): exact and robust regardless of coupling strength. `O(active³)`.
    ImplicitDirect,
    /// Matrix-free iterative (LSQR) solve of the same affine system (ADR 032):
    /// `O(active·neighbours)` memory, for very large active clouds where the dense
    /// direct solve is intractable.
    ImplicitIterative,
}

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
    /// How the inter-bubble coupling is solved (explicit / self-consistent
    /// fixed-point / direct linear solve). Default [`CouplingScheme::Explicit`];
    /// only relevant when `coupling_enabled` (ADR 028/030/031).
    pub coupling_scheme: CouplingScheme,
    /// Max fixed-point iterations for the self-consistent coupling solve.
    pub coupling_max_iterations: usize,
    /// Relative convergence tolerance for the self-consistent coupling solve.
    pub coupling_tolerance: f64,
    /// Include the driving-pressure **rate** `dp/dt` in the coupling source
    /// strengths (ADR 032). The Keller-Miksis acceleration is affine in both `p`
    /// and `dp/dt`; when enabled, the per-cell lagged finite-difference rate
    /// `(driving − prev_total)/dt` is fed into the source/affine acceleration so
    /// the radiated strengths carry the acoustic-radiation term. Opt-in (default
    /// `false`); the rate is explicit (lagged one step).
    pub couple_pressure_rate: bool,
    /// Use the instantaneous per-cell radius `R(t)` (not the equilibrium `R0`) for
    /// the Commander-Prosperetti resonance in the shielding screen (ADR 032).
    /// Quasi-static extension of the linear CP theory; opt-in (default `false`).
    pub shielding_radius_dependent: bool,
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
            cell_spacing: [1.0e-3; 3], // 1 mm
            coupling_enabled: false,   // opt-in (O(active²); see field docs)
            interaction_radius: f64::INFINITY,
            shielding_enabled: false, // opt-in (ADR 029)
            incident_axis: 2,         // z by default
            incident_from_high: false,
            coupling_scheme: CouplingScheme::Explicit, // ADR 028; opt into implicit
            coupling_max_iterations: 16,
            coupling_tolerance: 1e-3,
            couple_pressure_rate: false,       // opt-in (ADR 032)
            shielding_radius_dependent: false, // opt-in (ADR 032)
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

/// Euclidean distance [m] between two cell positions.
#[inline]
fn pair_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Matrix-free linear operator for the coupling system `M = I − D·G` (ADR 032).
///
/// `G_ab = ρ/d_ab` (within the cutoff, 0 on the diagonal) is computed on the fly
/// from cell positions, so no `n×n` matrix is stored: `O(n)` memory,
/// `O(n·neighbours)` per matvec. `G` is symmetric (`d_ab = d_ba`), so
/// `Mᵀ = (I − D·G)ᵀ = I − G·D`.
struct CouplingOperator {
    positions: Vec<[f64; 3]>,
    /// Per-cell affine slope `d_a = R_a²·∂R̈_a/∂p`.
    d: Vec<f64>,
    rho: f64,
    r_cut: f64,
}

impl CouplingOperator {
    /// `(G·s)_a = Σ_{b≠a, d_ab≤cut} (ρ/d_ab)·s_b`.
    fn apply_g(&self, s: &[f64]) -> Vec<f64> {
        let n = self.positions.len();
        (0..n)
            .map(|a| {
                let mut acc = 0.0_f64;
                for (b, (&pos_b, &s_b)) in self.positions.iter().zip(s.iter()).enumerate() {
                    if a == b {
                        continue;
                    }
                    let dist = pair_distance(self.positions[a], pos_b);
                    if dist > 0.0 && dist <= self.r_cut {
                        acc += self.rho / dist * s_b;
                    }
                }
                acc
            })
            .collect()
    }
}

impl MatFreeOperator for CouplingOperator {
    fn rows(&self) -> usize {
        self.positions.len()
    }
    fn cols(&self) -> usize {
        self.positions.len()
    }
    /// `y = M·x = x − D·(G·x)`.
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let gx = self.apply_g(x);
        for a in 0..self.positions.len() {
            y[a] = self.d[a].mul_add(-gx[a], x[a]);
        }
    }
    /// `x = Mᵀ·y = y − G·(D·y)` (`G` symmetric).
    fn t_matvec(&self, y: &[f64], x: &mut [f64]) {
        let dy: Vec<f64> = y
            .iter()
            .zip(self.d.iter())
            .map(|(&yi, &di)| di * yi)
            .collect();
        let g_dy = self.apply_g(&dy);
        for a in 0..self.positions.len() {
            x[a] = y[a] - g_dy[a];
        }
    }
}

/// Linear growth-rate **diagnostic** for cloud-interface instabilities (ADR 032).
///
/// The bubbly cloud (effective Wood mixture density `ρ_mix = (1−β)·ρ_L`) is the
/// **light** fluid; the surrounding liquid (`ρ_L`) is the **heavy** fluid. This is
/// a *linear* diagnostic — closed-form growth rates of a small perturbation at the
/// cloud edge — not a nonlinear interface simulation (no mushrooms / mixing).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceInstability {
    /// Representative cloud void fraction `β` used for the mixture density.
    pub void_fraction: f64,
    /// Atwood number `A = (ρ_L − ρ_mix)/(ρ_L + ρ_mix) = β/(2−β)`.
    pub atwood: f64,
    /// Rayleigh-Taylor growth rate `σ = √(A·k·a)` [1/s] (0 when stable, i.e. the
    /// heavy fluid is not accelerated into the light one, `a ≤ 0`). The
    /// perturbation grows as `exp(σ t)`.
    pub rayleigh_taylor_rate: f64,
    /// Richtmyer-Meshkov (impulsive) linear amplitude growth rate
    /// `ȧ = k·Δv·a₀·A` [m/s] (Richtmyer 1960); constant in the linear regime.
    pub richtmyer_meshkov_rate: f64,
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
        if self.density_field.shape() != pressure.shape() {
            self.density_field = Array3::zeros(pressure.shape());
            self.radius_field = Array3::from_elem(pressure.shape(), r0);
            self.velocity_field = Array3::zeros(pressure.shape());
            self.prev_total_pressure = Array3::zeros(pressure.shape());
            self.total_seeded = false;
        } else {
            // Reset the representative bubbles to equilibrium for a fresh run.
            self.radius_field.fill(r0);
            self.velocity_field.fill(0.0);
            self.total_seeded = false;
        }
        // Simple init: nucleate bubbles where pressure < threshold (-1 MPa) and near stone
        let threshold = -MPA_TO_PA;
        for ([i, j, k], p) in pressure.indexed_iter() {
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
        let [nx, ny, nz] = pressure.shape();
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
                        // CP resonance radius: equilibrium R₀, or the instantaneous
                        // R(t) when the quasi-static refinement is enabled (ADR 032).
                        let r_res = if self.parameters.shielding_radius_dependent {
                            r
                        } else {
                            r0
                        };
                        commander_prosperetti_attenuation(f, beta, r_res, c, rho, mu, p0, gamma)
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

    /// Active cells `(idx, position, R, Ṙ)` with seeded bubbles.
    fn active_cells(&self) -> Vec<([usize; 3], [f64; 3], f64, f64)> {
        let [nx, ny, nz] = self.density_field.shape();
        let mut cells = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = [i, j, k];
                    if self.density_field[idx] > 0.0 {
                        cells.push((
                            idx,
                            self.cell_position(i, j, k),
                            self.radius_field[idx].max(1e-12),
                            self.velocity_field[idx],
                        ));
                    }
                }
            }
        }
        cells
    }

    /// Keller-Miksis wall acceleration `R̈` of one bubble at `(R, Ṙ)` driven by
    /// `p_drive` with rate `dp_dt` (ADR 032; `dp_dt = 0` ⇒ instantaneous-only).
    /// `0` if non-finite.
    fn raw_acceleration(
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        r: f64,
        v: f64,
        p_drive: f64,
        dp_dt: f64,
        time: f64,
    ) -> f64 {
        let mut state = BubbleState::new(params);
        state.radius = r;
        state.wall_velocity = v;
        let a = solver
            .calculate_acceleration(&mut state, p_drive, dp_dt, time)
            .unwrap_or(0.0);
        if a.is_finite() {
            a
        } else {
            0.0
        }
    }

    /// Source strength `S = R²R̈ + 2RṘ²` of one bubble driven at `p_drive` with
    /// rate `dp_dt` (ADR 032).
    fn source_strength(
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        r: f64,
        v: f64,
        p_drive: f64,
        dp_dt: f64,
        time: f64,
    ) -> f64 {
        let accel = Self::raw_acceleration(solver, params, r, v, p_drive, dp_dt, time);
        let s = accel.mul_add(r * r, 2.0 * r * v * v);
        if s.is_finite() {
            s
        } else {
            0.0
        }
    }

    /// Inter-bubble acoustic coupling pressure per cell. Returns a zero field when
    /// coupling is disabled. The scheme is selected by `coupling_scheme`:
    /// [`CouplingScheme::Explicit`] (lagged single pass, ADR 028),
    /// [`CouplingScheme::ImplicitFixedPoint`] (under-relaxed self-consistent
    /// iteration, ADR 030/031), or [`CouplingScheme::ImplicitDirect`] (exact linear
    /// solve of `(I − D·G)·S = e`, robust in the strong regime, ADR 031).
    fn coupling_pressure_field(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        driving: &Array3<f64>,
        dt: f64,
        time: f64,
    ) -> Array3<f64> {
        let mut field = Array3::<f64>::zeros(self.density_field.shape());
        if !self.parameters.coupling_enabled {
            return field;
        }
        let cells = self.active_cells();
        let n = cells.len();
        if n == 0 {
            return field;
        }

        // Per-cell driving-rate dp/dt for the source strengths (ADR 032). Lagged
        // finite difference against the previous total pressure; 0 unless the
        // `couple_pressure_rate` refinement is enabled (then defaults reduce to
        // the instantaneous-only ADR 028/031 behaviour exactly).
        let rates: Vec<f64> = cells
            .iter()
            .map(|&(idx, ..)| {
                if self.parameters.couple_pressure_rate && self.total_seeded {
                    (driving[idx] - self.prev_total_pressure[idx]) / dt
                } else {
                    0.0
                }
            })
            .collect();

        // Precompute the pairwise coupling matrix G_ab = ρ/d_ab (within cutoff;
        // 0 on the diagonal). p_couple = G·S for any source-strength vector S.
        // (`ImplicitIterative` is matrix-free and never forms this dense matrix.)
        let g = self.coupling_matrix(&cells);
        let apply_g = |s: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|a| (0..n).map(|b| g[[a, b]] * s[b]).sum())
                .collect()
        };

        let p_couple: Vec<f64> = match self.parameters.coupling_scheme {
            CouplingScheme::Explicit => {
                // Lagged source strengths from the previous total pressure (ADR 028).
                let strengths: Vec<f64> = cells
                    .iter()
                    .enumerate()
                    .map(|(a, &(idx, _, r, v))| {
                        let p_src = if self.total_seeded {
                            self.prev_total_pressure[idx]
                        } else {
                            0.0
                        };
                        Self::source_strength(solver, params, r, v, p_src, rates[a], time)
                    })
                    .collect();
                apply_g(&strengths)
            }
            CouplingScheme::ImplicitFixedPoint { under_relaxation } => self.fixed_point_coupling(
                solver,
                params,
                driving,
                time,
                &g,
                &rates,
                under_relaxation,
            ),
            CouplingScheme::ImplicitDirect => {
                self.direct_coupling(solver, params, driving, time, &g, &rates, &apply_g)
            }
            CouplingScheme::ImplicitIterative => {
                self.iterative_coupling(solver, params, driving, time, &cells, &rates)
            }
        };

        for (a, &(idx, ..)) in cells.iter().enumerate() {
            field[idx] = p_couple[a];
        }
        field
    }

    /// Self-consistent coupling by under-relaxed fixed-point iteration (ADR 030/031).
    /// `cells`/`g` describe the active bubbles and their coupling matrix.
    // Coupling-solve inputs (solver/params/driving/time/G/rates) are intrinsic to
    // the affine system; bundling them buys no clarity here.
    #[allow(clippy::too_many_arguments)]
    fn fixed_point_coupling(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        driving: &Array3<f64>,
        time: f64,
        g: &Array2<f64>,
        rates: &[f64],
        under_relaxation: f64,
    ) -> Vec<f64> {
        let cells = self.active_cells();
        let n = cells.len();
        let omega = under_relaxation.clamp(1e-3, 1.0);
        let scale = driving.iter().fold(0.0_f64, |m, &p| m.max(p.abs())) + params.p0;
        let tol = self.parameters.coupling_tolerance * scale.max(1.0);
        let mut p_couple = vec![0.0_f64; n];
        for _ in 0..self.parameters.coupling_max_iterations.max(1) {
            let strengths: Vec<f64> = (0..n)
                .map(|a| {
                    let (idx, _, r, v) = cells[a];
                    Self::source_strength(
                        solver,
                        params,
                        r,
                        v,
                        driving[idx] + p_couple[a],
                        rates[a],
                        time,
                    )
                })
                .collect();
            let mut residual = 0.0_f64;
            for a in 0..n {
                let computed: f64 = (0..n).map(|b| g[[a, b]] * strengths[b]).sum();
                let updated = (1.0 - omega).mul_add(p_couple[a], omega * computed);
                residual = residual.max((updated - p_couple[a]).abs());
                p_couple[a] = updated;
            }
            if residual < tol {
                break;
            }
        }
        p_couple
    }

    /// Direct linear solve of the affine coupling system `(I − D·G)·S = e` (ADR 031),
    /// robust regardless of coupling strength. Falls back to an under-relaxed
    /// fixed point if the system is singular/ill-posed.
    /// Affine source-strength coefficients `(c_a, d_a)` with `S_a = c_a + d_a·p_a`
    /// for each active cell `a`. `R̈` is affine in the driving pressure (and in
    /// `dp/dt`), so two acceleration evaluations at `p = 0` and `p = p_ref` (the
    /// same `dp/dt = rates[a]`) reconstruct it exactly; the `dp/dt` term is folded
    /// into `c_a` and leaves the slope `d_a` unchanged (ADR 031/032).
    fn affine_coeffs(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        cells: &[([usize; 3], [f64; 3], f64, f64)],
        rates: &[f64],
        time: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let p_ref = params.p0.max(1.0);
        let mut c = vec![0.0_f64; cells.len()];
        let mut d = vec![0.0_f64; cells.len()];
        for (a, &(_, _, r, v)) in cells.iter().enumerate() {
            let accel0 = Self::raw_acceleration(solver, params, r, v, 0.0, rates[a], time);
            let accel_ref = Self::raw_acceleration(solver, params, r, v, p_ref, rates[a], time);
            let b_coef = (accel_ref - accel0) / p_ref;
            c[a] = (r * r).mul_add(accel0, 2.0 * r * v * v); // R²·R̈(0,dp/dt) + 2RṘ²
            d[a] = r * r * b_coef; // R²·∂R̈/∂p
        }
        (c, d)
    }

    // Coupling-solve inputs are intrinsic to the affine system (see above).
    #[allow(clippy::too_many_arguments)]
    fn direct_coupling(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        driving: &Array3<f64>,
        time: f64,
        g: &Array2<f64>,
        rates: &[f64],
        apply_g: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let cells = self.active_cells();
        let n = cells.len();
        let (c, d) = self.affine_coeffs(solver, params, &cells, rates, time);
        // M = I − D·G ; e = c + D·p_ext.
        let mut m = Array2::<f64>::eye(n);
        let mut e = Array1::<f64>::zeros(n);
        for a in 0..n {
            e[a] = d[a].mul_add(driving[cells[a].0], c[a]);
            for b in 0..n {
                m[[a, b]] -= d[a] * g[[a, b]];
            }
        }
        match solve(&m.view(), &e.view()) {
            Ok(s) => apply_g(s.as_slice().unwrap_or(&[])),
            Err(_) => {
                // Singular/ill-posed system: surface via the fallback path (a damped
                // fixed point), never silent zeros.
                self.fixed_point_coupling(solver, params, driving, time, g, rates, 0.5)
            }
        }
    }

    /// Matrix-free iterative solve of the affine coupling system `(I − D·G)·S = e`
    /// (ADR 032) via damped LSQR. Builds no `n×n` matrix: the operator applies
    /// `M·x = x − D·(G·x)` (and `Mᵀ = I − G·D`, `G` symmetric) computing
    /// `G_ab = ρ/d_ab` on the fly within the cutoff — `O(n)` memory,
    /// `O(n·neighbours)` per matvec — for very large active counts. Returns
    /// `p_couple = G·S`. Falls back to a damped fixed point on non-convergence.
    fn iterative_coupling(
        &self,
        solver: &KellerMiksisModel,
        params: &BubbleParameters,
        driving: &Array3<f64>,
        time: f64,
        cells: &[([usize; 3], [f64; 3], f64, f64)],
        rates: &[f64],
    ) -> Vec<f64> {
        let n = cells.len();
        let (c, d) = self.affine_coeffs(solver, params, cells, rates, time);
        let e: Vec<f64> = (0..n)
            .map(|a| d[a].mul_add(driving[cells[a].0], c[a]))
            .collect();
        let op = CouplingOperator {
            positions: cells.iter().map(|&(_, p, ..)| p).collect(),
            d,
            rho: params.rho_liquid,
            r_cut: self.parameters.interaction_radius,
        };
        let scale = e.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
        let tol = self.parameters.coupling_tolerance * scale;
        let config = LsqrConfig {
            max_iterations: self.parameters.coupling_max_iterations.max(1),
            tolerance: self.parameters.coupling_tolerance,
            damping: 0.0,
            atol: tol,
            btol: tol,
        };
        let result = solve_lsqr_matfree(&op, &e, &config);
        let s = result.solution;
        // p_couple = G·S (matrix-free apply, consistent with the dense `apply_g`).
        let g_s = op.apply_g(&s);
        // Fall back to a damped fixed point if the source strengths diverged.
        if g_s.iter().all(|x| x.is_finite()) {
            g_s
        } else {
            let g = self.coupling_matrix(cells);
            self.fixed_point_coupling(solver, params, driving, time, &g, rates, 0.5)
        }
    }

    /// Dense pairwise coupling matrix `G_ab = ρ/d_ab` (0 on the diagonal / beyond
    /// the cutoff) for the active `cells`.
    fn coupling_matrix(&self, cells: &[([usize; 3], [f64; 3], f64, f64)]) -> Array2<f64> {
        let n = cells.len();
        let rho = self.bubble_parameters().rho_liquid;
        let r_cut = self.parameters.interaction_radius;
        let mut g = Array2::<f64>::zeros((n, n));
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let dist = pair_distance(cells[a].1, cells[b].1);
                if dist > 0.0 && dist <= r_cut {
                    g[[a, b]] = rho / dist;
                }
            }
        }
        g
    }

    /// Representative cloud void fraction `β = mean over seeded cells of
    /// n·(4/3)π R(t)³`, clamped to `[0, 1)`. `0` if no cells are seeded.
    #[must_use]
    pub fn representative_void_fraction(&self) -> f64 {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for (&n, &r) in self.density_field.iter().zip(self.radius_field.iter()) {
            if n > 0.0 {
                sum += (n * (4.0 / 3.0) * PI * r.max(0.0).powi(3)).clamp(0.0, 1.0 - 1e-9);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Linear growth-rate **diagnostic** for cloud-interface (Rayleigh-Taylor /
    /// Richtmyer-Meshkov) instabilities (ADR 032).
    ///
    /// The cloud (Wood mixture density `ρ_mix = (1−β)·ρ_L`, the light fluid) borders
    /// the liquid (`ρ_L`, heavy), giving Atwood number `A = β/(2−β)`. For a
    /// perturbation of wavenumber `wavenumber` [1/m]:
    /// - Rayleigh-Taylor: `σ = √(A·k·a)` [1/s], real only when the interface
    ///   `acceleration` `a` [m/s²] drives the heavy fluid into the light
    ///   (`a > 0`); `a ≤ 0` ⇒ stable (rate `0`).
    /// - Richtmyer-Meshkov: `ȧ = k·Δv·a₀·A` [m/s] for impulsive velocity jump
    ///   `velocity_jump` `Δv` [m/s] and initial amplitude `initial_amplitude` `a₀`
    ///   [m] (Richtmyer 1960).
    ///
    /// This returns growth **rates**; it does not evolve the interface (the
    /// nonlinear evolution remains out of scope, CLD-1).
    #[must_use]
    pub fn interface_instability(
        &self,
        wavenumber: f64,
        acceleration: f64,
        velocity_jump: f64,
        initial_amplitude: f64,
    ) -> InterfaceInstability {
        let beta = self.representative_void_fraction();
        // ρ_mix = (1−β)·ρ_L ⇒ A = (ρ_L − ρ_mix)/(ρ_L + ρ_mix) = β/(2−β).
        let atwood = beta / (2.0 - beta);
        let k = wavenumber.max(0.0);
        let rt_arg = atwood * k * acceleration;
        let rayleigh_taylor_rate = if rt_arg > 0.0 { rt_arg.sqrt() } else { 0.0 };
        let richtmyer_meshkov_rate = k * velocity_jump * initial_amplitude * atwood;
        InterfaceInstability {
            void_fraction: beta,
            atwood,
            rayleigh_taylor_rate,
            richtmyer_meshkov_rate,
        }
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
        if pressure.shape() != self.density_field.shape() {
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
        let [nx, ny, nz] = self.density_field.shape();

        // Cloud-scale shielding (ADR 029): screen the incident field by the cloud's
        // void fraction before it drives the bubbles (avoids cloning when off).
        let shielded_field;
        let driving: &Array3<f64> = if self.parameters.shielding_enabled {
            shielded_field = self.shielded_pressure(pressure);
            &shielded_field
        } else {
            pressure
        };

        // Inter-bubble acoustic coupling field (ADR 028 explicit / ADR 030 implicit).
        let coupling = self.coupling_pressure_field(&solver, &params, driving, dt, time);

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

                    let p_total = driving[idx] + coupling[idx];
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
                    let integration = integrate_bubble_dynamics_adaptive(
                        &solver, &mut state, p_total, dp_dt, dt, time,
                    );

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
        assert!(
            r_deep > r_shallow && r_shallow > r0,
            "deeper tension must grow larger: {r_deep} > {r_shallow} > {r0}"
        );
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
        assert!(
            r_max_strong > r_max_weak,
            "R_max must increase with drive amplitude"
        );
    }

    #[test]
    fn test_inertial_collapse_energy_scales_with_drive() {
        // E = (4/3)π R_max³ Δp grows with the rarefactional amplitude (via R_max).
        let cloud = CavitationCloudDynamics::new(CloudParameters::default(), (1, 1, 1));
        let e_weak = cloud.inertial_collapse_energy(3.0 * MPA_TO_PA);
        let e_strong = cloud.inertial_collapse_energy(12.0 * MPA_TO_PA);
        assert!(
            e_weak > 0.0 && e_strong > e_weak,
            "collapse energy must rise with drive"
        );
    }

    // ── Inter-bubble acoustic coupling (ADR 028) ──────────────────────────────

    #[test]
    fn radiated_pressure_matches_closed_form() {
        // p_rad = (ρ/d)(r²r̈ + 2rṙ²).
        let (rho, d, r, rdot, rddot) = (1000.0, 2.0e-3, 5.0e-6, 0.3, 1.0e9);
        let expected = rho / d * (r * r * rddot + 2.0 * r * rdot * rdot);
        let got = bubble_radiated_pressure(rho, d, r, rdot, rddot);
        assert!(
            (got - expected).abs() < 1e-9 * expected.abs(),
            "expected {expected}, got {got}"
        );
        // Zero/negative distance guarded.
        assert_eq!(bubble_radiated_pressure(rho, 0.0, r, rdot, rddot), 0.0);
    }

    #[test]
    fn radiated_pressure_scales_inverse_distance() {
        let p1 = bubble_radiated_pressure(1000.0, 1.0e-3, 5.0e-6, 0.3, 1.0e9);
        let p2 = bubble_radiated_pressure(1000.0, 2.0e-3, 5.0e-6, 0.3, 1.0e9);
        assert!(
            (p1 - 2.0 * p2).abs() < 1e-9 * p1.abs(),
            "doubling distance must halve p_rad"
        );
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
        (0..n)
            .map(|m| amp * (2.0 * PI * f * (m as f64 * dt)).sin())
            .collect()
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
        assert_eq!(
            run(true),
            run(false),
            "a lone bubble must be unaffected by coupling"
        );
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

    // ── Self-consistent (implicit) coupling (ADR 030) ─────────────────────────

    #[test]
    fn implicit_coupling_field_is_self_consistent() {
        // KEYSTONE (ADR 030): the returned coupling field satisfies its own
        // fixed-point equation — recomputing p_couple from the final source
        // strengths reproduces it. Two bubbles, distance d.
        let d = 1.0e-3;
        let params = CloudParameters {
            coupling_enabled: true,
            coupling_scheme: CouplingScheme::ImplicitFixedPoint {
                under_relaxation: 1.0,
            },
            cell_spacing: [d, 1.0, 1.0],
            coupling_tolerance: 1e-9,
            coupling_max_iterations: 200,
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 1, 1));
        cloud.density_field.fill(params.bubble_density);
        cloud.radius_field[[0, 0, 0]] = 1.5e-6;
        cloud.velocity_field[[0, 0, 0]] = 2.0;
        cloud.radius_field[[1, 0, 0]] = 1.2e-6;
        cloud.velocity_field[[1, 0, 0]] = -1.5;

        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let p_in = 0.5e6;
        let driving = Array3::from_elem((2, 1, 1), p_in);
        let field = cloud.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);

        // Fixed point: field[i] = (ρ/d)·S_j(driving_j + field_j), j the other cell.
        let s = |idx: [usize; 3]| {
            CavitationCloudDynamics::source_strength(
                &solver,
                &bp,
                cloud.radius_field[idx],
                cloud.velocity_field[idx],
                p_in + field[idx],
                0.0,
                0.0,
            )
        };
        let coeff = bp.rho_liquid / d;
        let expect0 = coeff * s([1, 0, 0]);
        let expect1 = coeff * s([0, 0, 0]);
        let tol = 1e-6 * (field[[0, 0, 0]].abs().max(field[[1, 0, 0]].abs()).max(1.0));
        assert!(
            (field[[0, 0, 0]] - expect0).abs() < tol,
            "cell0 not self-consistent: {} vs {expect0}",
            field[[0, 0, 0]]
        );
        assert!(
            (field[[1, 0, 0]] - expect1).abs() < tol,
            "cell1 not self-consistent: {} vs {expect1}",
            field[[1, 0, 0]]
        );
    }

    /// Drive a 2-cell cloud with a given coupling scheme; return cell-0 radius.
    fn drive_two_cells_scheme(
        scheme: CouplingScheme,
        spacing: f64,
        amplitudes: &[f64],
        dt: f64,
    ) -> f64 {
        let params = CloudParameters {
            coupling_enabled: true,
            coupling_scheme: scheme,
            cell_spacing: [spacing, 1.0, 1.0],
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 1, 1));
        cloud.density_field.fill(params.bubble_density);
        for (n, &amp) in amplitudes.iter().enumerate() {
            cloud
                .evolve_cloud(dt, n as f64 * dt, &Array3::from_elem((2, 1, 1), amp))
                .unwrap();
        }
        cloud.cloud_radius()[[0, 0, 0]]
    }

    #[test]
    fn implicit_differs_from_explicit_under_coupling() {
        // The self-consistent solve removes the explicit lag, so the two schemes
        // give different trajectories for close (non-negligible) coupling.
        let f = CloudParameters::default().drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let seq = sinusoid(0.5e6, f, dt, 150);
        let r_impl = drive_two_cells_scheme(
            CouplingScheme::ImplicitFixedPoint {
                under_relaxation: 1.0,
            },
            0.5e-3,
            &seq,
            dt,
        );
        let r_expl = drive_two_cells_scheme(CouplingScheme::Explicit, 0.5e-3, &seq, dt);
        assert!(
            (r_impl - r_expl).abs() > 1e-12 * r_expl.max(1e-9),
            "implicit and explicit coupling must differ: impl={r_impl}, expl={r_expl}"
        );
    }

    // ── Strong-regime direct solver (ADR 031) ─────────────────────────────────

    /// Build a perturbed 2-bubble cloud at separation `spacing` with `scheme`.
    fn two_bubble_cloud(scheme: CouplingScheme, spacing: f64) -> CavitationCloudDynamics {
        let params = CloudParameters {
            coupling_enabled: true,
            coupling_scheme: scheme,
            cell_spacing: [spacing, 1.0, 1.0],
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 1, 1));
        cloud.density_field.fill(params.bubble_density);
        cloud.radius_field[[0, 0, 0]] = 1.6e-6;
        cloud.velocity_field[[0, 0, 0]] = 2.5;
        cloud.radius_field[[1, 0, 0]] = 1.3e-6;
        cloud.velocity_field[[1, 0, 0]] = -1.8;
        cloud
    }

    /// Max relative self-consistency residual of a coupling field: how far
    /// `field_i` is from `(ρ/d)·S_j(p_ext+field_j)` (the fixed-point equation).
    fn self_consistency_residual(
        cloud: &CavitationCloudDynamics,
        field: &Array3<f64>,
        spacing: f64,
        p_in: f64,
    ) -> f64 {
        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let s = |idx: [usize; 3]| {
            CavitationCloudDynamics::source_strength(
                &solver,
                &bp,
                cloud.radius_field[idx],
                cloud.velocity_field[idx],
                p_in + field[idx],
                0.0,
                0.0,
            )
        };
        let coeff = bp.rho_liquid / spacing;
        let r0 = (field[[0, 0, 0]] - coeff * s([1, 0, 0])).abs();
        let r1 = (field[[1, 0, 0]] - coeff * s([0, 0, 0])).abs();
        let scale = field[[0, 0, 0]].abs().max(field[[1, 0, 0]].abs()).max(1.0);
        r0.max(r1) / scale
    }

    #[test]
    fn direct_solve_is_self_consistent_even_in_strong_regime() {
        // KEYSTONE (ADR 031): the exact linear solve satisfies the fixed-point
        // equation to ~machine precision even at close (strong) coupling where the
        // plain Jacobi fixed point would diverge.
        let spacing = 2.0e-5; // 20 µm — strong coupling
        let cloud = two_bubble_cloud(CouplingScheme::ImplicitDirect, spacing);
        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let p_in = 0.5e6;
        let driving = Array3::from_elem((2, 1, 1), p_in);
        let field = cloud.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);
        let residual = self_consistency_residual(&cloud, &field, spacing, p_in);
        assert!(
            residual < 1e-9,
            "direct solve must be self-consistent: residual {residual}"
        );
    }

    #[test]
    fn direct_matches_fixed_point_in_weak_regime() {
        // Where the fixed point converges (weak coupling), the direct solve agrees.
        let spacing = 3.0e-3; // far apart — weak coupling, Jacobi converges
        let p_in = 0.5e6;
        let driving = Array3::from_elem((2, 1, 1), p_in);

        let direct = two_bubble_cloud(CouplingScheme::ImplicitDirect, spacing);
        let bp = direct.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let f_direct = direct.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);

        let fp = two_bubble_cloud(
            CouplingScheme::ImplicitFixedPoint {
                under_relaxation: 1.0,
            },
            spacing,
        );
        let f_fp = fp.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);

        for idx in [[0, 0, 0], [1, 0, 0]] {
            let scale = f_direct[idx].abs().max(1.0);
            assert!(
                (f_direct[idx] - f_fp[idx]).abs() <= 1e-3 * scale,
                "direct and converged fixed point must agree (weak): {} vs {}",
                f_direct[idx],
                f_fp[idx]
            );
        }
    }

    // ── dp/dt coupling (ADR 032) ──────────────────────────────────────────────

    #[test]
    fn source_strength_responds_to_pressure_rate() {
        // R̈ depends on dp/dt (acoustic-radiation term): a non-zero rate changes
        // the source strength; dp/dt = 0 reproduces the instantaneous-only value.
        let params = CloudParameters::default();
        let cloud = CavitationCloudDynamics::new(params, (1, 1, 1));
        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let (r, v, p) = (1.4e-6, 1.0, 0.5e6);
        let s0 = CavitationCloudDynamics::source_strength(&solver, &bp, r, v, p, 0.0, 0.0);
        let s_rate = CavitationCloudDynamics::source_strength(&solver, &bp, r, v, p, 1.0e12, 0.0);
        assert!(s0.is_finite() && s_rate.is_finite());
        assert!(
            (s0 - s_rate).abs() > 1e-12 * s0.abs().max(1e-30),
            "dp/dt must change the source strength: s0={s0}, s_rate={s_rate}"
        );
    }

    #[test]
    fn pressure_rate_coupling_changes_trajectory_and_is_opt_in() {
        // With couple_pressure_rate on, the coupled source strengths carry the
        // dp/dt term ⇒ a different two-bubble trajectory than rate-off; the default
        // (off) is unchanged from ADR 028/031.
        let f = CloudParameters::default().drive_frequency;
        let dt = (1.0 / f) / 200.0;
        let seq = sinusoid(0.5e6, f, dt, 150);
        let run = |rate: bool| {
            let params = CloudParameters {
                coupling_enabled: true,
                coupling_scheme: CouplingScheme::ImplicitDirect,
                cell_spacing: [0.5e-3, 1.0, 1.0],
                couple_pressure_rate: rate,
                ..CloudParameters::default()
            };
            let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 1, 1));
            cloud.density_field.fill(params.bubble_density);
            for (n, &amp) in seq.iter().enumerate() {
                cloud
                    .evolve_cloud(dt, n as f64 * dt, &Array3::from_elem((2, 1, 1), amp))
                    .unwrap();
            }
            cloud.cloud_radius()[[0, 0, 0]]
        };
        assert!(
            (run(true) - run(false)).abs() > 1e-12 * run(false).max(1e-9),
            "dp/dt coupling must change the trajectory"
        );
    }

    // ── R(t)-dependent shielding (ADR 032) ────────────────────────────────────

    #[test]
    fn radius_dependent_shielding_uses_instantaneous_radius() {
        use kwavers_physics::acoustics::bubble_dynamics::bubbly_medium::commander_prosperetti_attenuation;
        // A cloud expanded above R₀: the R(t)-dependent screen differs from the
        // R₀ screen, and at R = R₀ they coincide.
        // Moderate density / short path ⇒ partial (non-saturating) attenuation, so
        // the resonance-radius dependence is observable (a dense cloud screens to
        // ~0 either way and would mask it).
        let base = CloudParameters {
            shielding_enabled: true,
            incident_axis: 2,
            bubble_density: 1.0e11,
            drive_frequency: 3.0e6,
            ..CloudParameters::default()
        };
        let nz = 2;
        let p_in = 1.0e6;
        let pressure = Array3::from_elem((1, 1, nz), p_in);

        let screen = |radius_dependent: bool, r: f64| {
            let params = CloudParameters {
                shielding_radius_dependent: radius_dependent,
                ..base.clone()
            };
            let mut cloud = CavitationCloudDynamics::new(params.clone(), (1, 1, nz));
            cloud.density_field.fill(params.bubble_density);
            cloud.radius_field.fill(r);
            cloud.shielded_pressure(&pressure)[[0, 0, nz - 1]]
        };

        let bp = CavitationCloudDynamics::new(base.clone(), (1, 1, 1)).bubble_parameters();
        // At R = R₀ the two paths must agree (same resonance radius).
        assert!(
            (screen(true, bp.r0) - screen(false, bp.r0)).abs() <= 1e-9 * p_in,
            "R(t) screen at R=R₀ must equal the R₀ screen"
        );
        // At R = 2·R₀ the resonance shifts ⇒ different attenuation, and the
        // value must match an independent Beer-Lambert with the instantaneous R.
        let r_big = 2.0 * bp.r0;
        let differ = (screen(true, r_big) - screen(false, r_big)).abs();
        assert!(
            differ > 1e-6 * p_in,
            "R(t) screen must differ from R₀ screen when R≠R₀: true={}, false={}",
            screen(true, r_big),
            screen(false, r_big)
        );

        let beta = (base.bubble_density * (4.0 / 3.0) * PI * r_big.powi(3)).clamp(0.0, 1.0 - 1e-9);
        let alpha = commander_prosperetti_attenuation(
            bp.driving_frequency,
            beta,
            r_big, // instantaneous radius
            bp.c_liquid,
            bp.rho_liquid,
            bp.mu_liquid,
            bp.p0,
            bp.gamma,
        );
        let ds = base.cell_spacing[2];
        let expected = p_in * (-(alpha * ds * (nz as f64 - 1.0 + 0.5))).exp();
        assert!(
            (screen(true, r_big) - expected).abs() <= 1e-9 * p_in,
            "R(t) screen must match Beer-Lambert with the instantaneous radius"
        );
    }

    // ── Interface-instability diagnostic (ADR 032) ────────────────────────────

    #[test]
    fn rayleigh_taylor_rate_matches_closed_form() {
        // σ = √(A·k·a), A = β/(2−β); stable (0) when the heavy fluid is not
        // accelerated into the light one (a ≤ 0).
        let params = CloudParameters {
            bubble_density: 1.0e15,
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density); // radius at R₀
        let (k, accel, dv, a0) = (1.0e3, 1.0e8, 0.0, 0.0);
        let diag = cloud.interface_instability(k, accel, dv, a0);

        let beta = cloud.representative_void_fraction();
        let atwood = beta / (2.0 - beta);
        assert!(beta > 0.0 && (diag.void_fraction - beta).abs() < 1e-15);
        assert!((diag.atwood - atwood).abs() < 1e-15);
        let expected = (atwood * k * accel).sqrt();
        assert!(
            (diag.rayleigh_taylor_rate - expected).abs() <= 1e-9 * expected,
            "RT rate {} vs closed form {expected}",
            diag.rayleigh_taylor_rate
        );

        // Reversed acceleration ⇒ stable (no real growth).
        let stable = cloud.interface_instability(k, -accel, dv, a0);
        assert_eq!(
            stable.rayleigh_taylor_rate, 0.0,
            "light-over-heavy must be stable"
        );
    }

    #[test]
    fn richtmyer_meshkov_rate_matches_impulsive_form() {
        // ȧ = k·Δv·a₀·A.
        let params = CloudParameters {
            bubble_density: 1.0e15,
            ..CloudParameters::default()
        };
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density);
        let (k, dv, a0) = (2.0e3, 5.0, 1.0e-5);
        let diag = cloud.interface_instability(k, 0.0, dv, a0);
        let expected = k * dv * a0 * diag.atwood;
        assert!(
            (diag.richtmyer_meshkov_rate - expected).abs() <= 1e-12 * expected.abs(),
            "RM rate {} vs impulsive form {expected}",
            diag.richtmyer_meshkov_rate
        );
        assert!(
            diag.atwood > 0.0,
            "expected a positive Atwood number for a seeded cloud"
        );
    }

    // ── Sparse / matrix-free coupling solver (ADR 032) ────────────────────────

    #[test]
    fn iterative_solve_matches_direct_solve() {
        // The matrix-free LSQR solve of (I − D·G)·S = e reproduces the dense
        // direct solve to the solver tolerance on a moderate cloud.
        let spacing = 8.0e-4; // weak-moderate coupling
        let p_in = 0.5e6;
        let driving = Array3::from_elem((2, 1, 1), p_in);

        let mut direct = two_bubble_cloud(CouplingScheme::ImplicitDirect, spacing);
        direct.parameters.coupling_tolerance = 1e-10;
        direct.parameters.coupling_max_iterations = 500;
        let bp = direct.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let f_direct = direct.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);

        let mut iter = two_bubble_cloud(CouplingScheme::ImplicitIterative, spacing);
        iter.parameters.coupling_tolerance = 1e-10;
        iter.parameters.coupling_max_iterations = 500;
        let f_iter = iter.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);

        for idx in [[0, 0, 0], [1, 0, 0]] {
            let scale = f_direct[idx].abs().max(1.0);
            assert!(
                (f_direct[idx] - f_iter[idx]).abs() <= 1e-6 * scale,
                "iterative must match direct: {} vs {}",
                f_iter[idx],
                f_direct[idx]
            );
        }
    }

    #[test]
    fn iterative_solve_is_self_consistent() {
        // The matrix-free solution satisfies the fixed-point equation directly.
        let spacing = 6.0e-4;
        let p_in = 0.5e6;
        let driving = Array3::from_elem((2, 1, 1), p_in);
        let mut cloud = two_bubble_cloud(CouplingScheme::ImplicitIterative, spacing);
        cloud.parameters.coupling_tolerance = 1e-10;
        cloud.parameters.coupling_max_iterations = 500;
        let bp = cloud.bubble_parameters();
        let solver = KellerMiksisModel::new(bp.clone());
        let field = cloud.coupling_pressure_field(&solver, &bp, &driving, 1.0, 0.0);
        let residual = self_consistency_residual(&cloud, &field, spacing, p_in);
        assert!(
            residual < 1e-6,
            "iterative solve must be self-consistent: residual {residual}"
        );
    }
}
