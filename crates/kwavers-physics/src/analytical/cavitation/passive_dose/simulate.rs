//! True bubble-emission simulation via the production adaptive Keller–Miksis
//! solver.
//!
//! Unlike a fixed-step RK4 (which blows up above ~250 kPa), this drives the
//! full [`KellerMiksisModel`] — gas thermodynamics, mass transfer, compressible
//! radiation damping — with the
//! [`AdaptiveBubbleIntegrator`](crate::acoustics::bubble_dynamics::AdaptiveBubbleIntegrator)
//! whose Richardson-extrapolation step control sub-cycles through violent
//! inertial collapse. The radiated far-field acoustic emission
//! `p_sc(t) = ρ R/r_obs·(2Ṙ² + R R̈)` is recorded on a uniform output grid using
//! the *exact* wall acceleration the solver computes (not a finite difference),
//! so the harmonic / subharmonic / broadband content of the spectrum is an
//! emergent property of the simulated dynamics rather than an imposed shape.

use crate::acoustics::bubble_dynamics::{
    AdaptiveBubbleConfig, AdaptiveBubbleIntegrator, BubbleParameters, BubbleState,
    KellerMiksisModel, MarmottantModel, ShellProperties,
};
use kwavers_core::constants::numerical::TWO_PI;

/// Inputs for a single-bubble adaptive Keller–Miksis emission simulation.
#[derive(Debug, Clone, Copy)]
pub struct BubbleDriveConfig {
    /// Equilibrium radius `m`.
    pub r0_m: f64,
    /// Ambient pressure `Pa`.
    pub p0_pa: f64,
    /// Liquid density [kg/m³].
    pub rho: f64,
    /// Liquid sound speed [m/s].
    pub c_liquid: f64,
    /// Dynamic viscosity [Pa·s].
    pub mu: f64,
    /// Surface tension [N/m].
    pub sigma: f64,
    /// Vapour pressure `Pa`.
    pub pv: f64,
    /// Gas adiabatic index.
    pub gamma: f64,
    /// Drive frequency `Hz`.
    pub drive_freq_hz: f64,
    /// Peak acoustic drive pressure `Pa`.
    pub drive_amp_pa: f64,
    /// Number of drive cycles to simulate.
    pub n_cycles: f64,
    /// Output samples (uniform grid over the simulated window).
    pub n_out: usize,
    /// Observation distance for the far-field emission `m`.
    pub r_obs_m: f64,
    /// Enable gas thermodynamics + mass transfer (slower, more physical).
    pub thermal_effects: bool,
}

/// Result of an adaptive Keller–Miksis emission simulation.
#[derive(Debug, Clone)]
pub struct BubbleEmissionTrace {
    /// Output time grid `s`.
    pub time: Vec<f64>,
    /// Bubble radius `R(t)` `m`.
    pub radius: Vec<f64>,
    /// Wall velocity `Ṙ(t)` [m/s].
    pub wall_velocity: Vec<f64>,
    /// Far-field emitted pressure `p_sc(t)` `Pa`.
    pub emission: Vec<f64>,
    /// Maximum compression ratio `R₀/R_min` reached.
    pub max_compression: f64,
    /// Maximum wall Mach number reached.
    pub max_mach: f64,
    /// Number of inertial collapse events detected.
    pub collapse_count: u32,
    /// `true` if the adaptive integrator reached `t_end` without failing.
    pub converged: bool,
}

fn build_params(cfg: &BubbleDriveConfig) -> BubbleParameters {
    BubbleParameters {
        r0: cfg.r0_m,
        p0: cfg.p0_pa,
        rho_liquid: cfg.rho,
        c_liquid: cfg.c_liquid,
        mu_liquid: cfg.mu,
        sigma: cfg.sigma,
        pv: cfg.pv,
        gamma: cfg.gamma,
        driving_frequency: cfg.drive_freq_hz,
        driving_amplitude: cfg.drive_amp_pa,
        initial_gas_pressure: cfg.p0_pa,
        use_compressibility: true,
        use_thermal_effects: cfg.thermal_effects,
        use_mass_transfer: cfg.thermal_effects,
        ..BubbleParameters::default()
    }
}

/// Simulate the far-field acoustic emission of one bubble under sinusoidal
/// drive, using the production adaptive Keller–Miksis integrator.
///
/// The drive is `p_ac(t) = drive_amp·sin(ωt)` with `dp/dt = drive_amp·ω·cos(ωt)`,
/// held constant across each output step (chosen ≤ period/200) while the
/// integrator sub-cycles internally. At each output sample the exact wall
/// acceleration is evaluated to form the emission. If the integrator fails to
/// converge (e.g. a collapse stiffer than `dt_min` resolves), the trace is
/// returned truncated with `converged = false`.
#[must_use]
pub fn simulate_bubble_emission(cfg: &BubbleDriveConfig) -> BubbleEmissionTrace {
    let n_out = cfg.n_out.max(2);
    let mut trace = BubbleEmissionTrace {
        time: Vec::with_capacity(n_out),
        radius: Vec::with_capacity(n_out),
        wall_velocity: Vec::with_capacity(n_out),
        emission: Vec::with_capacity(n_out),
        max_compression: 1.0,
        max_mach: 0.0,
        collapse_count: 0,
        converged: true,
    };
    if !(cfg.drive_freq_hz > 0.0 && cfg.r0_m > 0.0 && cfg.n_cycles > 0.0 && cfg.r_obs_m > 0.0) {
        trace.converged = false;
        return trace;
    }

    let params = build_params(cfg);
    let model = KellerMiksisModel::new(params.clone());
    let period = 1.0 / cfg.drive_freq_hz;
    let base_cfg = AdaptiveBubbleConfig::default();
    // Cap the main step so the externally-held p_ac(t) tracks the sinusoid; the
    // adaptive sub-stepper still refines within each main step at collapse.
    let int_cfg = AdaptiveBubbleConfig {
        dt_max: (period / 200.0)
            .min(base_cfg.dt_max)
            .max(base_cfg.dt_min * 10.0),
        ..base_cfg
    };
    let mut integrator = AdaptiveBubbleIntegrator::new(&model, int_cfg);

    let mut state = BubbleState::new(&params);
    let omega = TWO_PI * cfg.drive_freq_hz;
    let t_end = cfg.n_cycles * period;
    let dt_main = t_end / (n_out as f64 - 1.0);

    let drive = |t: f64| -> (f64, f64) {
        (
            cfg.drive_amp_pa * (omega * t).sin(),
            cfg.drive_amp_pa * omega * (omega * t).cos(),
        )
    };

    // Record the initial (equilibrium) sample.
    let push_sample = |trace: &mut BubbleEmissionTrace, t: f64, st: &BubbleState, rddot: f64| {
        trace.time.push(t);
        trace.radius.push(st.radius);
        trace.wall_velocity.push(st.wall_velocity);
        trace.emission.push(
            cfg.rho * st.radius / cfg.r_obs_m
                * 2.0_f64.mul_add(st.wall_velocity * st.wall_velocity, st.radius * rddot),
        );
    };

    let (p0_ac, dp0) = drive(0.0);
    let r0ddot = model
        .calculate_acceleration(&mut state, p0_ac, dp0, 0.0)
        .unwrap_or(0.0);
    push_sample(&mut trace, 0.0, &state, r0ddot);

    for i in 1..n_out {
        let t0 = (i as f64 - 1.0) * dt_main;
        let t1 = i as f64 * dt_main;
        let (p_ac, dp_dt) = drive(t0);
        if integrator
            .integrate_adaptive(&mut state, p_ac, dp_dt, dt_main, t0)
            .is_err()
            || !state.radius.is_finite()
            || !state.wall_velocity.is_finite()
        {
            trace.converged = false;
            break;
        }
        // Exact wall acceleration at the output instant for the emission.
        let (p_ac1, dp1) = drive(t1);
        let rddot = model
            .calculate_acceleration(&mut state, p_ac1, dp1, t1)
            .unwrap_or(0.0);
        push_sample(&mut trace, t1, &state, rddot);

        trace.max_compression = trace.max_compression.max(state.compression_ratio);
        trace.max_mach = trace.max_mach.max(state.mach_number.abs());
        trace.collapse_count = trace.collapse_count.max(state.collapse_count);
    }
    trace
}

/// Inputs for a coated (encapsulated) microbubble emission simulation.
#[derive(Debug, Clone, Copy)]
pub struct ShellDriveConfig {
    /// Equilibrium radius `m`.
    pub r0_m: f64,
    /// Ambient pressure `Pa`.
    pub p0_pa: f64,
    /// Liquid density [kg/m³].
    pub rho: f64,
    /// Liquid sound speed [m/s].
    pub c_liquid: f64,
    /// Liquid dynamic viscosity [Pa·s].
    pub mu: f64,
    /// Gas adiabatic index.
    pub gamma: f64,
    /// Drive frequency `Hz`.
    pub drive_freq_hz: f64,
    /// Peak acoustic drive pressure `Pa`.
    pub drive_amp_pa: f64,
    /// Number of drive cycles to simulate.
    pub n_cycles: f64,
    /// RK4 sub-steps per drive cycle.
    pub steps_per_cycle: usize,
    /// Output samples.
    pub n_out: usize,
    /// Observation distance for the far-field emission `m`.
    pub r_obs_m: f64,
    /// Shell elastic compression modulus χ [N/m] (lipid ≈ 0.5–1.0).
    pub chi: f64,
    /// Shell shear viscosity [Pa·s] (lipid ≈ 0.5).
    pub shell_viscosity: f64,
    /// Shell thickness `m` (lipid ≈ 3 nm).
    pub shell_thickness: f64,
    /// Initial (unstressed) shell surface tension [N/m].
    pub sigma_initial: f64,
}

/// Simulate the acoustic emission of a *coated* (encapsulated) microbubble via
/// the Marmottant shell model.
///
/// The lipid/protein shell's buckling (R < R_buckling, σ→0) and rupture
/// (R > R_rupture, σ→σ_water) make the surface tension a piecewise-nonlinear
/// function of radius — the mechanism that lets clinical contrast microbubbles
/// emit a **subharmonic** at low drive pressures, which a free (uncoated) bubble
/// does not. The shell-damped Rayleigh–Plesset dynamics are integrated with a
/// fixed-step RK4 (stable at clinical pressures); the emitted far-field pressure
/// `p_sc = ρ R/r_obs·(2Ṙ² + R R̈)` is recorded on a uniform output grid.
/// The harmonic / subharmonic / broadband content is emergent.
#[must_use]
pub fn simulate_coated_bubble_emission(cfg: &ShellDriveConfig) -> BubbleEmissionTrace {
    let n_out = cfg.n_out.max(2);
    let mut trace = BubbleEmissionTrace {
        time: Vec::with_capacity(n_out),
        radius: Vec::with_capacity(n_out),
        wall_velocity: Vec::with_capacity(n_out),
        emission: Vec::with_capacity(n_out),
        max_compression: 1.0,
        max_mach: 0.0,
        collapse_count: 0,
        converged: true,
    };
    if !(cfg.drive_freq_hz > 0.0 && cfg.r0_m > 0.0 && cfg.n_cycles > 0.0 && cfg.r_obs_m > 0.0)
        || cfg.steps_per_cycle == 0
    {
        trace.converged = false;
        return trace;
    }

    let params = BubbleParameters {
        r0: cfg.r0_m,
        p0: cfg.p0_pa,
        rho_liquid: cfg.rho,
        c_liquid: cfg.c_liquid,
        mu_liquid: cfg.mu,
        gamma: cfg.gamma,
        driving_frequency: cfg.drive_freq_hz,
        driving_amplitude: cfg.drive_amp_pa,
        initial_gas_pressure: cfg.p0_pa,
        ..BubbleParameters::default()
    };
    let shell = ShellProperties {
        shear_viscosity: cfg.shell_viscosity,
        thickness: cfg.shell_thickness,
        sigma_initial: cfg.sigma_initial,
        ..ShellProperties::lipid_shell()
    };
    let model = MarmottantModel::new(params.clone(), shell, cfg.chi);
    let init = BubbleState::new(&params);

    let period = 1.0 / cfg.drive_freq_hz;
    let dt = period / cfg.steps_per_cycle as f64;
    let n_total = (cfg.n_cycles * cfg.steps_per_cycle as f64).round().max(1.0) as usize;
    // Output stride: record ~n_out evenly-spaced samples of the fine grid.
    let stride = (n_total / n_out).max(1);

    // R̈ as a function of (R, Ṙ, t) on a scratch state (Marmottant takes the
    // drive amplitude and computes the sinusoid internally).
    let amp = cfg.drive_amp_pa;
    let accel_at = |r: f64, v: f64, t: f64, scratch: &mut BubbleState| -> f64 {
        scratch.radius = r.max(1e-12);
        scratch.wall_velocity = v;
        model.calculate_acceleration(scratch, amp, t).unwrap_or(0.0)
    };

    let mut scratch = init;
    let mut r = init.radius;
    let mut v = init.wall_velocity;
    let mut t = 0.0_f64;
    for step in 0..=n_total {
        if step % stride == 0 || step == n_total {
            let a = accel_at(r, v, t, &mut scratch);
            trace.time.push(t);
            trace.radius.push(r);
            trace.wall_velocity.push(v);
            trace
                .emission
                .push(cfg.rho * r / cfg.r_obs_m * 2.0_f64.mul_add(v * v, r * a));
            trace.max_compression = trace.max_compression.max(cfg.r0_m / r.max(1e-12));
            trace.max_mach = trace.max_mach.max((v / cfg.c_liquid).abs());
        }
        if step == n_total {
            break;
        }
        // Classic RK4 on (R, Ṙ).
        let a1 = accel_at(r, v, t, &mut scratch);
        let (k1r, k1v) = (v, a1);
        let a2 = accel_at(
            r + 0.5 * dt * k1r,
            v + 0.5 * dt * k1v,
            t + 0.5 * dt,
            &mut scratch,
        );
        let (k2r, k2v) = (v + 0.5 * dt * k1v, a2);
        let a3 = accel_at(
            r + 0.5 * dt * k2r,
            v + 0.5 * dt * k2v,
            t + 0.5 * dt,
            &mut scratch,
        );
        let (k3r, k3v) = (v + 0.5 * dt * k2v, a3);
        let a4 = accel_at(r + dt * k3r, v + dt * k3v, t + dt, &mut scratch);
        let (k4r, k4v) = (v + dt * k3v, a4);
        r += dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        t += dt;
        // Reject physically impossible excursions (fixed-step RK4 can push a
        // violent collapse to absurd compression); truncate rather than emit
        // a spurious singular spike.
        if !r.is_finite() || !v.is_finite() || r < cfg.r0_m * 1.0e-3 || r > cfg.r0_m * 1.0e3 {
            trace.converged = false;
            break;
        }
        r = r.max(1e-12);
    }
    trace
}
