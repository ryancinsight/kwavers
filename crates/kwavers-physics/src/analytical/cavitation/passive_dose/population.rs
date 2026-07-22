//! Population-level passive-cavitation emission simulation.
//!
//! This module owns the Chapter 24 contrast-agent population path: draw a
//! log-normal bubble-radius population, simulate each bubble with the production
//! free or Marmottant-shell emission solver, reject non-physical traces, sum the
//! Hann-windowed single-sided spectra incoherently, and decompose the resulting
//! spectrum into passive-cavitation bands.

use super::spectrum::{hann_power_spectrum_fft, MAX_EXACT_F64_INTEGER};
use super::{
    decompose_emission_spectrum, simulate_bubble_emission, simulate_coated_bubble_emission,
    BubbleDriveConfig, CavitationBandEnergies, ShellDriveConfig,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};

const MAX_PHYSICAL_EMISSION_PA: f64 = 1.0e7;

/// Shared liquid and gas parameters for population emission simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationMedium {
    /// Ambient pressure `Pa`.
    pub p0_pa: f64,
    /// Liquid density [kg/m^3].
    pub rho: f64,
    /// Liquid sound speed [m/s].
    pub c_liquid: f64,
    /// Liquid dynamic viscosity [Pa s].
    pub mu: f64,
    /// Surface tension for free bubbles [N/m].
    pub sigma: f64,
    /// Vapour pressure `Pa`.
    pub pv: f64,
    /// Gas adiabatic index.
    pub gamma: f64,
}

/// Shell parameters for coated microbubble emission.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationShell {
    /// Use Marmottant-shell dynamics instead of free-bubble dynamics.
    pub coated: bool,
    /// Shell elastic compression modulus [N/m].
    pub chi: f64,
    /// Shell shear viscosity [Pa s].
    pub shell_viscosity: f64,
    /// Shell thickness `m`.
    pub shell_thickness: f64,
    /// Initial shell surface tension [N/m].
    pub sigma_initial: f64,
    /// RK4 sub-steps per drive cycle for coated bubbles.
    pub steps_per_cycle: usize,
}

/// Inputs for one population-level passive-cavitation emission simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationEmissionInput {
    /// Acoustic drive pressure `Pa`.
    pub drive_pa: f64,
    /// Fundamental drive frequency `Hz`.
    pub f0_hz: f64,
    /// Number of bubbles to draw and simulate.
    pub n_bubbles: usize,
    /// Deterministic RNG seed.
    pub seed: u64,
    /// Median equilibrium radius of the log-normal population `m`.
    pub r0_median_m: f64,
    /// Natural-log radius standard deviation.
    pub r0_sigma_ln: f64,
    /// Simulated drive cycles.
    pub n_cycles: f64,
    /// Output samples per single-bubble trace.
    pub n_out: usize,
    /// Observation distance for far-field emission `m`.
    pub r_obs_m: f64,
    /// Half-width of spectral line windows as a fraction of `f0_hz`.
    pub rel_halfwidth: f64,
    /// Passive-cavitation baseline PSD floor.
    pub noise_floor: f64,
    /// Enable thermal/mass-transfer terms in the free-bubble solver.
    pub thermal_effects: bool,
    /// Liquid/gas parameters.
    pub medium: PopulationMedium,
    /// Shell model parameters.
    pub shell: PopulationShell,
}

/// Population-level passive-cavitation emission result.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationEmission {
    /// Single-sided frequency axis `Hz`.
    pub freqs_hz: Vec<f64>,
    /// Incoherently summed population PSD.
    pub psd: Vec<f64>,
    /// Cavitation-band energies from the population PSD.
    pub bands: CavitationBandEnergies,
    /// Number of accepted finite, physical bubble traces.
    pub n_active: usize,
    /// Maximum compression among accepted traces.
    pub max_compression: f64,
    /// Maximum wall Mach number among accepted traces.
    pub max_mach: f64,
}

/// Inputs for a population-emission pressure sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationEmissionSweepInput<'a> {
    /// Drive-pressure samples `Pa`.
    pub pressures_pa: &'a [f64],
    /// Fundamental drive frequency `Hz`.
    pub f0_hz: f64,
    /// Number of bubbles to draw and simulate per pressure.
    pub n_bubbles: usize,
    /// Deterministic RNG seed for per-pressure population seeds.
    pub seed: u64,
    /// Median equilibrium radius of the log-normal population `m`.
    pub r0_median_m: f64,
    /// Natural-log radius standard deviation.
    pub r0_sigma_ln: f64,
    /// Simulated drive cycles.
    pub n_cycles: f64,
    /// Output samples per single-bubble trace.
    pub n_out: usize,
    /// Observation distance for far-field emission `m`.
    pub r_obs_m: f64,
    /// Half-width of spectral line windows as a fraction of `f0_hz`.
    pub rel_halfwidth: f64,
    /// Passive-cavitation baseline PSD floor.
    pub noise_floor: f64,
    /// Enable thermal/mass-transfer terms in the free-bubble solver.
    pub thermal_effects: bool,
    /// Liquid/gas parameters.
    pub medium: PopulationMedium,
    /// Shell model parameters.
    pub shell: PopulationShell,
}

/// Band-resolved population-emission pressure sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationEmissionSweep {
    /// Harmonic-comb emission power.
    pub harmonic: Vec<f64>,
    /// Subharmonic emission power.
    pub subharmonic: Vec<f64>,
    /// Ultraharmonic emission power.
    pub ultraharmonic: Vec<f64>,
    /// Stable emission power, `subharmonic + ultraharmonic`.
    pub stable: Vec<f64>,
    /// Broadband inertial-cavitation emission power.
    pub inertial: Vec<f64>,
    /// Aggregate cavitation monitor signal, `stable + inertial`.
    pub signal: Vec<f64>,
    /// Accepted bubble-trace count at each pressure.
    pub n_active: Vec<usize>,
    /// Maximum compression at each pressure.
    pub max_compression: Vec<f64>,
    /// Maximum wall Mach number at each pressure.
    pub max_mach: Vec<f64>,
}

/// Simulate passive-cavitation emission from a log-normal bubble population.
#[must_use]
pub fn simulate_population_emission(input: PopulationEmissionInput) -> Option<PopulationEmission> {
    validate_population_input(input)?;
    let distribution = LogNormal::new(input.r0_median_m.ln(), input.r0_sigma_ln).ok()?;
    let mut rng = ChaCha8Rng::seed_from_u64(input.seed);
    let mut traces = Vec::with_capacity(input.n_bubbles);
    let mut max_compression = 0.0_f64;
    let mut max_mach = 0.0_f64;

    for _ in 0..input.n_bubbles {
        let r0_m = distribution.sample(&mut rng);
        let trace = if input.shell.coated {
            simulate_coated_bubble_emission(&ShellDriveConfig {
                r0_m,
                p0_pa: input.medium.p0_pa,
                rho: input.medium.rho,
                c_liquid: input.medium.c_liquid,
                mu: input.medium.mu,
                gamma: input.medium.gamma,
                drive_freq_hz: input.f0_hz,
                drive_amp_pa: input.drive_pa,
                n_cycles: input.n_cycles,
                steps_per_cycle: input.shell.steps_per_cycle,
                n_out: input.n_out,
                r_obs_m: input.r_obs_m,
                chi: input.shell.chi,
                shell_viscosity: input.shell.shell_viscosity,
                shell_thickness: input.shell.shell_thickness,
                sigma_initial: input.shell.sigma_initial,
            })
        } else {
            simulate_bubble_emission(&BubbleDriveConfig {
                r0_m,
                p0_pa: input.medium.p0_pa,
                rho: input.medium.rho,
                c_liquid: input.medium.c_liquid,
                mu: input.medium.mu,
                sigma: input.medium.sigma,
                pv: input.medium.pv,
                gamma: input.medium.gamma,
                drive_freq_hz: input.f0_hz,
                drive_amp_pa: input.drive_pa,
                n_cycles: input.n_cycles,
                n_out: input.n_out,
                r_obs_m: input.r_obs_m,
                thermal_effects: input.thermal_effects,
            })
        };
        if !is_accepted_trace(&trace.emission) {
            continue;
        }
        max_compression = max_compression.max(trace.max_compression);
        max_mach = max_mach.max(trace.max_mach);
        traces.push(trace.emission);
    }

    if traces.is_empty() {
        return Some(empty_population_emission());
    }

    let full_len = traces.iter().map(Vec::len).max()?;
    if !(2..=MAX_EXACT_F64_INTEGER).contains(&full_len) {
        return None;
    }
    let n_active = traces.len();
    let full_len_f = full_len as f64;
    let dt_s = input.n_cycles / input.f0_hz / (full_len_f - 1.0);
    let n_pos = full_len / 2 + 1;
    let mut psd = vec![0.0_f64; n_pos];
    let mut freqs_hz = Vec::new();
    for trace in &traces {
        let mut padded = vec![0.0_f64; full_len];
        padded[..trace.len()].copy_from_slice(trace);
        let (freqs, trace_psd) = hann_power_spectrum_fft(&padded, dt_s)?;
        if freqs_hz.is_empty() {
            freqs_hz = freqs;
        }
        for (acc, value) in psd.iter_mut().zip(trace_psd) {
            *acc += value;
        }
    }

    let bands = decompose_emission_spectrum(
        &freqs_hz,
        &psd,
        input.f0_hz,
        input.rel_halfwidth,
        input.noise_floor,
    );
    Some(PopulationEmission {
        freqs_hz,
        psd,
        bands,
        n_active,
        max_compression,
        max_mach,
    })
}

/// Simulate a pressure sweep of population-level passive-cavitation emission.
#[must_use]
pub fn population_emission_sweep(
    input: PopulationEmissionSweepInput<'_>,
) -> Option<PopulationEmissionSweep> {
    validate_population_sweep_input(input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(input.seed);
    let sample_count = input.pressures_pa.len();
    let mut harmonic = Vec::with_capacity(sample_count);
    let mut subharmonic = Vec::with_capacity(sample_count);
    let mut ultraharmonic = Vec::with_capacity(sample_count);
    let mut stable = Vec::with_capacity(sample_count);
    let mut inertial = Vec::with_capacity(sample_count);
    let mut signal = Vec::with_capacity(sample_count);
    let mut n_active = Vec::with_capacity(sample_count);
    let mut max_compression = Vec::with_capacity(sample_count);
    let mut max_mach = Vec::with_capacity(sample_count);

    for &drive_pa in input.pressures_pa {
        let emission = simulate_population_emission(PopulationEmissionInput {
            drive_pa,
            f0_hz: input.f0_hz,
            n_bubbles: input.n_bubbles,
            seed: rng.next_u64(),
            r0_median_m: input.r0_median_m,
            r0_sigma_ln: input.r0_sigma_ln,
            n_cycles: input.n_cycles,
            n_out: input.n_out,
            r_obs_m: input.r_obs_m,
            rel_halfwidth: input.rel_halfwidth,
            noise_floor: input.noise_floor,
            thermal_effects: input.thermal_effects,
            medium: input.medium,
            shell: input.shell,
        })?;
        let stable_power = emission.bands.subharmonic + emission.bands.ultraharmonic;
        let signal_power = stable_power + emission.bands.broadband;
        harmonic.push(emission.bands.fundamental);
        subharmonic.push(emission.bands.subharmonic);
        ultraharmonic.push(emission.bands.ultraharmonic);
        stable.push(stable_power);
        inertial.push(emission.bands.broadband);
        signal.push(signal_power);
        n_active.push(emission.n_active);
        max_compression.push(emission.max_compression);
        max_mach.push(emission.max_mach);
    }

    Some(PopulationEmissionSweep {
        harmonic,
        subharmonic,
        ultraharmonic,
        stable,
        inertial,
        signal,
        n_active,
        max_compression,
        max_mach,
    })
}

fn validate_population_input(input: PopulationEmissionInput) -> Option<()> {
    if input.n_bubbles == 0
        || input.n_bubbles > MAX_EXACT_F64_INTEGER
        || input.n_out < 16
        || input.n_out > MAX_EXACT_F64_INTEGER
        || input.shell.steps_per_cycle == 0
        || !(input.drive_pa.is_finite()
            && input.drive_pa >= 0.0
            && input.f0_hz.is_finite()
            && input.f0_hz > 0.0
            && input.r0_median_m.is_finite()
            && input.r0_median_m > 0.0
            && input.r0_sigma_ln.is_finite()
            && input.r0_sigma_ln >= 0.0
            && input.n_cycles.is_finite()
            && input.n_cycles > 0.0
            && input.r_obs_m.is_finite()
            && input.r_obs_m > 0.0
            && input.rel_halfwidth.is_finite()
            && input.noise_floor.is_finite()
            && input.noise_floor >= 0.0
            && input.medium.p0_pa.is_finite()
            && input.medium.p0_pa > 0.0
            && input.medium.rho.is_finite()
            && input.medium.rho > 0.0
            && input.medium.c_liquid.is_finite()
            && input.medium.c_liquid > 0.0
            && input.medium.mu.is_finite()
            && input.medium.mu >= 0.0
            && input.medium.sigma.is_finite()
            && input.medium.sigma >= 0.0
            && input.medium.pv.is_finite()
            && input.medium.pv >= 0.0
            && input.medium.gamma.is_finite()
            && input.medium.gamma > 0.0
            && input.shell.chi.is_finite()
            && input.shell.chi >= 0.0
            && input.shell.shell_viscosity.is_finite()
            && input.shell.shell_viscosity >= 0.0
            && input.shell.shell_thickness.is_finite()
            && input.shell.shell_thickness >= 0.0
            && input.shell.sigma_initial.is_finite()
            && input.shell.sigma_initial >= 0.0)
    {
        return None;
    }
    Some(())
}

fn validate_population_sweep_input(input: PopulationEmissionSweepInput<'_>) -> Option<()> {
    if input.pressures_pa.is_empty()
        || !input
            .pressures_pa
            .iter()
            .all(|pressure| pressure.is_finite() && *pressure >= 0.0)
    {
        return None;
    }
    validate_population_input(PopulationEmissionInput {
        drive_pa: 0.0,
        f0_hz: input.f0_hz,
        n_bubbles: input.n_bubbles,
        seed: input.seed,
        r0_median_m: input.r0_median_m,
        r0_sigma_ln: input.r0_sigma_ln,
        n_cycles: input.n_cycles,
        n_out: input.n_out,
        r_obs_m: input.r_obs_m,
        rel_halfwidth: input.rel_halfwidth,
        noise_floor: input.noise_floor,
        thermal_effects: input.thermal_effects,
        medium: input.medium,
        shell: input.shell,
    })
}

fn is_accepted_trace(emission: &[f64]) -> bool {
    emission.len() >= 16
        && emission
            .iter()
            .all(|value| value.is_finite() && value.abs() <= MAX_PHYSICAL_EMISSION_PA)
}

fn empty_population_emission() -> PopulationEmission {
    PopulationEmission {
        freqs_hz: vec![0.0],
        psd: vec![0.0],
        bands: CavitationBandEnergies {
            fundamental: 0.0,
            subharmonic: 0.0,
            ultraharmonic: 0.0,
            broadband: 0.0,
        },
        n_active: 0,
        max_compression: 0.0,
        max_mach: 0.0,
    }
}
