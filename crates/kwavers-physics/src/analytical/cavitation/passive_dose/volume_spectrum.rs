//! Deterministic V_s-integrated passive-cavitation spectra.
//!
//! This module owns the Chapter 24 analytic pressure-sweep path: drive a fixed
//! radius population with the canonical fixed-step Keller-Miksis integrator,
//! convert each radius trajectory to far-field emission, estimate a
//! leakage-suppressed PSD, incoherently sum the radius spectra as the focal
//! volume `V_s` integral, and decompose the summed spectrum into cavitation
//! bands.

use super::{
    bubble_acoustic_emission_pressure, decompose_emission_spectrum, hann_windowed_power_spectrum,
    integrate_receiver_array_psd, CavitationBandEnergies,
};
use crate::analytical::cavitation::keller_miksis_shelled_rk4;

const MAX_EXACT_F64_INTEGER: usize = 1usize << 53;

/// Liquid and shell parameters for deterministic V_s spectrum aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeSpectrumMedium {
    /// Ambient pressure [Pa].
    pub p0_pa: f64,
    /// Liquid density [kg/m^3].
    pub rho: f64,
    /// Surface tension [N/m].
    pub sigma: f64,
    /// Gas polytropic index.
    pub gamma: f64,
    /// Liquid dynamic viscosity [Pa s].
    pub mu: f64,
    /// Vapour pressure [Pa].
    pub pv_pa: f64,
    /// Liquid sound speed [m/s].
    pub c_liquid: f64,
    /// Lumped shell viscosity [Pa s m]; zero is a bare bubble.
    pub xi_s: f64,
}

/// Inputs for one V_s-integrated emission spectrum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeEmissionSpectrumInput<'a> {
    /// Drive pressure [Pa].
    pub drive_pa: f64,
    /// Fundamental frequency [Hz].
    pub f0_hz: f64,
    /// Equilibrium radii in the sonication volume [m].
    pub r0_population_m: &'a [f64],
    /// Liquid and shell parameters.
    pub medium: VolumeSpectrumMedium,
    /// Simulated cycles.
    pub n_cycles: f64,
    /// Fixed RK4 steps per drive cycle.
    pub steps_per_cycle: usize,
    /// Observation distance [m].
    pub r_obs_m: f64,
    /// Target decimated sample count for the DFT.
    pub n_fft: usize,
    /// Fraction of the emission trace dropped as ring-up transient.
    pub transient_fraction: f64,
}

/// V_s-integrated emission spectrum.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeEmissionSpectrum {
    /// Frequency axis [Hz].
    pub freqs_hz: Vec<f64>,
    /// Incoherently summed V_s PSD.
    pub psd: Vec<f64>,
    /// Number of finite radius trajectories that contributed to the sum.
    pub n_active: usize,
}

/// Inputs for a V_s-integrated pressure sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeEmissionSweepInput<'a> {
    /// Drive-pressure samples [Pa].
    pub pressures_pa: &'a [f64],
    /// Fundamental frequency [Hz].
    pub f0_hz: f64,
    /// Equilibrium radii in the sonication volume [m].
    pub r0_population_m: &'a [f64],
    /// Liquid and shell parameters.
    pub medium: VolumeSpectrumMedium,
    /// Spectral line half-width as a fraction of `f0_hz`.
    pub rel_halfwidth: f64,
    /// PSD noise floor for band decomposition.
    pub noise_floor: f64,
    /// Simulated cycles.
    pub n_cycles: f64,
    /// Fixed RK4 steps per drive cycle.
    pub steps_per_cycle: usize,
    /// Observation distance [m].
    pub r_obs_m: f64,
    /// Target decimated sample count for the DFT.
    pub n_fft: usize,
    /// Fraction of the emission trace dropped as ring-up transient.
    pub transient_fraction: f64,
}

/// Band-resolved V_s-integrated pressure sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeEmissionSweep {
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
    /// Active radius-trajectory count for each pressure sample.
    pub n_active: Vec<usize>,
}

/// Compute a V_s-integrated passive-cavitation emission spectrum.
#[must_use]
pub fn volume_emission_spectrum(
    input: VolumeEmissionSpectrumInput<'_>,
) -> Option<VolumeEmissionSpectrum> {
    validate_volume_spectrum_input(input)?;
    let t_end_s = input.n_cycles / input.f0_hz;
    let n_steps = ((input.n_cycles * input.steps_per_cycle as f64).round()) as usize;
    if n_steps == 0 || n_steps > MAX_EXACT_F64_INTEGER {
        return None;
    }
    let dt_s = t_end_s / n_steps as f64;
    let time_s: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt_s).collect();
    let mut channel_psds = Vec::new();
    let mut freq_axis = Vec::new();
    let mut n_bins = usize::MAX;
    let mut n_active = 0_usize;

    for &r0_m in input.r0_population_m {
        let (radius, wall_velocity) = keller_miksis_shelled_rk4(
            r0_m,
            0.0,
            input.drive_pa,
            input.f0_hz,
            &time_s,
            input.medium.p0_pa,
            input.medium.rho,
            input.medium.sigma,
            input.medium.mu,
            input.medium.gamma,
            input.medium.pv_pa,
            input.medium.xi_s,
            input.medium.c_liquid,
        );
        if !all_finite(&radius) || !all_finite(&wall_velocity) {
            continue;
        }
        let emission = bubble_acoustic_emission_pressure(
            &radius,
            &wall_velocity,
            dt_s,
            input.medium.rho,
            input.r_obs_m,
        );
        if emission.is_empty() || !all_finite(&emission) {
            continue;
        }
        let start = (input.transient_fraction * emission.len() as f64) as usize;
        if start >= emission.len() {
            continue;
        }
        let steady = &emission[start..];
        let stride = (steady.len() / input.n_fft.max(1)).max(1);
        let decimated: Vec<f64> = steady.iter().step_by(stride).copied().collect();
        if decimated.len() < 2 {
            continue;
        }
        let (freqs, psd) =
            hann_windowed_power_spectrum(&decimated, dt_s * stride as f64, decimated.len());
        if freqs.is_empty() || psd.is_empty() || freqs.len() != psd.len() || !all_finite(&psd) {
            continue;
        }
        if freq_axis.is_empty() {
            freq_axis = freqs;
        }
        n_bins = n_bins.min(psd.len());
        channel_psds.push(psd);
        n_active += 1;
    }

    if channel_psds.is_empty() || freq_axis.is_empty() || n_bins == 0 {
        return Some(VolumeEmissionSpectrum {
            freqs_hz: vec![0.0],
            psd: vec![0.0],
            n_active: 0,
        });
    }

    let mut flat = Vec::with_capacity(channel_psds.len() * n_bins);
    for psd in &channel_psds {
        flat.extend_from_slice(&psd[..n_bins]);
    }
    let psd = integrate_receiver_array_psd(&flat, channel_psds.len(), n_bins);
    if psd.is_empty() {
        return None;
    }

    Some(VolumeEmissionSpectrum {
        freqs_hz: freq_axis[..n_bins].to_vec(),
        psd,
        n_active,
    })
}

/// Compute a band-resolved V_s-integrated pressure sweep.
#[must_use]
pub fn volume_emission_sweep(input: VolumeEmissionSweepInput<'_>) -> Option<VolumeEmissionSweep> {
    validate_volume_sweep_input(input)?;
    let sample_count = input.pressures_pa.len();
    let mut harmonic = Vec::with_capacity(sample_count);
    let mut subharmonic = Vec::with_capacity(sample_count);
    let mut ultraharmonic = Vec::with_capacity(sample_count);
    let mut stable = Vec::with_capacity(sample_count);
    let mut inertial = Vec::with_capacity(sample_count);
    let mut n_active = Vec::with_capacity(sample_count);

    for &drive_pa in input.pressures_pa {
        let spectrum = volume_emission_spectrum(VolumeEmissionSpectrumInput {
            drive_pa,
            f0_hz: input.f0_hz,
            r0_population_m: input.r0_population_m,
            medium: input.medium,
            n_cycles: input.n_cycles,
            steps_per_cycle: input.steps_per_cycle,
            r_obs_m: input.r_obs_m,
            n_fft: input.n_fft,
            transient_fraction: input.transient_fraction,
        })?;
        let bands = decompose_or_zero(
            &spectrum.freqs_hz,
            &spectrum.psd,
            input.f0_hz,
            input.rel_halfwidth,
            input.noise_floor,
        );
        harmonic.push(bands.fundamental);
        subharmonic.push(bands.subharmonic);
        ultraharmonic.push(bands.ultraharmonic);
        stable.push(bands.stable_emission());
        inertial.push(bands.inertial_emission());
        n_active.push(spectrum.n_active);
    }

    Some(VolumeEmissionSweep {
        harmonic,
        subharmonic,
        ultraharmonic,
        stable,
        inertial,
        n_active,
    })
}

fn validate_volume_sweep_input(input: VolumeEmissionSweepInput<'_>) -> Option<()> {
    if input.pressures_pa.is_empty()
        || !input
            .pressures_pa
            .iter()
            .all(|pressure| pressure.is_finite() && *pressure >= 0.0)
        || !(input.rel_halfwidth.is_finite()
            && input.noise_floor.is_finite()
            && input.noise_floor >= 0.0)
    {
        return None;
    }
    validate_volume_spectrum_input(VolumeEmissionSpectrumInput {
        drive_pa: 0.0,
        f0_hz: input.f0_hz,
        r0_population_m: input.r0_population_m,
        medium: input.medium,
        n_cycles: input.n_cycles,
        steps_per_cycle: input.steps_per_cycle,
        r_obs_m: input.r_obs_m,
        n_fft: input.n_fft,
        transient_fraction: input.transient_fraction,
    })
}

fn validate_volume_spectrum_input(input: VolumeEmissionSpectrumInput<'_>) -> Option<()> {
    if input.r0_population_m.is_empty()
        || input.steps_per_cycle == 0
        || input.n_fft < 2
        || !input
            .r0_population_m
            .iter()
            .all(|radius| radius.is_finite() && *radius > 0.0)
        || !(input.drive_pa.is_finite()
            && input.drive_pa >= 0.0
            && input.f0_hz.is_finite()
            && input.f0_hz > 0.0
            && input.n_cycles.is_finite()
            && input.n_cycles > 0.0
            && input.r_obs_m.is_finite()
            && input.r_obs_m > 0.0
            && input.transient_fraction.is_finite()
            && (0.0..1.0).contains(&input.transient_fraction)
            && input.medium.p0_pa.is_finite()
            && input.medium.p0_pa > 0.0
            && input.medium.rho.is_finite()
            && input.medium.rho > 0.0
            && input.medium.sigma.is_finite()
            && input.medium.sigma >= 0.0
            && input.medium.gamma.is_finite()
            && input.medium.gamma > 0.0
            && input.medium.mu.is_finite()
            && input.medium.mu >= 0.0
            && input.medium.pv_pa.is_finite()
            && input.medium.pv_pa >= 0.0
            && input.medium.c_liquid.is_finite()
            && input.medium.c_liquid > 0.0
            && input.medium.xi_s.is_finite()
            && input.medium.xi_s >= 0.0)
    {
        return None;
    }
    let n_steps = input.n_cycles * input.steps_per_cycle as f64;
    if !(n_steps.is_finite()
        && n_steps.round() >= 1.0
        && n_steps.round() <= MAX_EXACT_F64_INTEGER as f64)
    {
        return None;
    }
    Some(())
}

fn decompose_or_zero(
    freqs_hz: &[f64],
    psd: &[f64],
    f0_hz: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
) -> CavitationBandEnergies {
    if freqs_hz.len() <= 1 || psd.len() <= 1 {
        return CavitationBandEnergies {
            fundamental: 0.0,
            subharmonic: 0.0,
            ultraharmonic: 0.0,
            broadband: 0.0,
        };
    }
    decompose_emission_spectrum(freqs_hz, psd, f0_hz, rel_halfwidth, noise_floor)
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}
