//! Histotripsy therapy-delivery helpers shared by PyO3 chapter orchestration.
//!
//! These routines keep geometric safety, measured-spectrum scaling, and delivered
//! dose-response accounting in the Rust physics crate. Python callers may build
//! arrays and plot results, but the therapy semantics live here.

use super::{
    histotripsy_kill_fraction, simulate_population_emission, PopulationEmissionInput,
    PopulationMedium, PopulationShell,
};
use crate::analytical::transducer::electronic_steering_efficiency;
use crate::analytical::wave::{shock_formation_distance, shock_heat_source_density};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};

const MAX_EXACT_F64_INTEGER: usize = 1usize << 53;

/// Lateral semi-axis that keeps an anisotropic focal ellipsoid within an
/// isotropic clearance constraint.
///
/// If a boiling-histotripsy focus has axial/lateral semi-axis ratio `a_z/a_r`,
/// then a clearance bound applies to the largest semi-axis, not only the
/// transverse radius. This function returns `min(natural_lateral, clearance /
/// max(1, a_z/a_r))`.
#[must_use]
pub fn clipped_lateral_radius_for_clearance(
    natural_lateral_radius_m: f64,
    clearance_m: f64,
    axial_to_lateral_ratio: f64,
) -> f64 {
    if !(natural_lateral_radius_m.is_finite()
        && clearance_m.is_finite()
        && axial_to_lateral_ratio.is_finite())
        || natural_lateral_radius_m <= 0.0
        || clearance_m <= 0.0
    {
        return 0.0;
    }
    let ratio = axial_to_lateral_ratio.max(1.0);
    natural_lateral_radius_m.min(clearance_m / ratio).max(0.0)
}

/// Check that every voxel inside a focal ellipsoid is inside an allowed mask.
///
/// `allowed_mask` is row-major with dimensions `(nx, ny, nz)`. The beam axis is
/// `x`; the focal ellipsoid uses `axial_radius_m` on x and `lateral_radius_m`
/// on y/z. The function returns false when the center or any part of the
/// ellipsoid exits the grid or overlaps a false mask voxel.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ellipsoid_respects_allowed_mask(
    allowed_mask: &[bool],
    nx: usize,
    ny: usize,
    nz: usize,
    center_x: usize,
    center_y: usize,
    center_z: usize,
    lateral_radius_m: f64,
    axial_radius_m: f64,
    dx_m: f64,
) -> bool {
    if allowed_mask.len() != nx.saturating_mul(ny).saturating_mul(nz)
        || nx == 0
        || ny == 0
        || nz == 0
        || center_x >= nx
        || center_y >= ny
        || center_z >= nz
        || lateral_radius_m <= 0.0
        || axial_radius_m <= 0.0
        || dx_m <= 0.0
    {
        return false;
    }

    let rx = (axial_radius_m / dx_m).ceil() as isize;
    let rr = (lateral_radius_m / dx_m).ceil() as isize;
    let cx = center_x as isize;
    let cy = center_y as isize;
    let cz = center_z as isize;

    let x0 = cx - rx;
    let x1 = cx + rx;
    let y0 = cy - rr;
    let y1 = cy + rr;
    let z0 = cz - rr;
    let z1 = cz + rr;
    if x0 < 0 || y0 < 0 || z0 < 0 || x1 >= nx as isize || y1 >= ny as isize || z1 >= nz as isize {
        return false;
    }

    let inv_ax = 1.0 / axial_radius_m;
    let inv_lat = 1.0 / lateral_radius_m;
    for ix in x0..=x1 {
        let dxn = ((ix - cx) as f64 * dx_m * inv_ax).powi(2);
        for iy in y0..=y1 {
            let dyn_ = ((iy - cy) as f64 * dx_m * inv_lat).powi(2);
            for iz in z0..=z1 {
                let dzn = ((iz - cz) as f64 * dx_m * inv_lat).powi(2);
                if dxn + dyn_ + dzn <= 1.0 {
                    let idx = ((ix as usize) * ny + iy as usize) * nz + iz as usize;
                    if !allowed_mask[idx] {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Apply receive-path and tissue-state scaling to a passive cavitation PSD.
///
/// The same factors that scale the measured scalar cavitation signal must scale
/// the plotted spectrum. `receive_fraction` is an amplitude/energy transfer
/// fraction for the passive path; `susceptibility` accounts for local lesion
/// memory and interface-enhanced cavitation source strength.
#[must_use]
pub fn scale_measured_emission_spectrum(
    psd: &[f64],
    receive_fraction: f64,
    susceptibility: f64,
) -> Vec<f64> {
    let scale = receive_fraction.max(0.0) * susceptibility.max(0.0);
    psd.iter().map(|v| v.max(0.0) * scale).collect()
}

/// Convert a cumulative delivered histotripsy dose series into kill fractions.
///
/// Dose samples are clamped to nonnegative values. The returned value uses the
/// Weibull survival law implemented by [`histotripsy_kill_fraction`].
#[must_use]
pub fn delivered_histotripsy_progress(dose: &[f64], d0: f64, weibull_k: f64) -> Vec<f64> {
    dose.iter()
        .map(|&d| histotripsy_kill_fraction(d.max(0.0), d0, weibull_k))
        .collect()
}

/// Borrowed inputs for rastered per-spot passive-cavitation dose accounting.
#[derive(Clone, Copy, Debug)]
pub struct PerSpotCavitationDoseInput<'a> {
    /// Lateral steering offsets from the mechanical focus `m`.
    pub lateral_offsets_m: &'a [f64],
    /// Axial steering offsets from the mechanical focus `m`.
    pub axial_offsets_m: &'a [f64],
    /// Target peak pressure at the mechanical focus `Pa`.
    pub p_target_pa: f64,
    /// Drive frequency `Hz`.
    pub f0_hz: f64,
    /// Sound speed used by the steering-efficiency model [m/s].
    pub c_m_s: f64,
    /// Monotone pressure sweep samples `Pa`.
    pub pressures_pa: &'a [f64],
    /// Passive-cavitation emission power at each pressure sample.
    pub cavitation_power: &'a [f64],
    /// Number of pulses delivered at each raster spot.
    pub n_pulses_per_spot: usize,
    /// Effective pressure that defines the prescribed per-spot dose goal `Pa`.
    pub goal_pressure_pa: f64,
    /// One-way axial attenuation coefficient for positive axial offsets [Np/m].
    pub attenuation_np_m: f64,
    /// Whether electronic steering uses apodized aperture compensation.
    pub apodized: bool,
}

/// Row-major `(axial × lateral)` passive-cavitation dose grid.
#[derive(Clone, Debug, PartialEq)]
pub struct PerSpotCavitationDoseGrid {
    /// Per-spot cumulative cavitation dose.
    pub dose: Vec<f64>,
    /// Electronic-steering efficiency at each raster point.
    pub efficiency: Vec<f64>,
    /// Delivered peak pressure at each raster point `Pa`.
    pub p_spot_pa: Vec<f64>,
    /// Prescribed per-spot dose goal.
    pub goal_dose: f64,
    /// Number of axial rows in the row-major arrays.
    pub axial_count: usize,
    /// Number of lateral columns in the row-major arrays.
    pub lateral_count: usize,
}

/// Borrowed inputs for a curve-driven passive-cavitation monitor trace.
#[derive(Clone, Copy, Debug)]
pub struct CavitationMonitorTraceInput<'a> {
    /// Monotone pressure sweep samples `Pa`.
    pub pressures_pa: &'a [f64],
    /// Passive-cavitation emission power at each pressure sample.
    pub cavitation_power: &'a [f64],
    /// Number of controller pulses to simulate.
    pub n_pulses: usize,
    /// Pulse repetition frequency `Hz`.
    pub prf_hz: f64,
    /// Initial controller pressure `Pa`.
    pub p_start_pa: f64,
    /// Stable-emission target for the controller.
    pub target_signal: f64,
    /// Inertial-emission cap for the controller.
    pub inertial_cap: f64,
    /// Fractional pressure-step gain.
    pub gain: f64,
    /// Log-normal jitter sigma applied to the deterministic emission curve.
    pub jitter_sigma: f64,
    /// Fraction of final cumulative dose used as the plotted dose goal.
    pub goal_fraction: f64,
    /// Deterministic RNG seed for jitter.
    pub seed: u64,
}

/// Passive-cavitation monitor trace with controller pressure history.
#[derive(Clone, Debug, PartialEq)]
pub struct CavitationMonitorTrace {
    /// Time samples `s`.
    pub time_s: Vec<f64>,
    /// Measured cavitation signal per pulse.
    pub cavitation_signal: Vec<f64>,
    /// Applied acoustic power percentage, proportional to pressure squared.
    pub power_percent: Vec<f64>,
    /// Running cumulative cavitation dose.
    pub cumulative_dose: Vec<f64>,
    /// Prescribed plotted dose goal.
    pub goal_dose: f64,
}

/// Owned inputs for a simulated passive-cavitation monitor trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulatedPopulationMonitorInput {
    /// Fundamental drive frequency `Hz`.
    pub f0_hz: f64,
    /// Shared liquid and gas parameters.
    pub medium: PopulationMedium,
    /// Number of bubbles simulated per pulse.
    pub n_bubbles: usize,
    /// Number of controller pulses.
    pub n_pulses: usize,
    /// Pulse repetition frequency `Hz`.
    pub prf_hz: f64,
    /// Initial controller pressure `Pa`.
    pub p_start_pa: f64,
    /// Minimum controller pressure `Pa`.
    pub p_min_pa: f64,
    /// Maximum controller pressure `Pa`.
    pub p_max_pa: f64,
    /// Stable-emission target for the controller.
    pub target_signal: f64,
    /// Broadband-emission cap for the controller.
    pub inertial_cap: f64,
    /// Fractional pressure-step gain.
    pub gain: f64,
    /// Fraction of final cumulative dose used as the plotted dose goal.
    pub goal_fraction: f64,
    /// Deterministic RNG seed for per-pulse population seeds.
    pub seed: u64,
    /// Median equilibrium radius of the log-normal population `m`.
    pub r0_median_m: f64,
    /// Natural-log radius standard deviation.
    pub r0_sigma_ln: f64,
    /// Simulated drive cycles per pulse.
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
    /// Shell model parameters.
    pub shell: PopulationShell,
}

/// Simulated monitor trace with stable and broadband channels retained.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedPopulationMonitorTrace {
    /// Time samples `s`.
    pub time_s: Vec<f64>,
    /// Stable + broadband cavitation signal per pulse.
    pub cavitation_signal: Vec<f64>,
    /// Applied acoustic power percentage, proportional to pressure squared.
    pub power_percent: Vec<f64>,
    /// Running cumulative cavitation dose.
    pub cumulative_dose: Vec<f64>,
    /// Prescribed plotted dose goal.
    pub goal_dose: f64,
    /// Stable cavitation signal, subharmonic plus ultraharmonic.
    pub stable_signal: Vec<f64>,
    /// Broadband inertial-cavitation signal.
    pub broadband_signal: Vec<f64>,
}

/// Borrowed inputs for closed-loop passive-cavitation sonication control.
#[derive(Clone, Copy, Debug)]
pub struct ClosedLoopCavitationSonicationInput<'a> {
    /// Monotone pressure sweep samples `Pa`.
    pub pressures_pa: &'a [f64],
    /// Stable-cavitation emission power at each pressure sample.
    pub stable_power: &'a [f64],
    /// Inertial-cavitation emission power at each pressure sample.
    pub inertial_power: &'a [f64],
    /// Number of controller bursts.
    pub n_bursts: usize,
    /// Burst duration used for trapezoidal dose integration `s`.
    pub burst_duration_s: f64,
    /// Initial controller pressure `Pa`.
    pub p_start_pa: f64,
    /// Stable-emission target for the controller.
    pub stable_target: f64,
    /// Inertial-emission cap for the controller.
    pub inertial_limit: f64,
    /// Fractional pressure-step gain.
    pub gain: f64,
}

/// Closed-loop passive-cavitation sonication trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopCavitationSonicationTrace {
    /// Applied pressure per burst `Pa`.
    pub pressure_pa: Vec<f64>,
    /// Stable-cavitation emission sampled at each applied pressure.
    pub stable_emission: Vec<f64>,
    /// Inertial-cavitation emission sampled at each applied pressure.
    pub inertial_emission: Vec<f64>,
    /// Running stable-cavitation dose.
    pub stable_dose: Vec<f64>,
    /// Running inertial-cavitation dose.
    pub inertial_dose: Vec<f64>,
}

/// Firing order for rastered cavitation-dose pulsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterPulsingSchedule {
    /// Fire all pulses at one spot before moving to the next spot.
    Sequential,
    /// Round-robin over the raster spots for each pulse repetition.
    Interleaved,
}

/// Borrowed inputs for rastered cavitation-dose pulsing with thermal memory.
#[derive(Clone, Copy, Debug)]
pub struct RasterPulsingInput<'a> {
    /// Spot lateral offsets `m`.
    pub spot_lateral_m: &'a [f64],
    /// Spot axial offsets `m`.
    pub spot_axial_m: &'a [f64],
    /// Target pressure at the mechanical focus `Pa`.
    pub p_target_pa: f64,
    /// Drive frequency `Hz`.
    pub f0_hz: f64,
    /// Sound speed used by the steering-efficiency model [m/s].
    pub c_m_s: f64,
    /// Monotone cavitation-dose pressure sweep `Pa`.
    pub cav_pressures_pa: &'a [f64],
    /// Cavitation dose per pulse at each pressure sample.
    pub cav_dose_per_pulse: &'a [f64],
    /// Pulses delivered per raster spot.
    pub pulses_per_spot: usize,
    /// Pulse repetition frequency `Hz`.
    pub prf_hz: f64,
    /// Firing order.
    pub schedule: RasterPulsingSchedule,
    /// Interleave group size; zero means all spots.
    pub interleave_group: usize,
    /// One-way attenuation coefficient for positive axial offsets [Np/m].
    pub attenuation_np_m: f64,
    /// Whether electronic steering uses apodized aperture compensation.
    pub apodized: bool,
    /// Residual-bubble dissolution time `s`.
    pub tau_dissolution_s: f64,
    /// Residual-bubble shielding gain.
    pub shielding_g: f64,
    /// Thermal relaxation time constant `s`.
    pub tau_thermal_s: f64,
    /// Temperature rise at full pressure for one pulse `K`.
    pub thermal_gain_k_per_pulse: f64,
    /// Dose threshold used for coverage.
    pub goal_dose: f64,
    /// Number of compact output time samples.
    pub n_time_samples: usize,
}

/// Rastered pulsing trace with compact time-series and per-spot final state.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterPulsingTrace {
    /// Compact time axis `s`.
    pub time_s: Vec<f64>,
    /// Fraction of spots at or above the dose goal.
    pub coverage: Vec<f64>,
    /// Running sum of effective cavitation dose over all spots.
    pub cumulative_dose: Vec<f64>,
    /// Final effective cavitation dose per spot.
    pub per_spot_dose: Vec<f64>,
    /// Peak temperature rise per spot `K`.
    pub per_spot_peak_temp_k: Vec<f64>,
    /// Steady residual-bubble shielding efficacy for repeat pulses.
    pub efficacy: f64,
    /// Effective per-spot interval `s`.
    pub dt_spot_s: f64,
    /// Total treatment duration represented by the firing order `s`.
    pub treatment_s: f64,
    /// Delivered pressure per spot `Pa`.
    pub p_spot_pa: Vec<f64>,
}

/// Compute rastered per-spot passive-cavitation dose for a steered treatment grid.
///
/// Each spot uses the same pressure and emission-power contract as the Chapter
/// 24 passive-cavitation monitor:
///
/// ```text
/// p_spot = p_target · ε(Δlat, Δax) · exp(-α · max(Δax, 0))
/// dose   = n_pulses · interp_clamped(p_spot, pressures, cavitation_power)
/// ```
///
/// The interpolation matches NumPy `interp` endpoint clamping for a strictly
/// increasing pressure axis. Returns `None` when the input slices are empty,
/// mismatched, non-finite, non-monotone, or would require an inexact pulse-count
/// conversion into the floating-point dose model. This is an empirical
/// validation helper, not a machine-checked proof of BBB opening dynamics.
#[must_use]
pub fn per_spot_cavitation_dose_grid(
    input: PerSpotCavitationDoseInput<'_>,
) -> Option<PerSpotCavitationDoseGrid> {
    validate_per_spot_input(input)?;
    let lateral_count = input.lateral_offsets_m.len();
    let axial_count = input.axial_offsets_m.len();
    let sample_count = axial_count.checked_mul(lateral_count)?;

    if input.n_pulses_per_spot > MAX_EXACT_F64_INTEGER {
        return None;
    }
    // The bound above guarantees exact representation in the dose model.
    let pulse_count = input.n_pulses_per_spot as f64;

    let mut dose = Vec::with_capacity(sample_count);
    let mut efficiency = Vec::with_capacity(sample_count);
    let mut p_spot_pa = Vec::with_capacity(sample_count);
    for &dz_m in input.axial_offsets_m {
        let transmission = (-input.attenuation_np_m * dz_m.max(0.0)).exp();
        for &dx_m in input.lateral_offsets_m {
            let steering = electronic_steering_efficiency(
                dx_m,
                dz_m,
                input.f0_hz,
                input.c_m_s,
                input.apodized,
            );
            let p_spot = input.p_target_pa * steering * transmission;
            let spot_power =
                interpolate_clamped(p_spot, input.pressures_pa, input.cavitation_power)?;
            efficiency.push(steering);
            p_spot_pa.push(p_spot);
            dose.push(pulse_count * spot_power);
        }
    }
    let goal_power = interpolate_clamped(
        input.goal_pressure_pa,
        input.pressures_pa,
        input.cavitation_power,
    )?;

    Some(PerSpotCavitationDoseGrid {
        dose,
        efficiency,
        p_spot_pa,
        goal_dose: pulse_count * goal_power,
        axial_count,
        lateral_count,
    })
}

/// Simulate a curve-driven passive-cavitation monitor trace.
///
/// The measured cavitation signal for each pulse is the endpoint-clamped
/// interpolation of `cavitation_power(pressure)` multiplied by seeded log-normal
/// jitter. The controller receives that measured signal as both the stable and
/// inertial channel, matching the Chapter 24 aggregate cavitation-monitor
/// display. Cumulative dose uses the trapezoidal integration contract from
/// [`crate::analytical::cavitation::cumulative_cavitation_dose`].
#[must_use]
pub fn cavitation_monitor_trace(
    input: CavitationMonitorTraceInput<'_>,
) -> Option<CavitationMonitorTrace> {
    validate_pressure_power_sweep(input.pressures_pa, input.cavitation_power)?;
    if input.n_pulses == 0
        || !(input.prf_hz.is_finite()
            && input.prf_hz > 0.0
            && input.p_start_pa.is_finite()
            && input.p_start_pa >= 0.0
            && input.target_signal.is_finite()
            && input.target_signal >= 0.0
            && input.inertial_cap.is_finite()
            && input.inertial_cap >= 0.0
            && input.gain.is_finite()
            && input.gain >= 0.0
            && input.jitter_sigma.is_finite()
            && input.jitter_sigma >= 0.0
            && input.goal_fraction.is_finite()
            && input.goal_fraction >= 0.0)
    {
        return None;
    }
    let p_min_pa = input.pressures_pa[0];
    let p_max_pa = *input.pressures_pa.last()?;
    if p_max_pa <= 0.0 {
        return None;
    }
    let dt_s = 1.0 / input.prf_hz;
    let jitter_distribution = if input.jitter_sigma > 0.0 {
        Some(LogNormal::new(0.0, input.jitter_sigma).ok()?)
    } else {
        None
    };
    let mut rng = ChaCha8Rng::seed_from_u64(input.seed);
    let mut pressure_pa = input.p_start_pa.clamp(p_min_pa, p_max_pa);
    let mut signal = Vec::with_capacity(input.n_pulses);
    let mut power_percent = Vec::with_capacity(input.n_pulses);

    for _ in 0..input.n_pulses {
        let deterministic =
            interpolate_clamped(pressure_pa, input.pressures_pa, input.cavitation_power)?;
        let jitter = jitter_distribution
            .as_ref()
            .map_or(1.0, |distribution| distribution.sample(&mut rng));
        let measured = deterministic * jitter;
        signal.push(measured);
        power_percent.push((pressure_pa / p_max_pa).powi(2) * 100.0);
        pressure_pa = super::cavitation_controller_pressure(
            pressure_pa,
            measured,
            measured,
            input.target_signal,
            input.inertial_cap,
            input.gain,
            p_min_pa,
            p_max_pa,
        );
    }

    let time_s = (0..input.n_pulses).map(|i| i as f64 * dt_s).collect();
    let cumulative_dose = super::cumulative_cavitation_dose(&signal, dt_s);
    let final_dose = cumulative_dose.last().copied().unwrap_or(0.0);
    let goal_dose = if final_dose > 0.0 {
        input.goal_fraction * final_dose
    } else {
        1.0
    };

    Some(CavitationMonitorTrace {
        time_s,
        cavitation_signal: signal,
        power_percent,
        cumulative_dose,
        goal_dose,
    })
}

/// Simulate a passive-cavitation monitor trace from fresh bubble populations.
///
/// Each pulse draws a deterministic fresh population seed, runs the production
/// population-emission model at the current controller pressure, records stable
/// (`subharmonic + ultraharmonic`) and broadband emission, steps the
/// cavitation-dose controller, and integrates the combined signal with the
/// shared trapezoidal dose contract. This is empirical simulation evidence, not
/// a proof of BBB-opening dynamics.
#[must_use]
pub fn simulated_population_monitor_trace(
    input: SimulatedPopulationMonitorInput,
) -> Option<SimulatedPopulationMonitorTrace> {
    validate_simulated_population_monitor_input(input)?;
    let dt_s = 1.0 / input.prf_hz;
    let mut rng = ChaCha8Rng::seed_from_u64(input.seed);
    let mut pressure_pa = input.p_start_pa.clamp(input.p_min_pa, input.p_max_pa);
    let mut signal = Vec::with_capacity(input.n_pulses);
    let mut power_percent = Vec::with_capacity(input.n_pulses);
    let mut stable = Vec::with_capacity(input.n_pulses);
    let mut broadband = Vec::with_capacity(input.n_pulses);

    for _ in 0..input.n_pulses {
        let emission = simulate_population_emission(PopulationEmissionInput {
            drive_pa: pressure_pa,
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
        let stable_signal = emission.bands.subharmonic + emission.bands.ultraharmonic;
        let broadband_signal = emission.bands.broadband;
        stable.push(stable_signal);
        broadband.push(broadband_signal);
        signal.push(stable_signal + broadband_signal);
        power_percent.push((pressure_pa / input.p_max_pa).powi(2) * 100.0);
        pressure_pa = super::cavitation_controller_pressure(
            pressure_pa,
            stable_signal,
            broadband_signal,
            input.target_signal,
            input.inertial_cap,
            input.gain,
            input.p_min_pa,
            input.p_max_pa,
        );
    }

    let time_s = (0..input.n_pulses).map(|i| i as f64 * dt_s).collect();
    let cumulative_dose = super::cumulative_cavitation_dose(&signal, dt_s);
    let final_dose = cumulative_dose.last().copied().unwrap_or(0.0);
    let goal_dose = if final_dose > 0.0 {
        input.goal_fraction * final_dose
    } else {
        1.0
    };

    Some(SimulatedPopulationMonitorTrace {
        time_s,
        cavitation_signal: signal,
        power_percent,
        cumulative_dose,
        goal_dose,
        stable_signal: stable,
        broadband_signal: broadband,
    })
}

/// Run closed-loop passive-cavitation sonication from swept emission curves.
///
/// At each burst the stable and inertial emissions are endpoint-clamped
/// interpolations of the swept emission curves at the current pressure. The
/// pressure for the next burst is selected by the clinical cavitation-dose
/// controller, and both emission histories are integrated with the shared
/// trapezoidal cumulative-dose contract.
#[must_use]
pub fn closed_loop_cavitation_sonication(
    input: ClosedLoopCavitationSonicationInput<'_>,
) -> Option<ClosedLoopCavitationSonicationTrace> {
    validate_pressure_power_sweep(input.pressures_pa, input.stable_power)?;
    validate_pressure_power_sweep(input.pressures_pa, input.inertial_power)?;
    if input.n_bursts == 0
        || !(input.burst_duration_s.is_finite()
            && input.burst_duration_s > 0.0
            && input.p_start_pa.is_finite()
            && input.p_start_pa >= 0.0
            && input.stable_target.is_finite()
            && input.stable_target >= 0.0
            && input.inertial_limit.is_finite()
            && input.inertial_limit >= 0.0
            && input.gain.is_finite()
            && input.gain >= 0.0)
    {
        return None;
    }

    let p_min_pa = input.pressures_pa[0];
    let p_max_pa = *input.pressures_pa.last()?;
    let mut pressure_pa = input.p_start_pa.clamp(p_min_pa, p_max_pa);
    let mut pressure = Vec::with_capacity(input.n_bursts);
    let mut stable = Vec::with_capacity(input.n_bursts);
    let mut inertial = Vec::with_capacity(input.n_bursts);

    for _ in 0..input.n_bursts {
        let stable_emission =
            interpolate_clamped(pressure_pa, input.pressures_pa, input.stable_power)?;
        let inertial_emission =
            interpolate_clamped(pressure_pa, input.pressures_pa, input.inertial_power)?;
        pressure.push(pressure_pa);
        stable.push(stable_emission);
        inertial.push(inertial_emission);
        pressure_pa = super::cavitation_controller_pressure(
            pressure_pa,
            stable_emission,
            inertial_emission,
            input.stable_target,
            input.inertial_limit,
            input.gain,
            p_min_pa,
            p_max_pa,
        );
    }

    let stable_dose = super::cumulative_cavitation_dose(&stable, input.burst_duration_s);
    let inertial_dose = super::cumulative_cavitation_dose(&inertial, input.burst_duration_s);
    Some(ClosedLoopCavitationSonicationTrace {
        pressure_pa: pressure,
        stable_emission: stable,
        inertial_emission: inertial,
        stable_dose,
        inertial_dose,
    })
}

/// Simulate rastered cavitation-dose pulsing over a steered spot set.
///
/// Each spot receives pressure derating from electronic steering and positive
/// axial attenuation, dose per pulse from endpoint-clamped interpolation over
/// the measured cavitation-dose curve, residual-bubble shielding from
/// [`crate::analytical::cavitation::prf_efficacy_factor`], and a first-order
/// thermal relaxation memory. The compact time-series is linearly resampled from
/// the per-fire event history for plotting.
#[must_use]
pub fn raster_cavitation_pulsing(input: RasterPulsingInput<'_>) -> Option<RasterPulsingTrace> {
    validate_pressure_power_sweep(input.cav_pressures_pa, input.cav_dose_per_pulse)?;
    let n_spots = input.spot_lateral_m.len();
    if n_spots == 0
        || input.spot_axial_m.len() != n_spots
        || input.n_time_samples == 0
        || !(input.p_target_pa.is_finite()
            && input.p_target_pa >= 0.0
            && input.f0_hz.is_finite()
            && input.f0_hz > 0.0
            && input.c_m_s.is_finite()
            && input.c_m_s > 0.0
            && input.prf_hz.is_finite()
            && input.prf_hz > 0.0
            && input.attenuation_np_m.is_finite()
            && input.attenuation_np_m >= 0.0
            && input.tau_dissolution_s.is_finite()
            && input.tau_dissolution_s >= 0.0
            && input.shielding_g.is_finite()
            && input.shielding_g >= 0.0
            && input.tau_thermal_s.is_finite()
            && input.thermal_gain_k_per_pulse.is_finite()
            && input.thermal_gain_k_per_pulse >= 0.0
            && input.goal_dose.is_finite()
            && input.goal_dose >= 0.0)
        || !input.spot_lateral_m.iter().all(|v| v.is_finite())
        || !input.spot_axial_m.iter().all(|v| v.is_finite())
    {
        return None;
    }
    if n_spots > MAX_EXACT_F64_INTEGER || input.n_time_samples > MAX_EXACT_F64_INTEGER {
        return None;
    }

    let mut p_spot_pa = Vec::with_capacity(n_spots);
    let mut base_dose = Vec::with_capacity(n_spots);
    for (&lat_m, &ax_m) in input.spot_lateral_m.iter().zip(input.spot_axial_m.iter()) {
        let steering =
            electronic_steering_efficiency(lat_m, ax_m, input.f0_hz, input.c_m_s, input.apodized);
        let transmission = (-input.attenuation_np_m * ax_m.max(0.0)).exp();
        let pressure = input.p_target_pa * steering * transmission;
        p_spot_pa.push(pressure);
        base_dose.push(interpolate_clamped(
            pressure,
            input.cav_pressures_pa,
            input.cav_dose_per_pulse,
        )?);
    }

    let group = if input.interleave_group == 0 {
        n_spots
    } else {
        input.interleave_group.min(n_spots)
    };
    let dt_spot_s = match input.schedule {
        RasterPulsingSchedule::Sequential => 1.0 / input.prf_hz,
        RasterPulsingSchedule::Interleaved => group as f64 / input.prf_hz,
    };
    let efficacy = super::prf_efficacy_factor(
        &[1.0 / dt_spot_s],
        input.tau_dissolution_s,
        input.shielding_g,
    )
    .into_iter()
    .next()
    .unwrap_or(0.0);
    let thermal_tau = input.tau_thermal_s.max(1.0e-12);
    let pressure_scale = input.p_target_pa.max(1.0);
    let delta_t_per_pulse: Vec<f64> = p_spot_pa
        .iter()
        .map(|&p| input.thermal_gain_k_per_pulse * (p / pressure_scale).powi(2))
        .collect();

    let n_fire = n_spots.checked_mul(input.pulses_per_spot)?;
    if n_fire > MAX_EXACT_F64_INTEGER {
        return None;
    }
    let mut per_spot_dose = vec![0.0_f64; n_spots];
    let mut per_spot_temp = vec![0.0_f64; n_spots];
    let mut per_spot_peak_temp = vec![0.0_f64; n_spots];
    let mut last_fire_s = vec![None; n_spots];
    let mut event_time = Vec::with_capacity(n_fire);
    let mut cumulative = Vec::with_capacity(n_fire);
    let mut coverage = Vec::with_capacity(n_fire);
    let mut running_dose = 0.0_f64;

    for fire_index in 0..n_fire {
        let spot = match input.schedule {
            RasterPulsingSchedule::Sequential => fire_index / input.pulses_per_spot,
            RasterPulsingSchedule::Interleaved => fire_index % n_spots,
        };
        let t_s = fire_index as f64 / input.prf_hz;
        let repeat_efficacy = last_fire_s[spot].map_or(1.0, |_| efficacy);
        let effective_dose = base_dose[spot] * repeat_efficacy;
        per_spot_dose[spot] += effective_dose;
        running_dose += effective_dose;

        let cooling = last_fire_s[spot].map_or(1.0, |last| {
            let gap: f64 = t_s - last;
            (-gap / thermal_tau).exp()
        });
        per_spot_temp[spot] = per_spot_temp[spot] * cooling + delta_t_per_pulse[spot];
        per_spot_peak_temp[spot] = per_spot_peak_temp[spot].max(per_spot_temp[spot]);
        last_fire_s[spot] = Some(t_s);

        event_time.push(t_s);
        cumulative.push(running_dose);
        coverage.push(if input.goal_dose > 0.0 {
            per_spot_dose
                .iter()
                .filter(|&&dose| dose >= input.goal_dose)
                .count() as f64
                / n_spots as f64
        } else {
            0.0
        });
    }

    let treatment_s = event_time.last().copied().unwrap_or(0.0);
    let compact_t_end = treatment_s.max(1.0e-9);
    let time_s = linspace_inclusive(0.0, compact_t_end, input.n_time_samples);
    let compact_coverage = if n_fire > 0 {
        resample_linear_clamped(&time_s, &event_time, &coverage)?
    } else {
        vec![0.0; input.n_time_samples]
    };
    let compact_cumulative = if n_fire > 0 {
        resample_linear_clamped(&time_s, &event_time, &cumulative)?
    } else {
        vec![0.0; input.n_time_samples]
    };

    Some(RasterPulsingTrace {
        time_s,
        coverage: compact_coverage,
        cumulative_dose: compact_cumulative,
        per_spot_dose,
        per_spot_peak_temp_k: per_spot_peak_temp,
        efficacy,
        dt_spot_s,
        treatment_s,
        p_spot_pa,
    })
}

fn validate_per_spot_input(input: PerSpotCavitationDoseInput<'_>) -> Option<()> {
    if input.lateral_offsets_m.is_empty()
        || input.axial_offsets_m.is_empty()
        || !(input.p_target_pa.is_finite()
            && input.p_target_pa >= 0.0
            && input.f0_hz.is_finite()
            && input.f0_hz > 0.0
            && input.c_m_s.is_finite()
            && input.c_m_s > 0.0
            && input.goal_pressure_pa.is_finite()
            && input.goal_pressure_pa >= 0.0
            && input.attenuation_np_m.is_finite()
            && input.attenuation_np_m >= 0.0)
    {
        return None;
    }
    if !input.lateral_offsets_m.iter().all(|v| v.is_finite())
        || !input.axial_offsets_m.iter().all(|v| v.is_finite())
    {
        return None;
    }
    validate_pressure_power_sweep(input.pressures_pa, input.cavitation_power)?;
    Some(())
}

fn validate_simulated_population_monitor_input(
    input: SimulatedPopulationMonitorInput,
) -> Option<()> {
    if input.n_bubbles == 0
        || input.n_bubbles > MAX_EXACT_F64_INTEGER
        || input.n_pulses == 0
        || input.n_pulses > MAX_EXACT_F64_INTEGER
        || input.n_out < 16
        || input.n_out > MAX_EXACT_F64_INTEGER
        || input.shell.steps_per_cycle == 0
        || !(input.f0_hz.is_finite()
            && input.f0_hz > 0.0
            && input.prf_hz.is_finite()
            && input.prf_hz > 0.0
            && input.p_start_pa.is_finite()
            && input.p_start_pa >= 0.0
            && input.p_min_pa.is_finite()
            && input.p_min_pa >= 0.0
            && input.p_max_pa.is_finite()
            && input.p_max_pa > input.p_min_pa
            && input.target_signal.is_finite()
            && input.target_signal >= 0.0
            && input.inertial_cap.is_finite()
            && input.inertial_cap >= 0.0
            && input.gain.is_finite()
            && input.gain >= 0.0
            && input.goal_fraction.is_finite()
            && input.goal_fraction >= 0.0
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

fn validate_pressure_power_sweep(pressures_pa: &[f64], power: &[f64]) -> Option<()> {
    if pressures_pa.is_empty()
        || pressures_pa.len() != power.len()
        || !pressures_pa.iter().all(|v| v.is_finite() && *v >= 0.0)
        || !power.iter().all(|v| v.is_finite() && *v >= 0.0)
        || !pressures_pa.windows(2).all(|w| w[0] < w[1])
    {
        return None;
    }
    Some(())
}

fn interpolate_clamped(x: f64, xs: &[f64], ys: &[f64]) -> Option<f64> {
    if !(x.is_finite() && xs.len() == ys.len()) || xs.is_empty() {
        return None;
    }
    if x <= xs[0] {
        return Some(ys[0]);
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return Some(ys[last]);
    }
    let upper = xs.partition_point(|&v| v < x);
    let lower = upper.checked_sub(1)?;
    let span = xs[upper] - xs[lower];
    if span <= 0.0 {
        return None;
    }
    let t = (x - xs[lower]) / span;
    Some(ys[lower].mul_add(1.0 - t, ys[upper] * t))
}

fn linspace_inclusive(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

fn resample_linear_clamped(query: &[f64], xs: &[f64], ys: &[f64]) -> Option<Vec<f64>> {
    if xs.len() != ys.len() || xs.is_empty() {
        return None;
    }
    query
        .iter()
        .map(|&x| interpolate_clamped(x, xs, ys))
        .collect()
}

/// Agreement metrics for a cavitation-cloud erosion validation curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudErosionValidation {
    /// Non-negative least-squares scale applied to the model curve.
    pub model_scale: f64,
    /// Root-mean-square error after scaling, in the reference units.
    pub rmse: f64,
    /// `rmse / (max(reference) - min(reference))`; zero for a flat reference.
    pub normalized_rmse: f64,
    /// Maximum absolute pointwise error after scaling.
    pub max_abs_error: f64,
    /// Maximum relative pointwise error after scaling; ignores zero reference samples.
    pub max_relative_error: f64,
    /// Pearson correlation coefficient between the scaled model and reference.
    pub pearson_r: f64,
    /// Number of paired samples used.
    pub sample_count: usize,
}

/// Compare a modeled cavitation-cloud erosion curve with an experimental or
/// k-wave reference curve.
///
/// The model may be in arbitrary proportional units, so the comparator first
/// applies the single physically allowed calibration for an erosion-efficiency
/// coefficient: a non-negative least-squares scalar
/// `s = argmin_s ||s·model - reference||₂`. Shape agreement is then reported by
/// RMSE, range-normalized RMSE, maximum errors, and Pearson correlation. This is
/// an empirical-validation metric; it is not a proof of the erosion model.
#[must_use]
pub fn cloud_erosion_validation_metrics(
    reference_erosion: &[f64],
    model_erosion: &[f64],
) -> Option<CloudErosionValidation> {
    let n = reference_erosion.len().min(model_erosion.len());
    if n < 2 {
        return None;
    }

    let mut dot_rm = 0.0_f64;
    let mut dot_mm = 0.0_f64;
    let mut r_min = f64::INFINITY;
    let mut r_max = f64::NEG_INFINITY;
    let mut count = 0usize;
    for (&r, &m) in reference_erosion.iter().zip(model_erosion.iter()).take(n) {
        if !(r.is_finite() && m.is_finite()) {
            return None;
        }
        dot_rm += r * m;
        dot_mm += m * m;
        r_min = r_min.min(r);
        r_max = r_max.max(r);
        count += 1;
    }
    if count < 2 || dot_mm <= 0.0 {
        return None;
    }

    let scale = (dot_rm / dot_mm).max(0.0);
    let mut sum_sq = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    let mut mean_r = 0.0_f64;
    let mut mean_m = 0.0_f64;
    for (&r, &m) in reference_erosion.iter().zip(model_erosion.iter()).take(n) {
        let sm = scale * m;
        let err = sm - r;
        sum_sq += err * err;
        max_abs_error = max_abs_error.max(err.abs());
        if r != 0.0 {
            max_relative_error = max_relative_error.max((err / r).abs());
        }
        mean_r += r;
        mean_m += sm;
    }
    let inv_n = 1.0 / count as f64;
    mean_r *= inv_n;
    mean_m *= inv_n;

    let mut cov = 0.0_f64;
    let mut var_r = 0.0_f64;
    let mut var_m = 0.0_f64;
    for (&r, &m) in reference_erosion.iter().zip(model_erosion.iter()).take(n) {
        let dr = r - mean_r;
        let dm = scale * m - mean_m;
        cov += dr * dm;
        var_r += dr * dr;
        var_m += dm * dm;
    }

    let rmse = (sum_sq * inv_n).sqrt();
    let reference_range = r_max - r_min;
    Some(CloudErosionValidation {
        model_scale: scale,
        rmse,
        normalized_rmse: if reference_range > 0.0 {
            rmse / reference_range
        } else {
            0.0
        },
        max_abs_error,
        max_relative_error,
        pearson_r: if var_r > 0.0 && var_m > 0.0 {
            cov / (var_r * var_m).sqrt()
        } else {
            0.0
        },
        sample_count: count,
    })
}

/// Result of boiling-histotripsy lesion sizing from a resolved pressure profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoilingLesionPlan {
    /// Number of boiling pulses required to reach the requested coverage.
    pub pulses: usize,
    /// Lateral lesion semi-axis `m`.
    pub lateral_radius_m: f64,
    /// Axial lesion semi-axis `m`.
    pub axial_radius_m: f64,
    /// Single-pulse duration `ms`.
    pub pulse_ms: f64,
}

/// Size a boiling-histotripsy lesion from pressure samples generated by the
/// active transmit model.
///
/// `radius_m` and `normalized_pressure` are paired radial samples in the focal
/// transverse plane. The shock heat source, boiling time, conformal clearance
/// clipping, and per-spot pulse count are computed here so Python callers do not
/// own therapy-domain logic.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn boiling_lesion_from_pressure_profile(
    radius_m: &[f64],
    normalized_pressure: &[f64],
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
    tau_max_s: f64,
    axial_to_lateral_ratio: f64,
    clearance_m: f64,
    coverage_target: f64,
) -> Option<BoilingLesionPlan> {
    let n = radius_m.len().min(normalized_pressure.len());
    if n < 2
        || focal_pressure_pa <= 0.0
        || focal_depth_m <= 0.0
        || freq_hz <= 0.0
        || c_m_s <= 0.0
        || rho_kg_m3 <= 0.0
        || alpha_np_m < 0.0
        || heat_capacity_j_kg_k <= 0.0
        || delta_t_k <= 0.0
        || tau_max_s <= 0.0
        || clearance_m <= 0.0
        || !(0.0..1.0).contains(&coverage_target)
    {
        return None;
    }

    let z_shock = shock_formation_distance(
        focal_pressure_pa,
        freq_hz,
        c_m_s,
        rho_kg_m3,
        beta_nonlinearity,
    )
    .max(1.0e-12);
    let mut p_local = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    for &b in normalized_pressure.iter().take(n) {
        let bn = b.max(0.0);
        p_local.push(focal_pressure_pa * bn);
        sigma.push((focal_depth_m / z_shock) * bn);
    }
    let q_heat = shock_heat_source_density(&p_local, &sigma, alpha_np_m, rho_kg_m3, c_m_s);
    let mut natural_radius = 0.0_f64;
    for i in 0..n {
        if q_heat[i] <= 0.0 {
            continue;
        }
        let t_boil = rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[i];
        if t_boil <= tau_max_s {
            natural_radius = natural_radius.max(radius_m[i]);
        }
    }
    if natural_radius <= 0.0 {
        return None;
    }
    let lateral =
        clipped_lateral_radius_for_clearance(natural_radius, clearance_m, axial_to_lateral_ratio);
    if lateral <= 0.0 {
        return None;
    }
    let pulse_s = boiling_time_at_radius(
        radius_m,
        &q_heat,
        lateral,
        rho_kg_m3,
        heat_capacity_j_kg_k,
        delta_t_k,
    )
    .min(tau_max_s);
    let fraction_per_pulse = (0.10 + 0.25 * pulse_s / tau_max_s).clamp(0.05, 0.40);
    let pulses = ((1.0 - coverage_target).ln() / (1.0 - fraction_per_pulse).ln()).ceil();
    Some(BoilingLesionPlan {
        pulses: pulses.max(1.0) as usize,
        lateral_radius_m: lateral,
        axial_radius_m: lateral * axial_to_lateral_ratio.max(1.0),
        pulse_ms: pulse_s * 1.0e3,
    })
}

/// Boiling-onset time profile from normalized pressure samples.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn boiling_time_profile_from_pressure(
    normalized_pressure: &[f64],
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> Vec<f64> {
    if focal_pressure_pa <= 0.0
        || focal_depth_m <= 0.0
        || freq_hz <= 0.0
        || c_m_s <= 0.0
        || rho_kg_m3 <= 0.0
        || heat_capacity_j_kg_k <= 0.0
        || delta_t_k <= 0.0
    {
        return vec![f64::INFINITY; normalized_pressure.len()];
    }
    let z_shock = shock_formation_distance(
        focal_pressure_pa,
        freq_hz,
        c_m_s,
        rho_kg_m3,
        beta_nonlinearity,
    )
    .max(1.0e-12);
    let mut p_local = Vec::with_capacity(normalized_pressure.len());
    let mut sigma = Vec::with_capacity(normalized_pressure.len());
    for &b in normalized_pressure {
        let bn = b.max(0.0);
        p_local.push(focal_pressure_pa * bn);
        sigma.push((focal_depth_m / z_shock) * bn);
    }
    shock_heat_source_density(&p_local, &sigma, alpha_np_m, rho_kg_m3, c_m_s)
        .into_iter()
        .map(|q| {
            if q > 0.0 {
                rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

fn boiling_time_at_radius(
    radius_m: &[f64],
    q_heat: &[f64],
    target_radius_m: f64,
    rho_kg_m3: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> f64 {
    let n = radius_m.len().min(q_heat.len());
    if n == 0 {
        return f64::INFINITY;
    }
    if target_radius_m <= radius_m[0] {
        return rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[0].max(1.0e-300);
    }
    for i in 0..n - 1 {
        if radius_m[i] <= target_radius_m && target_radius_m <= radius_m[i + 1] {
            let denom = (radius_m[i + 1] - radius_m[i]).max(1.0e-300);
            let w = (target_radius_m - radius_m[i]) / denom;
            let qi = q_heat[i] * (1.0 - w) + q_heat[i + 1] * w;
            return rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / qi.max(1.0e-300);
        }
    }
    rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[n - 1].max(1.0e-300)
}

/// Propagate a cavitation source PSD to every passive receiver channel.
///
/// The PSD scaling uses the squared magnitude of the attenuating acoustic
/// Green function, `exp(-2 alpha r)/(4 pi r)^2`, per receiver. Output is
/// flattened row-major `(n_receivers, n_freq)`.
#[must_use]
pub fn receiver_channel_psd_from_source(
    source_psd: &[f64],
    source_xyz: [f64; 3],
    receiver_xyz: &[f64],
    alpha_np_m: f64,
) -> Vec<f64> {
    if !receiver_xyz.len().is_multiple_of(3) {
        return Vec::new();
    }
    let n_recv = receiver_xyz.len() / 3;
    let mut out = vec![0.0; n_recv * source_psd.len()];
    for ir in 0..n_recv {
        let j = 3 * ir;
        let dx = receiver_xyz[j] - source_xyz[0];
        let dy = receiver_xyz[j + 1] - source_xyz[1];
        let dz = receiver_xyz[j + 2] - source_xyz[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-9);
        let amp = (-alpha_np_m.max(0.0) * r).exp() / (4.0 * std::f64::consts::PI * r);
        let psd_gain = amp * amp;
        for (ifreq, &v) in source_psd.iter().enumerate() {
            out[ir * source_psd.len() + ifreq] = v.max(0.0) * psd_gain;
        }
    }
    out
}

/// Sum receiver-channel PSDs into a measured array spectrum.
#[must_use]
pub fn integrate_channel_psd(channel_psd: &[f64], n_receivers: usize, n_freq: usize) -> Vec<f64> {
    if channel_psd.len() != n_receivers.saturating_mul(n_freq) {
        return Vec::new();
    }
    let mut out = vec![0.0; n_freq];
    for ir in 0..n_receivers {
        for ifreq in 0..n_freq {
            out[ifreq] += channel_psd[ir * n_freq + ifreq].max(0.0);
        }
    }
    out
}

/// Backscatter coefficient of partially fractionated tissue (lesion B-mode).
///
/// Histotripsy mechanically homogenizes tissue: the sub-resolution acoustic
/// scatterers (cell nuclei, collagen fibres) that produce B-mode speckle are
/// progressively destroyed as the fractionation fraction `f ∈ [0, 1]` rises, so
/// the (incoherent) backscatter coefficient falls from the intact-tissue value
/// `σ_intact` toward the near-anechoic liquefied-homogenate value
/// `σ_liquefied`:
/// ```text
/// σ_bsc(f) = σ_liquefied + (σ_intact − σ_liquefied)·(1 − f)^γ
/// ```
/// The exponent `γ ≥ 1` controls how fast coherent scatterer structure is lost;
/// `γ = 2` matches the quadratic backscatter–scatterer-density scaling for
/// progressive homogenization. This is why a completed histotripsy lesion reads
/// **hypoechoic** on post-treatment B-mode while the surrounding tissue keeps
/// full speckle. `f` is clamped to [0, 1].
///
/// # Reference
/// Wang et al. (2018), *Ultrasound Med. Biol.* 44, 2466 (lesion echogenicity);
/// Insana et al. (1990), *J. Acoust. Soc. Am.* 87, 179 (backscatter ∝ scatterer
/// number density).
#[must_use]
pub fn fractionation_backscatter_coefficient(
    fractionation: &[f64],
    sigma_intact: f64,
    sigma_liquefied: f64,
    gamma: f64,
) -> Vec<f64> {
    let g = gamma.max(1.0);
    fractionation
        .iter()
        .map(|&f| {
            let f = f.clamp(0.0, 1.0);
            sigma_liquefied + (sigma_intact - sigma_liquefied) * (1.0 - f).powf(g)
        })
        .collect()
}

/// Acoustic impedance of partially fractionated tissue (lesion-rim echo).
///
/// As tissue liquefies its specific acoustic impedance `Z = ρc` migrates from
/// the intact value `z_intact` toward the water-like homogenate value
/// `z_liquefied` by linear volume mixing:
/// ```text
/// Z(f) = z_intact·(1 − f) + z_liquefied·f
/// ```
/// The spatial gradient of this map produces the **specular bright rim** seen at
/// the boundary of a histotripsy lesion (impedance mismatch between liquefied
/// core and intact rim). `f` is clamped to [0, 1].
///
/// # Reference
/// Bamber (1986), *Physical Principles of Medical Ultrasonics* (impedance
/// mixing); histotripsy lesion-boundary echogenicity (Wang et al. 2018).
#[must_use]
pub fn fractionation_acoustic_impedance(
    fractionation: &[f64],
    z_intact: f64,
    z_liquefied: f64,
) -> Vec<f64> {
    fractionation
        .iter()
        .map(|&f| {
            let f = f.clamp(0.0, 1.0);
            z_intact.mul_add(1.0 - f, z_liquefied * f)
        })
        .collect()
}

#[cfg(test)]
#[path = "therapy_delivery_tests.rs"]
mod tests;
