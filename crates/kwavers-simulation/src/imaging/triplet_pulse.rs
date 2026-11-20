//! Triplet-pulse bubble-cloud coupling (simulation orchestrator).
//!
//! Connects the pieces the prototype validated in Python (Stages 0–3,
//! `Triplet_Pulse_Prototype_Validation.md`) into native Rust:
//! a Keller–Miksis bubble driven by the phase-shifted imaging pulse,
//! monopole re-radiation to a receiver, and the 1P / PI / 3P coherent
//! sums from `kwavers-imaging`.
//!
//! The imaging pulse here is the *diagnostic* triplet pulse (Iwasaki
//! 2018: 1.74 MHz; Kuji 2024: 2.45 MHz), not the therapeutic trigger
//! pulse that creates the bubbles. The cloud is assumed to exist —
//! nucleated by the trigger — and this module answers what the imaging
//! pulse sees. Bubble *formation* (nucleation) is a separate model; see
//! the prototype's Stage 4.
//!
//! # Prototype invariants (must hold; each has a test below)
//!
//! 1. The transmit phase shift wraps modulo the **segment** length
//!    (`segment_cycles`, 6 cycles), not one period. The KM steady
//!    state carries a subharmonic (period 2T); wrapping modulo one
//!    period flips the fractional harmonics — the first prototype
//!    version made the 1.5th harmonic CANCEL in 3P, the opposite of
//!    the physics.
//! 2. Radiation uses a **late steady-state segment** (cycles 6–10 of
//!    a 12-cycle solve). The drive is **flat-top** — Hann rise over
//!    cycles 0–2, flat at full amplitude over cycles 2–10, Hann fall
//!    over cycles 10–12 — so the analysis window (cycles 6–10) sits
//!    entirely in the flat region at constant amplitude. A Hann
//!    envelope over the full solve puts the window on the falling
//!    half of the envelope and the response is never a periodic
//!    steady state (the prototype's Part 4 result). Cycles 0–6 carry
//!    startup transient plus an unsaturated subharmonic and break
//!    the phasor rule.
//! 3. The transmit **envelope stays unshifted** while the carrier
//!    shifts by the transmit phase.
//! 4. Bubble radius is always a sweep, never a single value: the
//!    papers give no bubble size. Prototype sweep: R0 ∈ {2, 3, 5} µm.
//! 5. All experimental values are chicken-breast research values, not
//!    Suizenji clinical settings.
//!
//! # Monopole radiation
//!
//! A bubble re-radiates as a monopole: the pressure at distance `d`
//! is `p = ρ / (4πd) · d²V/dt²` evaluated at the retarded time
//! `t − d/c`, with `V = 4πR³/3`, so
//! `d²V/dt² = 4π(2RṘ² + R²R̈)`. The `1/d` falloff and the travel-time
//! delay are both tested. Secondary Bjerknes (bubble–bubble)
//! coupling is out of scope here; the prototype's clouds are dilute
//! and the bubbles are driven independently.

use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::KwaversResult;
use kwavers_imaging::ultrasound::triplet_pulse::{
    combine_echoes, TripletPulseConfig, TripletPulseEchoes,
};
use kwavers_physics::acoustics::bubble_dynamics::{
    acoustic_pressure::{InstantaneousPressure, PressureAmplitude},
    adaptive_integration::integrate_bubble_dynamics_adaptive,
    bubble_state::{BubbleParameters, BubbleState},
    keller_miksis::KellerMiksisModel,
};

/// One radius sample of a solved bubble oscillation.
#[derive(Debug, Clone, Copy)]
pub struct RadiusSample {
    /// Bubble radius [m].
    pub radius: f64,
    /// Wall velocity dR/dt [m/s].
    pub wall_velocity: f64,
    /// Wall acceleration d²R/dt² [m/s²].
    pub wall_acceleration: f64,
}

/// The phase-shifted imaging drive for one transmission.
///
/// The drive is a flat-top tone burst: Hann rise over cycles 0–2,
/// flat at full amplitude over cycles 2–10, Hann fall over cycles
/// 10–12. The carrier `sin(2πft + φ)` shifts by the transmit phase
/// `φ`, the envelope does not. The solve runs `solve_cycles` cycles;
/// the analysis window is the flat region (cycles
/// `steady_start_cycle..steady_end_cycle`, prototype: 6–10).
#[derive(Debug, Clone)]
pub struct TripletPulseDrive {
    /// Imaging fundamental frequency [Hz].
    pub frequency_hz: f64,
    /// Drive pressure amplitude at the cloud — the slab PNP.
    pub amplitude_pa: PressureAmplitude,
    /// Cycles in the KM solve (prototype: 12).
    pub solve_cycles: usize,
    /// First cycle of the steady-state window (prototype: 6).
    pub steady_start_cycle: usize,
    /// Last cycle of the steady-state window (prototype: 10) — the
    /// end of the flat-top region.
    pub steady_end_cycle: usize,
    /// Cycles in one imaging segment (prototype: 6) — the phase
    /// wraps modulo this length, not one period (invariant 1).
    pub segment_cycles: usize,
}

impl TripletPulseDrive {
    /// Drive for the Iwasaki 2018 configuration at the given slab PNP.
    #[must_use]
    pub fn iwasaki_2018(amplitude_pa: PressureAmplitude) -> Self {
        Self {
            frequency_hz: 1.74e6,
            amplitude_pa,
            solve_cycles: 12,
            steady_start_cycle: 6,
            steady_end_cycle: 10,
            segment_cycles: 6,
        }
    }

    /// Solve duration [s].
    #[must_use]
    pub fn solve_duration_s(&self) -> f64 {
        self.solve_cycles as f64 / self.frequency_hz
    }

    /// Flat-top envelope over the solve window — unshifted by transmit
    /// phase. Hann rise over cycles 0–2, flat (1.0) over cycles 2–10,
    /// Hann fall over cycles 10–12; zero outside the solve.
    ///
    /// Flat-top is the standard steady-state practice in
    /// bubble-oscillation studies: a constant-amplitude drive with
    /// the initial transient discarded (dual-frequency KM studies
    /// discard the first 8 of 30 cycles; de Jong 2003: "driven by a
    /// constant amplitude sinusoidal excitation … transient … builds
    /// up to constant amplitude"). A Hann envelope over the full
    /// solve puts the analysis window on falling amplitude and the
    /// response is never a periodic steady state.
    #[must_use]
    pub fn envelope(&self, t: f64) -> f64 {
        let cyc = t * self.frequency_hz;
        if cyc < 0.0 || cyc > self.solve_cycles as f64 {
            0.0
        } else if cyc < 2.0 {
            0.5 * (1.0 - (std::f64::consts::PI * cyc / 2.0).cos())
        } else if cyc <= self.steady_end_cycle as f64 {
            1.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * (cyc - self.steady_end_cycle as f64) / 2.0).cos())
        }
    }

    /// Time derivative of the flat-top envelope.
    #[must_use]
    pub fn envelope_derivative(&self, t: f64) -> f64 {
        let cyc = t * self.frequency_hz;
        if cyc < 0.0 || cyc > self.solve_cycles as f64 {
            0.0
        } else if cyc < 2.0 {
            (std::f64::consts::PI * self.frequency_hz / 4.0)
                * (std::f64::consts::PI * cyc / 2.0).sin()
        } else if cyc <= self.steady_end_cycle as f64 {
            0.0
        } else {
            -(std::f64::consts::PI * self.frequency_hz / 4.0)
                * (std::f64::consts::PI * (cyc - self.steady_end_cycle as f64) / 2.0).sin()
        }
    }

    /// Drive pressure at time `t` for transmit phase `phase_deg`.
    ///
    /// The phase is applied to the carrier only. It wraps modulo the
    /// segment (`segment_cycles` cycles): a phase of `φ + 360°` is
    /// identical to `φ`, and the wrap length is the segment, never
    /// one period — see invariant 1.
    ///
    /// The envelope multiplies the tone, so the value is constructed
    /// at this waveform-definition boundary — the typed equivalent of
    /// `PressureAmplitude::sample` for an enveloped drive.
    #[must_use]
    pub fn pressure(&self, t: f64, phase_deg: f64) -> InstantaneousPressure {
        let phase = phase_deg.to_radians();
        let carrier = (2.0 * std::f64::consts::PI * self.frequency_hz * t + phase).sin();
        InstantaneousPressure::from_pascals(
            self.amplitude_pa.as_pascals() * self.envelope(t) * carrier,
        )
    }

    /// Time derivative of the drive pressure, for the KM integrator.
    #[must_use]
    pub fn pressure_derivative(&self, t: f64, phase_deg: f64) -> PressureRate {
        let phase = phase_deg.to_radians();
        let omega = 2.0 * std::f64::consts::PI * self.frequency_hz;
        let (sin, cos) = (omega * t + phase).sin_cos();
        PressureRate::from_base(
            self.amplitude_pa.as_pascals()
                * (self.envelope_derivative(t) * sin + self.envelope(t) * omega * cos),
        )
    }
}

/// Solve one bubble's radius history under the drive.
///
/// Runs `solve_cycles` at `steps_per_cycle` steps per cycle through
/// the native Keller–Miksis model (`KellerMiksisModel` +
/// `integrate_bubble_dynamics_adaptive`), one fixed step at a time
/// with the analytic drive pressure and its derivative.
///
/// Note: the native model also tracks thermodynamics, mass transfer
/// and shape modes — richer than the Python binding's KM solve the
/// prototype used. The thermal path is usable for multi-cycle drives
/// since the KW-PHYS-002 fix (Nov 18, 2026): with `p_sat` evaluated
/// at the liquid wall temperature (Kreider et al. 2011, JASA 130:3511,
/// Eq. 15) the bubble stays periodic instead of growing ~+14 R0 per
/// cycle. The parity tests still run thermal-off, matching the
/// prototype's adiabatic KM; thermal-on ratios are covered by the
/// boundedness test `bubble_only_3p_1p_ratio_thermal_on_bounded`.
///
/// # Errors
///
/// Propagates integration failures from the KM solver.
pub fn solve_radius_waveform(
    params: &BubbleParameters,
    drive: &TripletPulseDrive,
    phase_deg: f64,
    steps_per_cycle: usize,
) -> KwaversResult<Vec<RadiusSample>> {
    let model = KellerMiksisModel::new(params.clone());
    let mut state = BubbleState::new(params);
    let dt = 1.0 / (drive.frequency_hz * steps_per_cycle as f64);
    let n_steps = drive.solve_cycles * steps_per_cycle;
    let mut history = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let t = i as f64 * dt;
        history.push(RadiusSample {
            radius: state.radius,
            wall_velocity: state.wall_velocity,
            wall_acceleration: state.wall_acceleration,
        });
        if i == n_steps {
            break;
        }
        let p = drive.pressure(t, phase_deg);
        let dp_dt = drive.pressure_derivative(t, phase_deg);
        integrate_bubble_dynamics_adaptive(&model, &mut state, p, dp_dt, dt, t)?;
    }
    Ok(history)
}

/// The late steady-state window of a solved waveform: cycles
/// `steady_start_cycle..steady_end_cycle` (prototype: cycles 6–10,
/// the flat region of the flat-top drive).
/// Returns the sample range `[start, end)` into `history`.
#[must_use]
pub fn steady_state_range(drive: &TripletPulseDrive, steps_per_cycle: usize) -> (usize, usize) {
    (
        drive.steady_start_cycle * steps_per_cycle,
        drive.steady_end_cycle * steps_per_cycle,
    )
}

/// Monopole source strength `d²V/dt²` [m³/s²] per sample.
#[must_use]
pub fn monopole_source(samples: &[RadiusSample]) -> Vec<f64> {
    samples
        .iter()
        .map(|s| {
            4.0 * std::f64::consts::PI
                * (2.0 * s.radius * s.wall_velocity * s.wall_velocity
                    + s.radius * s.radius * s.wall_acceleration)
        })
        .collect()
}

/// Radiate a monopole source to a receiver at `distance_m`.
///
/// `p = ρ / (4πd) · source(t − d/c)`, with the retarded time
/// applied as an integer-plus-fractional sample delay (linear
/// interpolation). Samples before the first arrival are zero.
#[must_use]
pub fn radiate(
    source: &[f64],
    distance_m: f64,
    density: f64,
    sound_speed: f64,
    dt: f64,
) -> Vec<f64> {
    let delay_s = distance_m / sound_speed;
    let delay_samples = delay_s / dt;
    let whole = delay_samples.floor() as usize;
    let frac = delay_samples - whole as f64;
    let scale = density / (4.0 * std::f64::consts::PI * distance_m);
    (0..source.len())
        .map(|i| {
            if i < whole {
                0.0
            } else {
                let j = i - whole;
                let s0 = source[j];
                let s1 = if j + 1 < source.len() {
                    source[j + 1]
                } else {
                    0.0
                };
                scale * (s0 * (1.0 - frac) + s1 * frac)
            }
        })
        .collect()
}

/// A bubble cloud coupled to the triplet-pulse imaging sequence.
///
/// Bubbles are driven independently (no Bjerknes coupling — the
/// prototype's clouds are dilute) and their re-radiation sums
/// linearly at each receiver by superposition.
#[derive(Debug)]
pub struct TripletPulseCloud {
    /// Imaging configuration (frequency, phases).
    pub config: TripletPulseConfig,
    /// Imaging drive (amplitude = slab PNP).
    pub drive: TripletPulseDrive,
    /// Liquid density [kg/m³] (water 20 °C: 998).
    pub density: f64,
    /// Sound speed in the liquid [m/s] (water 20 °C: 1481).
    pub sound_speed: f64,
    /// Bubbles: (position [m; 3], equilibrium radius [m]).
    pub bubbles: Vec<([f64; 3], f64)>,
}

impl TripletPulseCloud {
    /// New empty cloud for the Iwasaki 2018 configuration.
    #[must_use]
    pub fn iwasaki_2018(amplitude_pa: PressureAmplitude) -> Self {
        let config = TripletPulseConfig::iwasaki_2018();
        let drive = TripletPulseDrive::iwasaki_2018(amplitude_pa);
        Self {
            config,
            drive,
            density: 998.0,
            sound_speed: 1481.0,
            bubbles: Vec::new(),
        }
    }

    /// Add a bubble at `position_m` with equilibrium radius `r0_m`.
    pub fn add_bubble(&mut self, position_m: [f64; 3], r0_m: f64) {
        self.bubbles.push((position_m, r0_m));
    }

    /// Water 20 °C parameters for one bubble (matches the prototype's
    /// KM constants: rho 998, sigma 0.0725, gamma 1.4, mu 1e-3,
    /// pv 2.34e3, c_l 1481, P0 1.013e5).
    #[must_use]
    pub fn bubble_params(&self, r0_m: f64) -> BubbleParameters {
        let mut params = BubbleParameters::default();
        params.r0 = r0_m;
        params.driving_frequency = self.config.frequency_hz;
        params.driving_amplitude = self.drive.amplitude_pa;
        params
    }

    /// Trace at one receiver for one transmit phase: the summed
    /// monopole re-radiation of every bubble, over the steady-state
    /// window only (invariant 2).
    ///
    /// # Errors
    ///
    /// Propagates KM integration failures.
    pub fn simulate_phase(
        &self,
        receiver_m: [f64; 3],
        phase_deg: f64,
        steps_per_cycle: usize,
    ) -> KwaversResult<Vec<f64>> {
        let dt = 1.0 / (self.config.frequency_hz * steps_per_cycle as f64);
        let (start, end) = steady_state_range(&self.drive, steps_per_cycle);
        let n_out = end - start;
        let mut trace = vec![0.0; n_out];
        for (position, r0) in &self.bubbles {
            let params = self.bubble_params(*r0);
            let history = solve_radius_waveform(&params, &self.drive, phase_deg, steps_per_cycle)?;
            let source = monopole_source(&history[start..end]);
            let dx = receiver_m[0] - position[0];
            let dy = receiver_m[1] - position[1];
            let dz = receiver_m[2] - position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let radiated = radiate(&source, distance, self.density, self.sound_speed, dt);
            for (acc, val) in trace.iter_mut().zip(radiated.iter()) {
                *acc += val;
            }
        }
        Ok(trace)
    }

    /// 1P / PI / 3P echoes at one receiver, from the four phase
    /// transmissions (0°, 120°, 240°, 180°).
    ///
    /// # Errors
    ///
    /// Propagates KM integration failures.
    pub fn simulate_receiver(
        &self,
        receiver_m: [f64; 3],
        steps_per_cycle: usize,
    ) -> KwaversResult<TripletPulseEchoes> {
        let ph = &self.config.phases;
        let phase_0 = self.simulate_phase(receiver_m, 0.0, steps_per_cycle)?;
        let phase_120 = self.simulate_phase(receiver_m, ph.phase_120_deg, steps_per_cycle)?;
        let phase_240 = self.simulate_phase(receiver_m, ph.phase_240_deg, steps_per_cycle)?;
        let phase_180 = self.simulate_phase(receiver_m, ph.phase_180_deg, steps_per_cycle)?;
        Ok(combine_echoes(&phase_0, &phase_120, &phase_240, &phase_180))
    }
}

#[cfg(test)]
mod tests;
