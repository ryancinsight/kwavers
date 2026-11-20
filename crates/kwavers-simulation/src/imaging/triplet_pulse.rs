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
//! 6. The cloud trace is on the receiver's clock and EXTENDED to cover
//!    the travel time: the full 12-cycle history is radiated and the
//!    steady-state window is taken afterwards, so each bubble's window
//!    occupies `[start + delay, end + delay)` and samples before the
//!    first arrival are exactly zero. Windowing before the delay
//!    returns all-zero traces for any receiver beyond ~3.4 mm (D1).
//! 7. The cloud path is thermal-off by default, for parity with the
//!    prototype's adiabatic KM solve; thermal-on is available via
//!    `TripletPulseCloud::with_thermal` and its ratios are recorded
//!    as observed values, never as parity targets (see "Thermal model"
//!    below).
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
//!
//! # Thermal model
//!
//! Binding default: thermal-OFF — matching the prototype's adiabatic
//! KM solve; the parity tests run thermal-off. Thermal-on is available
//! via `TripletPulseCloud::with_thermal(true)` and via the `thermal`
//! flag of the ratio helper, and its ratios are recorded below as
//! OBSERVED values, never as parity targets: the literature is silent
//! on thermal effects on the 1.5th fractional harmonic specifically
//! (but see the Sojahrood benchmark scoping below, which gives a
//! mechanism-level check the port has not yet run).
//!
//! Observed thermal-on 3P/1P ratios (0.26 MPa, flat-top drive, cycles
//! 6–10), re-measured Nov 19, 2026 after the KW-PHYS-003 latent-heat
//! fix: at 1.5f0, **0.248 / 0.302 / 0.143** for R0 = 2 / 3 / 5 µm
//! (thermal-off: 0.351 / 0.362 / 0.219); at f0, 0.136 / 0.034 / 0.001.
//! With the KW-PHYS-003 bug present the 1.5f0 values were 0.376 /
//! 0.358 / 0.142 — the fix moved R0 = 2 and 3 µm toward thermal-off
//! and left R0 = 5 µm unchanged, because the wrong-signed latent term
//! had added energy during collapse, inflating the fractional harmonic
//! where the collapse is most violent (near-resonant R0 = 2–3 µm);
//! the remaining gap to thermal-off is the genuine thermal-damping
//! effect, strongest at resonance.
//!
//! Accommodation sensitivity (D12(i)) — thermal-on 3P/1P at 1.5f0,
//! α = 0.04 (the default, the 1935 Alty value) vs α = 0.35 (the
//! mainstream value: Yasui, Storey & Szeri, Kreider, Hegedüs):
//!
//! | R0 | α = 0.04 | α = 0.35 |
//! |----|----------|----------|
//! | 2 µm | 0.248 | 0.640 |
//! | 3 µm | 0.302 | 0.300 |
//! | 5 µm | 0.143 | 0.167 |
//!
//! At f0 (R0 = 2 µm): 0.136 → 0.135. The near-resonant R0 = 2 µm
//! ratio is strongly α-dependent (0.248 → 0.640): how fast vapour
//! equilibrates decides how much fractional-harmonic content a
//! resonant bubble radiates. α = 0.04 stays the default (unchanged by
//! this work — a parameter choice, not a demonstrable error), and any
//! thermal-on use must state its α alongside its ratios.
//!
//! Sojahrood et al. 2021 benchmark — SCOPING ONLY, not run (D12(iii)).
//! Sojahrood, Haghi, Shirazi, Karshafian & Kolios, "On the threshold
//! of 1/2 order subharmonic emissions in the oscillations of
//! ultrasonically excited bubbles", Ultrasonics 112, 106363 (2021):
//! with full thermal damping, the pressure threshold for 1/2-order
//! subharmonic (period-doubling, P2) emissions has its minimum at
//! 0.5–0.6·f_r rather than the conventional 2·f_r, via 5/2-order
//! ultraharmonic resonance, and only bubbles with R0 ≥ 0.5 µm (≥1 µm
//! diameter) sustain stable SH emission there. The test the port
//! would need: for an air bubble in water, R0 ∈ {0.5, 1, 2, 5, 10} µm,
//! map the P2 threshold pressure vs f/f_r over 0.3–2.0, thermal-on vs
//! thermal-off — P2 detected by a Poincaré section (radius sampled
//! once per drive period; P2 = alternating values) with the 1/2-order
//! subharmonic confirmed in the radiated spectrum — and check (a) the
//! threshold minimum moves from ~2·f_r (thermal-off) toward
//! 0.5–0.6·f_r (thermal-on), and (b) R0 < 0.5 µm shows no stable SH
//! emission at 0.5–0.6·f_r. That map is out of reach in this session:
//! each threshold point needs a pressure sweep of multi-cycle KM
//! solves to steady state (hundreds of solves across radii ×
//! frequencies × thermal settings), plus a P2 detector that does not
//! exist in the port. Until it is run, thermal-on 1.5f0 ratios are
//! unvalidated against any benchmark and must not be quoted as
//! predictions of the experiment; thermal-off ratios are
//! code-consistency checks against the adiabatic prototype only. Note
//! the port's operating point sits where Sojahrood says thermal
//! damping matters most: at 1.74 MHz, R0 = 2 µm is near resonance
//! (Minnaert f_r ≈ 1.6 MHz, f/f_r ≈ 1.1).

use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::{KwaversError, KwaversResult};
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
/// The output has `out_len` samples: pass `source.len()` for the
/// source's own clock, or a longer length when the arrival window
/// extends past the source — the cloud path (`simulate_phase`) passes
/// an extended length because the ≥12 µs travel time exceeds the
/// 2.3 µs steady-state window (D1/C1: with `out_len == source.len()`
/// the whole windowed signal fell past the end of the trace and every
/// receiver beyond ~3.4 mm heard silence).
#[must_use]
pub fn radiate(
    source: &[f64],
    distance_m: f64,
    density: f64,
    sound_speed: f64,
    dt: f64,
    out_len: usize,
) -> Vec<f64> {
    let delay_s = distance_m / sound_speed;
    let delay_samples = delay_s / dt;
    let whole = delay_samples.floor() as usize;
    let frac = delay_samples - whole as f64;
    let scale = density / (4.0 * std::f64::consts::PI * distance_m);
    (0..out_len)
        .map(|i| {
            if i < whole {
                0.0
            } else {
                let j = i - whole;
                if j >= source.len() {
                    0.0
                } else {
                    let s0 = source[j];
                    let s1 = if j + 1 < source.len() {
                        source[j + 1]
                    } else {
                        0.0
                    };
                    scale * (s0 * (1.0 - frac) + s1 * frac)
                }
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
    /// Thermal/mass-transfer path for the cloud bubbles: `false`
    /// (default) runs thermal-off, matching the prototype's adiabatic
    /// KM solve and the parity tests; `true` runs the native
    /// thermal-on path (see the module docs — thermal-on ratios are
    /// observed values, not parity targets). Set with
    /// [`TripletPulseCloud::with_thermal`].
    pub thermal: bool,
}

impl TripletPulseCloud {
    /// New empty cloud for the Iwasaki 2018 configuration.
    ///
    /// Thermal-off by default, for parity with the prototype's
    /// adiabatic KM solve; use [`TripletPulseCloud::with_thermal`] for
    /// the native thermal-on path.
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
            thermal: false,
        }
    }

    /// Run the cloud bubbles with the native thermal/mass-transfer
    /// path (`true`) or thermal-off (`false`, the default — parity
    /// with the prototype's adiabatic KM).
    #[must_use]
    pub fn with_thermal(mut self, thermal: bool) -> Self {
        self.thermal = thermal;
        self
    }

    /// Add a bubble at `position_m` with equilibrium radius `r0_m`.
    pub fn add_bubble(&mut self, position_m: [f64; 3], r0_m: f64) {
        self.bubbles.push((position_m, r0_m));
    }

    /// Water 20 °C parameters for one bubble (matches the prototype's
    /// KM constants: rho 998, sigma 0.0725, gamma 1.4, mu 1e-3,
    /// pv 2.34e3, c_l 1481, P0 1.013e5). Thermal/mass transfer follow
    /// `self.thermal` (off by default, for parity with the prototype's
    /// adiabatic KM — the old code left the `BubbleParameters`
    /// defaults on, a second silent inconsistency in this path).
    #[must_use]
    pub fn bubble_params(&self, r0_m: f64) -> BubbleParameters {
        let mut params = BubbleParameters::default();
        params.r0 = r0_m;
        params.driving_frequency = self.config.frequency_hz;
        params.driving_amplitude = self.drive.amplitude_pa;
        params.use_thermal_effects = self.thermal;
        params.use_mass_transfer = self.thermal;
        params
    }

    /// Trace at one receiver for one transmit phase: the summed
    /// monopole re-radiation of every bubble's steady-state window.
    ///
    /// The trace is on the receiver's clock (index `i` ↔ time `i·dt`)
    /// and is EXTENDED to cover the travel time: the steady-state
    /// window (cycles 6–10, ~2.3 µs) arrives `d/c` after it is
    /// radiated, and at the prototype geometry (`d` ≥ 18 mm) that
    /// delay (≥12.15 µs) is five times the window. The old code
    /// windowed the source BEFORE applying the delay, so the whole
    /// signal fell past the end of the 2000-sample trace and every
    /// receiver beyond ~3.4 mm heard silence (D1/C1). The fix
    /// radiates the FULL 12-cycle history into an extended trace and
    /// windows afterwards: each bubble's window occupies
    /// `[start + delay, end + delay)` on the receiver's clock, and
    /// samples before the first arrival are exactly zero (causality).
    /// All four phases return the same length (set by the latest
    /// window-end arrival over the cloud), so `combine_echoes` still
    /// applies.
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
        let distance_to = |position: &[f64; 3]| {
            let dx = receiver_m[0] - position[0];
            let dy = receiver_m[1] - position[1];
            let dz = receiver_m[2] - position[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        // Extended output: long enough for the latest window-end
        // arrival over all bubbles.
        let max_delay_samples = self
            .bubbles
            .iter()
            .map(|(position, _)| distance_to(position) / self.sound_speed / dt)
            .fold(0.0_f64, f64::max);
        let n_total = end + max_delay_samples.ceil() as usize + 1;
        let mut trace = vec![0.0; n_total];
        for (position, r0) in &self.bubbles {
            let params = self.bubble_params(*r0);
            let history = solve_radius_waveform(&params, &self.drive, phase_deg, steps_per_cycle)?;
            let source = monopole_source(&history);
            let distance = distance_to(position);
            let radiated = radiate(
                &source,
                distance,
                self.density,
                self.sound_speed,
                dt,
                n_total,
            );
            // Window afterwards: keep only the steady-state window's
            // arrival, [start + delay, end + delay). The two `ceil`s
            // differ by exactly `end - start` samples (the window
            // length), since `end - start` is an integer.
            let delay_samples = distance / self.sound_speed / dt;
            let lo = (start as f64 + delay_samples).ceil() as usize;
            let hi = (end as f64 + delay_samples).ceil() as usize;
            for (acc, val) in trace[lo..hi].iter_mut().zip(radiated[lo..hi].iter()) {
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

/// Goertzel magnitude of `signal` at `freq_hz`, sampled at `dt`.
/// Single-frequency DFT — O(n) per frequency, no FFT dependency.
fn goertzel_magnitude(signal: &[f64], freq_hz: f64, dt: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * freq_hz * dt;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in signal {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let re = s1 - s2 * omega.cos();
    let im = s2 * omega.sin();
    (re * re + im * im).sqrt()
}

/// 3P/1P magnitude ratio at `analysis_freq_hz` between the coherent 3P sum
/// and the 1P trace, for one bubble's monopole source.
///
/// The bubble is driven by a 12-cycle flat-top tone burst at
/// `drive_frequency_hz` and amplitude `amplitude_pa`; the ratio is measured
/// on the steady-state window (cycles 6–10) of the monopole source
/// `ρ/(4πr) · d²V/dt²` at r = 1 mm. `thermal` selects the native model's
/// thermal/mass-transfer path: `false` turns both off, matching the
/// prototype's simple adiabatic KM solve; `true` leaves the defaults on.
/// `accommodation_coeff` is the Hertz–Knudsen accommodation coefficient α
/// (default 0.04; the mainstream value is 0.35 — see the
/// accommodation-sensitivity test, D12(i)).
///
/// Parity targets (thermal-off, 0.26 MPa, 1.74 MHz drive, 500 steps/cycle):
/// at 1.5f0 (2.61 MHz) **0.35 / 0.36 / 0.22** for R0 = 2 / 3 / 5 µm
/// (tolerance ±0.05); at f0 the ratio is < 0.05 (prototype: 0.011 at
/// R0 = 5 µm). Thermal-on ratios are OBSERVED, not targets: 1.5f0 =
/// 0.248 / 0.302 / 0.143, f0 = 0.136 / 0.034 / 0.001 (re-measured Nov 19,
/// 2026 after the KW-PHYS-003 latent-heat fix).
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when the radius, either
/// frequency, or the steps-per-cycle count is not positive, or the
/// amplitude is not finite; propagates KM integration failures.
pub fn three_p_over_one_p_ratio(
    r0_m: f64,
    amplitude_pa: PressureAmplitude,
    drive_frequency_hz: f64,
    analysis_freq_hz: f64,
    steps_per_cycle: usize,
    thermal: bool,
    accommodation_coeff: f64,
) -> KwaversResult<f64> {
    if !r0_m.is_finite() || r0_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: r0_m must be finite and > 0; got {r0_m}"
        )));
    }
    if !drive_frequency_hz.is_finite() || drive_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: drive_frequency_hz must be finite and > 0; got {drive_frequency_hz}"
        )));
    }
    if !analysis_freq_hz.is_finite() || analysis_freq_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: analysis_freq_hz must be finite and > 0; got {analysis_freq_hz}"
        )));
    }
    if steps_per_cycle == 0 {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: steps_per_cycle must be > 0".to_owned(),
        ));
    }
    if !amplitude_pa.as_pascals().is_finite() || amplitude_pa.as_pascals() < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: amplitude must be finite and >= 0; got {}",
            amplitude_pa.as_pascals()
        )));
    }
    let drive = TripletPulseDrive {
        frequency_hz: drive_frequency_hz,
        amplitude_pa,
        solve_cycles: 12,
        steady_start_cycle: 6,
        steady_end_cycle: 10,
        segment_cycles: 6,
    };
    let mut params = BubbleParameters::default();
    params.r0 = r0_m;
    params.driving_frequency = drive_frequency_hz;
    params.driving_amplitude = amplitude_pa;
    params.use_thermal_effects = thermal;
    params.use_mass_transfer = thermal;
    params.accommodation_coeff = accommodation_coeff;
    let (start, end) = steady_state_range(&drive, steps_per_cycle);
    let dt = 1.0 / (drive.frequency_hz * steps_per_cycle as f64);
    let mut phases = Vec::new();
    for phase_deg in [0.0, 120.0, 240.0] {
        let history = solve_radius_waveform(&params, &drive, phase_deg, steps_per_cycle)?;
        phases.push(monopole_source(&history[start..end]));
    }
    let one_p = goertzel_magnitude(&phases[0], analysis_freq_hz, dt);
    let three_p = goertzel_magnitude(
        &{
            let n = phases[0].len();
            (0..n)
                .map(|i| (phases[0][i] + phases[1][i] + phases[2][i]) / 3.0)
                .collect::<Vec<f64>>()
        },
        analysis_freq_hz,
        dt,
    );
    Ok(three_p / one_p)
}

#[cfg(test)]
mod tests;
