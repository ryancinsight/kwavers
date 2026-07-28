//! Ultrasonic-neuromodulation pulse-train protocol and dosimetry.
//!
//! Encodes the multi-timescale stimulus parameter hierarchy used to specify
//! ultrasonic neuromodulation sequences (Blackmore et al. 2019, Fig. 1 & Table 1)
//! and derives the standard dosimetry/safety quantities. Three nested timescales:
//!
//! ```text
//!  inner  pulses    — pulse length PL at pulse-repetition frequency PRF
//!  middle bursts    — burst duration BD separated by burst interval BI
//!  outer  experiment— N bursts over total time TT
//! ```
//!
//! Derived quantities (Blackmore Table 1):
//! - burst duty cycle      BDC = PL · PRF
//! - burst-on fraction          = BD / (BD + BI)
//! - total duty cycle      TDC = BDC · BD / (BD + BI)
//! - burst-repetition freq BRF = 1 / (BD + BI)
//! - total time            TT  = N · (BD + BI)
//!
//! Intensities (spatial-peak), from the in-pulse peak intensity I_SPPA:
//! - burst-averaged   I_SPBA = I_SPPA · BDC
//! - temporal-averaged I_SPTA = I_SPPA · TDC
//!
//! where the in-pulse intensity of a sinusoidal carrier of peak pressure `p` in a
//! medium of impedance `Z = ρc` is `I_SPPA = p²/(2Z)`.
//!
//! A continuous-wave pulsed protocol (no nesting) is the special case `BD = TT`,
//! `BI = 0`, `N = 1`, for which `TDC = BDC = PL·PRF`.
//!
//! # Why dosimetry-only (not carrier-resolved)
//!
//! Neuromodulation protocols last seconds to minutes; resolving the 0.1–1 MHz
//! carrier over that span (≫ 10⁸ steps) is infeasible — the [`super::nice`]
//! carrier-resolved integration is for single bursts/pulses, and the
//! cycle-averaged SONIC reduction is the route to whole-protocol membrane
//! simulation. This type therefore provides the protocol *specification*,
//! dosimetry, safety screening, and a pulse-envelope gate
//! ([`PulseTrainProtocol::pulse_active`]).
//!
//! # References
//!
//! - Blackmore, J. et al. (2019). Ultrasound neuromodulation: a review of
//!   results, mechanisms and safety. *Ultrasound Med. Biol.* 45(7), 1509-1536
//!   (Fig. 1, Table 1, safety guidelines).
//! - Manuel, T.J. et al. (2020). Ultrasound neuromodulation depends on pulse
//!   repetition frequency. *Sci. Rep.* 10, 15347 (PRF dependence).
//! - Atkinson-Clement, C. et al. (2025). Delay- and pressure-dependent
//!   neuromodulatory effects of transcranial ultrasound stimulation.
//!   *Neuromodulation* 28, 444-454 (theta-burst protocol).

use crate::analytical::safety::{fda_isppa_limit_w_cm2, fda_ispta_limit_mw_cm2};
use aequitas::systems::si::quantities::{
    Frequency, Intensity, MassDensity, Pressure, TemperatureDifference, Time, Velocity,
};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use kwavers_core::constants::medical::MI_LIMIT_SOFT_TISSUE;

/// W/m² → W/cm² conversion (1 W/cm² = 10⁴ W/m²).
const W_M2_TO_W_CM2: f64 = 1.0e-4;
/// W/cm² → mW/cm² conversion.
const W_CM2_TO_MW_CM2: f64 = 1.0e3;

/// A nested ultrasonic-neuromodulation pulse-train protocol (Blackmore Fig. 1).
#[derive(Debug, Clone, Copy)]
pub struct PulseTrainProtocol {
    /// Carrier (fundamental) frequency f `Hz`.
    pub carrier_frequency: Frequency<f64>,
    /// Pulse length PL (a.k.a. pulse duration) `s`.
    pub pulse_length: Time<f64>,
    /// Pulse repetition frequency PRF `Hz`.
    pub pulse_repetition_frequency: Frequency<f64>,
    /// Burst duration BD `s` (window over which pulses repeat at PRF).
    pub burst_duration: Time<f64>,
    /// Burst interval BI `s` (off time between consecutive bursts).
    pub burst_interval: Time<f64>,
    /// Number of bursts N.
    pub num_bursts: u32,
}

impl PulseTrainProtocol {
    /// Returns `true` if the protocol parameters are physically consistent:
    /// positive carrier/PRF/PL, the pulse fits within its repetition period
    /// (`PL ≤ 1/PRF`), positive burst duration, non-negative interval, `N ≥ 1`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let carrier_frequency = self.carrier_frequency.into_base();
        let pulse_length = self.pulse_length.into_base();
        let pulse_repetition_frequency = self.pulse_repetition_frequency.into_base();
        let burst_duration = self.burst_duration.into_base();
        let burst_interval = self.burst_interval.into_base();
        carrier_frequency > 0.0
            && pulse_length > 0.0
            && pulse_repetition_frequency > 0.0
            && pulse_length <= 1.0 / pulse_repetition_frequency + 1e-15
            && burst_duration > 0.0
            && burst_interval >= 0.0
            && self.num_bursts >= 1
    }

    /// Pulse repetition period 1/PRF `s`.
    #[inline]
    #[must_use]
    pub fn pulse_period(&self) -> Time<f64> {
        Time::from_base(1.0 / self.pulse_repetition_frequency.into_base())
    }

    /// Burst duty cycle BDC = PL · PRF (fraction of the burst with carrier on).
    #[inline]
    #[must_use]
    pub fn burst_duty_cycle(&self) -> f64 {
        self.pulse_length.into_base() * self.pulse_repetition_frequency.into_base()
    }

    /// Burst-on fraction BD / (BD + BI).
    #[inline]
    #[must_use]
    pub fn burst_on_fraction(&self) -> f64 {
        let burst_duration = self.burst_duration.into_base();
        let burst_interval = self.burst_interval.into_base();
        burst_duration / (burst_duration + burst_interval)
    }

    /// Total duty cycle TDC = BDC · BD / (BD + BI).
    #[inline]
    #[must_use]
    pub fn total_duty_cycle(&self) -> f64 {
        self.burst_duty_cycle() * self.burst_on_fraction()
    }

    /// Burst repetition frequency BRF = 1 / (BD + BI) `Hz`.
    #[inline]
    #[must_use]
    pub fn burst_repetition_frequency(&self) -> Frequency<f64> {
        Frequency::from_base(
            1.0 / (self.burst_duration.into_base() + self.burst_interval.into_base()),
        )
    }

    /// Total experiment time TT = N · (BD + BI) `s`.
    #[inline]
    #[must_use]
    pub fn total_time(&self) -> Time<f64> {
        Time::from_base(
            f64::from(self.num_bursts)
                * (self.burst_duration.into_base() + self.burst_interval.into_base()),
        )
    }

    /// Whether the carrier is active (a pulse is on) at absolute `time`.
    ///
    /// `true` inside a burst (`t mod (BD+BI) < BD`) and inside a pulse within
    /// that burst (`τ mod (1/PRF) < PL`); `false` during pulse-off, burst
    /// intervals, and after the final burst.
    #[must_use]
    pub fn pulse_active(&self, time: Time<f64>) -> bool {
        let time_s = time.into_base();
        if time_s < 0.0 || time_s >= self.total_time().into_base() {
            return false;
        }
        let burst_duration = self.burst_duration.into_base();
        let burst_period = burst_duration + self.burst_interval.into_base();
        let phase_in_burst = time_s % burst_period;
        if phase_in_burst >= burst_duration {
            return false; // in the burst interval
        }
        let phase_in_pulse = phase_in_burst % self.pulse_period().into_base();
        phase_in_pulse < self.pulse_length.into_base()
    }

    /// Compute spatial-peak dosimetry from the carrier peak pressure.
    ///
    /// `I_SPPA = p²/(2ρc)`; `I_SPBA = I_SPPA·BDC`; `I_SPTA = I_SPPA·TDC`;
    /// `MI = p_MPa/√f_MHz` (computed via
    /// [`crate::acoustics::analysis::calculate_mechanical_index`]).
    ///
    /// # Arguments
    /// * `peak_pressure` — carrier peak pressure amplitude
    /// * `density` — medium density ρ
    /// * `sound_speed` — medium sound speed c
    #[must_use]
    pub fn dosimetry(
        &self,
        peak_pressure: Pressure<f64>,
        density: MassDensity<f64>,
        sound_speed: Velocity<f64>,
    ) -> PulseTrainDosimetry {
        let peak_pressure_pa = peak_pressure.into_base();
        let density_kg_m3 = density.into_base();
        let sound_speed_m_s = sound_speed.into_base();
        let z = density_kg_m3 * sound_speed_m_s;
        let isppa_w_m2 = if z > 0.0 {
            (peak_pressure_pa * peak_pressure_pa) / (2.0 * z)
        } else {
            0.0
        };
        let bdc = self.burst_duty_cycle();
        let tdc = self.total_duty_cycle();
        PulseTrainDosimetry {
            isppa: Intensity::from_base(isppa_w_m2),
            ispba: Intensity::from_base(isppa_w_m2 * bdc),
            ispta: Intensity::from_base(isppa_w_m2 * tdc),
            mechanical_index: crate::acoustics::analysis::calculate_mechanical_index(
                peak_pressure_pa,
                self.carrier_frequency.into_base(),
            ),
            total_duty_cycle: tdc,
            total_time: self.total_time(),
        }
    }

    /// Atkinson-Clement et al. (2025) transcranial theta-burst protocol:
    /// 500 kHz carrier, 20 ms pulse, 5 Hz PRF (10 % duty cycle), 80 s total.
    /// Modelled as a single continuous pulsed train (one burst, no interval).
    ///
    /// # Examples
    ///
    /// ```
    /// use aequitas::systems::si::quantities::Pressure;
    /// use kwavers_physics::acoustics::therapy::neuromodulation::{
    ///     tissue_dosimetry, PulseTrainProtocol,
    /// };
    /// let p = PulseTrainProtocol::theta_burst_atkinson_2025();
    /// assert!((p.total_duty_cycle() - 0.10).abs() < 1e-9);
    /// // I_SPTA = I_SPPA · duty cycle.
    /// let d = tissue_dosimetry(&p, Pressure::from_base(300.0e3));
    /// assert!((d.ispta.into_base() / d.isppa.into_base() - 0.10).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn theta_burst_atkinson_2025() -> Self {
        Self {
            carrier_frequency: Frequency::from_base(500.0e3),
            pulse_length: Time::from_base(20.0e-3),
            pulse_repetition_frequency: Frequency::from_base(5.0),
            burst_duration: Time::from_base(80.0),
            burst_interval: Time::from_base(0.0),
            num_bursts: 1,
        }
    }
}

/// Spatial-peak dosimetry derived from a [`PulseTrainProtocol`] and peak pressure.
#[derive(Debug, Clone, Copy)]
pub struct PulseTrainDosimetry {
    /// Spatial-peak pulse-averaged intensity I_SPPA in canonical SI units.
    pub isppa: Intensity<f64>,
    /// Spatial-peak burst-averaged intensity I_SPBA in canonical SI units.
    pub ispba: Intensity<f64>,
    /// Spatial-peak temporal-averaged intensity I_SPTA in canonical SI units.
    pub ispta: Intensity<f64>,
    /// Mechanical index MI [-].
    pub mechanical_index: f64,
    /// Total duty cycle TDC [-].
    pub total_duty_cycle: f64,
    /// Total experiment time TT `s`.
    pub total_time: Time<f64>,
}

impl PulseTrainDosimetry {
    /// Whether this dosimetry is within the FDA diagnostic output limits used as
    /// a conservative reference for neuromodulation (Blackmore 2019 safety
    /// guidelines): `I_SPTA ≤ 720 mW/cm²`, `I_SPPA ≤ 190 W/cm²`, `MI ≤ 1.9`.
    #[must_use]
    pub fn within_fda_limits(&self) -> bool {
        self.ispta.into_base() * W_M2_TO_W_CM2 * W_CM2_TO_MW_CM2 <= fda_ispta_limit_mw_cm2()
            && self.isppa.into_base() * W_M2_TO_W_CM2 <= fda_isppa_limit_w_cm2()
            && self.mechanical_index <= MI_LIMIT_SOFT_TISSUE
    }
}

// ── ITRUSST neuromodulation-specific safety (Aubry et al. 2024) ──────────────

/// ITRUSST mechanical-index limit for non-significant risk (= FDA diagnostic).
pub const ITRUSST_MI_LIMIT: f64 = 1.9;
/// ITRUSST peak temperature-rise limit for non-significant thermal risk [°C].
pub const ITRUSST_TEMP_RISE_LIMIT_C: f64 = 2.0;
/// ITRUSST thermal-dose limit in brain tissue [CEM43 min].
pub const ITRUSST_CEM43_BRAIN: f64 = 2.0;
/// ITRUSST thermal-dose limit in bone tissue [CEM43 min].
pub const ITRUSST_CEM43_BONE: f64 = 16.0;
/// ITRUSST thermal-dose limit in skin tissue [CEM43 min].
pub const ITRUSST_CEM43_SKIN: f64 = 21.0;

/// ITRUSST biophysical-safety assessment of a transcranial-US exposure.
#[derive(Debug, Clone, Copy)]
pub struct ItrusstAssessment {
    /// Mechanical criterion met (`MI ≤ 1.9`).
    pub mechanical_ok: bool,
    /// Thermal criterion met (`ΔT ≤ 2 °C` **or** `CEM43 ≤ 2` in brain).
    pub thermal_ok: bool,
    /// Both criteria met ⇒ non-significant risk per the ITRUSST consensus.
    pub overall_ok: bool,
}

/// Assess an exposure against the ITRUSST consensus thresholds (Aubry et al.
/// 2024). The exposure is non-significant-risk if the mechanical index is within
/// 1.9 **and** at least one thermal criterion holds (peak temperature rise within
/// 2 °C, or brain thermal dose within 2 CEM43). The thermal-index exposure-time
/// table is not encoded here.
///
/// These are expert-consensus reference levels — informative, not regulatory
/// limits.
///
/// # Arguments
/// * `mechanical_index` — peak MI [-]
/// * `peak_temperature_rise` — peak focal temperature rise ΔT
/// * `cem43_brain_min` — cumulative brain thermal dose [CEM43 min]
#[must_use]
pub fn itrusst_assess(
    mechanical_index: f64,
    peak_temperature_rise: TemperatureDifference<f64>,
    cem43_brain_min: f64,
) -> ItrusstAssessment {
    let mechanical_ok = mechanical_index <= ITRUSST_MI_LIMIT;
    let thermal_ok = peak_temperature_rise.into_base() <= ITRUSST_TEMP_RISE_LIMIT_C
        || cem43_brain_min <= ITRUSST_CEM43_BRAIN;
    ItrusstAssessment {
        mechanical_ok,
        thermal_ok,
        overall_ok: mechanical_ok && thermal_ok,
    }
}

/// Spatial-peak dosimetry for the default tissue medium
/// (ρ = 1000 kg/m³, c = 1540 m/s).
#[must_use]
pub fn tissue_dosimetry(
    protocol: &PulseTrainProtocol,
    peak_pressure: Pressure<f64>,
) -> PulseTrainDosimetry {
    protocol.dosimetry(
        peak_pressure,
        MassDensity::from_base(DENSITY_WATER_NOMINAL),
        Velocity::from_base(SOUND_SPEED_TISSUE),
    )
}
