//! Histotripsy clinical exposure scenarios and pulse-pattern definitions.
//!
//! The intrinsic-threshold magnitude `p_t(f)` for water-rich soft tissue
//! follows Vlaisavljevich et al. (2015), who reported only a weak monotone
//! increase with frequency over 0.345–3 MHz. Their fit to bovine liver and
//! tissue phantoms is well-approximated by
//!
//! ```text
//!   p_t(f) [MPa] = p_t0 + k_f * log10(f / 1 MHz)
//! ```
//!
//! with `p_t0 = 28.2 MPa`, `k_f = 1.4 MPa/decade` for liver-equivalent
//! tissue (within the 1-σ band of Maxwell 2013 Table II at 1 MHz). For
//! stiffer media (Young's modulus `E > 10 kPa`) Vlaisavljevich reports an
//! additive correction `delta_p_t(E) ≈ 0.7 sqrt(E/E_ref)` MPa with
//! `E_ref = 1 kPa`; we expose the soft-tissue baseline only and let
//! callers add stiffness corrections explicitly if required.

mod benefit;
mod pulse;
mod regime;

use aequitas::systems::si::quantities::{Frequency, Intensity, Pressure, Time, Volume};
use aequitas::systems::si::units::{CubicMillimeter, Megahertz, Megapascal};
use kwavers_core::constants::fundamental::ACOUSTIC_IMPEDANCE_TISSUE_NOMINAL;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use leto_ops::application::special::erf;

pub use benefit::{intrinsic_threshold, BenefitDetriment};
pub use pulse::PulsePattern;
pub use regime::HistotripsyRegime;

/// Histotripsy exposure scenario.
///
/// Combines the regime, transducer drive parameters, pulse pattern, and a
/// short literature-derived benefit/detriment list. PNP is stored as a
/// negative pressure, PPP as positive.
#[derive(Debug, Clone, Copy)]
pub struct HistotripsyScenario {
    pub regime: HistotripsyRegime,
    pub frequency: Frequency<f64>,
    pub peak_negative_pressure: Pressure<f64>,
    pub peak_positive_pressure: Pressure<f64>,
    pub pulse: PulsePattern,
    pub treatment_duration: Time<f64>,
    pub focal_volume: Volume<f64>,
    pub bd: BenefitDetriment,
}

impl HistotripsyScenario {
    /// Mechanical Index `MI = |p^-_min(MPa)| / sqrt(f0(MHz))` (AIUM/NEMA).
    #[must_use]
    pub fn mechanical_index(&self) -> f64 {
        let pnp_mpa = self.peak_negative_pressure.into_base().abs() / MPA_TO_PA;
        let f0_mhz = self.frequency.into_base() / MHZ_TO_HZ;
        pnp_mpa / f0_mhz.sqrt()
    }

    /// Average duty cycle (dimensionless, in [0, 1]).
    #[must_use]
    pub fn duty_cycle(&self) -> f64 {
        let Some(prf) = self.pulse.average_prf() else {
            return f64::NAN;
        };
        self.pulse.pulse_on_time(self.frequency).into_base() * prf.into_base()
    }

    /// Intrinsic-threshold pressure magnitude at the carrier frequency.
    #[must_use]
    pub fn intrinsic_threshold(&self) -> Pressure<f64> {
        intrinsic_threshold(self.frequency)
    }

    /// Whether the focal PNP magnitude exceeds the soft-tissue intrinsic
    /// threshold at the carrier frequency.
    #[must_use]
    pub fn exceeds_intrinsic_threshold(&self) -> bool {
        self.peak_negative_pressure.into_base().abs() >= self.intrinsic_threshold().into_base()
    }

    /// Single-pulse cavitation probability `P_cav` from the Maxwell 2013
    /// erf-CDF with parameters `(p_t, sigma_t) = (28.2, 0.96) MPa`. The
    /// formula is monotone in PNP magnitude (Theorem 21.1).
    #[must_use]
    pub fn cavitation_probability(&self) -> f64 {
        const SIGMA_T_PA: f64 = 0.96 * MPA_TO_PA;
        let pnp_abs = self.peak_negative_pressure.into_base().abs();
        let pt = self.intrinsic_threshold().into_base();
        let arg = (pnp_abs - pt) / (SIGMA_T_PA * std::f64::consts::SQRT_2);
        0.5 * (1.0 + erf(arg))
    }

    /// In-situ spatial-peak pulse-average intensity at the focus,
    /// computed under the plane-wave approximation `I = p^2 / (2 ρ c)`
    /// using the peak-positive pressure as the worst-case envelope.
    #[must_use]
    pub fn pulse_average_intensity(&self) -> Intensity<f64> {
        let p = self
            .peak_positive_pressure
            .into_base()
            .max(self.peak_negative_pressure.into_base().abs());
        Intensity::from_base(p * p / (2.0 * ACOUSTIC_IMPEDANCE_TISSUE_NOMINAL))
    }

    // ---------------------- Library scenarios ----------------------

    /// Microsecond intrinsic-threshold liver histotripsy at 1 MHz
    /// (histotripsy-style clinical waveform; Smolock 2018, Vlaisavljevich 2015).
    #[must_use]
    pub fn intrinsic_threshold_liver_1mhz() -> Self {
        Self {
            regime: HistotripsyRegime::IntrinsicThreshold,
            frequency: Frequency::from_unit::<Megahertz>(1.0),
            peak_negative_pressure: Pressure::from_unit::<Megapascal>(-30.0),
            peak_positive_pressure: Pressure::from_unit::<Megapascal>(80.0),
            pulse: PulsePattern::ToneBurst { cycles: 2 },
            treatment_duration: Time::from_base(600.0),
            focal_volume: Volume::from_unit::<CubicMillimeter>(7.0),
            bd: BenefitDetriment {
                benefits: &[
                    "Sharp sub-cellular lesion boundary (~1 cell)",
                    "No bulk thermal accumulation; CEM43 << 1 min",
                    "Single-pulse nucleation independent of cycle count",
                    "Short total treatment time per focus (<1 s/cm^3)",
                    "Real-time feedback via cloud bubble cloud detection",
                ],
                detriments: &[
                    "Requires very high PNP (>= 28-30 MPa) — limits aperture/focal-gain margin",
                    "Aberration through skull/bone severely degrades focal PNP",
                    "Limited efficacy in stiff or calcified tissue (E > 10 kPa)",
                    "Pre-focal cavitation in skin/fat layers can shadow the focus",
                ],
            },
        }
    }

    /// Shock-scattering histotripsy at 1 MHz with sub-threshold PNP and
    /// strongly distorted positive shock (Maxwell 2011; Lin 2014).
    #[must_use]
    pub fn shock_scattering_1mhz() -> Self {
        Self {
            regime: HistotripsyRegime::ShockScattering,
            frequency: Frequency::from_unit::<Megahertz>(1.0),
            peak_negative_pressure: Pressure::from_unit::<Megapascal>(-20.0),
            peak_positive_pressure: Pressure::from_unit::<Megapascal>(90.0),
            pulse: PulsePattern::ToneBurst { cycles: 5 },
            treatment_duration: Time::from_base(600.0),
            focal_volume: Volume::from_unit::<CubicMillimeter>(8.0),
            bd: BenefitDetriment {
                benefits: &[
                    "Lower PNP requirement than intrinsic-threshold regime",
                    "Strong nonlinear shock supplies focal energy efficiently",
                    "Cavitation cloud sustained over full pulse length",
                ],
                detriments: &[
                    "Requires bubble seed nucleus — first-pulse latency",
                    "Stochastic shot-to-shot cloud morphology",
                    "Slightly larger thermal footprint than intrinsic-threshold",
                ],
            },
        }
    }

    /// Boiling histotripsy at 1 MHz, porcine liver protocol (Khokhlova 2019).
    /// 10 ms shock-formed pulses at 1 Hz PRF, duty cycle 1%.
    #[must_use]
    pub fn boiling_histotripsy_liver_1mhz() -> Self {
        Self {
            regime: HistotripsyRegime::Boiling,
            frequency: Frequency::from_unit::<Megahertz>(1.0),
            peak_negative_pressure: Pressure::from_unit::<Megapascal>(-15.0),
            peak_positive_pressure: Pressure::from_unit::<Megapascal>(85.0),
            pulse: PulsePattern::ShockFormed {
                duration: Time::from_base(10.0e-3),
            },
            treatment_duration: Time::from_base(1200.0),
            focal_volume: Volume::from_unit::<CubicMillimeter>(30.0),
            bd: BenefitDetriment {
                benefits: &[
                    "Lower PNP than intrinsic-threshold (skull/aberration tolerant)",
                    "Larger lesion volume per pulse via vapor-bubble seeding",
                    "Mechanically fractionated lesion with sub-cellular debris",
                    "Compatible with single-element therapy transducers",
                ],
                detriments: &[
                    "Per-pulse focal heating 50-100 K transient — must respect off-time",
                    "CEM43 several orders of magnitude above intrinsic-threshold regime",
                    "Bone interfaces accumulate heat at duty 1% — careful planning",
                    "Slower volumetric coverage (1-2 cm^3 / 10 min)",
                ],
            },
        }
    }

    /// Sub-threshold millisecond cavitation cloud regime at 500 kHz
    /// (Vlaisavljevich 2018). Long pulses below `p_t(f)`.
    #[must_use]
    pub fn millisecond_cavitation_500khz() -> Self {
        Self {
            regime: HistotripsyRegime::MillisecondCavitation,
            frequency: Frequency::from_unit::<Megahertz>(0.5),
            peak_negative_pressure: Pressure::from_unit::<Megapascal>(-18.0),
            peak_positive_pressure: Pressure::from_unit::<Megapascal>(35.0),
            pulse: PulsePattern::ShockFormed {
                duration: Time::from_base(5.0e-3),
            },
            treatment_duration: Time::from_base(1800.0),
            focal_volume: Volume::from_unit::<CubicMillimeter>(25.0),
            bd: BenefitDetriment {
                benefits: &[
                    "PNP well below 28 MPa — usable through transcranial windows",
                    "Lower frequency improves skull penetration (-6 dB at 500 kHz vs 1 MHz)",
                    "Cavitation cloud built from many-cycle inertial collapse",
                ],
                detriments: &[
                    "Longer treatment time per focus than intrinsic-threshold",
                    "Less sharp lesion boundary than microsecond regime",
                    "Higher cavitation-cloud heterogeneity shot-to-shot",
                ],
            },
        }
    }

    /// Microsecond histotripsy at 1.5 MHz for thrombolysis (clot
    /// fractionation; Maxwell 2009, Bader 2018).
    #[must_use]
    pub fn thrombolysis_1_5mhz() -> Self {
        Self {
            regime: HistotripsyRegime::IntrinsicThreshold,
            frequency: Frequency::from_unit::<Megahertz>(1.5),
            peak_negative_pressure: Pressure::from_unit::<Megapascal>(-32.0),
            peak_positive_pressure: Pressure::from_unit::<Megapascal>(70.0),
            pulse: PulsePattern::ToneBurst { cycles: 3 },
            treatment_duration: Time::from_base(300.0),
            focal_volume: Volume::from_unit::<CubicMillimeter>(4.0),
            bd: BenefitDetriment {
                benefits: &[
                    "Drug-free clot fractionation",
                    "Minimal collateral thermal damage",
                    "Compatible with intravascular imaging guidance",
                ],
                detriments: &[
                    "Requires very high PNP (>= 28 MPa) at 1.5 MHz",
                    "Vessel-wall safety window narrow — strict aiming required",
                ],
            },
        }
    }

    /// Optimal dual-PRF microsecond pattern (Macoskey 2018, Maeda 2018).
    /// Recovers cloud regeneration without thermal accumulation.
    #[must_use]
    pub fn intrinsic_threshold_dual_prf_1mhz() -> Self {
        let mut s = Self::intrinsic_threshold_liver_1mhz();
        s.pulse = PulsePattern::DualPrf {
            fast_prf: Frequency::from_base(1000.0),
            slow_prf: Frequency::from_base(50.0),
            fast_pulses: 5,
            cycles_per_pulse: 2,
        };
        s
    }

    /// Optimal dithered-PRF pattern (Mancia 2020). Reduces pre-focal
    /// pre-conditioning at fixed mean PRF.
    #[must_use]
    pub fn intrinsic_threshold_dithered_prf_1mhz() -> Self {
        let mut s = Self::intrinsic_threshold_liver_1mhz();
        s.pulse = PulsePattern::DitheredPrf {
            mean_prf: Frequency::from_base(200.0),
            jitter_frac: 0.3,
            cycles_per_pulse: 2,
        };
        s
    }
}
