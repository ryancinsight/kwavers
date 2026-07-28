/// Pulse-emission pattern at the transducer face.
///
/// Patterns encode the temporal envelope; cycle counts and durations are
/// stored in source-clock units (cycles for tone bursts, seconds for
/// shock-formed pulses). Optimal-pattern variants encode the dual-PRF and
/// dithered-PRF strategies that have been shown to improve cloud
/// regeneration and reduce pre-focal pre-conditioning effects.
use aequitas::systems::si::quantities::{Frequency, Time};
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PulsePattern {
    /// Constant-amplitude tone burst of `cycles` carrier cycles.
    ToneBurst { cycles: u32 },
    /// Shock-formed long sinusoid (boiling-histotripsy drive). The waveform
    /// distorts nonlinearly during propagation; `duration` is the on-time
    /// at the source.
    ShockFormed { duration: Time<f64> },
    /// Dual-PRF "burst-and-pause" pattern: `fast_pulses` micro-pulses at
    /// `fast_prf` followed by a quiescent gap of
    /// `1 / slow_prf - fast_pulses / fast_prf`.
    /// The fast burst maintains cloud coherence while the slow gap allows
    /// nucleus reset and thermal relaxation (Macoskey 2018).
    DualPrf {
        fast_prf: Frequency<f64>,
        slow_prf: Frequency<f64>,
        fast_pulses: u32,
        cycles_per_pulse: u32,
    },
    /// Dithered (jittered) PRF with mean rate `mean_prf` and uniform
    /// fractional jitter `jitter_frac ∈ [0, 1)`. Stochastic timing breaks
    /// pre-focal pre-conditioning lattices and improves cloud spatial
    /// homogeneity (Mancia 2020).
    DitheredPrf {
        mean_prf: Frequency<f64>,
        jitter_frac: f64,
        cycles_per_pulse: u32,
    },
}

impl PulsePattern {
    /// Effective on-time of a single emitted pulse at carrier frequency
    /// `carrier_frequency`. For dual-PRF patterns the on-time is the burst
    /// envelope length (fast micro-pulses sum); for dithered-PRF patterns
    /// it is the per-pulse on-time.
    #[must_use]
    pub fn pulse_on_time(self, carrier_frequency: Frequency<f64>) -> Time<f64> {
        let f0_hz = carrier_frequency.into_base();
        match self {
            Self::ToneBurst { cycles } => Time::from_base(f64::from(cycles) / f0_hz),
            Self::ShockFormed { duration } => duration,
            Self::DualPrf {
                fast_prf,
                fast_pulses,
                cycles_per_pulse,
                ..
            } => {
                let micro_on = f64::from(cycles_per_pulse) / f0_hz;
                let micro_period = 1.0 / fast_prf.into_base();
                Time::from_base(f64::from(fast_pulses - 1) * micro_period + micro_on)
            }
            Self::DitheredPrf {
                cycles_per_pulse, ..
            } => Time::from_base(f64::from(cycles_per_pulse) / f0_hz),
        }
    }

    /// Effective average pulse-repetition frequency.
    ///
    /// Single-pulse patterns have no repetition frequency and return `None`.
    #[must_use]
    pub fn average_prf(self) -> Option<Frequency<f64>> {
        match self {
            Self::ToneBurst { .. } | Self::ShockFormed { .. } => None,
            Self::DualPrf { slow_prf, .. } => Some(slow_prf),
            Self::DitheredPrf { mean_prf, .. } => Some(mean_prf),
        }
    }
}
