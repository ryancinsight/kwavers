//! The acoustic stimulation-protocol schema: the article-class single preset
//! ([`StimulationProgram`]) and the per-tile profile ([`TileStimulationProfile`]) the 96-channel
//! shield stack carries. Both expose the protocol-load proxy that the kwavers beam validation sums.

/// Article-class ultrasound stimulation program. The shape and unit conventions match the
/// operating point Javid et al. (MWSCAS 2024) used to drive their 16-element array: a fast
/// pulse-repetition cadence (PRF) inside each tone-burst, repeated for some sonication duration
/// (SD), separated by a longer inter-stimulus interval (ISI), accumulating to a total stimulation
/// time (TT). The dead-time enforces minimum PW≤TBD separation so a back-to-back burst cannot
/// bridge duty > 1; the output amplitude (vpp_v) and the carrier frequency (Hz) carry the actual
/// acoustic drive. Coupling these to the manifest makes the hardware what it actually drives
/// (vs. an article-only constants assumption).
#[derive(Debug, Clone, PartialEq)]
pub struct StimulationProgram {
    /// Pulse repetition frequency (Hz) inside the tone burst — the carrier of timing.
    pub prf_hz: f64,
    /// Tone-burst duration (s): how long each sonication's PW lasts.
    pub tbd_s: f64,
    /// Sonication duration (s): how long the protocol keeps the PW firing inside one ISI interval.
    pub sd_s: f64,
    /// Inter-stimulus interval (s): the off-time between sonication segments.
    pub isi_s: f64,
    /// Total stimulation time (s): the protocol wall-clock duration.
    pub tt_s: f64,
    /// Nominal pulse amplitude (V peak-to-peak) at the pulser output.
    pub vpp_v: f64,
    /// Minimum dead-time between consecutive tone bursts (s) so PW ≤ TBD always holds.
    pub dead_time_s: f64,
}

impl StimulationProgram {
    /// The article-class MWSCAS 2024 preset: 1 kHz PRF × 0.5 ms bursts × 300 ms sonication ×
    /// 3 s ISI × 18 s total, at 150 V, on a 2 MHz carrier. `dead_time` is set to a coherent
    /// 0.5 ms so back-to-back bursts stay bounded even if `tbd_s` is widened. This preset is the
    /// single source of truth for the article's acoustic protocol — acoustic simulations and
    /// kwovers validation consume it verbatim.
    #[must_use]
    pub fn article_default() -> Self {
        StimulationProgram {
            prf_hz: 1.0e3,
            tbd_s: 0.5e-3,
            sd_s: 300.0e-3,
            isi_s: 3.0,
            tt_s: 18.0,
            vpp_v: 150.0,
            dead_time_s: 0.5e-3,
        }
    }

    /// Number of sonication segments (ISI cycles) inside the total stimulation time:
    /// `ceil((tt - sd) / (sd + isi)) + 1`. The first segment always fires, so the formula
    /// counts it explicitly. Returns `1` for degenerate parameters (zero or negative ISI, etc.)
    /// so a downstream plot/index never divides by zero.
    #[must_use]
    pub fn sonication_count(&self) -> usize {
        if self.sd_s <= 0.0 || self.isi_s <= 0.0 || self.tt_s <= 0.0 {
            return 1;
        }
        let cycle = self.sd_s + self.isi_s;
        if self.tt_s <= self.sd_s {
            return 1;
        }
        1 + ((self.tt_s - self.sd_s) / cycle).ceil() as usize
    }

    /// Total energy budget proxy (J·s) = `vpp² × tt × sd / (sd + isi)` — a useful acoustic-load
    /// scalar when sweeping the program. Not the actual delivered energy (which depends on the
    /// transducer impedance and the burst count) but a first-order monotonic in the operating
    /// point that downstream thermal/budget accounting can consume as a single scalar.
    #[must_use]
    pub fn protocol_load_j_s(&self) -> f64 {
        let duty = if (self.sd_s + self.isi_s) > 0.0 {
            self.sd_s / (self.sd_s + self.isi_s)
        } else {
            0.0
        };
        self.vpp_v * self.vpp_v * self.tt_s * duty
    }
}

/// Track D v2 follow-up: per-tile stimulation program. Each HV tile in the 96-channel shield
/// stack (`4 tiles × 24 channels ⇒ TX_0..TX_95`) carries its own PRF, inter-tile start SHIFT,
/// carrier PHASE and amplitude RAMP profile. The inherited protocol fields (TBD/SD/ISI/TT/VPP/
/// dead-time) gate the per-tile timing exactly as the article-class [`StimulationProgram`] did,
/// so acoustic validation and kwavers beam propagation can run on a per-tile schedule.
///
/// SHIFT/PHASE/RAMP are macroscopic offsets relative to the stack reference (tile 0):
/// * **SHIFT** is the delayed protocol start (s). Tile `i` waits `shift_s` before its first
///   tone-burst fires. Stagger keeps the regulator's bulk-cap recharge cycle bounded.
/// * **PHASE** is the 2 MHz carrier phase offset (deg). Defines the inter-tile beamforming
///   reference a `tx_phase = π · i/N` scheme would normalise on.
/// * **RAMP** is the linear amplitude ramp (s) at the start and end of every tone burst (so
///   the matching network's slew-rate limit is never exceeded).
#[derive(Debug, Clone, PartialEq)]
pub struct TileStimulationProfile {
    /// Per-tile pulse repetition frequency (Hz) inside the tone burst.
    pub prf_hz: f64,
    /// Delayed protocol start relative to tile 0 (s).
    pub shift_s: f64,
    /// Carrier phase offset (deg).
    pub phase_deg: f64,
    /// Linear amplitude ramp duration at each tone burst boundary (s).
    pub ramp_s: f64,
    /// Tone-burst duration (s).
    pub tbd_s: f64,
    /// Sonication duration (s).
    pub sd_s: f64,
    /// Inter-stimulus interval (s).
    pub isi_s: f64,
    /// Total stimulation time (s).
    pub tt_s: f64,
    /// Nominal pulse amplitude (V peak-to-peak).
    pub vpp_v: f64,
    /// Minimum dead-time between consecutive tone bursts (s).
    pub dead_time_s: f64,
}

impl TileStimulationProfile {
    /// Project the article-class preset onto a tile with the supplied per-tile overrides; the
    /// protocol fields (TBD/SD/ISI/TT/VPP/dead-time) inherit the article values verbatim so the
    /// stack-level acoustic schedule stays comparable to the original 16-element MWSCAS 2024
    /// operating point.
    #[must_use]
    pub fn from_article_with(prf_hz: f64, shift_s: f64, phase_deg: f64, ramp_s: f64) -> Self {
        let s = StimulationProgram::article_default();
        TileStimulationProfile {
            prf_hz,
            shift_s,
            phase_deg,
            ramp_s,
            tbd_s: s.tbd_s,
            sd_s: s.sd_s,
            isi_s: s.isi_s,
            tt_s: s.tt_s,
            vpp_v: s.vpp_v,
            dead_time_s: s.dead_time_s,
        }
    }

    /// Per-tile protocol-load proxy (J·s); mirrors [`StimulationProgram::protocol_load_j_s`] so
    /// the stack-level kwavers validation can sum it across tiles without re-deriving the duty.
    #[must_use]
    pub fn protocol_load_j_s(&self) -> f64 {
        let duty = if (self.sd_s + self.isi_s) > 0.0 {
            self.sd_s / (self.sd_s + self.isi_s)
        } else {
            0.0
        };
        self.vpp_v * self.vpp_v * self.tt_s * duty
    }

    /// Frame duty (0..=1) of the per-tile pulser — the PRF-modulated fraction of wall-clock the
    /// tone bursts are firing. `frame_duty = tbd × prf` clipped to `[0, 1]`.
    #[must_use]
    pub fn frame_duty(&self) -> f64 {
        (self.tbd_s * self.prf_hz).clamp(0.0, 1.0)
    }
}
