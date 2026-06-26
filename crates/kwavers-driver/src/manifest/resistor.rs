//! [`ResistorPackage`] — the SMD series-damping resistor footprint, rated for continuous
//! dissipation at 70 °C ambient, and its pure rate-to-margin converter.

/// SMD resistor footprint rated for continuous dissipation at 70 °C ambient.
/// The per-channel series-damping resistor dissipates `per_tile_resistor_w`
/// of the dynamic pulser loss; the chosen footprint must rate above that
/// wattage on every tile's resistor, or the placement annealer cannot
/// satisfy IPC-7351 derating. Four footprints cover the article-class
/// HV7355 damping-resistor envelope; larger packages can be added as
/// variants without changing the call site.
///
/// **Article-class operating point envelope:** at the article's 50 pF
/// clamped load + 150 V V_pp + 2 MHz carrier + the per-tile PRF stagger
/// used by `four_tile_v2_manifest` (+50 Hz across 4 tiles ⇒ frame_duty
/// 0.50–0.575), the per-tile dissipation lands at 0.98–1.13 W. Tile\[3\]
/// exceeds `Smd2512 = 1.0 W` by ~13% (over-rates) but fits under
/// `Smd2512He = 1.5 W` with **+0.37 W margin on tile\[3\] WITHOUT
/// matching-cap retuning** -- the designer's "tight-but-still-fits"
/// middle envelope. `Smd4527 = 2 W` covers the envelope with comfortable
/// +0.87 W margin. Four design choices at the article envelope:
///
/// * **`Smd2512 = 1 W`** (`"2512"`) -- default if the matching cap is
///   retuned from 50 pF to 35 pF (drops dissipation to 0.69–0.79 W).
///   Over-rates tile\[3\] on the 50 pF envelope, NOT a valid choice
///   without cap retune.
/// * **`Smd2512He = 1.5 W`** (`"2512-HE"`) -- preferred for the
///   "tight-but-still-fits" path on the 50 pF / 150 V envelope; same 2512
///   land pattern as `Smd2512` (placement-friendly bump -- no board
///   re-layout) with ~50% more dissipation headroom.
/// * **`Smd4527 = 2 W`** (`"4527"`) -- designer-preferred default for the
///   unscoped article-class board; +0.87 W margin on tile\[3\] (Walsun /
///   Vishay RWC4527, 11.5 × 6.8 mm land pattern). Over-spec'd for the
///   article envelope but covers 2.5 W dissipation corner cases when a
///   future contributor compounds protocol tweaks past tile\[3\]'s
///   article-class dissipation.
/// * **`Smd1206 = 250 mW`** (`"1206"`) -- reserved for low-power surrogate
///   callers that size the dissipation envelope differently (always
///   over-rates the article-class operating point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResistorPackage {
    /// 1206 chip resistor: 250 mW at 70 °C ambient (IPC-7351 derating curve).
    Smd1206,
    /// 2512 chip resistor: 1 W at 70 °C ambient. Over-rates article-class
    /// tile\[3\] by ~13% on the 50 pF / 150 V envelope -- fits only after
    /// matching-cap retune to 35 pF.
    Smd2512,
    /// 2512-HE chip resistor: 1.5 W at 70 °C ambient. The "high-energy"
    /// 2512 footprint adds ~50% headroom over `Smd2512` (1 W) on the same
    /// land pattern -- the designer's middle envelope between the
    /// under-spec `Smd2512` (over-rates) and the over-spec `Smd4527`
    /// (comfortable 0.87 W margin at 2 W). Fits the article-class
    /// HV7355 envelope WITHOUT matching-cap retuning: 50 pF / 150 V /
    /// +50 Hz PRF stagger worst-case tile\[3\] dissipation ≈ 1.13 W
    /// ⇒ `Smd2512He` margin ≈ +0.37 W (positive slack on every tile).
    Smd2512He,
    /// 4527 chip resistor: 2 W at 70 °C ambient; covers the article-class
    /// HV7355 envelope with comfortable margin without matching-cap retuning.
    Smd4527,
}

impl ResistorPackage {
    /// Maximum continuous power dissipation (W) at 70 °C ambient.
    #[must_use]
    pub fn max_power_w(self) -> f64 {
        match self {
            Self::Smd1206 => 0.250,
            Self::Smd2512 => 1.0,
            Self::Smd2512He => 1.5,
            Self::Smd4527 => 2.0,
        }
    }

    /// Per-footprint dissipation margin (W) under the IPC-7351 70 °C rating.
    /// SIGNED — positive ⇒ headroom above the dissipation (`max_w − actual`),
    /// zero ⇒ exactly at the package limit, negative ⇒ footprint under-rates
    /// by `|margin|` W. The kwavers-side
    /// [`crate::validate::validate_against_budget`] 4th [`crate::validate::Check`]
    /// against `KWVERS_MIN_RESISTOR_MARGIN_W` is the sole
    /// gatekeeper — the inline rejection gate has been lifted out of
    /// [`crate::manifest::DriverManifest::validate_v2_energy_budget`] so this method is the
    /// pure rate-to-margin converter with no `Result` to thread; the consumer
    /// reads the signed value verbatim to plan footprint bumps (`Smd2512 ⇒
    /// Smd4527`) and matching-cap tightening without losing over-rate
    /// information at the validator boundary.
    #[must_use]
    pub fn power_margin_w(self, dissipation_w: f64) -> f64 {
        self.max_power_w() - dissipation_w
    }

    /// Human-readable footprint name (e.g. `"1206"`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Smd1206 => "1206",
            Self::Smd2512 => "2512",
            Self::Smd2512He => "2512-HE",
            Self::Smd4527 => "4527",
        }
    }
}
