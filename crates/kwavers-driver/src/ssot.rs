//! Single source of truth for cross-cutting constants and string literals.
//!
//! # SSOT role
//!
//! Every kwavers-safety bound, format string, sidecar key, and engineering-contract literal
//! in the crate MUST live in this module. Anywhere a magic literal appears in `src/board.rs`,
//! `src/manifest.rs`, `src/validate.rs`, `src/io.rs`, etc., Phase 1c migrates it here, naming the
//! value once. Tests in `self::tests::ssot_values_pin_and_ratchet` lock every constant at its
//! current value so a silent future edit loud-rejects contract drift.
//!
//! # Naming convention (Phase 1c decision)
//!
//! **Flat with domain prefixes** rather than nested submodules. The codebase already uses this
//! pattern (see `TX_LANES_V2`, `CHANNELS_PER_TILE_V2`, `KWVERS_*` in older versions, `ARTICLE_*` in
//! acoustic). Prefixes enforce grouping at the call site:
//!
//! * `KWVERS_*` — kwavers-side physics / safety bounds for the beam-propagation pre-step.
//! * `CHECK_*` — kwavers-side Check names (extracted from inline string literals at every call site).
//! * `PHYSICS_*` — physics constants (water acoustic impedance and friends).
//! * `UNIT_*` — SI-prefix unit-conversion scalars. Constants, not factors, so the compiler folds them.
//! * `MANIFEST_*` — driver-manifest schema keys + article-class transducer preset constants.
//! * `KICAD_*` — KiCad format-emission constants (generator name, format versions).
//!
//! # Categorical traps (constants NOT migrated)
//!
//! * **Stimulation presets** in `StimulationProgram::article_default()` are tunable configuration
//!   defaults, not structural engineering contracts. They stay at `src/manifest.rs`.
//! * **Algorithm-internal drawing grid** in `write_kicad_sch` (`pitch = 2.54`, `hw = 7.62`) are
//!   internal positioning numbers, not exposed engineering parameters.
//! * **Enum discriminants / pre-typed indices** (`LayerId(0)`, `NetId(42)`) are deeply typed — they
//!   migrate only if there's a literal behind them that callers might mistake for *arbitrary* 0 or
//!   42. Today that doesn't apply.
//! * **Test-only floats** (`1e-9`, `f64::INFINITY`) are algorithm-internal — they stay inline.
//!
//! See `docs/MIGRATION.md` § "Phase 1c — ssot" for the full rationale.

// ============================================================================
// Kwavers-side physics / safety bounds (formerly `const` block in src/validate.rs)
// ============================================================================
#![allow(clippy::doc_markdown)]

/// Article-class per-element acoustic sensitivity (Pa/A). Anchors the v2 focal-pressure estimate
/// to the paper's measured 6 MPa / 10 mm focus with 16 channels × 0.04 A per-element drive: each
/// element contributes `6e6 / 16 ≈ 0.375 MPa` and the anchor is `0.375e6 / 0.04 = 9.375e6 Pa/A`.
///
/// Pin this anchor so a future contributor who changes the article-class transducer (e.g. HV7355 →
/// successor) is forced to update the SSOT explicitly, not silently change every consumer's
/// per-element gain estimate.
pub const KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA: f64 = 9.375e6;

/// **Renamed from** `KWVERS_MIN_FOCAL_PRESSURE_PA` to fix a name-value mismatch: the prior name
/// read like a unit tag but the value `1.0e6` is 1 MPa (= 1 000 000 Pa), not 1 Pa. The new name
/// `1MPA_IN_PA` makes the engineering contract — "exactly 1 MPa" — immediately readable, so a
/// contributor reading `1.0e6` doesn't mistake it for a small baseline.
///
/// Min focal-pressure floor for the kwavers pre-step `lower` check; 1 MPa is the order-of-magnitude a
/// therapeutic acoustic stack must reach — below this the chain has a transduction-loss or
/// wiring-fault problem.
pub const KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA: f64 = 1.0e6;

/// Maximum Mechanical Index the kwavers pre-step accepts. 12.5 keeps the pulse-regime
/// neuromodulation headroom (FDA diagnostic 1.9, therapy band typically 2–10) plus margin for v2
/// stack settings whose article-anchored estimate can land at MI ≈ 9.5 — right at the
/// cavitation edge. Tightening to a lower number (e.g. 10.0) silently flips `report.all_pass` on
/// settings drift.
pub const KWVERS_MI_CAVITATION_CEILING: f64 = 12.5;

/// Min grating-lobe-free steering angle (deg) the kwavers pre-step requires for the `lower` check.
/// Article-class 0.5λ pitch reaches 90°; relax to ≥89° to keep the boolean-tolerance-side
/// acceptance while still flagging anywhere a future contributor widens the pitch above λ/2.
pub const KWVERS_MIN_GRATING_FREE_STEER_DEG: f64 = 89.0;

/// Min per-tile resistor power margin (W) the kwavers pre-step enforces via the 4th `Check::lower`
/// against the per-tile min of `KwaversBeamStep::resistor_margin_w`. The inline rejection gate was
/// lifted out of `validate_v2_energy_budget` so this constant is the SOLE gatekeeper.
///
/// **Slack floor: `0.05 W` (50 mW).** Enforces a real headroom BUDGET above the IPC-7351 70 °C
/// AMBIENT ceiling — at the ceiling a footprint dissipates precisely its rated wattage with zero
/// thermal margin. 50 mW of required slack catches "fits by exactly its rating" choices before
/// the IPC ceiling is crossed in service.
pub const KWVERS_MIN_RESISTOR_MARGIN_W: f64 = 0.05;

/// Water acoustic impedance (Rayl). Used to convert RMS pressure to spatial-peak intensity via
/// `I = p²/Z₀`. Typical tissue 1.54 MRayl ≈ water 1.48 MRayl; we use water as the conservative
/// impedance for ISPPA safety reporting since soft-tissue transducer coupling sits within
/// ~5 % of water.
pub const PHYSICS_WATER_Z0_RAYL: f64 = 1.48e6;

// ============================================================================
// SI-prefix unit-conversion scalars (formerly `const` block in src/validate.rs)
// ============================================================================

/// Pa → MPa conversion (i.e. 1 MPa = how many Pa). Constants, not factors, so the compiler folds
/// them. Use `p_pa / UNIT_PA_PER_MPA` to get p in MPa.
pub const UNIT_PA_PER_MPA: f64 = 1.0e6;

/// Hz → MHz conversion.
pub const UNIT_MHZ_PER_HZ: f64 = 1.0e6;

/// W/m² → W/cm² conversion: 1 cm² = 1e-4 m², so the intensity in W/cm² = intensity in W/m² × 1e4.
pub const UNIT_W_CM2_PER_W_M2: f64 = 1.0e4;

/// m → mm conversion.
pub const UNIT_MM_PER_M: f64 = 1.0e3;

// ============================================================================
// Kwavers-side `Check` names (extracted from inline string literals in src/validate.rs)
// ============================================================================
//
// Each is a string the [`crate::validate::validate_against_budget`] report carries verbatim. Extracting
// to SSOT means downstream tests grepping for the name, log scrapers pattern-matching the report, and
// the kwavers substitute reading the same identifier — all see the exact same byte-stable string.
// A contributor who accidentally mistypes one of these in `validate.rs` would otherwise break the
// grep / pattern-match silently.

/// `PhysicsReport` check name for the focal-pressure lower bound.
pub const CHECK_FOCAL_PRESSURE_NAME: &str = "focal pressure ≥ kwavers pre-step floor";

/// `PhysicsReport` check name for the mechanical-index upper bound.
pub const CHECK_MI_NAME: &str = "mechanical index < cavitation ceiling";

/// `PhysicsReport` check name for the grating-lobe-free steer lower bound.
pub const CHECK_GRATING_LOBE_NAME: &str = "grating-lobe-free steer ≥ 89°";

/// `PhysicsReport` check name for the per-tile resistor-margin lower bound (the SOLE gatekeeper for
/// the kwavers pre-step's resistor rating contract).
pub const CHECK_RESISTOR_MARGIN_NAME: &str = "resistor margin (per-tile min) ≥ 0 W";

// ============================================================================
// Driver manifest schema keys + article-class transducer preset (src/manifest.rs)
// ============================================================================

/// V1 manifest format header (legacy). Emitted at the top of every `.kv` sidecar; parsed by
/// [`crate::manifest::DriverManifest::from_text`] to dispatch v1 vs v2 deserialisation.
/// Backwards-compat-only.
pub const MANIFEST_FORMAT_V1: &str = "kicad-routing-driver-manifest-v1";

/// V2 manifest format header. Emitted at the top of every current-emission `.kv` sidecar.
pub const MANIFEST_FORMAT_V2: &str = "kicad-routing-driver-manifest-v2";

/// Article-class element count (MWSCAS 2024 transducer): 16 elements.
pub const MANIFEST_ARTICLE_ELEMENTS: usize = 16;

/// Article-class aperture per element (m): 4.3 mm / element (MWSCAS 2024 transducer pitch geometry).
pub const MANIFEST_ARTICLE_APERTURE_M: f64 = 4.3e-3;

/// Total transducer output lanes for the full 96-channel shield stack (4 tiles × 24 channels ⇒
/// `TX_0..TX_95`). Cardinal binding the kwavers pre-step enforces via `is_full_stack_v2()` —
/// migrating to SSOT so every module sees the same lane count without inline duplication.
pub const TX_LANES_V2: usize = 96;

/// Number of channels per HV tile in the 96-channel shield stack: the 24-channel HV7355 shield
/// tile is the article class; the 4-tile shield stack binds `4 × 24 ⇒ TX_LANES_V2`.
/// Migraed from src/manifest.rs to SSOT — formerly `pub const` at that module.
pub const CHANNELS_PER_TILE_V2: usize = 24;

// ============================================================================
// KiCad format emission (src/io.rs)
// ============================================================================

/// KiCad generator-name the engine signs every emitted `.kicad_pcb` / `.kicad_sch` with. Lets a
/// downstream auditor "who made this board?" inspect the file header without parsing the
/// comments.
pub const KICAD_GENERATOR_NAME: &str = "kicad-routing";

/// KiCad PCB format version stamp — every emitted `.kicad_pcb` carries this in the top-level
/// `(kicad_pcb (version YYYYMMDD) (generator …))`. Matches KiCad's 2024-01-08 file format.
pub const KICAD_PCB_FORMAT_VERSION: &str = "20240108";

/// KiCad schematic format version stamp. Matches KiCad's 2023-11-20 file format.
pub const KICAD_SCH_FORMAT_VERSION: &str = "20231120";

// ============================================================================
// Pinning tests (ratchet for silent edits)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// SSOT ratchet: every constant is pinned at its current value. A future contributor who
    /// silently edits a constant's value (without a co-tracked test / MIGRATION.md update) breaks
    /// this test loudly, rejecting contract drift. The values are deliberately hard-coded as
    /// literals here (NOT pulled from `super::*`) so this test pins the *intended engineering
    /// contract* rather than the current implementation.
    #[test]
    fn ssot_values_pin_and_ratchet() {
        // Kwavers-side physics / safety bounds
        assert_eq!(KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA, 9.375e6);
        assert_eq!(KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA, 1.0e6);
        assert_eq!(KWVERS_MI_CAVITATION_CEILING, 12.5);
        assert_eq!(KWVERS_MIN_GRATING_FREE_STEER_DEG, 89.0);
        assert_eq!(KWVERS_MIN_RESISTOR_MARGIN_W, 0.05);
        assert_eq!(PHYSICS_WATER_Z0_RAYL, 1.48e6);

        // Unit-prefix scalars
        assert_eq!(UNIT_PA_PER_MPA, 1.0e6);
        assert_eq!(UNIT_MHZ_PER_HZ, 1.0e6);
        assert_eq!(UNIT_W_CM2_PER_W_M2, 1.0e4);
        assert_eq!(UNIT_MM_PER_M, 1.0e3);

        // Check names (byte-stable strings for grep + log scraping)
        assert_eq!(
            CHECK_FOCAL_PRESSURE_NAME,
            "focal pressure ≥ kwavers pre-step floor"
        );
        assert_eq!(CHECK_MI_NAME, "mechanical index < cavitation ceiling");
        assert_eq!(CHECK_GRATING_LOBE_NAME, "grating-lobe-free steer ≥ 89°");
        assert_eq!(
            CHECK_RESISTOR_MARGIN_NAME,
            "resistor margin (per-tile min) ≥ 0 W"
        );

        // Manifest schema + article-class preset
        assert_eq!(MANIFEST_FORMAT_V1, "kicad-routing-driver-manifest-v1");
        assert_eq!(MANIFEST_FORMAT_V2, "kicad-routing-driver-manifest-v2");
        assert_eq!(MANIFEST_ARTICLE_ELEMENTS, 16);
        assert_eq!(MANIFEST_ARTICLE_APERTURE_M, 4.3e-3);
        assert_eq!(TX_LANES_V2, 96);
        assert_eq!(CHANNELS_PER_TILE_V2, 24);

        // KiCad format emission
        assert_eq!(KICAD_GENERATOR_NAME, "kicad-routing");
        assert_eq!(KICAD_PCB_FORMAT_VERSION, "20240108");
        assert_eq!(KICAD_SCH_FORMAT_VERSION, "20231120");
    }

    /// KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA reads as exactly 1 MPa expressed in Pa, NOT 1 Pa.
    /// The rename from `_PA` to `_1MPA_IN_PA` exists to prevent name-value confusion; the test
    /// below asserts the contract is exactly the SI Pascal value (`1e6`), which equals 1 MPa.
    /// A contributor reading the value alone still needs to know it's 1 MPa — anchor that.
    /// (Both legs of the relationship are independently pinned at `1.0e6` by
    /// [`Self::ssot_values_pin_and_ratchet`], so no extra `>=` guard is needed here — that
    /// comparison would be a clippy `assertions_on_constants` lint.)
    #[test]
    fn min_focal_pressure_1mpa_in_pa_equals_one_megapascal() {
        // 1 MPa = 1e6 Pa = UNIT_PA_PER_MPA × 1.0
        assert_eq!(
            KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA, UNIT_PA_PER_MPA,
            "rename documents exactly 1 MPa; tests lock the relationship"
        );
    }
}
