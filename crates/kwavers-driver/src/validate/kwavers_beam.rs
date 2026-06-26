//! The driver→transducer beam-propagation pre-step adapter — the typed seam the downstream
//! `crates/kwavers-transducer` simulator consumes.
//!
//! [`KwaversBeamStep`] is the *typed pre-step contract*: every field is a scalar this crate already
//! computes, which a future `crate::kwavers_transducer::simulate(&step)` would read verbatim to
//! produce a `PressureMap`. Until that wiring lands, [`validate_against_budget`] smoke-tests the same
//! predictions using the in-crate acoustic models in [`crate::physics::acoustic`] — the pre-step
//! shape stays identical, so wiring kwavers is "exchange the in-crate physics block for the kwavers
//! call", no struct migration. Kwavers safety bounds, `Check` names, water Z₀, and SI-prefix scalars
//! live in [`crate::ssot`] (the SSOT ratchet locks each at its engineering contract).

use crate::manifest::{DriverManifest, EnergyBudgetReport};
use crate::physics::acoustic::{
    acoustic_intensity_w_per_m2, f_number, focal_pressure_gain, max_grating_free_steer_deg,
    mechanical_index, near_field_distance_m, pitch_from_aperture_m, wavelength_m,
};
use crate::ssot::*;

use super::check::{Check, PhysicsReport};

/// Typed pre-step kwavers-transducer consumes. Every field is a scalar this
/// crate already computes; the kwavers-side simulator reads them verbatim
/// and produces a [`crate::physics::acoustic`] pressure map. The pre-step does NOT
/// duplicate the per-tile protocol-load proxy (that's [`EnergyBudgetReport`])
/// — kwavers consumers pull it from there.
///
/// Indexed by `lanes` so multi-stack configurations (>= 96) fold cleanly into
/// the same scalar schema. The `48`-lane or `192`-lane board class extends by
/// changing `lanes`; the only semantic shift is the coherent gain.
#[derive(Debug, Clone, PartialEq)]
pub struct KwaversBeamStep {
    /// Total transducer channels (96 for full-stack v2 = 4 tiles × 24 ch).
    pub lanes: usize,
    /// Aperture (m) — first-to-last element centre span.
    pub aperture_m: f64,
    /// Drive frequency (Hz).
    pub frequency_hz: f64,
    /// Medium sound speed (m/s).
    pub sound_speed_m_s: f64,
    /// Nominal focal depth (m).
    pub focal_m: f64,
    /// Hardware timing quantum (s).
    pub timing_step_s: f64,
    /// Centre-to-centre element pitch (m) = `aperture_m / (lanes - 1)`.
    pub pitch_m: f64,
    /// Acoustic wavelength (m) = `sound_speed_m_s / frequency_hz`.
    pub wavelength_m: f64,
    /// f-number of the focused aperture = `focal_m / aperture_m`.
    pub f_number: f64,
    /// Per-tile resistor power margin (W) under the chosen footprint's IPC-7351 70 °C rating.
    /// Mirrored verbatim from [`crate::manifest::EnergyBudgetReport::per_tile_resistor_margin_w`]
    /// — SIGNED after the inline rejection gate was lifted out of `validate_v2_energy_budget`:
    /// positive entry ⇒ headroom above the dissipation (`chosen_max_w − dissipation_i`),
    /// negative entry ⇒ footprint under-rates this tile by `|margin|` W. The kwavers-side
    /// 4th [`crate::validate::Check`] against `KWVERS_MIN_RESISTOR_MARGIN_W` is the sole
    /// gatekeeper (no longer redundant — it can actually fail now); the consumer reads the
    /// signed magnitude AND the headroom to plan footprint bumps (`Smd2512 ⇒ Smd4527`)
    /// on a per-tile basis, or matching-cap tightening, without re-deriving
    /// `per_tile_resistor_w[i]`. The constraint remains
    /// `new_footprint_max_w ≥ dissipation_i`; `resistor_margin_w[i]` quantifies the
    /// signed slack on it (negative entries are an explicit actionable signal). Surface
    /// consumed verbatim by the kwavers substitute via the `TODO(kwavers-transducer)`
    /// marker at the in-crate physics block.
    pub resistor_margin_w: Vec<f64>,
}

/// Build the typed pre-step from a verified full-stack v2 manifest + its
/// [`EnergyBudgetReport`]. Returns an `Err` if:
/// * the manifest is not the full-stack v2 shape (`is_full_stack_v2()` is
///   false: missing 96-lane binding, missing 4 tile profiles, or carrying
///   a legacy single-stim block), or
/// * the manifest's `tx_nets.len()` does not match `budget.lanes` (defence
///   against a stale budget crossed against a newer manifest).
#[must_use = "this function returns a Result; the Err case must be handled"]
pub fn manifest_to_kwavers_beam_step(
    manifest: &DriverManifest,
    budget: &EnergyBudgetReport,
) -> Result<KwaversBeamStep, String> {
    if !manifest.is_full_stack_v2() {
        return Err(format!(
            "kwavers pre-step requires full-stack v2 (96 lanes, 4 tiles, no legacy stim); \
             got {} lanes, {} tile profiles, stim={}",
            manifest.tx_nets.len(),
            manifest.tile_profiles.len(),
            if manifest.stimulation.is_some() {
                "Some"
            } else {
                "None"
            }
        ));
    }
    let lanes = manifest.tx_nets.len();
    if budget.lanes != lanes {
        return Err(format!(
            "kwavers pre-step manifest/budget lane mismatch: manifest={} budget={}",
            lanes, budget.lanes
        ));
    }
    let pitch = pitch_from_aperture_m(manifest.aperture_m, lanes);
    let lambda = wavelength_m(manifest.sound_speed_m_s, manifest.frequency_hz);
    let f_num = f_number(manifest.focal_m, manifest.aperture_m);
    if !pitch.is_finite() || !lambda.is_finite() || !f_num.is_finite() {
        return Err(format!(
            "kwavers pre-step non-finite geometry: pitch={pitch:.3e} lambda={lambda:.3e} f_number={f_num:.3e}"
        ));
    }
    Ok(KwaversBeamStep {
        lanes,
        aperture_m: manifest.aperture_m,
        frequency_hz: manifest.frequency_hz,
        sound_speed_m_s: manifest.sound_speed_m_s,
        focal_m: manifest.focal_m,
        timing_step_s: manifest.timing_step_s,
        pitch_m: pitch,
        wavelength_m: lambda,
        f_number: f_num,
        resistor_margin_w: budget.per_tile_resistor_margin_w.clone(),
    })
}

/// Output of the kwavers-side pre-step: every scalar kwavers would emit on a
/// real propagation call, plus a [`PhysicsReport`] that aggregates the
/// pruning-checks against kwavers-grade safety bounds (1 MPa transduction
/// floor, MI 10 cavitation ceiling, ±90° grating-lobe-free).
///
/// Today the predictions are smoke-tested from in-crate physics. When
/// kwavers-transducer is wired in, the same struct shape applies — the
/// kwavers consumer fills the same scalars (probably more accurately).
#[derive(Debug, Clone, PartialEq)]
pub struct KwaversBeamValidation {
    /// The pre-step kwavers was given (carried through for traceability so
    /// a downstream auditor can cross-check coherence with the sidecar).
    pub step: KwaversBeamStep,
    /// Estimated focal pressure (Pa) — coherent `N`-fold sum × per-element
    /// current × article-anchored acoustic sensitivity. For v2 article-class
    /// settings this peaks around 10–15 MPa; kwavers will refine it.
    pub focal_pressure_pa: f64,
    /// True iff the element pitch is grating-lobe-free over the full ±90°
    /// steering range (`max_grating_free_steer_deg ≥ 89°`). Article-class
    /// half-wavelength pitch ⇒ 90°.
    pub grating_lobe_free: bool,
    /// True iff the focus lies *beyond* the near-field distance (Fraunhofer
    /// regime). Information-only: focused-beam operation typically runs in
    /// the near-field (`focal_m < N`) and that is not a defect.
    pub in_far_field: bool,
    /// Spatial-peak pulse-average intensity (W/cm²) at the focus.
    pub isppa_w_cm2: f64,
    /// Mechanical Index at the focus = `p_focal_mpa / √f_mhz`.
    pub mechanical_index: f64,
    /// 6 dB axial extent (mm) proxy = `2 · f_number · λ` — the focused-beam
    /// axial intensity half-width on a uniform-illumination model.
    pub axial_extent_mm: f64,
    /// 6 dB lateral extent (mm) proxy = `λ · f_number` — upper-bound
    /// analytical single-element × full-array projection.
    pub lateral_extent_mm: f64,
    /// Per-tile resistor power margin (W). Mirrors [`KwaversBeamStep::resistor_margin_w`] —
    /// a duplicate field is intentional so the kwavers consumer can read the margins off
    /// the validation report (which carries the [`crate::physics::acoustic`] predictions) without
    /// having to walk the [`Self::step`] field again. SIGNED (see the `step` mirror's
    /// doc for full semantics); the 4th [`crate::validate::Check`] against
    /// `KWVERS_MIN_RESISTOR_MARGIN_W` is the sole gatekeeper.
    pub resistor_margin_w: Vec<f64>,
    /// All kwavers-derivable physics checks aggregated as a [`PhysicsReport`].
    pub report: PhysicsReport,
}

/// Per-element current (A) for the v2 stack — `peak_i_a` divided by the
/// 24-channel tile count. The kwavers-side call would use the same
/// per-element figure (it's the right unit for element-by-element arrays).
#[must_use]
fn per_element_peak_i_a(budget: &EnergyBudgetReport) -> f64 {
    budget.peak_i_a / (CHANNELS_PER_TILE_V2 as f64)
}

/// Estimated focal pressure (Pa) at the v2 stack's focus. Documented as:
/// `focal_pressure_gain(N) × per_element_peak_i × article_sensitivity`,
/// which is the coherent `N`-fold sum × article-anchored per-element
/// acoustic sensitivity. For v2 article-class settings this peaks ~10–15 MPa;
/// the kwavers-side refinement will substitute an actual simulated value.
#[must_use]
fn estimate_focal_pressure_pa(budget: &EnergyBudgetReport, lanes: usize) -> f64 {
    focal_pressure_gain(lanes)
        * per_element_peak_i_a(budget)
        * KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA
}

/// Validate a full-stack v2 manifest + its [`EnergyBudgetReport`] against
/// the kwavers-side pre-step:
///
/// 1. Re-assert the gate (`is_full_stack_v2`) — defence in depth against a
///    hand-constructed or stale manifest crossing the seam.
/// 2. Build the typed pre-step ([`KwaversBeamStep`]) the kwavers consumer
///    reads verbatim.
/// 3. **TODO(kwavers-transducer) anchor**: replace the in-crate physics
///    block with `crate::kwavers_transducer::simulate(&step) -> PressureMap`
///    once the kwavers dependency is available.
/// 4. Surface predictions + [`PhysicsReport`] aggregation to the caller.
///
/// Returns `Err` if the manifest is not full-stack v2 or the geometry is
/// non-physical (zero/NaN/inf aperture/frequency/sound-speed).
#[must_use = "this function returns a Result; the Err case must be handled"]
pub fn validate_against_budget(
    manifest: &DriverManifest,
    budget: &EnergyBudgetReport,
) -> Result<KwaversBeamValidation, String> {
    // 1. Re-assert the gate the validator already enforced. Defence in
    //    depth at this seam: the kwavers consumer trusts whatever the
    //    validate_against_budget caller hands in, and the seam must not
    //    silently accept a hand-built or stale manifest.
    if !manifest.is_full_stack_v2() {
        return Err(format!(
            "validate_against_budget requires full-stack v2 manifest (got {} lanes, {} tiles, stim={})",
            manifest.tx_nets.len(),
            manifest.tile_profiles.len(),
            if manifest.stimulation.is_some() { "Some" } else { "None" }
        ));
    }

    // 2. Build the typed pre-step kwavers reads verbatim.
    let step = manifest_to_kwavers_beam_step(manifest, budget)?;

    // 3. Acoustic + safety predictions using in-crate physics. The
    //    smoke-test today; replace with the real kwavers call below.
    // TODO(kwavers-transducer): replace with
    //       `crate::kwavers_transducer::simulate(&step) -> PressureMap`
    //       and re-derive `focal_pressure_pa` / `mechanical_index` /
    //       `isppa_w_cm2` from `pressure_map.focus_pressure_pa()` etc.
    //       The kwavers-side substitute should ALSO read `step.resistor_margin_w[i]`
    //       (SIGNED — positive = headroom, negative = under-rate magnitude) to size
    //       the next-package bump (`Smd2512 ⇒ Smd4527` etc.) and to plan matching-cap
    //       tightening WITHOUT re-deriving `per_tile_resistor_w[i]`. The margin vector
    //       replaced the upstream rejection pathway when the inline gate was lifted
    //       out of `validate_v2_energy_budget`, so negative entries are now an explicit
    //       signal the consumer must act on (bump package or retune cap) — they no
    //       longer vanish at the validate boundary.
    let focal_pressure_pa = estimate_focal_pressure_pa(budget, step.lanes);
    // ISPPA (W/cm²) = p² / Z₀ in water-impedance units, divided by
    // `UNIT_W_CM2_PER_W_M2` to convert from W/m² to the safety-meter W/cm².
    let isppa_w_cm2 =
        acoustic_intensity_w_per_m2(focal_pressure_pa, PHYSICS_WATER_Z0_RAYL) / UNIT_W_CM2_PER_W_M2;
    // Mechanical Index = p_mpa / √f_mhz; `mechanical_index` takes scaled inputs.
    let mi = mechanical_index(
        focal_pressure_pa / UNIT_PA_PER_MPA,
        step.frequency_hz / UNIT_MHZ_PER_HZ,
    );
    // Grating-lobe-free iff the pitch allows full ±90° steering within
    // 1° tolerance. Article-class λ/2 gives exactly 90°; v2's wider aperture
    // remains ≲ λ/2 ⇒ still grating-lobe-free at the article's 28°-per-side
    // aperture limit (4.3 mm × 95/15 = 27.2 mm ⇒ pitch 0.286 mm ≪ λ/2 0.385 mm).
    let grating_lobe_free = max_grating_free_steer_deg(step.pitch_m, step.wavelength_m)
        >= KWVERS_MIN_GRATING_FREE_STEER_DEG;
    // Far-field test (Fraunhofer): focus beyond N = D²/(4λ). v2 stack
    // operates in the focused near-field (focal_m = 10 mm << D²/(4λ) ~
    // 240 mm at λ=0.77 mm); this is informational, not a defect.
    let n_far = near_field_distance_m(step.aperture_m, step.wavelength_m);
    let in_far_field = step.focal_m >= n_far;
    // 6 dB intensity extents — analytical uniform-illumination proxies.
    // (Real extents tighten with element factor and array tapering; the
    //  kwavers consumer refines.) `UNIT_MM_PER_M` makes the m → mm conversion
    //  an explicit SI-prefix constant rather than an inline `1.0e3`.
    let axial_extent_mm = 2.0 * step.f_number * step.wavelength_m * UNIT_MM_PER_M;
    let lateral_extent_mm = step.wavelength_m * step.f_number * UNIT_MM_PER_M;
    // Per-tile minimum resistor margin (W) — the worst-case slack the chosen footprint has
    // over its IPC-7351 rating on the binding tile. This is the figure the kwavers-side
    // safety check aggregates into a single bound; kwavers-side consumes both the per-tile
    // vector (on `step` and on `self`) and this min scalar. The `f64::INFINITY` → `0.0`
    // fallback below is intentionally defensive (seam-contract defense-in-depth) — the
    // upstream `tile_profiles.len() != 4` and `lanes != manifest.tx_nets.len()` gates in
    // `manifest_to_kwavers_beam_step` already preclude an empty margin vector under any
    // reachable execution today, but the fallback keeps the lower-bound comparison
    // well-defined if a future contributor relaxes those gates for a non-standard stack shape.
    let min_resistor_margin_w = step
        .resistor_margin_w
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let min_resistor_margin_w = if min_resistor_margin_w == f64::INFINITY {
        0.0
    } else {
        min_resistor_margin_w
    };

    // 4. Physics checks against article-grade AND safety limits.
    let report = PhysicsReport::new(vec![
        Check::lower(
            CHECK_FOCAL_PRESSURE_NAME,
            focal_pressure_pa,
            KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA,
            "Pa",
        ),
        Check::upper(CHECK_MI_NAME, mi, KWVERS_MI_CAVITATION_CEILING, ""),
        Check::lower(
            CHECK_GRATING_LOBE_NAME,
            if grating_lobe_free { 1.0 } else { 0.0 },
            1.0,
            "bool",
        ),
        // Footprint headroom lock: kwavers-side sees the per-tile min margin vs the
        // IPC-7351 70 °C ceiling. Always passes today (validator rejects over-rated),
        // locks the safety contract at the seam so a future contributor who removes the
        // upstream rejection gate is caught at this Check.
        Check::lower(
            CHECK_RESISTOR_MARGIN_NAME,
            min_resistor_margin_w,
            KWVERS_MIN_RESISTOR_MARGIN_W,
            "W",
        ),
    ]);

    // Clone the per-tile margin out of `step` before the struct-literal move consumes `step`.
    // The Vec is duplicated onto `KwaversBeamValidation::resistor_margin_w` so the kwavers
    // consumer can read the margin off the validation report directly, without walking the
    // nested `validation.step.resistor_margin_w` field.
    let step_resistor_margin_w = step.resistor_margin_w.clone();
    Ok(KwaversBeamValidation {
        step,
        focal_pressure_pa,
        grating_lobe_free,
        in_far_field,
        isppa_w_cm2,
        mechanical_index: mi,
        axial_extent_mm,
        lateral_extent_mm,
        resistor_margin_w: step_resistor_margin_w,
        report,
    })
}
