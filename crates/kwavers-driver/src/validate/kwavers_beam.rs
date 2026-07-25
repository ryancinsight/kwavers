//! The driver→transducer beam-propagation pre-step adapter — the typed seam the downstream
//! `crates/kwavers-transducer` simulator consumes.
//!
//! [`KwaversBeamStep`] is the *typed pre-step contract*: physical fields are
//! Aequitas quantities and model flags remain scalars. With the `kwavers`
//! feature enabled, [`validate_against_budget`] sends that contract
//! into `kwavers-transducer` focused propagation. Without the feature, the same validation surface
//! remains available through the in-crate analytical fallback. Kwavers safety bounds, `Check` names,
//! water Z₀, and SI-prefix scalars live in [`crate::ssot`] (the SSOT ratchet locks each at its
//! engineering contract).

use crate::manifest::{DriverManifest, EnergyBudgetReport};
#[cfg(not(feature = "kwavers"))]
use crate::physics::acoustic::{
    acoustic_intensity_w_per_m2, f_number, focal_pressure_gain, max_grating_free_steer_deg,
    mechanical_index, near_field_distance_m, pitch_from_aperture_m, wavelength_m,
};
#[cfg(feature = "kwavers")]
use crate::physics::acoustic::{f_number, pitch_from_aperture_m, wavelength_m};
use crate::ssot::*;
use aequitas::systems::si::quantities::{
    Frequency, Intensity, Length, Power, Pressure, Time, Velocity,
};

use super::check::{Check, PhysicsReport};

/// Driver pre-step that kwavers-transducer consumes. Physical fields remain
/// Aequitas quantities through this public contract; scalar conversion occurs
/// at the manifest/numerical boundaries. The pre-step does NOT
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
    /// Aperture — first-to-last element centre span.
    pub aperture: Length,
    /// Drive frequency.
    pub frequency: Frequency,
    /// Medium sound speed.
    pub sound_speed: Velocity,
    /// Nominal focal depth.
    pub focal: Length,
    /// Hardware timing quantum.
    pub timing_step: Time,
    /// Centre-to-centre element pitch.
    pub pitch: Length,
    /// Acoustic wavelength.
    pub wavelength: Length,
    /// f-number of the focused aperture = `focal_m / aperture_m`.
    pub f_number: f64,
    /// Per-tile resistor power margin (W) under the chosen footprint's IPC-7351 70 °C rating.
    /// Converted from [`crate::manifest::EnergyBudgetReport::per_tile_resistor_margin_w`]
    /// — SIGNED after the inline rejection gate was lifted out of `validate_v2_energy_budget`:
    /// positive entry ⇒ headroom above the dissipation (`chosen_max_w − dissipation_i`),
    /// negative entry ⇒ footprint under-rates this tile by `|margin|` W. The kwavers-side
    /// 4th [`crate::validate::Check`] against `KWVERS_MIN_RESISTOR_MARGIN_W` is the sole
    /// gatekeeper (no longer redundant — it can actually fail now); the consumer reads the
    /// signed magnitude AND the headroom to plan footprint bumps (`Smd2512 ⇒ Smd4527`)
    /// on a per-tile basis, or matching-cap tightening, without re-deriving
    /// `per_tile_resistor_w`i``. The constraint remains
    /// `new_footprint_max_w ≥ dissipation_i`; `resistor_margin` quantifies the
    /// signed slack on it (negative entries are an explicit actionable signal).
    pub resistor_margin: Vec<Power>,
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
        aperture: Length::from_base(manifest.aperture_m),
        frequency: Frequency::from_base(manifest.frequency_hz),
        sound_speed: Velocity::from_base(manifest.sound_speed_m_s),
        focal: Length::from_base(manifest.focal_m),
        timing_step: Time::from_base(manifest.timing_step_s),
        pitch: Length::from_base(pitch),
        wavelength: Length::from_base(lambda),
        f_number: f_num,
        resistor_margin: budget
            .per_tile_resistor_margin_w
            .iter()
            .copied()
            .map(Power::from_base)
            .collect(),
    })
}

/// Output of the kwavers-side pre-step: every scalar kwavers would emit on a
/// real propagation call, plus a [`PhysicsReport`] that aggregates the
/// pruning-checks against kwavers-grade safety bounds (1 MPa transduction
/// floor, MI 10 cavitation ceiling, ±90° grating-lobe-free).
///
/// With the `kwavers` feature enabled, these scalars come from
/// `kwavers-transducer` focused propagation. The default build preserves the
/// same struct shape through the analytical fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct KwaversBeamValidation {
    /// The pre-step kwavers was given (carried through for traceability so
    /// a downstream auditor can cross-check coherence with the sidecar).
    pub step: KwaversBeamStep,
    /// Estimated focal pressure (Pa) — coherent `N`-fold sum × per-element
    /// current × article-anchored acoustic sensitivity in the fallback path;
    /// feature-enabled builds use the propagated pressure at the focus.
    pub focal_pressure: Pressure,
    /// True iff the element pitch is grating-lobe-free over the full ±90°
    /// steering range (`max_grating_free_steer_deg ≥ 89°`). Article-class
    /// half-wavelength pitch ⇒ 90°.
    pub grating_lobe_free: bool,
    /// True iff the focus lies *beyond* the near-field distance (Fraunhofer
    /// regime). Information-only: focused-beam operation typically runs in
    /// the near-field (`focal_m < N`) and that is not a defect.
    pub in_far_field: bool,
    /// Spatial-peak pulse-average intensity at the focus.
    pub isppa: Intensity,
    /// Mechanical Index at the focus = `p_focal_mpa / √f_mhz`.
    pub mechanical_index: f64,
    /// 6 dB axial extent proxy = `2 · f_number · λ` — the focused-beam
    /// axial intensity half-width on a uniform-illumination model.
    pub axial_extent: Length,
    /// 6 dB lateral extent proxy = `λ · f_number` — upper-bound
    /// analytical single-element × full-array projection.
    pub lateral_extent: Length,
    /// Per-tile resistor power margin. Mirrors [`KwaversBeamStep::resistor_margin`] —
    /// a duplicate field is intentional so the kwavers consumer can read the margins off
    /// the validation report (which carries the [`crate::physics::acoustic`] predictions) without
    /// having to walk the [`Self::step`] field again. SIGNED (see the `step` mirror's
    /// doc for full semantics); the 4th [`crate::validate::Check`] against
    /// `KWVERS_MIN_RESISTOR_MARGIN_W` is the sole gatekeeper.
    pub resistor_margin: Vec<Power>,
    /// All kwavers-derivable physics checks aggregated as a [`PhysicsReport`].
    pub report: PhysicsReport,
}

/// Per-element current (A) for the v2 stack — `peak_i_a` divided by the
/// 24-channel tile count. The kwavers-side call uses the same per-element
/// figure (it's the right unit for element-by-element arrays).
#[must_use]
fn per_element_peak_i_a(budget: &EnergyBudgetReport) -> f64 {
    budget.peak_i_a / (CHANNELS_PER_TILE_V2 as f64)
}

#[derive(Debug, Clone, PartialEq)]
struct BeamPropagationScalars {
    focal_pressure: Pressure,
    grating_lobe_free: bool,
    in_far_field: bool,
    isppa: Intensity,
    mechanical_index: f64,
    axial_extent: Length,
    lateral_extent: Length,
}

/// Estimated focal pressure (Pa) at the v2 stack's focus. Documented as:
/// `focal_pressure_gain(N) × per_element_peak_i × article_sensitivity`,
/// which is the coherent `N`-fold sum × article-anchored per-element
/// acoustic sensitivity. For v2 article-class settings this peaks ~10–15 MPa;
/// the kwavers-side refinement will substitute an actual simulated value.
#[cfg(not(feature = "kwavers"))]
#[must_use]
fn estimate_focal_pressure_pa(budget: &EnergyBudgetReport, lanes: usize) -> f64 {
    focal_pressure_gain(lanes)
        * per_element_peak_i_a(budget)
        * KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA
}

#[cfg(not(feature = "kwavers"))]
fn propagate_beam_step(
    step: &KwaversBeamStep,
    budget: &EnergyBudgetReport,
) -> Result<BeamPropagationScalars, String> {
    let focal_pressure_pa = estimate_focal_pressure_pa(budget, step.lanes);
    let isppa_w_m2 = acoustic_intensity_w_per_m2(focal_pressure_pa, PHYSICS_WATER_Z0_RAYL);
    let mi = mechanical_index(
        focal_pressure_pa / UNIT_PA_PER_MPA,
        step.frequency.into_base() / UNIT_MHZ_PER_HZ,
    );
    let grating_lobe_free =
        max_grating_free_steer_deg(step.pitch.into_base(), step.wavelength.into_base())
            >= KWVERS_MIN_GRATING_FREE_STEER_DEG;
    let n_far = near_field_distance_m(step.aperture.into_base(), step.wavelength.into_base());
    Ok(BeamPropagationScalars {
        focal_pressure: Pressure::from_base(focal_pressure_pa),
        grating_lobe_free,
        in_far_field: step.focal.into_base() >= n_far,
        isppa: Intensity::from_base(isppa_w_m2),
        mechanical_index: mi,
        axial_extent: Length::from_base(2.0 * step.f_number * step.wavelength.into_base()),
        lateral_extent: Length::from_base(step.wavelength.into_base() * step.f_number),
    })
}

#[cfg(feature = "kwavers")]
fn propagate_beam_step(
    step: &KwaversBeamStep,
    budget: &EnergyBudgetReport,
) -> Result<BeamPropagationScalars, String> {
    use aequitas::systems::si::quantities::{
        AcousticImpedance, ElectricCurrent, Length, PressurePerElectricCurrent,
    };
    use kwavers_transducer::{
        ApertureDesignSpec, CartesianPosition, ChannelWiring, DEFAULT_KERF_FRACTION,
        FocusedLinearArrayPropagationSpec, design_array, propagate_focused_linear_array,
    };

    let pitch_fraction = (step.pitch.into_base() / step.wavelength.into_base()).clamp(1e-9, 2.0);
    let design = design_array(&ApertureDesignSpec {
        aperture_x: Length::from_base(0.0),
        aperture_y: Length::from_base(step.aperture.into_base() + step.pitch.into_base()),
        frequency: step.frequency,
        sound_speed: step.sound_speed,
        max_pitch_fraction: pitch_fraction,
        kerf_fraction: DEFAULT_KERF_FRACTION,
        wiring: ChannelWiring::ColumnsAsChannels,
    })
    .map_err(|e| format!("kwavers-transducer design_array: {e}"))?;
    let map = propagate_focused_linear_array(&FocusedLinearArrayPropagationSpec {
        design,
        center: CartesianPosition::from_base([0.0, 0.0, 0.0])
            .map_err(|e| format!("CartesianPosition center: {e}"))?,
        focus: CartesianPosition::from_base([0.0, 0.0, step.focal.into_base()])
            .map_err(|e| format!("CartesianPosition focus: {e}"))?,
        frequency: step.frequency,
        sound_speed: step.sound_speed,
        per_channel_peak_current: ElectricCurrent::from_base(per_element_peak_i_a(budget)),
        pressure_per_current: PressurePerElectricCurrent::from_base(
            KWVERS_ARTICLE_FOCAL_PRESSURE_PER_AMP_PA,
        ),
        acoustic_impedance: AcousticImpedance::from_base(PHYSICS_WATER_Z0_RAYL),
    })
    .map_err(|e| format!("kwavers-transducer propagation: {e}"))?;
    Ok(BeamPropagationScalars {
        focal_pressure: map.focal_pressure,
        grating_lobe_free: map.grating_lobe_free,
        in_far_field: map.in_far_field,
        isppa: map.isppa,
        mechanical_index: map.mechanical_index,
        axial_extent: map.axial_extent,
        lateral_extent: map.lateral_extent,
    })
}

/// Validate a full-stack v2 manifest + its [`EnergyBudgetReport`] against
/// the kwavers-side pre-step:
///
/// 1. Re-assert the gate (`is_full_stack_v2`) — defence in depth against a
///    hand-constructed or stale manifest crossing the seam.
/// 2. Build the typed pre-step ([`KwaversBeamStep`]) the kwavers consumer
///    reads verbatim.
/// 3. Propagate the focused beam through `kwavers-transducer` when the
///    `kwavers` feature is enabled, otherwise use the analytical fallback.
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
            if manifest.stimulation.is_some() {
                "Some"
            } else {
                "None"
            }
        ));
    }

    // 2. Build the typed pre-step kwavers reads verbatim.
    let step = manifest_to_kwavers_beam_step(manifest, budget)?;

    // 3. Acoustic + safety predictions from the selected propagation backend.
    let propagated = propagate_beam_step(&step, budget)?;
    // Per-tile minimum resistor margin (W) — the worst-case slack the chosen footprint has
    // over its IPC-7351 rating on the binding tile. This is the figure the kwavers-side
    // safety check aggregates into a single bound; kwavers-side consumes both the per-tile
    // vector (on `step` and on `self`) and this min scalar. The `f64::INFINITY` → `0.0`
    // fallback below is intentionally defensive (seam-contract defense-in-depth) — the
    // upstream `tile_profiles.len() != 4` and `lanes != manifest.tx_nets.len()` gates in
    // `manifest_to_kwavers_beam_step` already preclude an empty margin vector under any
    // reachable execution today, but the fallback keeps the lower-bound comparison
    // well-defined if a future contributor relaxes those gates for a non-standard stack shape.
    let min_resistor_margin = step
        .resistor_margin
        .iter()
        .map(|margin| margin.into_base())
        .fold(f64::INFINITY, f64::min);
    let min_resistor_margin = if min_resistor_margin == f64::INFINITY {
        Power::from_base(0.0)
    } else {
        Power::from_base(min_resistor_margin)
    };

    // 4. Physics checks against article-grade AND safety limits.
    let report = PhysicsReport::new(vec![
        Check::lower(
            CHECK_FOCAL_PRESSURE_NAME,
            propagated.focal_pressure.into_base(),
            KWVERS_MIN_FOCAL_PRESSURE_1MPA_IN_PA,
            "Pa",
        ),
        Check::upper(
            CHECK_MI_NAME,
            propagated.mechanical_index,
            KWVERS_MI_CAVITATION_CEILING,
            "",
        ),
        Check::lower(
            CHECK_GRATING_LOBE_NAME,
            if propagated.grating_lobe_free {
                1.0
            } else {
                0.0
            },
            1.0,
            "bool",
        ),
        // Footprint headroom lock: kwavers-side sees the per-tile min margin vs the
        // IPC-7351 70 °C ceiling. Always passes today (validator rejects over-rated),
        // locks the safety contract at the seam so a future contributor who removes the
        // upstream rejection gate is caught at this Check.
        Check::lower(
            CHECK_RESISTOR_MARGIN_NAME,
            min_resistor_margin.into_base(),
            KWVERS_MIN_RESISTOR_MARGIN_W,
            "W",
        ),
    ]);

    // Clone the per-tile margin out of `step` before the struct-literal move consumes `step`.
    // The Vec is duplicated onto `KwaversBeamValidation::resistor_margin` so the kwavers
    // consumer can read the margin off the validation report directly, without walking the
    // nested `validation.step.resistor_margin` field.
    let step_resistor_margin = step.resistor_margin.clone();
    Ok(KwaversBeamValidation {
        step,
        focal_pressure: propagated.focal_pressure,
        grating_lobe_free: propagated.grating_lobe_free,
        in_far_field: propagated.in_far_field,
        isppa: propagated.isppa,
        mechanical_index: propagated.mechanical_index,
        axial_extent: propagated.axial_extent,
        lateral_extent: propagated.lateral_extent,
        resistor_margin: step_resistor_margin,
        report,
    })
}
