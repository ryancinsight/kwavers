//! Adversarial DFM / physics critic.
//!
//! The optimiser (placer + router) minimises a cost; this critic plays the adversary, attacking a
//! routed board to surface the weaknesses the cost did not capture, and emitting a per-region
//! **weakness field** that the next placement is biased away from. Iterating the two is the min-max
//! (adversarial) loop: generate a layout, attack it, repair, repeat.
//!
//! Attack axes, each grounded in standard PCB engineering:
//! * **Lane crossings** — proper crossings of different nets' flight lines (MST topology). Every
//!   topological crossing must be resolved by a layer change, so fewer crossings ⇒ fewer vias and
//!   less congestion (classic routability metric).
//! * **Clearance violations** — different-net copper closer than the manufacturing rule permits.
//! * **Near-short / fault risk** — different-net copper closer than a wider risk margin (a graded
//!   signal beyond binary DRC), weighted higher across the HV↔LV boundary where a short is
//!   catastrophic.
//! * **Crosstalk** — long parallel adjacent runs of different nets couple capacitively/inductively
//!   (coupling ∝ length / spacing); HV↔LV adjacency is doubly penalised (crosstalk + creepage).
//! * **Antenna / dangling** — a track end not landing on a pad, via, or another track is an
//!   etch/ESD antenna and a likely open fault.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::board::{
    split_domain_from_name, Board, LayerId, NetClassKind, NetId, SplitDomain, Track, ZoneFill,
};
use crate::geom::{
    dist_point_seg, dist_seg_seg, distance_to_polygon_boundary, point_in_polygon, segments_cross,
    GridSpec, Nm, Point,
};
use crate::place::component::is_surge_suppressor_refdes;
use crate::place::{Component, CongestionField, FootprintDef, Role};
use crate::rules::DesignRules;
use crate::verify::{parasitic_ac_coupling_check, schematic_isolation_bfs};

/// Structured result of an adversarial audit.
#[derive(Debug, Clone, Default)]
pub struct FaultReport {
    /// Proper crossings between different nets' flight lines.
    pub crossings: usize,
    /// Different-net copper features inside the manufacturing clearance rule.
    pub clearance_violations: usize,
    /// Different-net copper features within the soft near-short risk margin.
    pub near_shorts: usize,
    /// Parallel-adjacent different-net track pairs (crosstalk-prone).
    pub crosstalk: usize,
    /// Different-net via pairs too close to hold annular-ring clearance (drill spacing fault).
    pub via_adjacency: usize,
    /// Acute-angle (`< 90°`) junctions between connected same-net segments — etch acid-trap sites.
    pub acid_traps: usize,
    /// Total drilled vias — each is a plating/registration defect site and a per-via fab cost; a
    /// secondary objective the optimiser minimises among equally clean boards.
    pub via_count: usize,
    /// Track ends not landing on a pad/via/other track (antenna / open-fault risk).
    pub dangling: usize,
    /// Galvanic isolation boundary violations.
    pub isolation_violations: usize,
    /// Parasitic AC coupling violations.
    pub ac_coupling_violations: usize,
    /// Connected same-net tracks meeting at exactly 90 degrees (sharp bend).
    pub sharp_bends: usize,
    /// Same-layer different-net track segments that physically cross between grid nodes.
    pub track_crossing_violations: usize,
    /// Drilled holes (via barrels) whose edge is closer than the hole-to-copper clearance to a
    /// **foreign-net** track or pad on a layer the barrel passes through. Mirrors kicad-cli's
    /// `hole_clearance` class — previously unmodelled, so escape-via boards passed the internal audit
    /// while failing the external oracle.
    pub hole_clearance_violations: usize,
    /// Adjacent parallel same-net segments closer than 4W.
    pub serpentine_spacing_violations: usize,
    /// Routing segments with length less than 1.5W.
    pub serpentine_length_violations: usize,
    /// Serpentine length-compensation runs placed farther than the configured bend-root distance.
    pub serpentine_compensation_distance_violations: usize,
    /// Different-net via pad outer boundary spacing closer than 15 mils.
    pub via_spacing_violations: usize,
    /// Same-net non-ground via pad outer boundary spacing closer than 15 mils, creating plane hot spots.
    pub plane_hotspot_via_spacing_violations: usize,
    /// Vias/pads/components placed inside a differential pair's parallel trace corridor, or large coupling cap packages.
    pub diff_pair_violations: usize,
    /// Differential-pair members routed on different copper-layer sets.
    pub diff_pair_layer_mismatch_violations: usize,
    /// Differential pairs in one indexed interface routed on different copper-layer sets.
    pub diff_pair_interface_layer_mismatch_violations: usize,
    /// Differential pairs in one indexed interface using different total routed via counts.
    pub diff_pair_interface_via_count_mismatch_violations: usize,
    /// Differential-pair members using different routed via counts.
    pub diff_pair_via_count_violations: usize,
    /// Differential-pair members whose routed lengths differ by more than the configured tolerance.
    pub diff_pair_length_mismatch_violations: usize,
    /// Differential-pair members whose via-delimited routed segments differ by more than tolerance.
    pub diff_pair_segment_length_mismatch_violations: usize,
    /// Indexed parallel-bus nets whose routed lengths exceed the configured group skew budget.
    pub parallel_bus_length_mismatch_violations: usize,
    /// Differential-pair members whose routed P/N spacing varies beyond the configured tolerance.
    pub diff_pair_spacing_variation_violations: usize,
    /// Differential-pair vias whose P/N station placement is not symmetric.
    pub diff_pair_via_symmetry_violations: usize,
    /// Differential-pair AC-coupling capacitors whose P/N placements are not symmetric.
    pub diff_pair_coupling_cap_symmetry_violations: usize,
    /// Differential-pair AC-coupling capacitors whose package is larger than the configured 0603-class budget.
    pub diff_pair_coupling_cap_package_violations: usize,
    /// Differential-pair power-reference stitching capacitors whose P/N placements are not symmetric.
    pub diff_pair_stitching_cap_symmetry_violations: usize,
    /// Differential-pair pad-entry breakout distances whose P/N lengths are not symmetric.
    pub diff_pair_pad_entry_mismatch_violations: usize,
    /// Differential-pair pad-entry breakouts whose absolute uncoupled length exceeds the budget.
    pub diff_pair_pad_entry_length_violations: usize,
    /// Aggregate *excess* length-mismatch (mm) summed over every routed diff pair — the
    /// per-mm diff-pair-tolerance fee that [`FaultReport::risk_score`] folds in at 60.0/mm. Lets
    /// the co-optimiser distinguish a 0.6 mm intra-pair skew (acceptable on a long route,
    /// marginal fee) from a 6 mm skew (catastrophic) without depending solely on a hard count.
    pub diff_pair_total_length_mismatch_mm: f64,
    /// Charge-recycling not enabled on a board with N-level pulser ICs.
    pub charge_recycling_violations: usize,
    /// Pulse-skipping pressure error exceeding 5% tolerance.
    pub pulse_skip_violations: usize,
    /// Unrelated signal copper routed inside a differential-pair keepout corridor.
    pub diff_pair_keepout_violations: usize,
    /// High-speed tracks routed closer to the board edges than high_speed_edge_clearance.
    pub high_speed_edge_violations: usize,
    /// Active high-speed IC courtyards placed inside the component edge keepout.
    pub high_speed_component_edge_violations: usize,
    /// High-speed termination resistors placed too far from an active IC pad on their high-speed net.
    pub high_speed_termination_placement_violations: usize,
    /// High-speed tracks crossing a split plane boundary without a nearby stitching capacitor.
    pub split_plane_crossings: usize,
    /// High-speed tracks inside a reference zone but closer than 3W to that zone's boundary.
    pub reference_plane_margin_violations: usize,
    /// High-speed tracks without adjacent-layer ground/power reference-plane coverage.
    pub reference_plane_absence_violations: usize,
    /// Inner-layer high-speed tracks without ground planes on both adjacent layers.
    pub inner_layer_dual_ground_reference_violations: usize,
    /// High-speed tracks using only a power-plane reference without endpoint stitching capacitors.
    pub power_reference_stitching_cap_violations: usize,
    /// Non-plane signal tracks routed through a ground/power reference-plane zone on the same layer.
    pub reference_plane_intrusion_violations: usize,
    /// Ground reference-plane layers split into multiple same-net pour islands.
    pub ground_plane_fragmentation_violations: usize,
    /// Analog/digital signals routed over the opposite split ground-domain plane.
    pub split_domain_reference_violations: usize,
    /// Analog/digital signal tracks whose return currents overlap on the same ground reference zone.
    pub mixed_domain_shared_reference_violations: usize,
    /// Analog/digital signal tracks crossing the inferred virtual split-domain boundary.
    pub virtual_split_crossing_violations: usize,
    /// High-speed routed copper branch nodes that form stub/T topologies instead of daisy chains.
    pub high_speed_stub_violations: usize,
    /// High-speed layer-transition vias without a nearby ground transition via.
    pub high_speed_transition_ground_via_violations: usize,
    /// Differential-pair layer-transition ground vias whose stations are not symmetric.
    pub diff_pair_transition_ground_via_symmetry_violations: usize,
    /// High-speed source/sink pads without nearby ground return copper.
    pub high_speed_terminal_ground_via_violations: usize,
    /// High-speed signal vias placed too far from any same-net pad.
    pub high_speed_via_pad_proximity_violations: usize,
    /// High-speed vias whose outer diameter exceeds the selected via-size rule.
    pub high_speed_via_diameter_violations: usize,
    /// Blind/buried vias whose drill exceeds the blind/buried fabrication limit.
    pub blind_buried_via_drill_violations: usize,
    /// HDI laser micro-vias whose build-up dielectric aspect ratio (drill ÷ dielectric thickness)
    /// exceeds [`crate::rules::DesignRules::max_microvia_ar`] — the laser fabricator's
    /// manufacturing gate. Each offending [`crate::board::ViaKind::Micro`] via contributes 1 to
    /// this count, wrapping the per-board aggregate boolean in
    /// [`crate::validate::microvia_aspect_check`] into a per-via field strength so the
    /// optimiser's [`FaultReport::risk_score`] weights HDI reject rates per offending via, and
    /// so `dirty_fields_mirrors_every_hard_drc_clean_clause` fires when this field
    /// silently drifts to non-zero on a board whose `max_microvia_ar` was tightened below 1.0
    /// or whose build-up dielectric pushes the AR above the limit.
    pub microvia_aspect_violations: usize,
    /// Decoupling capacitor ground pads without a nearby ground via.
    pub decoupling_ground_via_violations: usize,
    /// Decoupling capacitor power pads that cannot reach the associated IC power pin on a shared layer.
    pub decoupling_power_layer_violations: usize,
    /// Associated decoupling capacitors whose IC/cap commutation loop exceeds the area budget.
    pub decoupling_loop_area_violations: usize,
    /// Active IC power/ground pads without a same-net internal plane underneath.
    pub active_ic_power_plane_violations: usize,
    /// **Charge-reservoir sufficiency** violations — per active IC **or buck converter**, the sum of every
    /// associated decoupling capacitor's incremental drive current `I_k = C_k · dV/dt`
    /// over the board's switching window (via [`crate::physics::emi::capacitive_drive_current_a`],
    /// using board-uniform `dV`/`dt` from
    /// [`crate::rules::DesignRules::ic_switching_dv_v`] /
    /// [`crate::rules::DesignRules::ic_switching_risetime_s`]) failed to meet the active
    /// IC's datasheet `I_dd` rating set on
    /// [`crate::place::footprint::FootprintDef::i_dd_a`]. Each under-provisioned IC
    /// contributes 1, so [`FaultReport::risk_score`] (folded at 20.0 per violation —
    /// the fab-reject / rail-collapse tier that `microvia_aspect_violations` also sits
    /// on) learns to add caps rather than eke out a sub-budget solution. ICs whose
    /// `i_dd_a == 0.0` (no rating set) or whose board has `dv ≤ 0` /
    /// `risetime ≤ 0` are vacuous: the detector silently skips them. Caps whose
    /// `capacitance_f == 0.0` (no datasheet value set) contribute 0 A to the per-IC sum
    /// — leaving the violation intentionally loud when the library is incomplete.
    /// For `Role::Power` (buck, bulk cap) the check covers the input-side bulk
    /// reservoir — the convention is that `assoc_ic` ties only the caps supplying the
    /// buck's own input-side switching current draw `i_dd_a`, mirroring how [`crate::physics::emi::commutation_loops`]
    /// treats assoc-IC ties symmetrically (it is a geometric / loop-inductance
    /// measurement, not directional). Output-side caps should be sized against
    /// [`crate::physics::pdn::holdup_capacitance_f`] (`I·Δt/ΔV`) separately — they are not
    /// the buck's `i_dd_a` demand and would otherwise force this detector into a
    /// spurious under-provisioned verdict on a board whose output cap bank is
    /// sized correctly for its load step instead.
    pub charge_reservoir_violations: usize,
    /// High-speed vias whose physical barrel extends beyond the layers used by the signal.
    pub high_speed_via_stub_violations: usize,
    /// Unfilled vias placed directly inside non-ground SMD pads.
    pub unfilled_via_in_pad_violations: usize,
    /// Same-net vias placed in the connector-to-surge-suppressor clamp path.
    pub surge_suppressor_via_violations: usize,
    /// Unrelated high-speed parallel traces closer than the configured width-derived spacing.
    pub high_speed_parallel_spacing_violations: usize,
    /// Unrelated high-speed traces routed in parallel on adjacent copper layers.
    pub high_speed_adjacent_layer_parallel_violations: usize,
    /// Aggregate risk score (higher = worse); HV-involved faults weighted up.
    pub risk_score: f64,
    /// Board locations of the worst weaknesses (drives the feedback field).
    pub hotspots: Vec<Point>,
}

impl FaultReport {
    /// True iff every hard manufacturing, routing, and high-speed integrity DRC field is clean.
    #[must_use]
    pub fn hard_drc_clean(&self) -> bool {
        self.clearance_violations == 0
            && self.via_adjacency == 0
            && self.acid_traps == 0
            && self.dangling == 0
            && self.sharp_bends == 0
            && self.track_crossing_violations == 0
            && self.hole_clearance_violations == 0
            && self.serpentine_spacing_violations == 0
            && self.serpentine_length_violations == 0
            && self.serpentine_compensation_distance_violations == 0
            && self.via_spacing_violations == 0
            && self.plane_hotspot_via_spacing_violations == 0
            && self.diff_pair_violations == 0
            && self.diff_pair_layer_mismatch_violations == 0
            && self.diff_pair_interface_layer_mismatch_violations == 0
            && self.diff_pair_interface_via_count_mismatch_violations == 0
            && self.diff_pair_via_count_violations == 0
            && self.diff_pair_length_mismatch_violations == 0
            && self.diff_pair_segment_length_mismatch_violations == 0
            && self.parallel_bus_length_mismatch_violations == 0
            && self.diff_pair_spacing_variation_violations == 0
            && self.diff_pair_via_symmetry_violations == 0
            && self.diff_pair_coupling_cap_symmetry_violations == 0
            && self.diff_pair_coupling_cap_package_violations == 0
            && self.diff_pair_stitching_cap_symmetry_violations == 0
            && self.diff_pair_pad_entry_mismatch_violations == 0
            && self.diff_pair_pad_entry_length_violations == 0
            && self.diff_pair_keepout_violations == 0
            && self.high_speed_edge_violations == 0
            && self.high_speed_component_edge_violations == 0
            && self.high_speed_termination_placement_violations == 0
            && self.high_speed_parallel_spacing_violations == 0
            && self.high_speed_adjacent_layer_parallel_violations == 0
            && self.reference_plane_margin_violations == 0
            && self.reference_plane_absence_violations == 0
            && self.inner_layer_dual_ground_reference_violations == 0
            && self.power_reference_stitching_cap_violations == 0
            && self.reference_plane_intrusion_violations == 0
            && self.ground_plane_fragmentation_violations == 0
            && self.split_domain_reference_violations == 0
            && self.mixed_domain_shared_reference_violations == 0
            && self.virtual_split_crossing_violations == 0
            && self.high_speed_stub_violations == 0
            && self.high_speed_transition_ground_via_violations == 0
            && self.diff_pair_transition_ground_via_symmetry_violations == 0
            && self.high_speed_terminal_ground_via_violations == 0
            && self.high_speed_via_pad_proximity_violations == 0
            && self.high_speed_via_diameter_violations == 0
            && self.blind_buried_via_drill_violations == 0
            && self.microvia_aspect_violations == 0
            && self.decoupling_ground_via_violations == 0
            && self.decoupling_power_layer_violations == 0
            && self.decoupling_loop_area_violations == 0
            && self.active_ic_power_plane_violations == 0
            && self.charge_reservoir_violations == 0
            && self.high_speed_via_stub_violations == 0
            && self.unfilled_via_in_pad_violations == 0
            && self.surge_suppressor_via_violations == 0
            && self.split_plane_crossings == 0
    }
}

pub(crate) fn is_hv(board: &Board, net: NetId) -> bool {
    matches!(board.class_of(net), NetClassKind::Hv)
}
