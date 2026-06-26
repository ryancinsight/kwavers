//! Manufacturing and high-voltage design rules.
//!
//! These are the *manufacturing best practices* the engine routes to satisfy: a fabricator's
//! minimum track/clearance/annular-ring capability, plus the high-voltage **creepage** rule that
//! keeps HV copper away from low-voltage features. The router consumes them through
//! [`crate::cost::PhysicsCost`], which turns the creepage distance into a spatial gradient rather
//! than a post-route DRC check.

use crate::geom::Nm;

/// Via-construction technology the board is built with — the dominant cost/manufacturability lever
/// after layer count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViaPolicy {
    /// **Standard** stackup: every routed via is a single mechanically-drilled **through-hole**
    /// (F.Cu→B.Cu), the cheapest process. A via-in-pad escape is a filled+capped through-hole
    /// (VIPPO). This is the default — an ordinary 4-layer board.
    #[default]
    ThroughHole,
    /// **HDI** stackup: sequential build-up with laser micro-vias on the outer layer pairs and
    /// mechanical blind/buried vias in the core. The router's actual layer spans are honoured and
    /// classified, so dense (fine-pitch BGA) escapes route on fewer layers — at higher per-via cost.
    Hdi,
}

/// A fabricator capability / design-rule profile. Defaults follow common 4-layer process floors;
/// per-class track and clearance are selected by [`crate::board::NetClassKind`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DesignRules {
    /// Via-construction technology (through-hole vs HDI) — governs how routed vias are built/classified.
    pub via_policy: ViaPolicy,
    /// Smallest manufacturable track width.
    pub min_track: Nm,
    /// Smallest copper-to-copper clearance between different nets.
    pub min_clearance: Nm,
    /// Smallest via drill (hole) diameter.
    pub min_via_drill: Nm,
    /// Smallest annular ring (radial copper around a via hole).
    pub min_annular: Nm,
    /// **HDI micro-via** laser-drill diameter (build-up via between adjacent layers). Smaller than a
    /// mechanical drill; the aspect ratio (dielectric thickness / drill) is capped (`max_microvia_ar`).
    pub microvia_drill: Nm,
    /// Micro-via annular ring (laser vias hold a tighter ring than mechanical).
    pub microvia_annular: Nm,
    /// Maximum micro-via aspect ratio (layer-pair dielectric thickness ÷ drill) a laser process holds
    /// (≈ 0.75–1.0 typical); a manufacturability gate for HDI build-up vias.
    pub max_microvia_ar: f64,
    /// Maximum drill diameter for blind and buried vias. Section 7.4.1 lists 6 mil
    /// (150 µm) as the IPC limit for blind/buried vias.
    pub max_blind_buried_via_drill: Nm,
    /// Default signal track width.
    pub signal_track: Nm,
    /// Default high-voltage track width.
    pub hv_track: Nm,
    /// Default power track width.
    pub power_track: Nm,
    /// Minimum copper-to-board-edge clearance. Copper inside this margin risks exposure at the
    /// routed/V-scored outline and is a fab reject; a best-practice keepout ring.
    pub edge_clearance: Nm,
    /// Minimum component-courtyard spacing for assembly, inspection, rework, and 3D package
    /// envelope clearance. Independent of copper-to-copper clearance.
    pub assembly_clearance: Nm,
    /// Minimum space between the outer boundaries of two vias to avoid plane hot spots.
    pub min_via_to_via_spacing: Nm,
    /// Minimum board edge clearance for high-speed switching tracks.
    pub high_speed_edge_clearance: Nm,
    /// Minimum board-edge clearance for active ICs carrying high-speed nets. This is intentionally
    /// larger than ordinary component keep-in because section 7.2.2.2 of the high-speed guide calls
    /// for sensitive high-speed devices to stay away from board edges and toward the board centre.
    pub high_speed_component_edge_clearance: Nm,
    /// Maximum distance from a high-speed termination resistor pad to an active IC pad on the same
    /// high-speed net. Section 7.2.3 requires termination resistors to be placed with the relevant
    /// receiver/driver instead of squeezed in later; this local policy uses a 2 mm placement budget.
    pub high_speed_termination_distance: Nm,
    /// Minimum reference-plane copper margin on each side of a high-speed trace, expressed as a
    /// multiplier of that trace's routed width. The high-speed guide's 3W rule maps to `3.0`.
    pub high_speed_reference_plane_margin_widths: f64,
    /// Maximum distance from a high-speed layer-transition via to a ground transition via. The guide
    /// requires close stitching/ground transition vias; this project policy uses 2 mm as the local
    /// return-path search radius.
    pub high_speed_transition_ground_via_distance: Nm,
    /// Maximum distance from a high-speed source/sink pad to local ground copper. The guide calls
    /// for ground return vias at source and sink positions; this project policy reuses the 2 mm
    /// return-path search radius.
    pub high_speed_terminal_ground_via_distance: Nm,
    /// Maximum distance from a high-speed signal via to a same-net pad. Section 7.2.6 states that
    /// high-speed vias should be placed close to their respective pads to avoid parasitic
    /// capacitance; this project policy uses the same 2 mm local-placement budget.
    pub high_speed_via_pad_distance: Nm,
    /// Maximum distance from a decoupling capacitor's ground pad to a ground via. Section 7.3.1.1
    /// routes decoupling capacitors early and places their ground vias to control local loop
    /// inductance; this project policy uses a 1 mm via-placement budget.
    pub decoupling_ground_via_distance: Nm,
    /// Maximum commutation-loop area for an associated decoupling capacitor, in mm². Section 6.4.2.5
    /// ties trace-loop inductance to PDN impedance; this local policy keeps bypass loops in the
    /// few-nH regime by limiting the cap-to-IC V/G loop envelope.
    pub max_decoupling_loop_area_mm2: f64,
    /// Maximum distance from a signal entry/exit point to a capacitor tying a power-plane reference
    /// back to ground. Section 7.3.1.7 requires stitching/bypass capacitors at sink/source when a
    /// high-speed signal uses a power plane as its reference; this project policy uses 2 mm.
    pub power_reference_stitching_cap_distance: Nm,
    /// Maximum distance from a split-plane crossing point to a stitching capacitor. Section 7.3.1.7
    /// requires stitching capacitors close to the signal path when a split/obstruction must be
    /// crossed; this project policy uses a 2 mm local path budget.
    pub split_plane_stitching_cap_distance: Nm,
    /// Maximum unused physical via span beyond the layers actually used by a high-speed signal.
    /// `0` means any unused high-speed via barrel is treated as a stub that must be removed by
    /// changing via construction or the routed layer transition.
    pub high_speed_max_via_stub_layers: u16,
    /// Maximum routed length mismatch between differential-pair members. The guide requires length
    /// matching; this project policy uses 0.5 mm as the intra-pair skew budget.
    pub diff_pair_length_tolerance: Nm,
    /// Maximum routed length mismatch for each via-delimited differential-pair segment. The guide
    /// requires local segment matching as well as total length matching; this project policy uses
    /// the same 0.5 mm skew budget per routed layer segment.
    pub diff_pair_segment_length_tolerance: Nm,
    /// Maximum routed length skew across indexed high-speed parallel-bus nets. Section 7.3.1.6
    /// requires parallel-bus data signals to stay inside the receiver timing-skew budget; this local
    /// policy uses a 2 mm routed-length skew budget for explicitly named bus groups.
    pub parallel_bus_length_tolerance: Nm,
    /// Maximum distance from a serpentine length-compensation run to the local bend/mismatch point.
    /// Section 7.3.1.6 calls for length compensation close to the root of the mismatch and gives a
    /// 15 mm maximum distance.
    pub serpentine_compensation_bend_distance: Nm,
    /// Maximum variation in center-to-center spacing across routed parallel differential-pair
    /// segments. Section 7.3.1.5 requires constant pair distance to preserve differential
    /// impedance; this local policy allows 0.25 mm routing-grid and pad-entry variation.
    pub diff_pair_spacing_tolerance: Nm,
    /// Maximum station mismatch between P-side and N-side differential-pair vias, measured along
    /// the pair routing axis. Section 7.3.1.5 requires differential-pair vias to be symmetric.
    pub diff_pair_via_symmetry_tolerance: Nm,
    /// Maximum station mismatch between the P-side and N-side AC-coupling capacitors of a
    /// differential pair, measured along the pair's routed axis. Section 7.3.1.5 requires serial
    /// coupling capacitors to be placed symmetrically.
    pub diff_pair_coupling_cap_symmetry_tolerance: Nm,
    /// Maximum package courtyard dimension for a differential-pair AC-coupling capacitor. Section
    /// 7.3.1.5 prefers 0402 capacitors, accepts 0603, and rejects larger 0805/C-pack packages; this
    /// local policy permits 0603-class courtyards while rejecting 0805-class bodies.
    pub diff_pair_coupling_cap_max_courtyard: Nm,
    /// Maximum mismatch between P-side and N-side differential-pair pad-entry distances. Section
    /// 7.3.1.6 calls out unequal pad-entry lengths as a source of CAD length-accounting mismatch;
    /// this project policy uses the same 0.5 mm local symmetry budget.
    pub diff_pair_pad_entry_tolerance: Nm,
    /// Maximum absolute uncoupled pad-entry length for each member of a differential pair. Section
    /// 7.3.1.6 requires equal pad-entry lengths, and this budget also keeps the breakout itself
    /// local so matched-but-long single-ended stubs do not pass by symmetry alone.
    pub diff_pair_pad_entry_max_length: Nm,
    /// Minimum keepout from a differential pair to unrelated signal copper. The guide's 30 mil
    /// rule maps to 0.762 mm.
    pub diff_pair_signal_keepout: Nm,
    /// Minimum keepout from a clock differential pair to unrelated signal copper. The guide's
    /// 50 mil clock rule maps to 1.27 mm.
    pub diff_pair_clock_keepout: Nm,
    /// Minimum spacing between adjacent differential pairs, expressed as a multiplier of the wider
    /// trace width. The guide's adjacent-pair rule maps to `5.0`.
    pub diff_pair_pair_spacing_widths: f64,
    /// Minimum spacing between unrelated high-speed parallel traces, expressed as a multiplier of
    /// the wider trace. The guide's crosstalk guidance maps to a conservative 3W spacing rule.
    pub high_speed_parallel_spacing_widths: f64,
    /// Minimum edge-to-edge spacing when an unrelated parallel high-speed run involves a clock-like
    /// net. The guide uses a wider 50 mil spacing rule for clock coupling; this maps to 1.27 mm.
    pub high_speed_clock_parallel_keepout: Nm,
    /// Preferred soft spacing from existing high-speed copper, expressed as a multiplier of the
    /// existing trace width. Section 7.3.1.4 says spacing should be increased beyond the minimum
    /// outside bottlenecks; this cost-only budget guides routing without making 5W a hard DRC.
    pub high_speed_preferred_parallel_spacing_widths: f64,
    /// Minimum parallel overlap length before the high-speed spacing rule is counted as a coupled
    /// run. Short pad-entry adjacency below this length is left to ordinary clearance checks.
    pub high_speed_parallel_coupling_length: Nm,
    /// Whether 45° (diagonal) routing moves are enabled on this board. Diagonal moves reduce
    /// wire length by up to 29% on open stretches and improve BGA escape channel utilisation.
    /// Enabled by default; disable only for strictly rectilinear fab requirements.
    pub diagonal_routing: bool,
    /// Build-up dielectric thickness (mm) for HDI laser-drilled micro-vias: the physical depth a
    /// micro-via must span through the build-up layer. Used by the micro-via aspect-ratio gate
    /// (`microvia_aspect_check`). Typical: 0.1 mm for a single-layer build-up (100 µm prepreg).
    pub build_up_mm: f64,
    /// Board-uniform **rail voltage excursion** (V) during an active IC's switching edge — the
    /// `dV/dt` numerator in [`crate::physics::emi::capacitive_drive_current_a`]. Typical: 3.3 V for a 3.3 V
    /// CMOS rail falling to GND on a 0–3.3 V step. Board-uniform (per IC-family datasheets are
    /// effectively fixed by the rail design) so it lives on `DesignRules` rather than
    /// `FootprintDef`. `0.0` (default in tests) makes `detect_charge_reservoir_violations`
    /// vacuous — matching the validator's `microvia_aspect_check` pass-vacuous pattern.
    pub ic_switching_dv_v: f64,
    /// Board-uniform **switching-edge rise time** (s) for an active IC's power-pin step. The
    /// `dV/dt` denominator in [`crate::physics::emi::capacitive_drive_current_a`]. Typical: 5e-9 s for a
    /// CMOS-class load (5 ns). `≤ 0` makes the charge-reservoir detector vacuous (cannot
    /// compute transient current).
    pub ic_switching_risetime_s: f64,
    /// Maximum pulse-skip fraction configured for this operating point (0 = check vacuous).
    /// `0.0` makes `detect_pulse_skip_violations` vacuous — identical to how
    /// [`DesignRules::ic_switching_dv_v`] being `0.0` makes the charge-reservoir check vacuous.
    pub max_skip_fraction: f64,
    /// Tolerable RMS pressure error fraction from pulse skipping (0..1). Default `0.05` (5%)
    /// per the TBME-2025 threshold used in [`crate::pulse_skip::rms_pressure_error_fraction`].
    pub pressure_error_tol: f64,
    /// Characteristic signal frequency (Hz) used for the λ/10 transmission-line length check.
    /// A trace longer than c₀ / (10·f·√εr_eff) must be treated as a controlled-impedance
    /// transmission line and requires impedance matching. `0.0` makes the check vacuous.
    /// Typical: 1e8 (100 MHz USB/GPIO) … 5e9 (5 GHz PCIE/WiFi).
    pub high_speed_frequency_hz: f64,
    /// Relative permittivity of the board dielectric — both for the λ/10 effective-medium
    /// calculation and the microstrip impedance formula. FR4 ≈ 4.5; Rogers RO4003 ≈ 3.55.
    pub dielectric_er: f64,
    /// Prepreg/core thickness (mm) of the signal layer above its adjacent reference plane.
    /// Used with [`Self::dielectric_er`] to compute microstrip characteristic impedance for
    /// the antenna trace impedance mismatch check. Typical 4-layer: 0.2 mm.
    pub dielectric_height_mm: f64,
    /// Maximum Euclidean distance (nm) from a decoupling capacitor to its associated IC power
    /// pin. The article mandates placement "as close as possible"; this is the hard ceiling
    /// enforced by `detect_decoupling_cap_distance_violations`. Typical: 3 mm.
    pub max_decoupling_cap_distance: Nm,
    /// Target characteristic impedance (Ω) for antenna-connected traces. Standard RF practice
    /// is 50 Ω for the trace connecting a transceiver to its antenna. `0.0` makes the check
    /// vacuous.
    pub antenna_impedance_ohm: f64,
    /// Tolerance (Ω) around [`Self::antenna_impedance_ohm`] before a trace is flagged. A
    /// computed |Z − target| > tolerance triggers `detect_antenna_impedance_mismatch`. Default
    /// 10 Ω (±20 % of the 50 Ω standard target).
    pub antenna_impedance_tolerance_ohm: f64,
}

impl DesignRules {
    /// The holohv board's committed rule floor: 0.15 mm track, 0.13 mm clearance/annular,
    /// 0.2 mm via drill — matching `holohv16.kicad_dru`.
    #[must_use]
    pub fn holohv() -> Self {
        DesignRules {
            via_policy: ViaPolicy::ThroughHole,
            min_track: Nm::from_mm(0.15),
            min_clearance: Nm::from_mm(0.13),
            min_via_drill: Nm::from_mm(0.2),
            min_annular: Nm::from_mm(0.13),
            microvia_drill: Nm::from_mm(0.1),
            microvia_annular: Nm::from_mm(0.075),
            max_microvia_ar: 1.0,
            max_blind_buried_via_drill: Nm::from_mm(0.15),
            signal_track: Nm::from_mm(0.15),
            hv_track: Nm::from_mm(0.25),
            power_track: Nm::from_mm(0.25),
            edge_clearance: Nm::from_mm(0.5),
            assembly_clearance: Nm::from_mm(2.0),
            min_via_to_via_spacing: Nm::from_mm(0.381), // 15 mils
            high_speed_edge_clearance: Nm::from_mm(1.0),
            high_speed_component_edge_clearance: Nm::from_mm(3.0),
            high_speed_termination_distance: Nm::from_mm(2.0),
            high_speed_reference_plane_margin_widths: 3.0,
            high_speed_transition_ground_via_distance: Nm::from_mm(2.0),
            high_speed_terminal_ground_via_distance: Nm::from_mm(2.0),
            high_speed_via_pad_distance: Nm::from_mm(2.0),
            decoupling_ground_via_distance: Nm::from_mm(1.0),
            max_decoupling_loop_area_mm2: 10.0,
            power_reference_stitching_cap_distance: Nm::from_mm(2.0),
            split_plane_stitching_cap_distance: Nm::from_mm(2.0),
            high_speed_max_via_stub_layers: 0,
            diff_pair_length_tolerance: Nm::from_mm(0.5),
            diff_pair_segment_length_tolerance: Nm::from_mm(0.5),
            parallel_bus_length_tolerance: Nm::from_mm(2.0),
            serpentine_compensation_bend_distance: Nm::from_mm(15.0),
            diff_pair_spacing_tolerance: Nm::from_mm(0.25),
            diff_pair_via_symmetry_tolerance: Nm::from_mm(0.5),
            diff_pair_coupling_cap_symmetry_tolerance: Nm::from_mm(0.5),
            diff_pair_coupling_cap_max_courtyard: Nm::from_mm(1.7),
            diff_pair_pad_entry_tolerance: Nm::from_mm(0.5),
            diff_pair_pad_entry_max_length: Nm::from_mm(2.0),
            diff_pair_signal_keepout: Nm::from_mm(0.762),
            diff_pair_clock_keepout: Nm::from_mm(1.27),
            diff_pair_pair_spacing_widths: 5.0,
            high_speed_parallel_spacing_widths: 3.0,
            high_speed_clock_parallel_keepout: Nm::from_mm(1.27),
            high_speed_preferred_parallel_spacing_widths: 5.0,
            high_speed_parallel_coupling_length: Nm::from_mm(5.0),
            diagonal_routing: true,
            build_up_mm: 0.1,
            ic_switching_dv_v: 3.3,
            ic_switching_risetime_s: 5e-9,
            max_skip_fraction: 0.0, // vacuous — caller sets this for a real operating point
            pressure_error_tol: 0.05, // 5 % per TBME-2025
            high_speed_frequency_hz: 1.0e8, // 100 MHz — conservative threshold for USB / GPIO
            dielectric_er: 4.5,             // FR4 nominal
            dielectric_height_mm: 0.2,      // typical 4-layer prepreg/core thickness
            max_decoupling_cap_distance: Nm::from_mm(3.0), // "as close as possible" ceiling
            antenna_impedance_ohm: 50.0,    // universal RF 50-Ω convention
            antenna_impedance_tolerance_ohm: 10.0, // ±20 % of 50 Ω
        }
    }

    /// Default track width for a net class.
    #[must_use]
    pub fn track_for(&self, class: crate::board::NetClassKind) -> Nm {
        use crate::board::NetClassKind::*;
        match class {
            Hv => self.hv_track,
            Power | Ground => self.power_track,
            Signal => self.signal_track,
        }
    }

    /// Canonical routing-grid pitch: the **signal clearance quantum** `signal_track + min_clearance`.
    ///
    /// One net per cell on a grid of this pitch is clearance-correct *by construction* for the
    /// dominant signal class: two signal tracks in orthogonally-adjacent cells sit exactly
    /// `pitch` apart centre-to-centre, leaving `pitch - signal_track = min_clearance` of copper
    /// gap — DRC-legal with no halo. Diagonal neighbours are `√2·pitch` apart (wider still).
    ///
    /// Picking the grid pitch *below* this quantum (as a hand-tuned fine grid does to resolve a
    /// pad channel) makes adjacency itself a clearance violation, so a sub-quantum grid **must**
    /// run a clearance halo ([`crate::route::grid::Grid::set_clearance_radius`]) sized
    /// `ceil(quantum / pitch) - 1` to keep foreign copper out of the adjacent cells. Wider classes
    /// (HV: `hv_track + hv_clearance`) are not separated by this pitch alone and rely on that halo
    /// plus the per-class pad obstacles.
    #[must_use]
    pub fn routing_pitch(&self) -> Nm {
        self.signal_track + self.min_clearance
    }

    /// Hole-to-copper clearance: the minimum gap a drilled hole's edge must hold from foreign-net
    /// copper. Held to the copper-to-copper clearance (`min_clearance`) so the rule the internal
    /// audit checks, the rule emitted into the `.kicad_dru`, and the rule kicad-cli enforces are the
    /// *same* value (SSOT) — rather than letting kicad fall back to its conservative 0.25 mm default,
    /// which flagged vias the engine considered legal and split the internal/external verdict.
    #[must_use]
    pub fn hole_clearance(&self) -> Nm {
        self.min_clearance
    }

    /// Default via outer diameter (drill + annular ring on both sides).
    #[must_use]
    pub fn via_diameter(&self) -> Nm {
        self.min_via_drill + self.min_annular * 2
    }

    /// HDI micro-via outer diameter (laser drill + micro annular ring on both sides).
    #[must_use]
    pub fn microvia_diameter(&self) -> Nm {
        self.microvia_drill + self.microvia_annular * 2
    }

    /// Resolve the physical span, construction class, drill and outer diameter of a via that
    /// *electrically* connects layers `lo..=hi`, honouring the board's [`ViaPolicy`].
    ///
    /// Under [`ViaPolicy::ThroughHole`] every via is a single full-stack mechanical drill
    /// (`0..=nlayers-1`, `Through`), regardless of the connected span — the cheapest standard process.
    /// Under [`ViaPolicy::Hdi`] the actual span is kept and classified into micro/blind/buried/through,
    /// with micro-vias taking the laser drill/diameter. Returns `(from, to, kind, drill, diameter)`.
    #[must_use]
    pub fn resolve_via(
        &self,
        lo: u16,
        hi: u16,
        nlayers: usize,
    ) -> (
        crate::board::LayerId,
        crate::board::LayerId,
        crate::board::ViaKind,
        Nm,
        Nm,
    ) {
        use crate::board::{LayerId, Via, ViaKind};
        match self.via_policy {
            ViaPolicy::ThroughHole => (
                LayerId(0),
                LayerId((nlayers - 1) as u16),
                ViaKind::Through,
                self.min_via_drill,
                self.via_diameter(),
            ),
            ViaPolicy::Hdi => {
                let (from, to) = (LayerId(lo.min(hi)), LayerId(lo.max(hi)));
                let kind = Via::classify(from, to, nlayers);
                let (drill, dia) = if matches!(kind, ViaKind::Micro) {
                    (self.microvia_drill, self.microvia_diameter())
                } else {
                    (self.min_via_drill, self.via_diameter())
                };
                (from, to, kind, drill, dia)
            }
        }
    }
}

impl Default for DesignRules {
    fn default() -> Self {
        DesignRules::holohv()
    }
}

/// The high-voltage creepage rule: minimum surface clearance an HV net must hold from any
/// low-voltage feature. Larger than the bare electrical clearance because surface tracking, not
/// just air breakdown, governs at the HV/LV boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreepageRule {
    /// Minimum HV-to-LV surface clearance.
    pub hv_clearance: Nm,
}

impl CreepageRule {
    /// The holohv creepage rule: 0.60 mm HV-to-LV (matching the `HV_creepage` DRU rule and
    /// IPC-2221B Table 6-1 B1 external-uncoated ≤150 V spacing per
    /// [`crate::physics::dielectric::ipc2221_min_spacing_mm`]). The earlier 0.5 mm floor left a 0.1 mm
    /// deficit against the B1 rule which `detect_creepage_violations` would still flag.
    #[must_use]
    pub fn holohv() -> Self {
        CreepageRule {
            hv_clearance: Nm::from_mm(0.60),
        }
    }
}

impl Default for CreepageRule {
    fn default() -> Self {
        CreepageRule::holohv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::ViaKind;

    #[test]
    fn through_hole_policy_makes_every_via_a_full_stack_through() {
        let r = DesignRules::holohv(); // default ViaPolicy::ThroughHole
                                       // Even an F.Cu→In1 connection becomes a full-stack mechanical through-hole on a standard board.
        let (from, to, kind, drill, dia) = r.resolve_via(0, 1, 4);
        assert_eq!((from.0, to.0), (0, 3), "spans the full stack");
        assert_eq!(kind, ViaKind::Through);
        assert_eq!(drill, r.min_via_drill);
        assert_eq!(dia, r.via_diameter());
    }

    #[test]
    fn diagonal_routing_enabled_by_default() {
        assert!(
            DesignRules::holohv().diagonal_routing,
            "diagonal routing should default to enabled"
        );
    }

    #[test]
    fn hdi_policy_keeps_the_span_and_uses_laser_microvias() {
        let r = DesignRules {
            via_policy: ViaPolicy::Hdi,
            ..DesignRules::holohv()
        };
        // Outer→adjacent-inner stays a laser micro-via (small drill) under HDI.
        let (from, to, kind, drill, dia) = r.resolve_via(0, 1, 4);
        assert_eq!((from.0, to.0), (0, 1), "HDI keeps the shallow span");
        assert_eq!(kind, ViaKind::Micro);
        assert_eq!(drill, r.microvia_drill);
        assert_eq!(dia, r.microvia_diameter());
        // A core inner→inner span is a buried mechanical via (not laser).
        let (_, _, kind2, drill2, _) = r.resolve_via(1, 2, 4);
        assert_eq!(kind2, ViaKind::Buried);
        assert_eq!(drill2, r.min_via_drill);
    }
}
