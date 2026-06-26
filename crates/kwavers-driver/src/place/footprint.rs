//! Footprint definitions and constrained orientation math.
//!
//! A footprint is a courtyard rectangle plus pads at offsets from its centre. Placement is
//! axis-aligned: components translate freely, but rotations are filtered by each footprint's
//! `RotationPolicy` (see [`super::rotation`]) so mechanical mating, pin-1 escape, and assembly
//! orientation constraints are not lost during annealing.
//!
//! Phase 2c: the 4-variant `Rot` ZST marker + the `RotationPolicy` placement-freedom enum
//! (and their impls) were carved out into [`super::rotation`] per the spec's
//! `place/{mod, anneal, energy, footprint, import, rotation, tests}.rs` layout. This file
//! keeps the footprint-shape and placement-class concerns: `Role`, `IsolationDomain`,
//! `PadDef`, `FootprintDef` + its builder methods, and the `Model3D` type alias.

use super::rotation::RotationPolicy;
use crate::board::LayerId;
use crate::geom::{Nm, Point};

/// Ceramic capacitor dielectric material grade (TI SLYP173 §5-17/5-20).
///
/// Each grade has materially different capacitance stability over temperature, voltage, and
/// frequency. The default [`DielectricGrade::Unknown`] leaves the check vacuous so existing
/// footprints without a grade assigned are never falsely flagged — add a grade only when the
/// datasheet is known.
///
/// Ordering (best→worst): `Cog > X7r > Z5u > Y5v`. `Unknown` is neutral (vacuous).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DielectricGrade {
    /// Unknown or unspecified; check is vacuous for this component.
    #[default]
    Unknown,
    /// **COG (NP0)** — ±30 ppm/°C, flat vs. frequency and voltage; best grade.
    Cog,
    /// **X7R** — ±15 % over −55 °C … +125 °C; acceptable mid-grade for bypass.
    X7r,
    /// **Z5U** — +22 %/−56 % over +10 °C … +85 °C; variable, lower quality.
    Z5u,
    /// **Y5V** — +22 %/−82 % over −30 °C … +85 °C; worst grade; avoid for high-speed.
    Y5v,
}

impl DielectricGrade {
    /// `true` if this grade meets or exceeds `min_required`.
    ///
    /// `Unknown` is always accepted (no grade information → vacuous pass). The ordering is
    /// `Cog (4) ≥ X7r (3) ≥ Z5u (2) ≥ Y5v (1)`; `Unknown` (0) bypasses the check.
    #[must_use]
    pub fn meets_minimum(self, min_required: Self) -> bool {
        if self == Self::Unknown || min_required == Self::Unknown {
            return true;
        }
        self.rank() >= min_required.rank()
    }

    fn rank(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Y5v => 1,
            Self::Z5u => 2,
            Self::X7r => 3,
            Self::Cog => 4,
        }
    }
}

/// Physical package form factor of a component.
///
/// Through-hole/leaded packages carry higher lead inductance than SMT equivalents (TI
/// SLYP173 §5-20/5-21, §5-30). The default [`PackageFormFactor::Unknown`] leaves the check vacuous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackageFormFactor {
    /// Unknown or unspecified; through-hole check is vacuous for this component.
    #[default]
    Unknown,
    /// **SMT** (surface-mount): 0402, 0603, QFN, SOIC, BGA, etc. Preferred for high-speed.
    Smt,
    /// **Through-hole / leaded**: DIP, TO-220, axial/radial resistors, leaded caps.
    /// High lead inductance; avoid for high-speed signal paths (TI SLYP173 §5-21, §5-30).
    ThroughHole,
}

/// Placement role — drives where a part *wants* to sit in the placement energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// An active IC (HV7355 pulser, isolator) — pulled to the board core.
    ActiveIc,
    /// A board-edge interface (transducer / control / power connector) — pulled to the periphery.
    Connector,
    /// A decoupling/bypass capacitor — pinned next to its IC's power pin.
    Decoupling,
    /// A power-conversion part (buck, bulk cap).
    Power,
    /// A generic passive (series R, ferrite) — placed by wirelength only.
    Passive,
}

/// LV↔HV isolation domain of a footprint/component. Drives the placement
/// [`crate::place::energy::EnergyTerms::isolation_drift`] term so the annealer lifts HV
/// components across to the HV side of [`crate::place::energy::PlaceConfig::isolation_axis`]
/// and parks LV ones on the LV side, producing a clean floor-planned isolation barrier instead
/// of a mixed scatter. Default [`IsolationDomain::Lv`] because the placement's prevailing
/// usage is LV-only designs (controllers, FPGAs, signal chains); HV is opt-in per component via
/// [`crate::place::component::Component::with_isolation_domain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationDomain {
    /// Low-voltage side (≤ ~50 V) — drift toward the axis-min line on the LV side.
    #[default]
    Lv,
    /// High-voltage side (> ~50 V) — drift toward the axis-max line on the HV side.
    Hv,
}

/// A pad within a footprint, offset from the footprint centre at `R0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadDef {
    /// Offset from the footprint centre (before rotation).
    pub offset: Point,
    /// Copper-pad `(width, height)` at `R0`.
    pub size: (Nm, Nm),
    /// Copper layers the pad is accessible on.
    pub layers: Vec<LayerId>,
    /// Whether this is a power/ground pin (used by decoupling proximity).
    pub power_pin: bool,
}

/// A 3D body model attachment:
/// `(model path, (dx,dy,dz) mm offset, (rx,ry,rz) deg rotation, optional (w,h) mm envelope)`.
pub type Model3D = (String, (f64, f64, f64), (f64, f64, f64), Option<(f64, f64)>);

/// A footprint: courtyard size and pads. Net assignment is per *instance*, not here.
#[derive(Debug, Clone, PartialEq)]
pub struct FootprintDef {
    /// Human-readable name (e.g. `"HV7355K6-G"`).
    pub name: String,
    /// Courtyard `(width, height)` at `R0`.
    pub courtyard: (Nm, Nm),
    /// Placement role.
    pub role: Role,
    /// Rotation freedom allowed during automated placement.
    pub rotation_policy: RotationPolicy,
    /// Pads, at offsets from the centre.
    pub pads: Vec<PadDef>,
    /// Pad names/numbers parallel to `pads` (the real footprint's pin identifiers, e.g. `"1"`, `"A1"`,
    /// `"EP"`). Empty for synthesized abstractions (which are indexed positionally); populated by the
    /// `.kicad_mod` importer so nets can be wired by pin name against the exact manufacturer footprint.
    pub pad_names: Vec<String>,
    /// Optional 3D body model for rendering. Emitted as a KiCad `(model …)` so `kicad-cli pcb
    /// render` shows the real component body.
    pub model: Option<Model3D>,
    /// Ball pitch (mm, as `Nm`) if this is a **BGA** — pads are a ball grid that needs escape
    /// routing (via-in-pad fanout to an inner layer). `None` for ordinary leaded/QFN/connector
    /// footprints whose pads route directly on the top layer.
    pub ball_pitch: Option<Nm>,
    /// Datasheet **steady-state supply current** (A) this active IC draws during its worst-case
    /// switching window — the peak `I_dd` the rail must deliver. Drives the per-IC power-integrity
    /// budget in `detect_charge_reservoir_violations`: each IC's
    /// `Σ C_k·dV/dt` (across all associated decoupling caps) must meet this rating or the rail
    /// collapses during the switching edge. `0.0` (default) ⇒ IC has no datasheet rating set and
    /// the charge-reservoir detector treats it as vacuous (silently skipped), matching the
    /// pass-vacuous pattern in [`crate::validate::microvia_aspect_check`]. Set per-IC: a 0 is the
    /// right default because every active IC family has its own I_dd (it is a part-level value,
    /// not a board-uniform one — see [`crate::rules::DesignRules::ic_switching_dv_v`] /
    /// [`crate::rules::DesignRules::ic_switching_risetime_s`] for the board-uniform edge
    /// parameters that combine with each cap's `capacitance_f`). For a `Role::Power` footprint (buck converter) the same field carries the buck's own
    /// input-side switching current draw — `assoc_ic` is expected to tie only the
    /// input-side bulk reservoir; output-cap hold-up sizing is delegated to [`crate::physics::pdn::holdup_capacitance_f`].
    pub i_dd_a: f64,
    /// Capacitance (F) of this **decoupling capacitor footprint** — the total reservoir of charge
    /// each cap can supply during an IC's switching edge. Drives
    /// `detect_charge_reservoir_violations` via
    /// [`crate::physics::emi::capacitive_drive_current_a`] (`I_per_cap = C·dV/dt`). `0.0` (default) ⇒
    /// caps whose footprint has no value set contribute zero to the per-IC sum, so a board
    /// that hasn't yet wired datasheet cap values into its library is automatically flagged
    /// as under-provisioned (`Σ_i_i_cap = 0 ≪ I_dd`) so the operator notices the gap.
    /// Pass `with_capacitance_f(f64)` on the footprint builder to silence.
    pub capacitance_f: f64,
    /// Ceramic dielectric material grade (TI SLYP173 §5-17/5-20). Default
    /// [`DielectricGrade::Unknown`] makes the grade check vacuous — set only for caps whose
    /// datasheet grade is known. The audit flags decoupling caps whose grade is worse than
    /// [`crate::rules::DesignRules::min_decoupling_cap_grade`].
    pub dielectric_grade: DielectricGrade,
    /// Physical package form factor. Default [`PackageFormFactor::Unknown`] makes the through-hole
    /// check vacuous — set to [`PackageFormFactor::ThroughHole`] for DIP, TO-220, axial, and
    /// radial leaded components so `detect_through_hole_high_speed_violations` can flag them when
    /// connected to high-speed nets (TI SLYP173 §5-20/5-21, §5-30).
    pub package_form_factor: PackageFormFactor,
}

impl FootprintDef {
    /// Build a rectangular footprint with an explicit pad list.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        courtyard: (Nm, Nm),
        role: Role,
        pads: Vec<PadDef>,
    ) -> Self {
        FootprintDef {
            name: name.into(),
            courtyard,
            role,
            rotation_policy: RotationPolicy::for_role(role),
            pads,
            pad_names: Vec::new(),
            model: None,
            ball_pitch: None,
            // Per-part electrical specs default to `0.0` / enum defaults so existing
            // FootprintDef::new(…) call sites continue to compile unchanged.
            i_dd_a: 0.0,
            capacitance_f: 0.0,
            dielectric_grade: DielectricGrade::Unknown,
            package_form_factor: PackageFormFactor::Unknown,
        }
    }

    /// Set the active-IC datasheet steady-state `I_dd` (A) for the charge-reservoir audit.
    /// Builder style so the `new(...)` call site doesn't break.
    #[must_use]
    pub fn with_i_dd_a(mut self, i_dd_a: f64) -> Self {
        self.i_dd_a = i_dd_a;
        self
    }

    /// Assign human-readable pad identifiers (schematic pin numbers / names) in pad-index order.
    /// Used by the SVG renderer for pad tooltips and by [`Self::pad_index`] for net-by-name wiring.
    /// The length should equal `self.pads.len()`; missing trailing names are shown as `?`.
    #[must_use]
    pub fn with_pad_names(mut self, names: Vec<String>) -> Self {
        self.pad_names = names;
        self
    }

    /// Set the decoupling-cap capacitance (F) for the charge-reservoir audit. Builder style so
    /// the `new(...)` call site doesn't break.
    #[must_use]
    pub fn with_capacitance_f(mut self, c_f: f64) -> Self {
        self.capacitance_f = c_f;
        self
    }

    /// Set the ceramic capacitor dielectric grade for the grade-quality audit.
    #[must_use]
    pub fn with_dielectric_grade(mut self, grade: DielectricGrade) -> Self {
        self.dielectric_grade = grade;
        self
    }

    /// Set the physical package form factor for the through-hole detection audit.
    #[must_use]
    pub fn with_package_form_factor(mut self, form: PackageFormFactor) -> Self {
        self.package_form_factor = form;
        self
    }

    /// Index of the pad named `name` (the real footprint's pin identifier), or `None`. Used to wire a
    /// net to a pin by name on an imported manufacturer footprint.
    #[must_use]
    pub fn pad_index(&self, name: &str) -> Option<usize> {
        self.pad_names.iter().position(|n| n == name)
    }

    /// Override the role-derived rotation policy.
    #[must_use]
    pub fn with_rotation_policy(mut self, policy: RotationPolicy) -> Self {
        self.rotation_policy = policy;
        self
    }

    /// Override the courtyard size of the footprint.
    #[must_use]
    pub fn with_courtyard(mut self, width: Nm, height: Nm) -> Self {
        self.courtyard = (width, height);
        self
    }

    /// Smallest centre-to-centre distance between any two pads (the footprint's pad pitch). `None` for
    /// a footprint with fewer than two pads. Drives fine-pitch escape detection ([`Self::needs_escape`]).
    #[must_use]
    pub fn min_pad_pitch(&self) -> Option<Nm> {
        let mut best: Option<i64> = None;
        for (i, a) in self.pads.iter().enumerate() {
            for b in self.pads.iter().skip(i + 1) {
                let d = a.offset.euclid(b.offset) as i64;
                best = Some(best.map_or(d, |m| m.min(d)));
            }
        }
        best.map(Nm)
    }

    /// Whether this footprint's pads are too fine-pitch to escape on the top layer and must instead
    /// fan out **via-in-pad to an inner layer** (the BGA escape path, generalised). A pad row whose
    /// pitch is below `direct_pitch` cannot fit a track + two clearances in the channel between
    /// adjacent pads, so a perimeter QFN/QSOP/fine-QFP needs the same escape a BGA does. A part
    /// explicitly marked BGA (`ball_pitch`) always escapes.
    #[must_use]
    pub fn needs_escape(&self, direct_pitch: Nm) -> bool {
        self.ball_pitch.is_some() || self.min_pad_pitch().is_some_and(|p| p < direct_pitch)
    }

    /// Mark an **already-imported** manufacturer footprint as a BGA with the given ball pitch. The
    /// geometry file itself does not declare a pitch (the .kicad_mod only carries per-ball positions);
    /// without an explicit pitch the importer cannot know every ball is on a uniform grid, so by
    /// default `ball_pitch = None` and the BGA falls back to top-layer routing — and a 1 mm pitch
    /// 484-ball package cannot route its buried balls on top. Calling `with_bga_pitch(1.0 mm)` after
    /// `import_kicad_mod` flips `needs_escape(FINE_PITCH_ESCAPE)` to true and `place_to_board`
    /// fans each used ball out via-in-pad (VIPPO) to the first inner layer where the router
    /// escapes it through the channels between the ball vias. Corner-depopulated grids (e.g.
    /// FG(G)BGA484 on a 22×22 grid missing a handful of corners) are still routed correctly because
    /// `needs_escape` only governs the *escape* policy, not the pad count.
    #[must_use]
    pub fn with_bga_pitch(mut self, pitch: Nm) -> Self {
        self.ball_pitch = Some(pitch);
        self
    }

    /// Build a **BGA** footprint: a `rows × cols` ball grid at `pitch`, centred on the origin. The
    /// resulting footprint is marked for escape routing (`ball_pitch = Some(pitch)`), so the router
    /// fans each used ball out via-in-pad to an inner layer instead of trying to reach it on the
    /// congested top layer. `power` indexes (row-major) the supply balls.
    #[must_use]
    pub fn bga(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        pitch: Nm,
        power: &[usize],
    ) -> Self {
        let (x0, y0) = (
            -(pitch.0 * (cols as i64 - 1)) / 2,
            -(pitch.0 * (rows as i64 - 1)) / 2,
        );
        let mut pads = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let k = r * cols + c;
                pads.push(PadDef {
                    offset: Point::new(Nm(x0 + pitch.0 * c as i64), Nm(y0 + pitch.0 * r as i64)),
                    size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                    layers: vec![LayerId(0)],
                    power_pin: power.contains(&k),
                });
            }
        }
        let span = (Nm(pitch.0 * cols as i64), Nm(pitch.0 * rows as i64));
        let mut fp = FootprintDef::new(name, span, Role::ActiveIc, pads);
        fp.ball_pitch = Some(pitch);
        fp
    }

    /// Attach a 3D body model (path + offset mm + rotation deg) for rendering. Builder style so the
    /// `new`/`two_row_header` call sites need no change.
    #[must_use]
    pub fn with_model(
        mut self,
        path: impl Into<String>,
        offset: (f64, f64, f64),
        rotate: (f64, f64, f64),
    ) -> Self {
        self.model = Some((path.into(), offset, rotate, None));
        self
    }

    /// Attach a 3D body model plus a package envelope when the KiCad model filename does not encode
    /// body size. The envelope is an assembly-verification input only; KiCad emission still uses the
    /// path, offset, and rotation fields.
    #[must_use]
    pub fn with_model_envelope(
        mut self,
        path: impl Into<String>,
        offset: (f64, f64, f64),
        rotate: (f64, f64, f64),
        envelope_mm: (f64, f64),
    ) -> Self {
        self.model = Some((path.into(), offset, rotate, Some(envelope_mm)));
        self
    }

    /// Build a two-row 0.1″-class pin header (the board-to-board **stacking** connector). `per_row`
    /// pins on each of two rows at `pitch`; pad index order is column-major
    /// (`row0col0, row1col0, row0col1, …`) so a `(signal, GND)` pinout interleaves a ground return
    /// beside every signal pin. Through-hole pads (all layers) so the bus passes through stacked
    /// tiles. `power_pins` marks the supply pads for the placement's power-fanout heuristic.
    #[must_use]
    pub fn two_row_header(
        name: impl Into<String>,
        per_row: usize,
        pitch: Nm,
        power_pins: &[usize],
    ) -> Self {
        let rows = 2;
        let w = pitch.0 * (per_row as i64 - 1) + Nm::from_mm(2.0).0;
        let h = pitch.0 * (rows as i64 - 1) + Nm::from_mm(2.0).0;
        let (x0, y0) = (-(w / 2) + Nm::from_mm(1.0).0, -(h / 2) + Nm::from_mm(1.0).0);
        let mut pads = Vec::with_capacity(rows * per_row);
        for col in 0..per_row {
            for row in 0..rows {
                pads.push(PadDef {
                    offset: Point::new(
                        Nm(x0 + pitch.0 * col as i64),
                        Nm(y0 + pitch.0 * row as i64),
                    ),
                    size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
                    layers: vec![LayerId(0), LayerId(1)],
                    power_pin: power_pins.contains(&(col * rows + row)),
                });
            }
        }
        FootprintDef::new(name, (Nm(w), Nm(h)), Role::Connector, pads)
            .with_pad_names((1..=(rows * per_row)).map(|i| i.to_string()).collect())
    }
}
