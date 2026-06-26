//! The board model: layers, nets, pads, and the routed copper (tracks and vias).
//!
//! # Phase 1a migration roadmap
//!
//! The board model's spatial quantities (track width, via drill/diameter, pad offsets) already
//! use [`crate::units::Nm`] from `geom`. **No f64 length sites remain in this module.** Pad and
//! track layer indices use [`LayerId`] (a `u16` newtype). The remaining soft-unit `f64` sites
//! (`IrDrop` / voltage / current / power in the per-net thermal co-analysis) live in
//! [`crate::physics::thermal`] / [`crate::physics::pdn`] and migrate when those vertical slices are carved out.
//! **This module is essentially migration-complete at Phase 1a.**
//!
//! This is a pure domain model with no infrastructure dependency — it can be built in memory by a
//! test, a generator, or a future `.kicad_pcb` parser without any of them leaking into the model.

use crate::geom::{GridSpec, Nm, Point};

/// A copper-layer index. Layer `0` is the top (F.Cu); `nlayers - 1` is the bottom (B.Cu).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(pub u16);

/// A net identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NetId(pub u32);

/// The electrical class of a net — a closed set, so it is dispatched as an exhaustive enum rather
/// than a trait object. The class drives both design-rule selection and physics cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetClassKind {
    /// High-voltage net (VPP/VDDH/TX outputs) — wide track, large clearance, creepage-policed.
    Hv,
    /// Low-voltage logic/control signal.
    Signal,
    /// A power rail (5 V, 3.3 V) routed as wide copper or a plane.
    Power,
    /// Ground / return.
    Ground,
}

/// Analog/digital floorplan domain inferred from schematic net naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDomain {
    /// Analog section / analog return.
    Analog,
    /// Digital section / digital return.
    Digital,
}

impl NetClassKind {
    /// Whether this class is a low-voltage domain (the side HV must keep creepage from).
    #[must_use]
    pub fn is_low_voltage(self) -> bool {
        matches!(
            self,
            NetClassKind::Signal | NetClassKind::Power | NetClassKind::Ground
        )
    }
}

/// Classify schematic net names into analog/digital floorplan domains.
#[must_use]
pub fn split_domain_from_name(name: &str) -> Option<SplitDomain> {
    let upper = name.to_ascii_uppercase();
    let tokens: Vec<&str> = upper
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect();
    if tokens
        .iter()
        .any(|token| matches!(*token, "AGND" | "AGROUND" | "ANALOG" | "ANA"))
    {
        Some(SplitDomain::Analog)
    } else if tokens.iter().any(|token| {
        matches!(
            *token,
            "DGND" | "DGROUND" | "DIGITAL" | "DIG" | "FPGA" | "SPI" | "I2C" | "UART"
        )
    }) || upper.starts_with("CLK")
    {
        Some(SplitDomain::Digital)
    } else {
        None
    }
}

/// A named net with an electrical class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Net {
    /// Stable identifier.
    pub id: NetId,
    /// Net name as it appears in the schematic.
    pub name: String,
    /// Electrical class.
    pub class: NetClassKind,
}

/// A component pad — a routing terminal. SMD pads list a single accessible layer; through-hole
/// pads list every layer they connect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pad {
    /// Pad centre on the board.
    pub pos: Point,
    /// Copper layers on which this pad is accessible.
    pub layers: Vec<LayerId>,
    /// Net this pad belongs to, if any (unconnected pads are `None`).
    pub net: Option<NetId>,
}

/// A routed copper track segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    /// Segment start.
    pub start: Point,
    /// Segment end.
    pub end: Point,
    /// Track width.
    pub width: Nm,
    /// Layer the segment is on.
    pub layer: LayerId,
    /// Net the segment carries.
    pub net: NetId,
}

/// Construction class of a via — the HDI pathway it represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViaKind {
    /// Mechanically drilled, spanning the full stack outer-to-outer.
    Through,
    /// Mechanically drilled, one outer layer to an inner layer (or non-adjacent inner span touching
    /// an outer).
    Blind,
    /// Mechanically drilled, inner-to-inner only (does not reach an outer layer).
    Buried,
    /// Laser-drilled micro-via between two *adjacent* layers touching an outer surface — the HDI
    /// build-up via; small diameter, low aspect ratio.
    Micro,
}

/// A routed via connecting two layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Via {
    /// Via centre.
    pub pos: Point,
    /// Drill (hole) diameter.
    pub drill: Nm,
    /// Outer pad diameter.
    pub diameter: Nm,
    /// Net the via carries.
    pub net: NetId,
    /// Span: lower layer.
    pub from: LayerId,
    /// Span: upper layer.
    pub to: LayerId,
    /// Construction class (through / blind / buried / micro) — set from the layer span by
    /// [`Via::classify`].
    pub kind: ViaKind,
    /// **VIPPO** — via-in-pad, plated-over (filled + capped). True for an escape via sitting inside a
    /// solderable pad (BGA ball / fine-pitch lead), so the pad stays flat and solderable.
    pub filled: bool,
}

impl Via {
    /// Classify a via from its layer span on an `nlayers` stack: outer-to-outer ⇒ `Through`; a single
    /// adjacent hop touching an outer surface ⇒ `Micro` (laser build-up via); any other span that
    /// reaches an outer layer ⇒ `Blind`; an inner-only span ⇒ `Buried`.
    #[must_use]
    pub fn classify(from: LayerId, to: LayerId, nlayers: usize) -> ViaKind {
        let (lo, hi) = (from.0.min(to.0), from.0.max(to.0));
        let last = (nlayers - 1) as u16;
        let touches_outer = lo == 0 || hi == last;
        if lo == 0 && hi == last {
            ViaKind::Through
        } else if hi - lo == 1 && touches_outer {
            ViaKind::Micro
        } else if touches_outer {
            ViaKind::Blind
        } else {
            ViaKind::Buried
        }
    }
}

/// How a copper [`Zone`]'s fill connects to same-net pads — the manufacturing trade-off between
/// solderability (thermal relief spokes) and a solid low-impedance connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneFill {
    /// Thermal-relief spokes: a small gap with bridging spokes around pads, so reflow heat is not
    /// sunk into the plane. The right choice for a ground/power **pour**.
    ThermalRelief,
    /// Solid: copper fills right up to the pad. The right choice for a **teardrop** fillet, whose
    /// whole purpose is a continuous mechanical/electrical transition into the pad.
    Solid,
}

/// A filled copper region on one layer, assigned to a net. KiCad's zone filler carves design-rule
/// clearance around every foreign-net feature automatically, so a zone is the clearance-safe way to
/// add copper (ground pour for EMI return / shielding / copper balance, or teardrop reinforcement).
#[derive(Debug, Clone)]
pub struct Zone {
    /// Net the fill belongs to.
    pub net: NetId,
    /// Copper layer the fill sits on.
    pub layer: LayerId,
    /// Closed outline polygon (board-space points; the filler intersects it with copper rules).
    pub polygon: Vec<Point>,
    /// Pad-connection style.
    pub fill: ZoneFill,
}

/// A board: its routing grid, electrical nets, component pads, and the routed copper produced by
/// the engine.
#[derive(Debug, Clone)]
pub struct Board {
    /// The routing grid that discretises this board.
    pub spec: GridSpec,
    /// All nets, indexed by `NetId.0`.
    pub nets: Vec<Net>,
    /// All component pads.
    pub pads: Vec<Pad>,
    /// Routed track segments (filled by the router).
    pub tracks: Vec<Track>,
    /// Routed vias (filled by the router).
    pub vias: Vec<Via>,
    /// Copper fill regions (ground pour, teardrops) — emitted as KiCad zones.
    pub zones: Vec<Zone>,
}

impl Board {
    /// Create an empty board over the given routing grid.
    #[must_use]
    pub fn new(spec: GridSpec) -> Self {
        Board {
            spec,
            nets: Vec::new(),
            pads: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            zones: Vec::new(),
        }
    }

    /// Find a net by exact name (e.g. the ground net for a copper pour).
    #[must_use]
    pub fn net_by_name(&self, name: &str) -> Option<NetId> {
        self.nets.iter().find(|n| n.name == name).map(|n| n.id)
    }

    /// Add a net, returning its id.
    pub fn add_net(&mut self, name: impl Into<String>, class: NetClassKind) -> NetId {
        let id = NetId(self.nets.len() as u32);
        self.nets.push(Net {
            id,
            name: name.into(),
            class,
        });
        id
    }

    /// Add a pad.
    pub fn add_pad(&mut self, pad: Pad) {
        self.pads.push(pad);
    }

    /// Electrical class of a net.
    #[must_use]
    pub fn class_of(&self, net: NetId) -> NetClassKind {
        self.nets[net.0 as usize].class
    }

    /// Pads belonging to a net.
    pub fn pads_of(&self, net: NetId) -> impl Iterator<Item = &Pad> {
        self.pads.iter().filter(move |p| p.net == Some(net))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_names_classify_split_domains() {
        assert_eq!(
            split_domain_from_name("ANALOG_SIG"),
            Some(SplitDomain::Analog),
            "explicit ANALOG token maps to the analog placement domain"
        );
        assert_eq!(
            split_domain_from_name("BUS_SPI_MOSI"),
            Some(SplitDomain::Digital),
            "SPI control nets map to the digital placement domain"
        );
        assert_eq!(
            split_domain_from_name("CLK_MAIN"),
            Some(SplitDomain::Digital),
            "clock nets map to the digital placement domain"
        );
        assert_eq!(
            split_domain_from_name("TX_0"),
            None,
            "unmarked high-speed output nets are not forced into an analog/digital split domain"
        );
    }
}
