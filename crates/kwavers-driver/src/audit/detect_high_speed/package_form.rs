//! Through-hole / leaded package violation detector for high-speed nets.
//!
//! TI Application Note SLYP173 §5-20/5-21, §5-30: "Avoid any leaded devices for
//! high-frequency designs" and "Choose SOIC or smaller packages over DIP packages."
//! Leaded through-hole packages introduce 2–20 nH of lead inductance per pin, which
//! disrupts signal integrity and decoupling action at high frequencies.
//!
//! The check is vacuous when a component's [`crate::place::PackageFormFactor`] is
//! [`crate::place::PackageFormFactor::Unknown`] (no package metadata available).

use crate::board::Board;
use crate::geom::Point;
use crate::place::{Component, FootprintDef, PackageFormFactor, Role};

use super::super::net_util::{is_clock_like_net, is_high_speed_net};

/// Flags active ICs, decoupling capacitors, or high-speed-net components that use a
/// through-hole / leaded package (TI SLYP173 §5-20/5-21, §5-30).
///
/// A component is flagged when **all** of the following hold:
/// 1. Its `package_form_factor` is [`PackageFormFactor::ThroughHole`].
/// 2. It is an `ActiveIc`, a `Decoupling` cap, or any component whose pads are wired to
///    at least one high-speed or clock-like net on the board.
///
/// This captures:
/// * High-speed ICs in DIP packages (high lead inductance on every pin).
/// * Through-hole bypass caps (lead inductance defeats high-frequency bypass action).
/// * Any other component connected to a high-speed/clock net with a leaded body.
///
/// # Vacuous condition
///
/// Returns `(0, [])` when the board has no through-hole components.
pub(crate) fn detect_through_hole_high_speed_violations(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> (usize, Vec<Point>) {
    let mut count = 0;
    let mut pts = Vec::new();

    for comp in comps {
        if comp.fp >= lib.len() {
            continue;
        }
        let fp = &lib[comp.fp];
        if fp.package_form_factor != PackageFormFactor::ThroughHole {
            continue;
        }
        // Active ICs and decoupling caps are always flagged regardless of net names —
        // both categories are dominated by high-frequency concerns.
        let flagged_by_role = matches!(fp.role, Role::ActiveIc | Role::Decoupling);
        // Any other through-hole component that lands on a high-speed or clock net.
        let flagged_by_net = comp
            .nets
            .iter()
            .any(|n| n.is_some_and(|id| is_high_speed_net(board, id) || is_clock_like_net(board, id)));

        if flagged_by_role || flagged_by_net {
            count += 1;
            pts.push(comp.placement.pos);
        }
    }
    (count, pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, NetClassKind};
    use crate::geom::{GridSpec, Nm};
    use crate::place::Placement;
    use crate::place::rotation::Rot;

    fn make_board() -> Board {
        let spec = GridSpec::cover(
            Nm::from_mm(50.0),
            Nm::from_mm(50.0),
            Nm::from_mm(0.5),
            2,
        )
        .unwrap();
        Board::new(spec)
    }

    fn make_fp(role: Role, form: PackageFormFactor) -> FootprintDef {
        FootprintDef::new(
            "TEST",
            (Nm::from_mm(2.0), Nm::from_mm(2.0)),
            role,
            vec![],
        )
        .with_package_form_factor(form)
    }

    fn make_comp(fp: usize, net: Option<crate::board::NetId>) -> Component {
        Component {
            fp,
            refdes: String::new(),
            nets: vec![net],
            placement: Placement {
                pos: crate::geom::Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
                rot: Rot::R0,
            },
            assoc_ic: None,
            ..Component::default()
        }
    }

    /// A through-hole ActiveIc must be flagged.
    #[test]
    fn flags_through_hole_active_ic() {
        let b = make_board();
        let lib = vec![make_fp(Role::ActiveIc, PackageFormFactor::ThroughHole)];
        let comps = vec![make_comp(0, None)];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 1, "through-hole ActiveIc must be flagged");
    }

    /// A through-hole Decoupling cap must be flagged.
    #[test]
    fn flags_through_hole_decoupling_cap() {
        let b = make_board();
        let lib = vec![make_fp(Role::Decoupling, PackageFormFactor::ThroughHole)];
        let comps = vec![make_comp(0, None)];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 1, "through-hole Decoupling cap must be flagged");
    }

    /// A through-hole Passive on a CLK net must be flagged.
    #[test]
    fn flags_through_hole_passive_on_clk_net() {
        let mut b = make_board();
        let clk_net = b.add_net("CLK", NetClassKind::Signal);
        let lib = vec![make_fp(Role::Passive, PackageFormFactor::ThroughHole)];
        let comps = vec![make_comp(0, Some(clk_net))];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 1, "through-hole passive on CLK net must be flagged");
    }

    /// An SMT ActiveIc on a CLK net must NOT be flagged.
    #[test]
    fn passes_smt_active_ic() {
        let mut b = make_board();
        let clk_net = b.add_net("CLK", NetClassKind::Signal);
        let lib = vec![make_fp(Role::ActiveIc, PackageFormFactor::Smt)];
        let comps = vec![make_comp(0, Some(clk_net))];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 0, "SMT ActiveIc must not be flagged");
    }

    /// A through-hole Passive on a low-speed net must NOT be flagged.
    #[test]
    fn passes_through_hole_passive_on_low_speed_net() {
        let mut b = make_board();
        let uart_net = b.add_net("UART_TX", NetClassKind::Signal);
        let lib = vec![make_fp(Role::Passive, PackageFormFactor::ThroughHole)];
        let comps = vec![make_comp(0, Some(uart_net))];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 0, "through-hole passive on low-speed net must pass");
    }

    /// A component with Unknown package form factor must not be flagged.
    #[test]
    fn passes_unknown_package_form_factor() {
        let b = make_board();
        let lib = vec![make_fp(Role::ActiveIc, PackageFormFactor::Unknown)];
        let comps = vec![make_comp(0, None)];
        let (n, _) = detect_through_hole_high_speed_violations(&b, &comps, &lib);
        assert_eq!(n, 0, "Unknown package form factor is vacuous — must not flag");
    }
}
