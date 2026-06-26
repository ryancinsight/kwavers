//! Transmission-line length checks for high-frequency signal traces.
//!
//! A trace becomes a transmission line (requiring controlled impedance / termination) when
//! its physical length exceeds **λ/10** of the highest-frequency component of the signal it
//! carries. Beyond that threshold, signal reflections, standing waves, and EMI radiation
//! become significant — the root cause of PCB article mistake #7 ("excessive trace length").
//!
//! # Threshold derivation
//!
//! The wavelength in a PCB medium with relative permittivity εᵣ is:
//!
//! ```text
//! λ = c₀ / (f · √εᵣ)
//! ```
//!
//! where c₀ = 2.998 × 10⁸ m/s and εᵣ ≈ 4.5 for FR4 (the √εᵣ = 2.12 factor reduces the
//! free-space wavelength by the velocity factor of the dielectric). The threshold is λ/10:
//!
//! ```text
//! L_max = c₀ / (10 · f · √εᵣ)   [metres]
//!       = 29.98 mm / (f_GHz · √εᵣ)
//! ```
//!
//! For FR4 (εᵣ = 4.5, √εᵣ ≈ 2.121):
//! - f = 100 MHz → L_max ≈ 141 mm
//! - f = 1 GHz  → L_max ≈ 14.1 mm
//! - f = 5 GHz  → L_max ≈ 2.83 mm
//! - f = 10 GHz → L_max ≈ 1.41 mm
//!
//! # Evidence tier
//!
//! Compile-time derivation from Maxwell's equations + measured FR4 permittivity. The λ/10
//! boundary is the industry-standard rule-of-thumb cited in IPC-2141A and most SI textbooks.

use crate::board::{Board, NetId};
use crate::validate::board_checks::net_length_mm;

/// Speed of light in vacuum [m/s].
const C0_M_S: f64 = 2.997_924_58e8;

/// Relative permittivity of FR4 (IPC-4101C nominal, loss-tangent 0.02).
const FR4_EPSILON_R: f64 = 4.5;

/// Velocity factor of FR4: 1/√εᵣ.
const FR4_VELOCITY_FACTOR: f64 = 1.0 / 2.121_320_344; // 1/√4.5

/// Maximum routed length (mm) before a trace must be treated as a transmission line.
///
/// Derived from λ/10 = c₀ / (10 · f · √εᵣ). The result is a physical constraint in the
/// PCB medium, not a heuristic — the derivation is the evidence.
///
/// # Arguments
///
/// * `freq_hz` — the highest signal frequency (Hz). For clocks, use the fundamental; for
///   fast digital edges, use `0.35 / t_rise` where `t_rise` is the 20–80% rise time in seconds.
/// * `epsilon_r` — relative permittivity of the PCB substrate (pass `None` to use the FR4
///   default of 4.5).
#[must_use]
pub fn transmission_line_threshold_mm(freq_hz: f64, epsilon_r: Option<f64>) -> f64 {
    if freq_hz <= 0.0 {
        return f64::INFINITY;
    }
    let er = epsilon_r.unwrap_or(FR4_EPSILON_R);
    let vf = if er > 0.0 { 1.0 / er.sqrt() } else { FR4_VELOCITY_FACTOR };
    // λ/10 in metres, then converted to mm.
    (C0_M_S * vf / (10.0 * freq_hz)) * 1000.0
}

/// A net whose routed length exceeds the λ/10 transmission-line threshold.
///
/// The `margin_mm` is negative (excess over threshold); a positive value means the net is
/// safely below the threshold. Callers can use `is_violation()` to filter.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionLineViolation {
    /// The net that is too long.
    pub net: NetId,
    /// Net name for display.
    pub net_name: String,
    /// Routed length of the net (mm).
    pub length_mm: f64,
    /// λ/10 threshold (mm).
    pub threshold_mm: f64,
}

impl TransmissionLineViolation {
    /// True when the net exceeds the threshold (the normal case after filtering).
    #[must_use]
    pub fn is_violation(&self) -> bool {
        self.length_mm > self.threshold_mm
    }

    /// Excess length over the threshold (mm); negative when within budget.
    #[must_use]
    pub fn excess_mm(&self) -> f64 {
        self.length_mm - self.threshold_mm
    }
}

/// Check every routed net in `board` against the λ/10 transmission-line threshold and return
/// the subset that exceed it.
///
/// Only nets with non-zero routed length are evaluated. Unconnected or unrouted nets are
/// silently skipped. Pass `signal_nets` to restrict the check to a specific subset (e.g. clock
/// or differential-pair nets); pass `&[]` or the full `board.nets` slice to check all nets.
///
/// # Arguments
///
/// * `board` — the board to check.
/// * `freq_hz` — highest signal frequency in Hz (see [`transmission_line_threshold_mm`]).
/// * `signal_nets` — the subset of `NetId`s to evaluate; if empty, all nets are checked.
/// * `epsilon_r` — PCB substrate permittivity (`None` → FR4 default 4.5).
///
/// # Returns
///
/// A vector of [`TransmissionLineViolation`]s, one per net that exceeds λ/10. Empty if no
/// net violates the constraint.
#[must_use]
pub fn check_transmission_line_lengths(
    board: &Board,
    freq_hz: f64,
    signal_nets: &[NetId],
    epsilon_r: Option<f64>,
) -> Vec<TransmissionLineViolation> {
    let threshold = transmission_line_threshold_mm(freq_hz, epsilon_r);
    let nets_to_check: Vec<NetId> = if signal_nets.is_empty() {
        board.nets.iter().map(|n| n.id).collect()
    } else {
        signal_nets.to_vec()
    };
    nets_to_check
        .into_iter()
        .filter_map(|net| {
            let length = net_length_mm(board, net);
            if length <= 0.0 {
                return None; // unrouted — not a transmission-line concern
            }
            let net_name = board
                .nets
                .get(net.0 as usize)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            Some(TransmissionLineViolation {
                net,
                net_name,
                length_mm: length,
                threshold_mm: threshold,
            })
        })
        .filter(|v| v.is_violation())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, NetClassKind, Track};
    use crate::geom::{GridSpec, Nm, Point};

    fn tiny_board() -> (Board, NetId) {
        let spec = GridSpec::cover(Nm::from_mm(200.0), Nm::from_mm(200.0), Nm(250_000), 2)
            .expect("invariant: valid board dimensions");
        let mut b = Board::new(spec);
        let n = b.add_net("CLK", NetClassKind::Signal);
        (b, n)
    }

    // ── threshold derivation ──────────────────────────────────────────────

    #[test]
    fn threshold_100mhz_fr4_near_141mm() {
        // λ/10 at 100 MHz in FR4 (εᵣ = 4.5): c₀/(10·f·√εᵣ) = 2.998e8/(10·1e8·2.121) ≈ 141.3 mm
        let t = transmission_line_threshold_mm(100e6, None);
        assert!((t - 141.3).abs() < 1.0, "expected ≈141 mm, got {t:.2}");
    }

    #[test]
    fn threshold_1ghz_fr4_near_14mm() {
        let t = transmission_line_threshold_mm(1e9, None);
        assert!((t - 14.13).abs() < 0.2, "expected ≈14.1 mm, got {t:.2}");
    }

    #[test]
    fn threshold_zero_freq_is_infinity() {
        assert!(transmission_line_threshold_mm(0.0, None).is_infinite());
    }

    #[test]
    fn threshold_scales_with_epsilon_r() {
        // Higher εᵣ → shorter λ → tighter threshold.
        let fr4 = transmission_line_threshold_mm(1e9, None); // εᵣ = 4.5
        let higher = transmission_line_threshold_mm(1e9, Some(9.0)); // εᵣ = 9.0
        assert!(
            higher < fr4,
            "higher εᵣ must produce a shorter threshold; fr4={fr4:.2}, higher={higher:.2}"
        );
    }

    // ── check_transmission_line_lengths ──────────────────────────────────

    #[test]
    fn net_below_threshold_not_flagged() {
        let (mut board, net) = tiny_board();
        // Add a 5 mm track — well below the 1 GHz / FR4 threshold of ~14 mm.
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            end: Point::new(Nm::from_mm(5.0), Nm::from_mm(0.0)),
            width: Nm(250_000),
            layer: crate::board::LayerId(0),
            net,
        });
        let violations = check_transmission_line_lengths(&board, 1e9, &[], None);
        assert!(violations.is_empty(), "5 mm trace at 1 GHz must not trigger");
    }

    #[test]
    fn net_above_threshold_flagged() {
        let (mut board, net) = tiny_board();
        // Add a 30 mm track — exceeds the 1 GHz / FR4 threshold of ~14 mm.
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            end: Point::new(Nm::from_mm(30.0), Nm::from_mm(0.0)),
            width: Nm(250_000),
            layer: crate::board::LayerId(0),
            net,
        });
        let violations = check_transmission_line_lengths(&board, 1e9, &[], None);
        assert_eq!(violations.len(), 1, "30 mm trace at 1 GHz must trigger");
        let v = &violations[0];
        assert_eq!(v.net_name, "CLK");
        assert!((v.length_mm - 30.0).abs() < 0.1, "length: {}", v.length_mm);
        assert!(v.excess_mm() > 0.0, "excess must be positive: {}", v.excess_mm());
    }

    #[test]
    fn signal_nets_filter_limits_scope() {
        let (mut board, net) = tiny_board();
        let other = board.add_net("POWER", NetClassKind::Power);
        // POWER has a long trace, CLK has a short one.
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            end: Point::new(Nm::from_mm(50.0), Nm::from_mm(0.0)),
            width: Nm(500_000),
            layer: crate::board::LayerId(0),
            net: other,
        });
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(0.0), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(5.0), Nm::from_mm(5.0)),
            width: Nm(250_000),
            layer: crate::board::LayerId(0),
            net,
        });
        // Check only CLK — POWER should not appear even though it violates.
        let violations = check_transmission_line_lengths(&board, 1e9, &[net], None);
        assert!(
            violations.is_empty(),
            "CLK is short — filter to signal_nets must exclude POWER"
        );
    }

    #[test]
    fn unrouted_net_not_flagged() {
        let (board, net) = tiny_board();
        // No tracks for this net — zero length.
        let violations = check_transmission_line_lengths(&board, 100e6, &[net], None);
        assert!(violations.is_empty(), "unrouted net must not be flagged");
    }

    #[test]
    fn violation_fields_are_correct() {
        let (mut board, net) = tiny_board();
        board.tracks.push(Track {
            start: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            end: Point::new(Nm::from_mm(20.0), Nm::from_mm(0.0)),
            width: Nm(250_000),
            layer: crate::board::LayerId(0),
            net,
        });
        let violations = check_transmission_line_lengths(&board, 1e9, &[], None);
        assert_eq!(violations.len(), 1);
        let v = &violations[0];
        assert_eq!(v.net, net);
        assert!(v.is_violation());
        // threshold ≈ 14.13 mm; length = 20 mm; excess ≈ 5.87 mm
        let expected_excess = 20.0 - transmission_line_threshold_mm(1e9, None);
        assert!(
            (v.excess_mm() - expected_excess).abs() < 0.01,
            "excess: {} vs expected {}",
            v.excess_mm(),
            expected_excess
        );
    }
}
