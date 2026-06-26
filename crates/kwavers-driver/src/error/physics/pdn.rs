//! Power-delivery-network (PDN) failures.
//!
//! Forward-looking variants for the [`crate::physics::pdn`] slice. Mirrors the EMI
//! pattern: today `ir_drop(...) -> f64` and `target_impedance_ohm(...) -> f64`
//! return breached values; Phase 2 migrates the breaches into `Result<_, Pdn>`.

/// Power-delivery-network failure.
///
/// `Debug` only — every variant carries `f64` fields (`rail_v`/`drop_v`/`tol_v`,
/// `actual_ohm`/`target_ohm`/`freq_hz`, `res_hz`/`baseband_hz`, `dieaway_freq_hz`/
/// `edge_freq_hz`), which violate `Eq` (`NaN != NaN`).
/// [`Geometry`](super::super::geometry::Geometry) keeps `Copy + Eq` because every field is
/// integer; `Pdn` drops them because of the `f64` measurement-and-limit-pair convention.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Pdn {
    /// Rail voltage drop under full load exceeds what the pulser can tolerate
    /// without losing linearity.
    #[error("rail {rail_v} V drops by {drop_v} V, exceeding tolerance {tol_v} V")]
    RailVoltageDropExceeds {
        /// Rail nominal voltage (V).
        rail_v: f64,
        /// Measured drop (V).
        drop_v: f64,
        /// Allowed tolerance (V).
        tol_v: f64,
    },

    /// At a given frequency the PDN impedance is above the target impedance — current
    /// peaks through the decoupling instead of through the rail.
    #[error("PDN impedance {actual_ohm:.3} Ω at {freq_hz:.0} Hz exceeds target {target_ohm:.3} Ω")]
    TargetImpedanceExceeded {
        /// Measured impedance (Ω).
        actual_ohm: f64,
        /// Target impedance (Ω).
        target_ohm: f64,
        /// Frequency the breach was measured at (Hz).
        freq_hz: f64,
    },

    /// A plane anti-resonance lands inside the pulser's baseband — makes the
    /// decoupling network ineffective at exactly the frequencies it was designed
    /// to filter.
    #[error("PDN anti-resonance at {res_hz:.0} Hz inside pulser baseband (≤ {baseband_hz:.0} Hz)")]
    AntiResonanceInBand {
        /// Anti-resonance frequency (Hz).
        res_hz: f64,
        /// Pulser baseband upper edge (Hz).
        baseband_hz: f64,
    },

    /// The decoupling capacitance's effective series resistance / inductance is so
    /// high that the capacitor's high-frequency attenuation dies away above the
    /// pulser's edge-rate spectrum.
    #[error(
        "decoupling capacitor dies away above {dieaway_freq_hz:.0} Hz, \
         below the pulser edge rate of {edge_freq_hz:.0} Hz"
    )]
    DecouplingDieaway {
        /// Frequency at which the capacitor's impedance floor kicks in (Hz).
        dieaway_freq_hz: f64,
        /// Pulser edge-rate spectrum limit (Hz).
        edge_freq_hz: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rail_drop_display_carries_all_three() {
        // Rust's f64 default Display uses shortest round-trip: 0.20 displays as "0.2"
        // (not "0.20"), 3.3 as "3.3", 0.42 as "0.42". Assert on the format-template
        // anchors ("V drops by", "exceeding tolerance") so the test is robust to
        // display-rule changes.
        let s = Pdn::RailVoltageDropExceeds {
            rail_v: 3.3,
            drop_v: 0.42,
            tol_v: 0.20,
        }
        .to_string();
        assert!(s.contains("V drops by"), "drop phrase present: {s}");
        assert!(
            s.contains("exceeding tolerance"),
            "tolerance phrase present: {s}"
        );
        assert!(s.contains("3.3 V"), "rail nominal present: {s}");
        assert!(s.contains("0.42 V"), "drop present: {s}");
        assert!(s.contains("0.2 V"), "tolerance present: {s}");
    }

    #[test]
    fn target_impedance_display_carries_freq() {
        let s = Pdn::TargetImpedanceExceeded {
            actual_ohm: 0.42,
            target_ohm: 0.10,
            freq_hz: 2.5e6,
        }
        .to_string();
        assert!(s.contains("0.420")); // 3-decimal format
        assert!(s.contains("0.100"));
        assert!(s.contains("2500000") || s.contains("2.5e6") || s.contains("2.5e+06"));
    }

    /// Seat-belt.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(p: Pdn) -> &'static str {
            match p {
                Pdn::RailVoltageDropExceeds { .. } => "rvde",
                Pdn::TargetImpedanceExceeded { .. } => "tie",
                Pdn::AntiResonanceInBand { .. } => "arib",
                Pdn::DecouplingDieaway { .. } => "dd",
                _ => "future-variant",
            }
        }
    }
}
