//! EMI / commutation-loop inductance failures.
//!
//! Forward-looking variants for the [`crate::physics::emi`] slice. The slice currently
//! returns inductance / loss values directly; the variants below are the typed
//! migration surface for Phase 2 once `loop_inductance_nh(...) > budget` and
//! similar breaches stop being returned as `Option<f64>` and become
//! `Result<_, Emi>`.

/// EMI physics failure.
///
/// `Debug` only — every variant carries `f64` fields (`nh`/`budget_nh`,
/// `dbuv_m`/`limit_dbuv_m`, `clearance_mm`/`min_clearance_mm`/`working_v`, `w`/`envelope_w`),
/// which violate `Eq` (`NaN != NaN`).
/// [`Geometry`](super::super::geometry::Geometry) keeps `Copy + Eq` because every field is
/// integer; `Emi` drops them because of the `f64` measurement-and-limit-pair convention.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Emi {
    /// Commutation loop inductance exceeds the device's `dV/dt` survival budget
    /// (or the layout's noise-target budget).
    #[error("loop inductance {nh:.2} nH exceeds budget {budget_nh:.2} nH")]
    LoopInductanceExceeds {
        /// Measured loop inductance (nH).
        nh: f64,
        /// Allowed budget (nH).
        budget_nh: f64,
    },

    /// Radiated EMI exceeds the configured compliance limit (CISPR / FCC / custom).
    #[error("radiated EMI {dbuv_m:.1} dBµV/m exceeds limit {limit_dbuv_m:.1} dBµV/m")]
    RadiatedEmiExceeds {
        /// Measured level (dBµV/m).
        dbuv_m: f64,
        /// Compliance limit (dBµV/m).
        limit_dbuv_m: f64,
    },

    /// HV-to-LV creepage distance is too short to withstand the working voltage
    /// (IEC 60664). Different from `loop inductance` — this is a clearance breach.
    #[error(
        "HV creepage {clearance_mm} mm below minimum {min_clearance_mm} mm at V={working_v} V"
    )]
    CreepageViolated {
        /// Actual edge-to-edge clearance (mm).
        clearance_mm: f64,
        /// Required minimum (mm, IEC 60664 table).
        min_clearance_mm: f64,
        /// Working RMS voltage on the HV copper (V).
        working_v: f64,
    },

    /// Reverse-recovery dissipation of a co-packaged diode exceeds the thermal
    /// envelope of the heatsink.
    #[error("reverse-recovery dissipation {w} W exceeds thermal envelope {envelope_w} W")]
    ReverseRecoveryOverheats {
        /// Computed dissipation (W).
        w: f64,
        /// Available thermal envelope (W).
        envelope_w: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_inductance_exceeds_names_both_sides() {
        assert!(Emi::LoopInductanceExceeds {
            nh: 200.5,
            budget_nh: 150.0,
        }
        .to_string()
        .contains("200.50 nH"));
    }

    #[test]
    fn radiated_emi_names_measured_and_limit() {
        let s = Emi::RadiatedEmiExceeds {
            dbuv_m: 48.5,
            limit_dbuv_m: 40.0,
        }
        .to_string();
        assert!(s.contains("48.5"));
        assert!(s.contains("40.0"));
    }

    #[test]
    fn creepage_carries_voltage_context() {
        // The Display format uses Rust's default f64 Display (shortest round-trip
        // decimal), so 0.8 displays as "0.8", 2.5 as "2.5", and 150.0 as "150".
        // We assert substrings that are stable across that formatting, not exact
        // formatted literals (which would be brittle under display-rule changes).
        let s = Emi::CreepageViolated {
            clearance_mm: 0.8,
            min_clearance_mm: 2.5,
            working_v: 150.0,
        }
        .to_string();
        assert!(s.contains("150 V"), "working_v voltage context: {s}");
        assert!(
            s.contains("0.8"),
            "actual clearance appears in display: {s}"
        );
        assert!(
            s.contains("2.5"),
            "minimum clearance appears in display: {s}"
        );
        assert!(s.contains("mm"), "mm unit appears in display: {s}");
        assert!(
            s.contains(" below minimum "),
            "the breach phrase appears: {s}"
        );
    }

    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(e: Emi) -> &'static str {
            match e {
                Emi::LoopInductanceExceeds { .. } => "li",
                Emi::RadiatedEmiExceeds { .. } => "re",
                Emi::CreepageViolated { .. } => "cv",
                Emi::ReverseRecoveryOverheats { .. } => "rro",
                _ => "future-variant",
            }
        }
    }
}
