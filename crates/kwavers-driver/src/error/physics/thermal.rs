//! Thermal physics failures.
//!
//! Forward-looking variants for the [`crate::physics::thermal`] slice. Currently the
//! slice returns `f64` ΔT values, not errors; the variants below are the typed
//! migration surface for Phase 2 once the slice stops absorbing budget
//! breaches into ad-hoc numerical heuristics.

/// Thermal physics failure.
///
/// `Debug` only — every variant carries `f64` fields (`cool_dt_k`, `tj_k`/`tj_max_k`,
/// `dt_k`/`margin_k`, `tau_s`), which violate `Eq` (`NaN != NaN`).
/// [`Geometry`](super::super::geometry::Geometry) keeps `Copy + Eq` because every field is
/// integer; `Thermal` drops them because the variants carry `f64` loss measurements.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Thermal {
    /// No board count in the search range hits the thermal rise budget — physically,
    /// the dissipated power cannot leave the stack at any feasible stack height.
    #[error("thermal: no stack configuration meets ΔT budget ({dt_k} K)")]
    CoolingInfeasible {
        /// Maximum allowed per-board temperature rise (K).
        dt_k: f64,
    },

    /// Junction temperature exceeds the part's datasheet absolute maximum.
    #[error("junction temperature {tj_k} K exceeds Tj_max {tj_max_k} K")]
    JunctionExceeded {
        /// Junction temperature the device reaches (K).
        tj_k: f64,
        /// Datasheet absolute maximum (K).
        tj_max_k: f64,
    },

    /// Transient thermal rise over the largest pulsers-on / pulsers-off pulse exceeds
    /// the steady-state budget by more than the configured transient margin.
    #[error("transient thermal rise {dt_k:.2} K exceeds margin {margin_k:.2} K")]
    TransientRiseExceeded {
        /// Peak transient rise (K).
        dt_k: f64,
        /// Allowed transient margin (K).
        margin_k: f64,
    },

    /// The thermal-time-constant estimate fell below the physics floor (τ < 0).
    #[error("thermal time constant {tau_s} s is non-physical")]
    NonPhysicalTimeConstant {
        /// Computed time constant (s); should be > 0.
        tau_s: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooling_infeasible_names_dt_budget() {
        assert_eq!(
            Thermal::CoolingInfeasible { dt_k: 30.0 }.to_string(),
            "thermal: no stack configuration meets ΔT budget (30 K)"
        );
    }

    #[test]
    fn junction_exceeded_carries_tj_and_limit() {
        let s = Thermal::JunctionExceeded {
            tj_k: 425.0,
            tj_max_k: 398.15,
        }
        .to_string();
        assert!(s.contains("425"));
        assert!(s.contains("398.15"));
    }

    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(t: Thermal) -> &'static str {
            match t {
                Thermal::CoolingInfeasible { .. } => "ci",
                Thermal::JunctionExceeded { .. } => "je",
                Thermal::TransientRiseExceeded { .. } => "tre",
                Thermal::NonPhysicalTimeConstant { .. } => "nptc",
                _ => "future-variant",
            }
        }
    }
}
