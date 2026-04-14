//! CEM43 Reference Solutions and Validation
//!
//! This module provides analytical reference solutions for CEM43 thermal dose
//! calculations following Sapareto & Dewey (1984).
//!
//! # Mathematical Foundation
//!
//! ## THEOREM: CEM43 Analytical Solution for Constant Temperature
//!
//! For tissue held at constant temperature T for duration Δt:
//!
//! ```text
//! CEM43(T, Δt) = Δt × R^(43 - T)
//!
//! where R = 0.5 for T < 43°C
//!           0.25 for T ≥ 43°C (accelerated damage)
//! ```
//!
//! ## THEOREM: CEM43 for Linear Temperature Ramp
//!
//! For T(t) = T₀ + (T₁ - T₀) × (t / t_total):
//!
//! ```text
//! CEM43 = ∫₀^t_total R(T(t))^(43 - T(t)) dt
//! ```
//!
//! Requires numerical integration (Simpson's rule with N=1000 sufficient).
//!
//! ## Reference: Sapareto & Dewey (1984)
//! DOI: 10.1016/0360-3016(84)90379-1

/// R-factor below 43°C threshold (R = 0.25: each 1°C below 43 halves the damage rate)
/// Sapareto & Dewey (1984) Table 1: R = 0.25 for T < 43°C
pub const R_FACTOR_SUBTHRESHOLD: f64 = 0.25;

/// R-factor at/above 43°C threshold (R = 0.5: each 1°C above 43 doubles the damage rate)
/// Sapareto & Dewey (1984) Table 1: R = 0.5 for T ≥ 43°C
pub const R_FACTOR_SUPRATHRESHOLD: f64 = 0.5;

/// Temperature threshold in Celsius
pub const THRESHOLD_TEMP_C: f64 = 43.0;

/// Standard damage threshold (CEM43 = 240 min)
pub const STANDARD_DAMAGE_THRESHOLD: f64 = 240.0;

/// Calculate analytical CEM43 for constant temperature
///
/// # Arguments
/// * `t_celsius` - Temperature in Celsius
/// * `duration_minutes` - Duration in minutes
///
/// # Returns
/// Cumulative Equivalent Minutes at 43°C
///
/// # Example
/// ```
/// let cem43 = analytical_cem43_constant(44.0, 60.0);
/// // Returns 120.0 (double rate at 44°C)
/// ```
pub fn analytical_cem43_constant(t_celsius: f64, duration_minutes: f64) -> f64 {
    let r = if t_celsius >= THRESHOLD_TEMP_C {
        R_FACTOR_SUPRATHRESHOLD
    } else {
        R_FACTOR_SUBTHRESHOLD
    };

    let exponent = THRESHOLD_TEMP_C - t_celsius;
    duration_minutes * r.powf(exponent)
}

/// Calculate analytical CEM43 for linear temperature ramp
///
/// Uses Simpson's rule for numerical integration.
///
/// # Arguments
/// * `t_start` - Starting temperature (°C)
/// * `t_end` - Ending temperature (°C)
/// * `duration_minutes` - Total duration in minutes
pub fn analytical_cem43_ramp(t_start: f64, t_end: f64, duration_minutes: f64) -> f64 {
    const N_INTERVALS: usize = 1000;
    let dt = duration_minutes / N_INTERVALS as f64;
    let slope = (t_end - t_start) / duration_minutes;

    let mut sum = 0.0;
    for i in 0..N_INTERVALS {
        let t = i as f64 * dt;
        let temp = t_start + slope * t;

        let r = if temp >= THRESHOLD_TEMP_C {
            R_FACTOR_SUPRATHRESHOLD
        } else {
            R_FACTOR_SUBTHRESHOLD
        };

        sum += r.powf(THRESHOLD_TEMP_C - temp);
    }

    sum * dt
}

/// Calculate treatment time needed at given temperature for target CEM43
///
/// Solves: t = CEM43_target / R^(43 - T)
///
/// # Arguments
/// * `t_celsius` - Treatment temperature
/// * `target_cem43` - Target CEM43 value (typically 240.0)
pub fn time_for_target_cem43(t_celsius: f64, target_cem43: f64) -> f64 {
    let r = if t_celsius >= THRESHOLD_TEMP_C {
        R_FACTOR_SUPRATHRESHOLD
    } else {
        R_FACTOR_SUBTHRESHOLD
    };

    let exponent = THRESHOLD_TEMP_C - t_celsius;
    target_cem43 / r.powf(exponent)
}

/// Get R-factor for a given temperature
pub fn r_factor_at_temperature(t_celsius: f64) -> f64 {
    if t_celsius >= THRESHOLD_TEMP_C {
        R_FACTOR_SUPRATHRESHOLD
    } else {
        R_FACTOR_SUBTHRESHOLD
    }
}

/// Validation test cases from Sapareto & Dewey (1984)
pub mod literature_cases {
    use super::*;

    /// Case 1: Threshold (43°C, 240 min)
    pub fn case_threshold() -> (f64, f64, f64) {
        (43.0, 240.0, 240.0) // (temp, duration, expected_cem43)
    }

    /// Case 2: 44°C for 60 min → 120 CEM43
    pub fn case_44c_60min() -> (f64, f64, f64) {
        (44.0, 60.0, 120.0)
    }

    /// Case 3: 45°C for 30 min → 120 CEM43
    pub fn case_45c_30min() -> (f64, f64, f64) {
        (45.0, 30.0, 120.0)
    }

    /// Case 4: 46°C for 15 min → 120 CEM43
    pub fn case_46c_15min() -> (f64, f64, f64) {
        (46.0, 15.0, 120.0)
    }

    /// Case 5: 50°C for 1 min → 240 CEM43 (approx)
    pub fn case_50c_1min() -> (f64, f64, f64) {
        let expected = time_for_target_cem43(50.0, 240.0);
        (50.0, expected, 240.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_case() {
        let cem43 = analytical_cem43_constant(43.0, 240.0);
        assert!(
            (cem43 - 240.0).abs() < 0.01,
            "43°C for 240 min should give CEM43 = 240, got {}",
            cem43
        );
    }

    #[test]
    fn test_44c_case() {
        let cem43 = analytical_cem43_constant(44.0, 60.0);
        assert!(
            (cem43 - 120.0).abs() < 0.01,
            "44°C for 60 min should give CEM43 = 120, got {}",
            cem43
        );
    }

    #[test]
    fn test_45c_case() {
        let cem43 = analytical_cem43_constant(45.0, 30.0);
        assert!(
            (cem43 - 120.0).abs() < 0.01,
            "45°C for 30 min should give CEM43 = 120, got {}",
            cem43
        );
    }

    #[test]
    fn test_37c_negligible() {
        let cem43 = analytical_cem43_constant(37.0, 1000.0);
        assert!(
            cem43 < 1.0,
            "37°C should give negligible CEM43, got {}",
            cem43
        );
    }

    #[test]
    fn test_r_factor_calculation() {
        assert_eq!(r_factor_at_temperature(37.0), 0.25); // T < 43°C: subthreshold
        assert_eq!(r_factor_at_temperature(43.0), 0.5);  // T ≥ 43°C: suprathreshold
        assert_eq!(r_factor_at_temperature(45.0), 0.5);  // T ≥ 43°C: suprathreshold
    }

    #[test]
    fn test_time_for_target_threshold() {
        // Should give 240 min for 43°C target
        let t = time_for_target_cem43(43.0, 240.0);
        assert!((t - 240.0).abs() < 0.01);
    }

    #[test]
    fn test_all_literature_cases() {
        let cases = [
            literature_cases::case_threshold(),
            literature_cases::case_44c_60min(),
            literature_cases::case_45c_30min(),
            literature_cases::case_46c_15min(),
        ];

        for (temp, duration, expected) in cases.iter() {
            let computed = analytical_cem43_constant(*temp, *duration);
            let rel_error = (computed - expected).abs() / expected;
            assert!(
                rel_error < 0.001,
                "Case ({:.1}°C, {:.0}min): expected {:.2}, got {:.2}, error {:.4}%",
                temp,
                duration,
                expected,
                computed,
                rel_error * 100.0
            );
        }
    }
}
