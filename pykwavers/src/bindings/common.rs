use kwavers::core::error::KwaversError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Convert kwavers errors to Python exceptions.
pub(crate) fn kwavers_error_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}

/// Convert k-Wave absorption units dB/(MHz^y·cm) to Np/m at the given frequency.
///
/// This follows the standard scalar conversion
/// `alpha_np_m = alpha_db_cm * f_mhz^y * 100 / (20 / ln(10))`.
/// It is used by the GPU PSTD paths, which currently apply absorption using a
/// centre-frequency attenuation model rather than the full spectral Treeby/Cox
/// formulation used by the CPU PSTD solver.
#[cfg(feature = "gpu")]
pub(crate) fn alpha_db_cm_to_np_m(alpha_db_cm: f64, frequency_mhz: f64, alpha_power: f64) -> f64 {
    let db_to_np = 20.0 / std::f64::consts::LN_10;
    alpha_db_cm * frequency_mhz.powf(alpha_power) * 100.0 / db_to_np
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu")]
    use super::alpha_db_cm_to_np_m;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_alpha_db_cm_to_np_m_matches_scalar_reference_at_1mhz() {
        let alpha_db_cm = 0.75;
        let got = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let expected = alpha_db_cm * 100.0 / (20.0 / std::f64::consts::LN_10);
        assert!(
            (got - expected).abs() < 1e-12,
            "conversion mismatch: got {got}, expected {expected}"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_alpha_db_cm_to_np_m_respects_power_law_frequency_scaling() {
        let alpha_db_cm = 0.5;
        let at_1mhz = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let at_2mhz = alpha_db_cm_to_np_m(alpha_db_cm, 2.0, 1.5);
        let expected_ratio = 2.0_f64.powf(1.5);
        let got_ratio = at_2mhz / at_1mhz;
        assert!(
            (got_ratio - expected_ratio).abs() < 1e-12,
            "power-law scaling mismatch: got {got_ratio}, expected {expected_ratio}"
        );
    }
}
