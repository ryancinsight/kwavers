//! IMEX-RK type definitions: scheme enum and configuration.

/// Types of IMEX-RK schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IMEXRKType {
    /// IMEX-SSP2(2,2,2) - 2nd order SSP scheme
    SSP2_222,
    /// IMEX-SSP3(3,3,3) - 3rd order SSP scheme
    SSP3_333,
    /// IMEX-ARK3 - 3rd order additive Runge-Kutta
    ARK3,
    /// IMEX-ARK4 - 4th order additive Runge-Kutta
    ARK4,
}

/// Configuration for IMEX-RK schemes
#[derive(Debug, Clone)]
pub struct IMEXRKConfig {
    /// Type of IMEX-RK scheme
    pub scheme_type: IMEXRKType,
    /// Whether to use embedded error estimation
    pub embedded_error: bool,
}

impl Default for IMEXRKConfig {
    fn default() -> Self {
        Self {
            scheme_type: IMEXRKType::ARK3,
            embedded_error: true,
        }
    }
}
