//! Conservation validation metrics.

/// Conservation validation metrics for a single timestep.
#[derive(Debug, Clone)]
pub struct ConservationMetrics {
    /// Relative energy error: `|E(t) - E(0)| / E(0)`.
    pub energy_error: f64,
    /// Maximum pointwise mass continuity residual [kg m^-3 s^-1].
    pub mass_error: f64,
    /// Maximum pointwise linearised momentum residual per axis [N m^-3].
    pub momentum_error: (f64, f64, f64),
    /// Volumetric irreversible entropy production rate [W/K].
    pub entropy_production_rate: f64,
    /// True when numerical residuals satisfy tolerances and entropy production is nonnegative.
    pub is_conserved: bool,
}
