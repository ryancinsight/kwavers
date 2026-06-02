/// Maximum shape mode index tracked: n = 2, 3, ..., `N_MODES + 1`.
///
/// Modes above n approximately 6 have high capillary damping in this model.
pub const N_MODES: usize = 5;

/// Fraction of bubble radius at which a mode is considered unstable.
pub const BREAKUP_FRACTION: f64 = 0.3;

/// Stand-off ratio below which a jet forms near a rigid wall.
pub const JET_STANDOFF_CRITICAL: f64 = 2.0;
