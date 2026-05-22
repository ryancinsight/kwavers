//! Physical constants for Mie scattering calculations

/// Speed of light in vacuum [m/s] — re-exported from `crate::core::constants::fundamental`.
pub use crate::core::constants::fundamental::SPEED_OF_LIGHT as C;
/// Vacuum permittivity [F/m] — re-exported from `crate::core::constants::fundamental`.
pub const EPSILON_0: f64 = crate::core::constants::fundamental::VACUUM_PERMITTIVITY;
/// Vacuum permeability [H/m] — re-exported from `crate::core::constants::fundamental`.
pub use crate::core::constants::fundamental::VACUUM_PERMEABILITY as MU_0;
