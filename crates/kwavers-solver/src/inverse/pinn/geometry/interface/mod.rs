//! Heterogeneous PINN region interfaces.

mod condition;
mod domain;
mod error;

pub use condition::PinnGeometryInterfaceCondition;
pub use domain::MultiRegionDomain;
pub use error::MultiRegionError;
