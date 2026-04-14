//! Validated archive access and owned conversion using `rkyv`.

#[cfg(feature = "zero-copy")]
mod grid_archive;
#[cfg(feature = "zero-copy")]
mod simulation_archive;

#[cfg(feature = "zero-copy")]
pub use grid_archive::{
    access_serializable_grid, deserialize_grid, serialize_grid, SerializableGrid,
};
#[cfg(feature = "zero-copy")]
pub use simulation_archive::SimulationData;

#[cfg(all(test, feature = "zero-copy"))]
mod tests;
