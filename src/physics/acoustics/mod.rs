//! Acoustic physics module

pub mod analysis;
pub mod analytical;
pub mod conservation;
// field_indices moved to domain/field/indices.rs
// field_mapping moved to domain/field/mapping.rs
pub mod functional;
pub mod imaging;
pub mod mechanics;
pub mod bubble_dynamics;
pub mod wave_propagation;
pub mod skull;
pub mod state;
pub mod therapy;
pub mod traits;
pub mod transcranial;
// pub mod validation; // Moved to solver/validation/physics_benchmarks
// wave_fields moved to domain/field/wave.rs

pub use conservation::*;
// field_indices moved
// field_mapping moved
pub use state::*;
pub use traits::*;
// wave_fields moved
