//! Biological-response provider adapters for thermal field storage.

mod cem43;

pub use cem43::{
    cem43_equivalent_minutes, cem43_rate_per_minute, cem43_reference_celsius,
    checked_cem43_increments, CelsiusStorage, Cem43Accumulator, KelvinStorage,
    StoredTemperatureScale,
};
