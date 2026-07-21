//! Biological-response provider adapters for thermal field storage.

mod cem43;

pub(crate) use cem43::{checked_cem43_increments, CelsiusStorage, KelvinStorage};
