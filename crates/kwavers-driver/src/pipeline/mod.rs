//! Placement → routing bridge and the place↔route co-optimization loop.
//!
//! After the placer fixes component positions, this derives what the router needs, then runs the
//! adversarial place↔route loop. Carved by role: `config`, `result`, `place_board`,
//! `cooptimize`.

mod config;
mod cooptimize;
mod place_board;
mod result;

#[cfg(test)]
mod tests;

pub use config::{role_footprint_dissipation_w, CoOpt};
pub use cooptimize::{cooptimize, cooptimize_min_area, cooptimize_min_layers};
pub use place_board::{place_to_board, RoutingInputs};
pub use result::CoOptResult;
