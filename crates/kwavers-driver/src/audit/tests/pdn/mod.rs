//! PDN (Power Delivery Network) audit tests, split into three bounded-context sub-modules:
//!
//! - `copper`: copper-balance / stackup-warp checks
//! - `decoupling`: decoupling-cap placement (ground-via proximity, power-layer match, loop area,
//!   active-IC internal power plane)
//! - `charge`: charge-reservoir, charge-recycling, and pulse-skip checks

use super::*;

mod charge;
mod copper;
mod decoupling;
