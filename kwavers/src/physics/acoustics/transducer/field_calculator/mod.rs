//! Multi-Element Transducer Field Calculator
//!
//! ## References
//!
//! - Jensen & Svendsen (1992). Calculation of pressure fields from arbitrarily shaped transducers
//! - Tupholme (1969) / Stepanishen (1971). Transient radiation from pistons
//! - Zeng & McGough (2008). Evaluation of the angular spectrum approach
//! - Nyborg (1981). Heat generation by ultrasound

pub mod angular_spectrum;
pub mod geometry;
pub mod heating;
pub mod plugin;
pub mod spatial_impulse_response;

pub use geometry::TransducerGeometry;
pub use plugin::TransducerFieldCalculatorPlugin;
