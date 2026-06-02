//! Perfusion Modeling for Contrast-Enhanced Ultrasound
//!
//! ## Theory — Advection-Diffusion-Reaction Transport
//!
//! The contrast-agent concentration C [bubbles m⁻³] obeys the
//! advection-diffusion-reaction (ADR) equation:
//!
//! ```text
//! ∂C/∂t = −u·∇C − k_perf·C + S
//! ```
//!
//! where:
//! - `u` = blood flow velocity field [m s⁻¹]
//! - `k_perf` = effective trans-capillary clearance rate [s⁻¹]
//!   computed as `permeability [m s⁻¹] / dx (m)` (Patlak 1983)
//! - `S` = source term [bubbles m⁻³ s⁻¹] (inflow BC at i=0)
//!
//! ## References
//!
//! - Patlak CS, Blasberg RG, Fenstermacher JD (1983).
//!   "Graphical evaluation of blood-to-brain transfer constants from
//!   multiple-time uptake data." *J Cereb Blood Flow Metab* 3, 1–7.
//! - Levenspiel O (1999). *Chemical Reaction Engineering*, 3rd ed.
//!   Wiley, Ch. 13 (residence-time distribution).

mod kinetics;
mod model;

pub use kinetics::{FlowKinetics, PerfusionParameters, TissueUptake};
pub use model::CeusPerfusionModel;
