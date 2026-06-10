//! Bilayer-sonophore (BLS) intramembrane-cavitation models.
//!
//! The deep-vertical subtree for the mechanical/electrical leaflet model that
//! underlies the NICE pathway, split by concern:
//! - [`capacitance`] — the curved-dome capacitance geometry `C_m(Z)` (Plaksin
//!   Eq. 8) and the kinematic [`BilayerSonophore`] source.
//! - [`pressures`] — the intramembrane force balance (intermolecular, elastic,
//!   electrical, gas), the rest-gap solver, and the quasi-static
//!   [`BilayerSonophoreQuasistatic`] source.
//! - [`dynamics`] — the full transient leaflet Rayleigh–Plesset ODE (Plaksin
//!   Eq. 2) and the [`BilayerSonophoreDynamic`] source.

pub mod capacitance;
pub mod dynamics;
pub mod pressures;

pub use capacitance::{bls_capacitance, BilayerSonophore, LEAFLET_GAP_M, SONOPHORE_RADIUS_M};
pub use dynamics::BilayerSonophoreDynamic;
pub use pressures::{quasistatic_deflection, rest_gap, BilayerSonophoreQuasistatic};
