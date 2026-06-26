//! Physics slice errors.
//!
//! Each physics vertical slice owns one sub-enum; this mod is just the namespace.
//! The slices mirror the slice tree under `src/physics/`:
//!
//! * [`thermal::Thermal`] — thermal physics (steady-state ΔT, transient rise, junction
//!   temperature).
//! * [`emi::Emi`] — EMI / commutation-loop inductance / radiated noise.
//! * [`pdn::Pdn`] — power-delivery-network (rail drop, anti-resonance, decoupling).
//! * [`si::Si`] — signal integrity (microstrip/stripline impedance, crosstalk, skew).
//! * [`acoustic::Acoustic`] — acoustic domain (pulser profile, focal mismatch).
//!
//! Each sub-enum exposes the forward-looking variants the corresponding slice will
//! migrate into as Phase 2 / Phase 3 unfold. Today they are placeholders with
//! `#[non_exhaustive]`, so the slice can grow without breaking the aggregator.

pub mod acoustic;
pub mod emi;
pub mod pdn;
pub mod si;
pub mod thermal;

pub use acoustic::Acoustic;
pub use emi::Emi;
pub use pdn::Pdn;
pub use si::Si;
pub use thermal::Thermal;
