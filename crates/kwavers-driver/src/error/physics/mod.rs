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
//! Each sub-enum declares the variants its slice will return once that slice's kernels
//! stop absorbing budget breaches and start returning `Result`. Until then they carry no
//! constructed variants; `#[non_exhaustive]` lets each grow without breaking the
//! aggregator. The adoption order is tracked in `docs/MIGRATION.md`.

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
