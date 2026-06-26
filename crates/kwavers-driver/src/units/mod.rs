//! # Unit newtypes — type-level unit safety at zero runtime cost.
//!
//! The unit-bearing quantities in this crate (lengths, frequencies, impedances,
//! powers, voltages, currents, capacitances, inductances, charges, temperatures)
//! are encapsulated in `#[repr(transparent)]` Rust newtypes. The compiler refuses to
//! add an `Ohm` to a `Volt`, and silently passing a farad where a siemens belongs
//! becomes a type error rather than a runtime surprise. Conversion to `f64` is
//! explicit (`value()`), so the source/SSOT of every unit-bearing value is grep-able.
//!
//! # SSOT role
//!
//! Every board-space length lives in `Nm` (exact integer nm — the same integer KiCad
//! uses internally, so design-rule clearances are exact distances, not floating
//! values rounded at the rule boundary). Every per-domain quantity in the physics
//! kernels lives in a `Float(pub f64)` wrapper whose `value()` extraction is the
//! canonical escape hatch to `f64` for displaying, plotting, or handing to a 2.5D
//! field solver. All conversions between units go through `From`/`Into` so the type
//! signatures document the physical dimension at the call site.
//!
//! # Arithmetic
//!
//! Each newtype implements the operators that are dimensionally valid:
//!
//! * Same-unit sum/diff: `Ohm + Ohm`, `Watt + Watt`, `Hz + Hz`, `Celsius + Celsius`.
//! * Cross-unit products (defined for physical dimensions only):
//!   `Ohm * Amp = Volt` (Ohm's law), `Volt * Amp = Watt`, `Farad * Volt = Coulomb`.
//! * Scalar scaling: `Hz * f64 → Hz`, `Watt / f64 → Watt`.
//!
//! Arithmetic that would silently mix dimensions (`Hz + Ohm`, `Watt + Volt`) is
//! not implemented; the compiler will report a type mismatch. This is the point.
//!
//! # Slice layout
//!
//! Carved by **role** (Phase 4k). Plain backticks name the slice-private submodules; the public
//! items each hosts stay clickable.
//! * `length` — the [`Nm`] integer board-space length newtype (independent of the soft-unit system).
//! * `quantity` — the [`Unit`] trait, the `#[repr(transparent)]` [`Float<U>`] wrapper, and
//!   [`approx_eq`].
//! * `kinds` — the per-unit ZST kind markers + the concrete aliases ([`Hz`], [`Ohm`], …).
//! * `factories` — SI-prefix constructors + the temperature (K/°C/°F) bridge (impls only).
//! * `arithmetic` — same-unit/scalar/cross-unit dimensional algebra (impls only).

mod arithmetic;
mod factories;
mod kinds;
mod length;
mod quantity;

#[cfg(test)]
mod tests;

pub use kinds::{
    Amp, AmpKind, Celsius, CelsiusKind, Coulomb, CoulombKind, Farad, FaradKind, Henry, HenryKind,
    Hz, HzKind, Kelvin, KelvinKind, Ohm, OhmKind, Volt, VoltKind, Watt, WattKind,
};
pub use length::Nm;
pub use quantity::{approx_eq, Float, Unit};
