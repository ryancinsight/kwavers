//! The unit kind-markers (one zero-sized [`Unit`] ZST per SI unit) and the concrete newtype aliases
//! over [`Float<Kind>`]. Each `Kind` carries only its `Display` symbol; the alias is the name the
//! rest of the crate uses (`Hz`, `Ohm`, …).

use super::quantity::{Float, Unit};

/// Marker for the [`Hz`] newtype — cycles per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HzKind;
impl Unit for HzKind {
    const SYMBOL: &'static str = "Hz";
}

/// Marker for the [`Ohm`] newtype — electrical impedance (resistance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OhmKind;
impl Unit for OhmKind {
    const SYMBOL: &'static str = "Ω";
}

/// Marker for the [`Watt`] newtype — power dissipation (joules per second).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WattKind;
impl Unit for WattKind {
    const SYMBOL: &'static str = "W";
}

/// Marker for the [`Kelvin`] newtype — absolute temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KelvinKind;
impl Unit for KelvinKind {
    const SYMBOL: &'static str = "K";
}

/// Marker for the [`Celsius`] newtype — a separate dimension from Kelvin so the
/// type system catches accidental `K + °C` invalid arithmetic; conversions go
/// through the explicit `to_kelvin()` / `from_kelvin()` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CelsiusKind;
impl Unit for CelsiusKind {
    const SYMBOL: &'static str = "°C";
}

/// Marker for [`Volt`] — potential / EMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltKind;
impl Unit for VoltKind {
    const SYMBOL: &'static str = "V";
}

/// Marker for [`Amp`] — current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmpKind;
impl Unit for AmpKind {
    const SYMBOL: &'static str = "A";
}

/// Marker for [`Henry`] — inductance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HenryKind;
impl Unit for HenryKind {
    const SYMBOL: &'static str = "H";
}

/// Marker for [`Farad`] — capacitance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaradKind;
impl Unit for FaradKind {
    const SYMBOL: &'static str = "F";
}

/// Marker for [`Coulomb`] — charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoulombKind;
impl Unit for CoulombKind {
    const SYMBOL: &'static str = "C";
}

/// Frequency in hertz. Construct with `Hz::from(50e3)` or the kHz/MHz/GHz
/// factories; `Hz` + `Hz` is valid (e.g. sideband sum), `Hz * f64` is a scalar
/// rate multiplier.
pub type Hz = Float<HzKind>;

/// Electrical impedance / resistance in ohms. `Ohm * Amp = Volt` (Ohm's law).
pub type Ohm = Float<OhmKind>;

/// Power in watts. `Volt * Amp = Watt`.
pub type Watt = Float<WattKind>;

/// Absolute temperature in kelvin. Use `Kelvin::from_celsius(c)` /
/// `Kelvin::from_fahrenheit(f)` to construct from common scales.
pub type Kelvin = Float<KelvinKind>;

/// Relative temperature in degrees Celsius. Celsius↔Kelvin conversions are
/// explicit (`to_kelvin()`); the type system bars `Celsius + Kelvin` mixing.
pub type Celsius = Float<CelsiusKind>;

/// Electric potential in volts.
pub type Volt = Float<VoltKind>;

/// Electric current in amperes.
pub type Amp = Float<AmpKind>;

/// Inductance in henries.
pub type Henry = Float<HenryKind>;

/// Capacitance in farads.
pub type Farad = Float<FaradKind>;

/// Electric charge in coulombs.
pub type Coulomb = Float<CoulombKind>;
