//! SI-prefix construction factories (kHz/MHz/GHz, mΩ/kΩ, mW/µW, nF/pF/µF, nH/µH, nC/µC) and the
//! Kelvin↔Celsius↔Fahrenheit temperature bridge — the only cross-scale conversions, kept explicit so
//! the type system still bars `K + °C` mixing.

use std::marker::PhantomData;

use super::kinds::{Celsius, Coulomb, Farad, Henry, Hz, Kelvin, Ohm, Watt};
use super::quantity::Float;

impl Hz {
    /// Construct from kilohertz.
    #[must_use]
    pub fn from_khz(v: f64) -> Self {
        Float(v * 1.0e3, PhantomData)
    }
    /// Construct from megahertz.
    #[must_use]
    pub fn from_mhz(v: f64) -> Self {
        Float(v * 1.0e6, PhantomData)
    }
    /// Construct from gigahertz.
    #[must_use]
    pub fn from_ghz(v: f64) -> Self {
        Float(v * 1.0e9, PhantomData)
    }
    /// Value in kilohertz.
    #[must_use]
    pub fn to_khz(self) -> f64 {
        self.0 * 1.0e-3
    }
    /// Value in megahertz.
    #[must_use]
    pub fn to_mhz(self) -> f64 {
        self.0 * 1.0e-6
    }
    /// Value in gigahertz.
    #[must_use]
    pub fn to_ghz(self) -> f64 {
        self.0 * 1.0e-9
    }
}

impl Ohm {
    /// Construct from milliohms.
    #[must_use]
    pub fn from_mohm(v: f64) -> Self {
        Float(v * 1.0e-3, PhantomData)
    }
    /// Construct from kilohms.
    #[must_use]
    pub fn from_kohm(v: f64) -> Self {
        Float(v * 1.0e3, PhantomData)
    }
}

impl Watt {
    /// Construct from milliwatts.
    #[must_use]
    pub fn from_mw(v: f64) -> Self {
        Float(v * 1.0e-3, PhantomData)
    }
    /// Construct from microwatts.
    #[must_use]
    pub fn from_uw(v: f64) -> Self {
        Float(v * 1.0e-6, PhantomData)
    }
}

impl Farad {
    /// Construct from nanofarads.
    #[must_use]
    pub fn from_nf(v: f64) -> Self {
        Float(v * 1.0e-9, PhantomData)
    }
    /// Construct from picofarads.
    #[must_use]
    pub fn from_pf(v: f64) -> Self {
        Float(v * 1.0e-12, PhantomData)
    }
    /// Construct from microfarads.
    #[must_use]
    pub fn from_uf(v: f64) -> Self {
        Float(v * 1.0e-6, PhantomData)
    }
}

impl Henry {
    /// Construct from nanohenries.
    #[must_use]
    pub fn from_nh(v: f64) -> Self {
        Float(v * 1.0e-9, PhantomData)
    }
    /// Construct from microhenries.
    #[must_use]
    pub fn from_uh(v: f64) -> Self {
        Float(v * 1.0e-6, PhantomData)
    }
}

impl Coulomb {
    /// Construct from nanocoulombs.
    #[must_use]
    pub fn from_nc(v: f64) -> Self {
        Float(v * 1.0e-9, PhantomData)
    }
    /// Construct from microcoulombs.
    #[must_use]
    pub fn from_uc(v: f64) -> Self {
        Float(v * 1.0e-6, PhantomData)
    }
}

impl Kelvin {
    /// Construct from a Celsius reading, shifting by the absolute-zero offset.
    #[must_use]
    pub fn from_celsius(c: f64) -> Self {
        Float(c + 273.15, PhantomData)
    }
    /// Construct from a Fahrenheit reading.
    #[must_use]
    pub fn from_fahrenheit(f: f64) -> Self {
        Float((f - 32.0) * 5.0 / 9.0 + 273.15, PhantomData)
    }
    /// Value in Celsius (relative to 273.15 K).
    #[must_use]
    pub fn to_celsius(self) -> f64 {
        self.0 - 273.15
    }
    /// Value in Fahrenheit.
    #[must_use]
    pub fn to_fahrenheit(self) -> f64 {
        (self.0 - 273.15) * 9.0 / 5.0 + 32.0
    }
}

impl Celsius {
    /// Construct from Kelvin (shifts by −273.15).
    #[must_use]
    pub fn from_kelvin(k: f64) -> Self {
        Float(k - 273.15, PhantomData)
    }
    /// Construct from Fahrenheit.
    #[must_use]
    pub fn from_fahrenheit(f: f64) -> Self {
        Float((f - 32.0) * 5.0 / 9.0, PhantomData)
    }
    /// Value in Kelvin.
    #[must_use]
    pub fn to_kelvin(self) -> f64 {
        self.0 + 273.15
    }
    /// Value in Fahrenheit.
    #[must_use]
    pub fn to_fahrenheit(self) -> f64 {
        self.0 * 9.0 / 5.0 + 32.0
    }
}
