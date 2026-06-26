//! The authoritative board-space length newtype, [`Nm`] — exact integer nanometres (the same
//! integer KiCad uses internally), `#[repr(transparent)]` over `i64` so design-rule clearances are
//! exact distances rather than floating values rounded at the rule boundary. Independent of the
//! `Float<Unit>` soft-unit machinery in [`super::quantity`].

use std::ops::{Add, Mul, Neg, Sub};

/// A length in **nanometres**. `#[repr(transparent)]` so it is zero-overhead over
/// `i64` — a 1:1 match for KiCad's internal board-coordinate integer. Re-exported
/// by [`crate::geom`] so all `crate::geom::Nm` references compile unchanged.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Nm(pub i64);

impl Nm {
    /// Construct from millimetres, rounding to the nearest nanometre.
    #[must_use]
    pub fn from_mm(mm: f64) -> Self {
        Nm((mm * 1.0e6).round() as i64)
    }

    /// Construct from microns, rounding to the nearest nanometre.
    #[must_use]
    pub fn from_um(um: f64) -> Self {
        Nm((um * 1.0e3).round() as i64)
    }

    /// Value in millimetres (f64). The escape hatch for display / emission /
    /// plotting / division by a non-`Nm` quantity.
    #[must_use]
    pub fn to_mm(self) -> f64 {
        (self.0 as f64) * 1.0e-6
    }

    /// Value in microns (f64).
    #[must_use]
    pub fn to_um(self) -> f64 {
        (self.0 as f64) * 1.0e-3
    }

    /// Absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        Nm(self.0.abs())
    }
}

impl Add for Nm {
    type Output = Nm;
    fn add(self, rhs: Nm) -> Nm {
        Nm(self.0 + rhs.0)
    }
}
impl Sub for Nm {
    type Output = Nm;
    fn sub(self, rhs: Nm) -> Nm {
        Nm(self.0 - rhs.0)
    }
}
impl Mul<i64> for Nm {
    type Output = Nm;
    fn mul(self, rhs: i64) -> Nm {
        Nm(self.0 * rhs)
    }
}
impl Neg for Nm {
    type Output = Nm;
    fn neg(self) -> Nm {
        Nm(-self.0)
    }
}
