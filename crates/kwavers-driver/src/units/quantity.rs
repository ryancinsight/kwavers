//! The soft-unit core: the [`Unit`] kind-marker trait and the `#[repr(transparent)]`
//! [`Float<U>`] wrapper every SI newtype ([`super::Hz`], [`super::Ohm`], …) is a type alias over,
//! plus the tolerance-aware [`approx_eq`] comparison. The concrete kinds live in [`super::kinds`];
//! construction and arithmetic in [`super::factories`] / [`super::arithmetic`].

use std::fmt;
use std::marker::PhantomData;

/// Internal trait: the kind-marker that distinguishes each unit newtype. The
/// `Float` newtype is parameterised over a [`Unit`] kind, so the cross-unit
/// arithmetic implementations are localised here.
pub trait Unit: Copy + Clone + 'static {
    /// SI symbol used by `Display` (e.g. "Hz", "Ω", "W", "°C").
    const SYMBOL: &'static str;
}

/// A `#[repr(transparent)]` wrapper around `f64` parameterised by a unit kind. Every
/// concrete SI newtype below (`Hz`, `Ohm`, …) is implemented as a type alias for
/// `Float<{tag kind}>` plus utility impls.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Float<U: Unit>(pub f64, pub PhantomData<U>);

impl<U: Unit> Float<U> {
    /// Wrap a raw `f64` value. The escape hatch `value()` is the only `f64`
    /// exposure; arithmetic and conversions stay within the newtype.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }

    /// Construct from `f64`.
    #[must_use]
    pub fn from_f64(v: f64) -> Self {
        Float(v, PhantomData)
    }
}

impl<U: Unit> Default for Float<U> {
    fn default() -> Self {
        Float(0.0, PhantomData)
    }
}

impl<U: Unit> From<f64> for Float<U> {
    fn from(v: f64) -> Self {
        Float(v, PhantomData)
    }
}

impl<U: Unit> fmt::Display for Float<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, U::SYMBOL)
    }
}

/// Approximate equality for f64-backed newtypes (sloppy compare for tolerances
/// where exact bit-equality is not informative — `Hz ≈ Hz` within a tolerance).
#[must_use]
pub fn approx_eq<U: Unit>(a: Float<U>, b: Float<U>, tol: f64) -> bool {
    (a.0 - b.0).abs() <= tol.max(0.0)
}
