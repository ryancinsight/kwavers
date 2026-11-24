//! Centered second derivatives on a uniform axis, with spacing factored out.
//! Pair entries multiply the sum at offsets ±1, ±2, … in that order.

use super::FdAccuracyOrder;

pub(in crate::operators) struct SecondDerivative<const R: usize> {
    pub(in crate::operators) center: f64,
    pub(in crate::operators) pairs: [f64; R],
}

pub(in crate::operators) const SECOND: SecondDerivative<1> = SecondDerivative {
    center: -2.0,
    pairs: [1.0],
};
pub(in crate::operators) const FOURTH: SecondDerivative<2> = SecondDerivative {
    center: -5.0 / 2.0,
    pairs: [4.0 / 3.0, -1.0 / 12.0],
};
pub(in crate::operators) const SIXTH: SecondDerivative<3> = SecondDerivative {
    center: -49.0 / 18.0,
    pairs: [3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0],
};
pub(in crate::operators) const EIGHTH: SecondDerivative<4> = SecondDerivative {
    center: -205.0 / 72.0,
    pairs: [8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0],
};

pub(super) fn weights(order: FdAccuracyOrder) -> (f64, &'static [f64]) {
    match order {
        FdAccuracyOrder::Second => (SECOND.center, &SECOND.pairs),
        FdAccuracyOrder::Fourth => (FOURTH.center, &FOURTH.pairs),
        FdAccuracyOrder::Sixth => (SIXTH.center, &SIXTH.pairs),
        FdAccuracyOrder::Eighth => (EIGHTH.center, &EIGHTH.pairs),
    }
}
