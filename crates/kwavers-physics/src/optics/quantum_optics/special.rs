//! Special functions required by quantum-optics approximations.

/// Modified Bessel function of the second kind, order zero: `K0(x)` for `x > 0`.
///
/// # Approximation
///
/// Abramowitz-Stegun 9.8.5 is used for `0 < x <= 2`; 9.8.6 is used for
/// `x > 2`. Both polynomial forms are positive on their domains and are
/// accurate to about `1.6e-7` relative error for the free-free Gaunt-factor
/// range used here.
#[must_use]
pub(super) fn bessel_k0(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    if x <= 2.0 {
        let t1 = (x / 3.75) * (x / 3.75);
        let i0 = 1.0
            + t1 * (3.515_632_9
                + t1 * (3.089_942_4
                    + t1 * (1.206_749_2
                        + t1 * (0.265_973_2 + t1 * (0.036_076_8 + t1 * 0.004_581_3)))));
        let t2 = (x * 0.5) * (x * 0.5);
        let corr = -0.577_215_66
            + t2 * (0.422_784_20
                + t2 * (0.230_697_56
                    + t2 * (0.034_885_90
                        + t2 * (0.002_626_98 + t2 * (0.000_107_50 + t2 * 7.4e-6)))));
        (x * 0.5).ln().mul_add(-i0, corr)
    } else {
        let t = 2.0 / x;
        let poly = 1.253_314_14
            + t * (-0.078_323_58
                + t * (0.021_895_68
                    + t * (-0.010_624_46
                        + t * (0.005_878_72 + t * (-0.002_515_40 + t * 0.000_532_08)))));
        (-x).exp() / x.sqrt() * poly
    }
}
