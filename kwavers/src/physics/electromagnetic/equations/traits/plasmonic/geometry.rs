//! Geometry kernels for quasistatic plasmonic depolarization factors.
//!
//! # Theorem
//!
//! For a spheroid with transverse radius `r` and symmetry radius `c`, the
//! principal depolarization factors satisfy
//!
//! ```text
//! L_x = L_y = (1 - L_z)/2,    L_x + L_y + L_z = 1.
//! ```
//!
//! In the prolate case `c > r`, with eccentricity
//! `e = sqrt(1 - r^2/c^2)`,
//!
//! ```text
//! L_z = (1 - e^2)/(2e^3) [ln((1+e)/(1-e)) - 2e].
//! ```
//!
//! In the oblate case `c < r`, with eccentricity
//! `e = sqrt(1 - c^2/r^2)`,
//!
//! ```text
//! L_z = (1 + e^2)/e^3 [e - atan(e)].
//! ```
//!
//! Proof: these are the closed-form surface-integral solutions for ellipsoids
//! of revolution. Both formulas return finite limits as `e -> 0`; the spherical
//! limit is `L_x=L_y=L_z=1/3`. The transverse factors follow from symmetry and
//! the electrostatic identity `trace(N)=1` for the depolarization tensor.

#[must_use]
pub(crate) fn ellipsoid_depolarization_factors(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let r = 0.5 * (a + b);
    if r <= 0.0 || c <= 0.0 {
        return spherical();
    }

    if (c - r).abs() <= f64::EPSILON {
        return spherical();
    }

    if c > r {
        prolate_depolarization_factors(r, c)
    } else {
        oblate_depolarization_factors(r, c)
    }
}

fn spherical() -> (f64, f64, f64) {
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
}

fn prolate_depolarization_factors(r: f64, c: f64) -> (f64, f64, f64) {
    let e = (1.0 - (r * r) / (c * c)).sqrt();
    let denom = 2.0 * e * e * e;
    let lz = if denom > 0.0 {
        let term = ((1.0 + e) / (1.0 - e)).ln() - 2.0 * e;
        (1.0 - e * e) / denom * term
    } else {
        1.0 / 3.0
    };
    transverse_pair(lz)
}

fn oblate_depolarization_factors(r: f64, c: f64) -> (f64, f64, f64) {
    let e = (1.0 - (c * c) / (r * r)).sqrt();
    let denom = e * e * e;
    let lz = if denom > 0.0 {
        (1.0 + e * e) / denom * (e - e.atan())
    } else {
        1.0 / 3.0
    };
    transverse_pair(lz)
}

fn transverse_pair(lz: f64) -> (f64, f64, f64) {
    let lx = 0.5 * (1.0 - lz);
    (lx, lx, lz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_limit_is_isotropic() {
        let (lx, ly, lz) = ellipsoid_depolarization_factors(2.0, 2.0, 2.0);
        assert!((lx - 1.0 / 3.0).abs() < 1e-15);
        assert!((ly - 1.0 / 3.0).abs() < 1e-15);
        assert!((lz - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn depolarization_factors_sum_to_one() {
        for (a, b, c) in [(1.0, 1.0, 3.0), (3.0, 3.0, 1.0), (2.0, 3.0, 5.0)] {
            let (lx, ly, lz) = ellipsoid_depolarization_factors(a, b, c);
            let sum = lx + ly + lz;
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "depolarization factors must sum to one; ({lx}, {ly}, {lz}) sum={sum}"
            );
            assert!(lx > 0.0 && ly > 0.0 && lz > 0.0);
        }
    }
}
