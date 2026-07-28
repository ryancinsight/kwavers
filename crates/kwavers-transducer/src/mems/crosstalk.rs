//! Inter-element acoustic crosstalk for MUT arrays (fluid-borne coupling).
//!
//! Closes the "inter-element cross-coupling" gap flagged in Chapter 33 §33.8.
//! Adjacent CMUT/PMUT cells couple acoustically through the loading fluid: a
//! vibrating element radiates a pressure that drives its neighbours. This module
//! models that **fluid path** via the **baffled monopole (point-source)
//! approximation**, which is exact in the limit of well-separated, small
//! elements (`d ≫ a`, `ka ≲ 1`).
//!
//! # Model
//!
//! A membrane element of area `A` with surface-velocity amplitude `u` is a
//! baffled simple source of volume velocity `Q = u·A`. In an infinite rigid
//! baffle (radiation into a half-space) the far-field pressure at range `r` is
//!
//! ```text
//! p(r) = jωρ Q /(2π r) · e^{-jkr},   k = ω/c.
//! ```
//!
//! The force this induces on a second element `i` of area `A_i` at separation
//! `d` from element `j` gives the **mutual radiation impedance** (force on `i`
//! per unit velocity of `j`):
//!
//! ```text
//! Z_ij = F_ij / u_j = jωρ A_i A_j /(2π d) · e^{-jkd}.
//! ```
//!
//! Reciprocity (`Z_ij = Z_ji`) is structural. The `1/(2π)` (not `1/(4π)`) is the
//! baffle half-space factor.
//!
//! # Scope (honest)
//!
//! This captures the **fluid** crosstalk path only. The **substrate** path
//! (dispersive Lamb/Stoneley waves on the CMUT membrane support) and full
//! coupled-field FEM are out of scope and require a meshed model. The monopole
//! approximation degrades for closely-packed elements (`d ≈ a`) where the finite
//! piston directivity matters.
//!
//! # References
//! - Pritchard, R.L. (1960). "Mutual acoustic impedance between radiators in an
//!   infinite rigid plane." *J. Acoust. Soc. Am.* 32(6), 730–737.
//! - Kinsler, Frey, Coppens & Sanders, *Fundamentals of Acoustics* (4th ed.), §7.

use aequitas::systems::si::{
    quantities::{AcousticImpedance, Area, Frequency, Length, MassDensity, Velocity},
    units::{Hertz, KilogramPerCubicMeter, Meter, MeterPerSecond},
};
use eunomia::Complex64;
use leto::Array2;
use std::f64::consts::{FRAC_PI_2, TAU};

/// Mutual radiation impedance `Z_ij = jωρ A_i A_j /(2π d) · e^{-jkd}` [kg/s]
/// between two baffled membrane elements (baffled monopole approximation).
///
/// `area_i`, `area_j` are the element face areas, `separation` the centre-to-
/// centre distance `d > 0`, `angular_frequency` the angular frequency, and
/// `rho`/`c` the fluid density and sound speed. The arguments carry SI units;
/// radians are dimensionless, so angular frequency is represented by the SI
/// frequency quantity.
///
/// Returns `0` for a non-positive separation (degenerate / coincident elements;
/// the monopole formula is invalid there — use the single-element radiation
/// impedance for the self term).
#[must_use]
pub fn mutual_radiation_impedance(
    area_i: Area,
    area_j: Area,
    separation: Length,
    angular_frequency: Frequency,
    rho: MassDensity,
    c: Velocity,
) -> AcousticImpedance<Complex64> {
    let separation_m = separation.in_unit::<Meter>();
    let angular_frequency_hz = angular_frequency.in_unit::<Hertz>();
    let sound_speed_m_s = c.in_unit::<MeterPerSecond>();
    if separation_m <= 0.0 || sound_speed_m_s <= 0.0 {
        return AcousticImpedance::from_base(Complex64::new(0.0, 0.0));
    }
    let k = angular_frequency_hz / sound_speed_m_s;
    let magnitude = angular_frequency_hz
        * rho.in_unit::<KilogramPerCubicMeter>()
        * area_i.into_base()
        * area_j.into_base()
        / (TAU * separation_m);
    // jωρAA/(2πd)·e^{-jkd} = magnitude · e^{j(π/2 − kd)}.
    AcousticImpedance::from_base(Complex64::from_polar(
        magnitude,
        FRAC_PI_2 - k * separation_m,
    ))
}

/// Build the inter-element crosstalk coupling matrix `Z` for an array.
///
/// `positions` are the element centres (3-D, for conformal arrays); `areas`
/// the per-element face areas (must match `positions` in length). Entry
/// `Z`i``J`` (`i≠j`) is [`mutual_radiation_impedance`]; the **diagonal is zero**
/// — this is the *cross*-coupling matrix, the self radiation impedance is a
/// separate single-element property. The matrix is complex-symmetric
/// (`Z`i``J` = Z`J``i``, reciprocity).
///
/// Returns an empty matrix if `positions`/`areas` lengths disagree.
#[must_use]
pub fn crosstalk_matrix(
    positions: &[[Length; 3]],
    areas: &[Area],
    angular_frequency: Frequency,
    rho: MassDensity,
    c: Velocity,
) -> Array2<AcousticImpedance<Complex64>> {
    let n = positions.len();
    if areas.len() != n {
        return Array2::from_elem(
            [0, 0],
            AcousticImpedance::from_base(Complex64::new(0.0, 0.0)),
        );
    }
    let mut z = Array2::from_elem(
        [n, n],
        AcousticImpedance::from_base(Complex64::new(0.0, 0.0)),
    );
    for i in 0..n {
        for j in (i + 1)..n {
            let d = separation(positions[i], positions[j]);
            let zij = mutual_radiation_impedance(areas[i], areas[j], d, angular_frequency, rho, c);
            z[[i, j]] = zij;
            z[[j, i]] = zij;
        }
    }
    z
}

/// Euclidean centre-to-centre distance between two element positions `m`.
#[inline]
#[must_use]
fn separation(a: [Length; 3], b: [Length; 3]) -> Length {
    let dx = a[0].into_base() - b[0].into_base();
    let dy = a[1].into_base() - b[1].into_base();
    let dz = a[2].into_base() - b[2].into_base();
    Length::from_base(dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, SquareMeter};

    // Water-immersed element scale: a = 50 µm membrane, 5 MHz.
    fn area(radius: f64) -> Area {
        Area::from_unit::<SquareMeter>(std::f64::consts::PI * radius * radius)
    }
    fn rho() -> MassDensity {
        MassDensity::from_unit::<KilogramPerCubicMeter>(1000.0)
    }
    fn sound_speed() -> Velocity {
        Velocity::from_unit::<MeterPerSecond>(1500.0)
    }
    fn length(value: f64) -> Length {
        Length::from_unit::<Meter>(value)
    }
    fn omega(freq: f64) -> Frequency {
        Frequency::from_unit::<Hertz>(TAU * freq)
    }

    /// Closed-form magnitude `|Z_ij| = ωρ A_i A_j /(2π d)` and retardation phase
    /// `arg Z_ij = π/2 − k d`.
    #[test]
    fn mutual_impedance_matches_closed_form() {
        let (ai, aj) = (area(50e-6), area(40e-6));
        let d = 150e-6;
        let w = omega(5e6);
        let z = mutual_radiation_impedance(ai, aj, length(d), w, rho(), sound_speed());

        let expected_mag =
            w.in_unit::<Hertz>() * 1000.0 * ai.into_base() * aj.into_base() / (TAU * d);
        assert!(
            (z.into_base().norm() - expected_mag).abs() <= 1e-12 * expected_mag,
            "magnitude {} vs {expected_mag}",
            z.into_base().norm()
        );
        let expected_phase = FRAC_PI_2 - (w.in_unit::<Hertz>() / 1500.0) * d;
        // Compare via the complex value to avoid atan2 branch issues.
        let expected = Complex64::from_polar(expected_mag, expected_phase);
        assert!(
            (z.into_base() - expected).norm() <= 1e-9 * expected_mag,
            "phase mismatch"
        );
    }

    /// Reciprocity: `Z_ij == Z_ji`.
    #[test]
    fn mutual_impedance_is_reciprocal() {
        let (ai, aj) = (area(50e-6), area(35e-6));
        let d = 120e-6;
        let w = omega(4e6);
        let zij = mutual_radiation_impedance(ai, aj, length(d), w, rho(), sound_speed());
        let zji = mutual_radiation_impedance(aj, ai, length(d), w, rho(), sound_speed());
        assert!((zij.into_base() - zji.into_base()).norm() <= 1e-15);
    }

    /// Magnitude scales `∝ ω` and `∝ 1/d`, and decays toward zero with range.
    #[test]
    fn magnitude_scales_with_frequency_and_inverse_distance() {
        let a = area(50e-6);
        let base =
            mutual_radiation_impedance(a, a, length(100e-6), omega(2e6), rho(), sound_speed())
                .into_base()
                .norm();

        // Double frequency → double magnitude.
        let double_f =
            mutual_radiation_impedance(a, a, length(100e-6), omega(4e6), rho(), sound_speed())
                .into_base()
                .norm();
        assert!((double_f / base - 2.0).abs() <= 1e-9);

        // Double distance → half magnitude.
        let double_d =
            mutual_radiation_impedance(a, a, length(200e-6), omega(2e6), rho(), sound_speed())
                .into_base()
                .norm();
        assert!((double_d / base - 0.5).abs() <= 1e-9);

        // Far separation → vanishing coupling.
        let far = mutual_radiation_impedance(a, a, length(1.0), omega(2e6), rho(), sound_speed())
            .into_base()
            .norm();
        assert!(
            far < base * 1e-3,
            "coupling must vanish at large separation"
        );
    }

    /// The coupling matrix is complex-symmetric with a zero diagonal, and its
    /// off-diagonal magnitudes match the pairwise closed form.
    #[test]
    fn crosstalk_matrix_is_symmetric_zero_diagonal() {
        // 1-D pitch array of 4 equal elements.
        let a = area(50e-6);
        let pitch = 130e-6;
        let positions: Vec<[Length; 3]> = (0..4)
            .map(|i| [length(i as f64 * pitch), length(0.0), length(0.0)])
            .collect();
        let areas = vec![a; 4];
        let w = omega(5e6);
        let z = crosstalk_matrix(&positions, &areas, w, rho(), sound_speed());

        assert_eq!(z.shape(), [4, 4]);
        for i in 0..4 {
            assert!(
                z[[i, i]].into_base().norm() <= 1e-300,
                "diagonal must be zero"
            );
            for j in 0..4 {
                assert!((z[[i, j]].into_base() - z[[j, i]].into_base()).norm() <= 1e-15);
            }
        }
        // Nearest neighbours (d = pitch) couple more strongly than next-nearest
        // (d = 2·pitch): magnitude ∝ 1/d ⇒ factor 2.
        let nn = z[[0, 1]].into_base().norm();
        let nnn = z[[0, 2]].into_base().norm();
        assert!((nn / nnn - 2.0).abs() <= 1e-9, "nn/nnn magnitude ratio");

        // Cross-check against the standalone closed form.
        let expected = mutual_radiation_impedance(a, a, length(pitch), w, rho(), sound_speed());
        assert!((z[[0, 1]].into_base() - expected.into_base()).norm() <= 1e-12);
    }

    /// Degenerate inputs: coincident elements and mismatched lengths.
    #[test]
    fn degenerate_inputs_are_handled() {
        let a = area(50e-6);
        let coincident =
            mutual_radiation_impedance(a, a, length(0.0), omega(5e6), rho(), sound_speed());
        assert_eq!(coincident.into_base(), Complex64::new(0.0, 0.0));

        let positions = vec![
            [length(0.0), length(0.0), length(0.0)],
            [length(1e-4), length(0.0), length(0.0)],
        ];
        let areas = vec![a]; // wrong length
        let z = crosstalk_matrix(&positions, &areas, omega(5e6), rho(), sound_speed());
        assert_eq!(z.shape(), [0, 0]);
    }
}
