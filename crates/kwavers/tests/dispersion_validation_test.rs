//! Numerical dispersion validated against the library's own relations.
//!
//! This file previously implemented `fdtd_dispersion_relation` locally and
//! tested that, exercising no library code. The local version carried two
//! errors. Its second-order branch wrote `asin(sin(x))`, which cancels to `x`
//! and drops the temporal discretization entirely -- that is why the group
//! velocity came out as exactly `c cos(kh/2)`, an error of 1.23% where the
//! scheme's is 0.93%. Its fourth-order branch used `sin(kh/2)` where the
//! stencil calls for `sin(kh)`, so its modified wavenumber tended to `k/3`
//! rather than `k` and reported a 59.7% dispersion error.
//!
//! `test_anisotropic_dispersion` was worse than wrong: it computed
//! `sqrt(kx^2 + ky^2)` from `kx = 2 pi cos(theta) / lambda` and
//! `ky = 2 pi sin(theta) / lambda`, which is `2 pi / lambda` at every angle. It
//! measured the same number three times and passed. Its own comment said the
//! analysis was "Simplified here for demonstration". The library exposes no
//! 2-D dispersion relation, so there is nothing here to point a rewritten test
//! at; it is dropped rather than reproduced, and 2-D anisotropy is recorded as
//! uncovered.
//!
//! What follows tests `kwavers_physics::analytical::wave`, where these
//! relations live.
//!
//! Reference: Taflove & Hagness (2005) *Computational Electrodynamics*, 4.5.

use kwavers_physics::analytical::wave::{centered_fd_modified_wavenumber, fdtd_phase_error_1d};
use std::f64::consts::PI;

/// Points per wavelength converted to `kh = k dx`.
fn kh_for(points_per_wavelength: f64) -> f64 {
    2.0 * PI / points_per_wavelength
}

/// Wavenumbers from the resolved end out to Nyquist.
fn kh_to_nyquist() -> Vec<f64> {
    (1..=20).map(|i| PI * f64::from(i) / 20.0).collect()
}

/// A centered stencil's modified wavenumber must converge at its stated order.
///
/// This is the property worth testing, and it is stronger than any single
/// magic threshold: for a centered difference of order `p`, the relative error
/// `(k*h - kh) / kh` scales as `(kh)^p`, so halving `kh` divides the error by
/// `2^p`. Measuring that exponent falsifies a wrong stencil regardless of its
/// magnitude -- the fourth-order coefficients this file used to carry were
/// wrong by a factor of three at leading order, which a threshold might absorb
/// but an order measurement cannot.
///
/// The observed order is taken between successive halvings and required within
/// 2% of nominal. The residual is the next term in the expansion, `O((kh)^2)`
/// relative to the leading one, which at these `kh` is well inside 2%.
#[test]
fn centered_stencils_converge_at_their_stated_order() {
    const ORDER_TOLERANCE: f64 = 0.02;
    // Small enough that the leading term dominates, so the measured exponent is
    // the stencil's order rather than a blend of it and the next term.
    let kh_samples = [0.2_f64, 0.1, 0.05];

    for order in [2_u32, 4, 6] {
        let modified = centered_fd_modified_wavenumber(&kh_samples, order)
            .expect("2, 4 and 6 are the orders this stencil family defines");

        let errors: Vec<f64> = kh_samples
            .iter()
            .zip(&modified)
            .map(|(&kh, &star)| ((star - kh) / kh).abs())
            .collect();

        for pair in errors.windows(2) {
            let observed = (pair[0] / pair[1]).log2();
            let nominal = f64::from(order);
            assert!(
                (observed - nominal).abs() / nominal < ORDER_TOLERANCE,
                "order-{order} stencil converges at observed order {observed:.4}, \
                 not {nominal}; the coefficients do not match the order claimed"
            );
        }
    }
}

/// Higher order must resolve better at the same sampling.
///
/// A separate claim from the convergence one: a stencil can carry the right
/// exponent with the wrong constant. At a fixed `kh` the errors must be
/// strictly ordered, which pins the constants relative to each other.
#[test]
fn higher_order_stencils_resolve_better_at_equal_sampling() {
    let kh = [kh_for(10.0)]; // 10 points per wavelength: coarse enough to separate

    let error_at = |order: u32| {
        let modified = centered_fd_modified_wavenumber(&kh, order).expect("supported order");
        ((modified[0] - kh[0]) / kh[0]).abs()
    };

    let (second, fourth, sixth) = (error_at(2), error_at(4), error_at(6));
    assert!(
        second > fourth && fourth > sixth,
        "stencil errors at 10 PPW are not ordered by order: \
         2nd = {second:.3e}, 4th = {fourth:.3e}, 6th = {sixth:.3e}"
    );
}

/// The 1-D magic time step: FDTD is dispersion-free at `CFL = 1`.
///
/// `k'h = 2 arcsin(CFL sin(kh/2)) / CFL` collapses to `kh` exactly when
/// `CFL = 1`, for every `kh` including the poorly resolved ones. It is a sharp
/// property -- the error is not small there, it is zero -- so it falsifies any
/// relation that has drifted from the scheme's, which is how the spurious
/// `asin(sin(x))` in the old local helper would have been caught.
///
/// The bound is round-off through one `sin` and one `arcsin`, a few `eps`;
/// `1e-14` is two orders above that and far below any structural error.
#[test]
fn fdtd_is_dispersion_free_at_the_magic_time_step() {
    const ROUND_OFF: f64 = 1.0e-14;
    // Out to Nyquist, so the claim covers the range where dispersion is
    // otherwise severe, not just the resolved end.
    let kh = kh_to_nyquist();

    for (sample, error) in kh.iter().zip(fdtd_phase_error_1d(&kh, 1.0)) {
        assert!(
            error.abs() < ROUND_OFF,
            "phase error {error:.3e} at kh = {sample:.4} with CFL = 1; the magic \
             time step must be dispersion-free at every wavenumber"
        );
    }
}

/// Below the magic step, phase error grows monotonically toward Nyquist.
#[test]
fn fdtd_phase_error_grows_toward_nyquist() {
    let kh = kh_to_nyquist();
    let errors = fdtd_phase_error_1d(&kh, 0.5);

    for (index, pair) in errors.windows(2).enumerate() {
        assert!(
            pair[1].abs() > pair[0].abs(),
            "phase error falls from {:.3e} to {:.3e} between kh = {:.4} and \
             {:.4}; FDTD dispersion is monotone in kh below CFL = 1",
            pair[0],
            pair[1],
            kh[index],
            kh[index + 1]
        );
    }
}

/// Group velocity error is three times the phase velocity error.
///
/// Energy travels at `v_g = d omega / dk`, not at the phase velocity, and the
/// two carry different errors. With `c~(k) = c (1 + e(kh))` and `e ~ A (kh)^2`
/// in the resolved limit,
///
/// ```text
/// v_g / c = (1 + e) + kh de/d(kh) = 1 + 3e + O((kh)^4)
/// ```
///
/// so the group velocity error is asymptotically `3e`. That ratio is a derived
/// property of the scheme rather than a threshold, so it cannot be satisfied
/// by a relation that merely happens to be small -- but a ratio alone can hold
/// between two wrong numbers, so the magnitude is anchored as well.
///
/// `e` comes from `fdtd_phase_error_1d`; the derivative is a centered
/// difference in `kh`, whose truncation error is `O(h^2)` relative and
/// negligible beside the tolerances here.
#[test]
fn group_velocity_error_is_three_times_the_phase_error() {
    const RATIO_TOLERANCE: f64 = 0.03;
    // At 20 PPW with CFL 0.5 the scheme's group velocity error is 0.928%,
    // from v_g / c = cos(kh/2) / sqrt(1 - CFL^2 sin^2(kh/2)).
    const EXPECTED_GROUP_ERROR: f64 = 0.00928;
    let cfl = 0.5;
    let kh = kh_for(20.0);
    let step = kh * 1.0e-4;

    let samples = [kh - step, kh, kh + step];
    let errors = fdtd_phase_error_1d(&samples, cfl);
    let (below, at, above) = (errors[0], errors[1], errors[2]);

    let derivative = (above - below) / (2.0 * step);
    let group_error = at + kh * derivative;
    let ratio = group_error / at;

    assert!(
        (ratio - 3.0).abs() < RATIO_TOLERANCE,
        "group velocity error is {ratio:.4}x the phase error, not 3x \
         (phase {at:.3e}, group {group_error:.3e} at 20 PPW, CFL = {cfl})"
    );
    assert!(
        (group_error.abs() - EXPECTED_GROUP_ERROR).abs() < 1.0e-4,
        "group velocity error {:.5} does not match the {EXPECTED_GROUP_ERROR} \
         the scheme gives at 20 PPW and CFL 0.5",
        group_error.abs()
    );
}
