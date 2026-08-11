//! Value-semantic verification of the power-law relaxation fit.
//!
//! The acceptance bound used throughout is **1 % relative error in `α(f)`
//! across the fit band**. That is not a tuned tolerance: reported `α₀` for soft
//! tissue carries roughly 10 % inter-study spread (Duck 1990, Ch. 4), so a
//! model error an order of magnitude below the measurement uncertainty of the
//! coefficient it reproduces cannot dominate any downstream result. The tests
//! additionally assert that the error *falls* with arm count, which a fit that
//! merely happened to clear a threshold would not do.

use super::*;

/// 1 dB = 1/8.685889638 Np, and cm⁻¹ → m⁻¹ multiplies by 100.
const NEPER_PER_DB: f64 = 1.0 / 8.685_889_638_065_035;

/// `α₀` in dB·cm⁻¹·MHz⁻ᵞ (the clinical literature convention) as the amplitude
/// absorption in Np·m⁻¹ at 1 MHz.
fn alpha_np_m(alpha_db_cm_mhz: f64) -> f64 {
    alpha_db_cm_mhz * NEPER_PER_DB * 100.0
}

const F_REF: f64 = 1.0e6;
const C_WATER_LIKE: f64 = 1540.0;
const RHO_TISSUE: f64 = 1000.0;

fn target(alpha_db: f64, gamma: f64) -> PowerLawTarget {
    PowerLawTarget {
        alpha_ref_np_m: alpha_np_m(alpha_db),
        exponent: gamma,
        f_ref: F_REF,
        sound_speed: C_WATER_LIKE,
        density: RHO_TISSUE,
    }
}

/// Fullwave 2.5's validated heterogeneous-attenuation envelope — `α₀` from 0.25
/// to 0.75 dB·cm⁻¹·MHz⁻ᵞ and `γ` from 0.4 to 1.6 — is reproduced to under 1 %
/// relative error over a 0.5–5 MHz band with six relaxation arms.
#[test]
fn fits_fullwave_attenuation_envelope_within_one_percent() {
    let band = FitBand::new(0.5e6, 5.0e6, 6).expect("valid band");
    for &alpha_db in &[0.25, 0.5, 0.75] {
        for &gamma in &[0.4, 0.7, 1.0, 1.3, 1.6] {
            let t = target(alpha_db, gamma);
            let fit = fit_power_law(&t, &band).expect("fit converges");
            assert!(
                fit.max_relative_error() < 1.0e-2,
                "α₀={alpha_db} dB/cm/MHz^γ, γ={gamma}: max relative error \
                 {:.4} exceeds 1 %",
                fit.max_relative_error()
            );
            // Spot-check the exact absorption against the law itself, not just
            // the fit's self-reported error.
            for &f in &[0.6e6, 1.0e6, 2.5e6, 4.5e6] {
                let got = fit.attenuation(TWO_PI * f);
                let want = t.alpha_at(f);
                assert!(
                    ((got - want) / want).abs() < 1.0e-2,
                    "γ={gamma} at {f:e} Hz: α = {got:e} vs target {want:e}"
                );
            }
        }
    }
}

/// Fit quality improves with the number of relaxation arms — the signature of a
/// genuine least-squares fit rather than a threshold that happens to pass.
///
/// The claim is refinement, not strict monotonicity: past the point where the
/// arms already span the band, the Tikhonov floor (see `RIDGE`) rather than the
/// basis limits the residual, so adding arms plateaus instead of improving.
#[test]
fn error_falls_with_arm_count_then_plateaus() {
    let t = target(0.5, 1.3);
    let errors: Vec<f64> = [2usize, 3, 4, 6, 8, 12]
        .into_iter()
        .map(|n_arms| {
            let band = FitBand::new(0.5e6, 5.0e6, n_arms).expect("valid band");
            fit_power_law(&t, &band)
                .expect("fit converges")
                .max_relative_error()
        })
        .collect();

    // Coarse-to-adequate refinement must buy at least an order of magnitude.
    assert!(
        errors[0] > 10.0 * errors[3],
        "2 arms {:.5} vs 6 arms {:.5}: refinement bought under 10×",
        errors[0],
        errors[3]
    );
    // Every adequately resolved grid clears 0.5 %, and none diverges — an
    // unregularized active-set solve blows past 10 % on the dense grids.
    for (n_arms, error) in [6usize, 8, 12].into_iter().zip(&errors[3..]) {
        assert!(*error < 5.0e-3, "{n_arms} arms gave {error:.5}");
    }
}

/// The equilibrium modulus is calibrated so the *dispersive* phase velocity at
/// `f_ref` equals the prescribed `c₀`. Without the calibration the medium would
/// run fast by the full Kramers–Krönig increment, so this also checks that the
/// naive `M_∞ = ρc²` choice is genuinely wrong here.
#[test]
fn phase_velocity_matches_prescribed_speed_at_reference() {
    let band = FitBand::new(0.5e6, 5.0e6, 6).expect("valid band");
    for &gamma in &[0.4, 1.0, 1.6] {
        let t = target(0.75, gamma);
        let fit = fit_power_law(&t, &band).expect("fit converges");
        let c_p = fit.phase_velocity(TWO_PI * F_REF);
        assert!(
            ((c_p - C_WATER_LIKE) / C_WATER_LIKE).abs() < 1.0e-9,
            "γ={gamma}: c_p({F_REF:e}) = {c_p} ≠ {C_WATER_LIKE}"
        );
        assert!(
            fit.equilibrium_modulus() < RHO_TISSUE * C_WATER_LIKE * C_WATER_LIKE,
            "γ={gamma}: calibration must lower M_∞ below the lossless ρc²"
        );
    }
}

/// Absorption is dispersive in the Kramers–Krönig sense: phase velocity rises
/// with frequency whenever `α > 0`.
#[test]
fn dispersion_is_causal_and_increasing() {
    let band = FitBand::new(0.5e6, 5.0e6, 6).expect("valid band");
    let fit = fit_power_law(&target(0.75, 1.1), &band).expect("fit converges");
    let mut previous = 0.0;
    for &f in &[0.5e6, 1.0e6, 2.0e6, 4.0e6, 5.0e6] {
        let c_p = fit.phase_velocity(TWO_PI * f);
        assert!(c_p > previous, "phase velocity fell at {f:e} Hz");
        previous = c_p;
    }
}

/// A lossless voxel takes no arms and the exact lossless modulus.
#[test]
fn lossless_voxel_is_exact() {
    let band = FitBand::new(0.5e6, 5.0e6, 4).expect("valid band");
    let t = PowerLawTarget {
        alpha_ref_np_m: 0.0,
        ..target(0.0, 1.0)
    };
    let fit = fit_power_law(&t, &band).expect("fit converges");
    assert_eq!(fit.weights(), &[0.0; 4]);
    assert_eq!(
        fit.equilibrium_modulus(),
        RHO_TISSUE * C_WATER_LIKE * C_WATER_LIKE
    );
    assert_eq!(fit.attenuation(TWO_PI * F_REF), 0.0);
}

/// All arm strengths are non-negative — a negative `ΔMₗ` is an unstable
/// (energy-generating) Maxwell element in the time-domain solver.
#[test]
fn arm_strengths_are_non_negative() {
    let band = FitBand::new(0.5e6, 5.0e6, 8).expect("valid band");
    for &gamma in &[0.4, 0.8, 1.2, 1.6] {
        let fit = fit_power_law(&target(0.5, gamma), &band).expect("fit converges");
        assert!(
            fit.weights().iter().all(|&w| w >= 0.0),
            "γ={gamma} produced a negative arm strength: {:?}",
            fit.weights()
        );
    }
}

/// The headline capability: a heterogeneous medium in which the **exponent**
/// varies voxel to voxel is reproduced on one shared `τ` grid, each voxel
/// following its own law.
#[test]
fn heterogeneous_exponent_field_is_fitted_per_voxel() {
    let shape = [3usize, 1, 1];
    let gammas = [0.4, 1.0, 1.6];
    let alphas_db = [0.25, 0.5, 0.75];
    let mut alpha = Array3::<f64>::zeros(shape);
    let mut gamma = Array3::<f64>::zeros(shape);
    let mut c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);
    for i in 0..3 {
        alpha[[i, 0, 0]] = alpha_np_m(alphas_db[i]);
        gamma[[i, 0, 0]] = gammas[i];
        c[[i, 0, 0]] = 1480.0 + 40.0 * i as f64;
    }

    let band = FitBand::new(0.5e6, 5.0e6, 6).expect("valid band");
    let fit =
        fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band).expect("field fit converges");

    assert_eq!(fit.relaxation_times().len(), 6);
    assert_eq!(fit.weights().len(), 6);
    assert!(
        fit.max_relative_error() < 1.0e-2,
        "worst voxel error {:.4}",
        fit.max_relative_error()
    );

    // Reconstruct each voxel's spectrum from the field fit and check it against
    // that voxel's own power law.
    for i in 0..3 {
        let weights: Vec<f64> = fit.weights().iter().map(|w| w[[i, 0, 0]]).collect();
        let m_inf = fit.equilibrium_modulus()[[i, 0, 0]];
        for &f in &[0.6e6, 2.0e6, 4.5e6] {
            let omega = TWO_PI * f;
            let m = complex_modulus(m_inf, &weights, fit.relaxation_times(), omega);
            let got = wavenumber(RHO_TISSUE, m, omega).im.abs();
            let want = alpha[[i, 0, 0]] * (f / F_REF).powf(gammas[i]);
            assert!(
                ((got - want) / want).abs() < 1.0e-2,
                "voxel {i} (γ={}) at {f:e} Hz: α = {got:e} vs {want:e}",
                gammas[i]
            );
        }
        // Each voxel keeps its own prescribed speed.
        let m = complex_modulus(m_inf, &weights, fit.relaxation_times(), TWO_PI * F_REF);
        let c_p = TWO_PI * F_REF / wavenumber(RHO_TISSUE, m, TWO_PI * F_REF).re;
        assert!(
            ((c_p - c[[i, 0, 0]]) / c[[i, 0, 0]]).abs() < 1.0e-9,
            "voxel {i}: c_p = {c_p} ≠ {}",
            c[[i, 0, 0]]
        );
    }
}

/// Bit-identical voxels reuse one fit, so a labelled medium costs one solve per
/// distinct tissue; the reused entries must be exactly equal, not merely close.
#[test]
fn identical_voxels_share_one_fit_exactly() {
    let shape = [4usize, 2, 1];
    let alpha = Array3::<f64>::from_elem(shape, alpha_np_m(0.5));
    let gamma = Array3::<f64>::from_elem(shape, 1.1);
    let c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);
    let band = FitBand::new(0.5e6, 5.0e6, 5).expect("valid band");
    let fit = fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band).expect("fit");

    let reference: Vec<f64> = fit.weights().iter().map(|w| w[[0, 0, 0]]).collect();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for (l, w) in fit.weights().iter().enumerate() {
                assert_eq!(w[[i, j, 0]], reference[l]);
            }
        }
    }
}

/// Arm fields are grid-shaped and carry the shared relaxation times.
#[test]
fn arm_fields_expose_shared_relaxation_times() {
    let shape = [2usize, 2, 2];
    let alpha = Array3::<f64>::from_elem(shape, alpha_np_m(0.5));
    let gamma = Array3::<f64>::from_elem(shape, 1.0);
    let c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);
    let band = FitBand::new(1.0e6, 4.0e6, 3).expect("valid band");
    let fit = fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band).expect("fit");

    let arms = fit.arm_fields();
    assert_eq!(arms.len(), 3);
    for (l, (dm, tau)) in arms.iter().enumerate() {
        assert_eq!(dm.shape(), shape);
        assert_eq!(tau.shape(), shape);
        assert_eq!(tau[[1, 1, 1]], fit.relaxation_times()[l]);
    }
}

/// Relaxation times are log-spaced, ascending, and padded beyond the band.
#[test]
fn relaxation_times_span_the_padded_band() {
    let band = FitBand::new(1.0e6, 10.0e6, 5).expect("valid band");
    let taus = band.relaxation_times();
    assert_eq!(taus.len(), 5);
    for pair in taus.windows(2) {
        assert!(pair[1] > pair[0], "τ grid must ascend: {taus:?}");
    }
    let pad = 10.0_f64.powf(band.tau_padding_decades);
    assert!((taus[0] - 1.0 / (TWO_PI * 10.0e6 * pad)).abs() < 1e-18);
    assert!((taus[4] - pad / (TWO_PI * 1.0e6)).abs() < 1e-18);
    // Log-spacing: equal ratios between neighbours.
    let ratio = taus[1] / taus[0];
    for pair in taus.windows(2) {
        assert!((pair[1] / pair[0] - ratio).abs() < 1e-12);
    }
}

#[test]
fn rejects_invalid_bands_and_targets() {
    assert!(FitBand::new(0.0, 1.0e6, 4).is_err());
    assert!(FitBand::new(2.0e6, 1.0e6, 4).is_err());
    assert!(FitBand::new(1.0e6, 2.0e6, 0).is_err());

    let band = FitBand::new(1.0e6, 2.0e6, 3).expect("valid band");
    for bad in [
        PowerLawTarget {
            density: 0.0,
            ..target(0.5, 1.0)
        },
        PowerLawTarget {
            sound_speed: -1.0,
            ..target(0.5, 1.0)
        },
        PowerLawTarget {
            alpha_ref_np_m: -1.0,
            ..target(0.5, 1.0)
        },
        PowerLawTarget {
            exponent: f64::NAN,
            ..target(0.5, 1.0)
        },
        PowerLawTarget {
            f_ref: 0.0,
            ..target(0.5, 1.0)
        },
    ] {
        assert!(fit_power_law(&bad, &band).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn rejects_mismatched_field_shapes() {
    let band = FitBand::new(1.0e6, 2.0e6, 3).expect("valid band");
    let a = Array3::<f64>::from_elem([2, 1, 1], alpha_np_m(0.5));
    let g = Array3::<f64>::from_elem([3, 1, 1], 1.0);
    let c = Array3::<f64>::from_elem([2, 1, 1], C_WATER_LIKE);
    let r = Array3::<f64>::from_elem([2, 1, 1], RHO_TISSUE);
    assert!(fit_power_law_fields(&a, &g, &c, &r, F_REF, &band).is_err());
}
