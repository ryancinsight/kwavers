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

fn band_with(n_arms: usize, placement: RelaxationTimePlacement) -> FitBand {
    let mut band = FitBand::new(0.5e6, 5.0e6, n_arms).expect("valid band");
    band.placement = placement;
    band
}

/// **Two relaxation mechanisms suffice.** Fullwave 2.5 ships a database fitted
/// at `num_relax = 2`; matching that arm count is what makes the memory cost of
/// a heterogeneous run tractable, since the time-domain solver carries one field
/// per arm per voxel.
///
/// On a fixed log-spaced grid two arms cannot span a decade — the error is ~30 %.
/// Optimizing the times brings the same two arms to under 3 %, which is the
/// capability this placement exists to provide.
#[test]
fn two_optimized_arms_cover_the_envelope() {
    let optimized = band_with(2, RelaxationTimePlacement::Optimized);
    let log_spaced = band_with(2, RelaxationTimePlacement::LogSpaced);

    let mut worst_optimized = 0.0_f64;
    let mut worst_log_spaced = 0.0_f64;
    for &gamma in &[0.4, 0.7, 1.0, 1.3, 1.6] {
        let t = target(0.5, gamma);
        worst_optimized = worst_optimized.max(
            fit_power_law(&t, &optimized)
                .expect("fit converges")
                .max_relative_error(),
        );
        worst_log_spaced = worst_log_spaced.max(
            fit_power_law(&t, &log_spaced)
                .expect("fit converges")
                .max_relative_error(),
        );
    }

    assert!(
        worst_optimized < 3.0e-2,
        "two optimized arms gave {worst_optimized:.4}"
    );
    // The improvement is the point: an order of magnitude, not a rounding.
    assert!(
        worst_log_spaced > 10.0 * worst_optimized,
        "optimizing the times bought only {:.1}x ({worst_log_spaced:.4} -> \
         {worst_optimized:.4})",
        worst_log_spaced / worst_optimized
    );
}

/// Optimized placement starts from the log-spaced grid and keeps it unless it
/// finds better, so it can never be the worse of the two at any arm count.
#[test]
fn optimizing_never_degrades_the_fit() {
    for n_arms in [1usize, 2, 3, 4, 6] {
        for &gamma in &[0.4, 1.0, 1.6] {
            let t = target(0.5, gamma);
            let optimized =
                fit_power_law(&t, &band_with(n_arms, RelaxationTimePlacement::Optimized))
                    .expect("fit converges")
                    .max_relative_error();
            let log_spaced =
                fit_power_law(&t, &band_with(n_arms, RelaxationTimePlacement::LogSpaced))
                    .expect("fit converges")
                    .max_relative_error();
            assert!(
                optimized <= log_spaced,
                "{n_arms} arms, gamma={gamma}: optimized {optimized:.5} is worse \
                 than log-spaced {log_spaced:.5}"
            );
        }
    }
}

/// The search is deterministic: no RNG, a fixed initial simplex, and a canonical
/// (sorted) parameterization. Two identical calls must agree bit for bit, or a
/// simulation is not reproducible from its inputs.
#[test]
fn optimized_placement_is_deterministic() {
    let band = band_with(3, RelaxationTimePlacement::Optimized);
    let t = target(0.6, 1.2);
    let first = fit_power_law(&t, &band).expect("fit converges");
    let second = fit_power_law(&t, &band).expect("fit converges");
    assert_eq!(first.relaxation_times(), second.relaxation_times());
    assert_eq!(first.weights(), second.weights());
    assert_eq!(first.equilibrium_modulus(), second.equilibrium_modulus());
}

/// A heterogeneous medium gets **one** relaxation-time grid, chosen against
/// every distinct voxel at once. This is the invariant the solver depends on —
/// one memory field per arm for the whole domain — and the reason the field fit
/// cannot simply optimize each voxel independently.
#[test]
fn heterogeneous_field_shares_one_optimized_grid() {
    let shape = [4usize, 1, 1];
    let gammas = [0.5, 0.8, 1.2, 1.5];
    let alphas_db = [0.3, 0.5, 0.6, 0.75];
    let mut alpha = Array3::<f64>::zeros(shape);
    let mut gamma = Array3::<f64>::zeros(shape);
    let c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);
    for i in 0..4 {
        alpha[[i, 0, 0]] = alpha_np_m(alphas_db[i]);
        gamma[[i, 0, 0]] = gammas[i];
    }

    // Three arms — a count that the log-spaced grid cannot serve well, so the
    // ensemble search is doing real work here.
    let band = band_with(3, RelaxationTimePlacement::Optimized);
    let fit =
        fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band).expect("field fit converges");

    assert_eq!(fit.relaxation_times().len(), 3);

    // Sharing one grid across four exponents costs something relative to giving
    // each voxel its own; measure that cost rather than asserting a round
    // number. The best any voxel could do with a private 3-arm grid is its
    // single-target fit, and the shared grid must stay within a small multiple
    // of the worst of those.
    let best_private = (0..4)
        .map(|i| {
            let t = PowerLawTarget {
                alpha_ref_np_m: alpha[[i, 0, 0]],
                exponent: gammas[i],
                f_ref: F_REF,
                sound_speed: C_WATER_LIKE,
                density: RHO_TISSUE,
            };
            fit_power_law(&t, &band)
                .expect("private fit")
                .max_relative_error()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        fit.max_relative_error() <= 10.0 * best_private.max(1e-4),
        "sharing one grid cost {:.4} against {best_private:.4} per-voxel",
        fit.max_relative_error()
    );
    // Absolute ceiling: still an order below the ~10 % inter-study spread in
    // reported tissue alpha_0, so the sharing penalty cannot dominate a result.
    assert!(
        fit.max_relative_error() < 2.0e-2,
        "worst voxel on the shared grid: {:.4}",
        fit.max_relative_error()
    );

    // Every arm field carries the *same* time for all voxels.
    for (l, (_, tau_field)) in fit.arm_fields().iter().enumerate() {
        for i in 0..4 {
            assert_eq!(tau_field[[i, 0, 0]], fit.relaxation_times()[l]);
        }
    }

    // And each voxel still follows its own law on that shared grid.
    for i in 0..4 {
        let weights: Vec<f64> = fit.weights().iter().map(|w| w[[i, 0, 0]]).collect();
        let m_inf = fit.equilibrium_modulus()[[i, 0, 0]];
        for &f in &[0.6e6, 2.0e6, 4.5e6] {
            let omega = TWO_PI * f;
            let m = complex_modulus(m_inf, &weights, fit.relaxation_times(), omega);
            let got = wavenumber(RHO_TISSUE, m, omega).im.abs();
            let want = alpha[[i, 0, 0]] * (f / F_REF).powf(gammas[i]);
            assert!(
                ((got - want) / want).abs() < 2.0e-2,
                "voxel {i} (gamma={}) at {f:e} Hz: {got:e} vs {want:e}",
                gammas[i]
            );
        }
    }
}

/// A lossless voxel alongside lossy ones neither breaks the ensemble search nor
/// acquires arms of its own.
#[test]
fn lossless_voxel_alongside_lossy_takes_no_arms() {
    let shape = [2usize, 1, 1];
    let mut alpha = Array3::<f64>::zeros(shape);
    alpha[[1, 0, 0]] = alpha_np_m(0.5);
    let gamma = Array3::<f64>::from_elem(shape, 1.1);
    let c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);

    let band = band_with(3, RelaxationTimePlacement::Optimized);
    let fit = fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band).expect("field fit");

    for field in fit.weights() {
        assert_eq!(field[[0, 0, 0]], 0.0, "lossless voxel gained an arm");
        assert!(field[[1, 0, 0]] >= 0.0);
    }
    assert_eq!(
        fit.equilibrium_modulus()[[0, 0, 0]],
        RHO_TISSUE * C_WATER_LIKE * C_WATER_LIKE
    );
    assert!(fit.max_relative_error() < 1.0e-2);
}

/// **Air inclusion in soft tissue** — the contrast case Fullwave 2.5 exercises
/// in four separate examples, and the hardest thing to ask of a *shared*
/// relaxation grid: air and tissue differ by 3600:1 in impedance, 4.5x in sound
/// speed, two orders of magnitude in absorption, and carry different exponents,
/// yet the solver can only hold one set of relaxation times for the domain.
#[test]
fn air_inclusion_shares_the_tissue_grid() {
    const C_AIR: f64 = 343.0;
    const RHO_AIR: f64 = 1.2;
    const ALPHA_AIR_NP_M: f64 = 20.0;
    const GAMMA_AIR: f64 = 1.8;
    const GAMMA_TISSUE: f64 = 1.1;

    let shape = [2usize, 1, 1];
    let mut alpha = Array3::<f64>::from_elem(shape, alpha_np_m(0.5));
    let mut gamma = Array3::<f64>::from_elem(shape, GAMMA_TISSUE);
    let mut c = Array3::<f64>::from_elem(shape, C_WATER_LIKE);
    let mut rho = Array3::<f64>::from_elem(shape, RHO_TISSUE);
    alpha[[1, 0, 0]] = ALPHA_AIR_NP_M;
    gamma[[1, 0, 0]] = GAMMA_AIR;
    c[[1, 0, 0]] = C_AIR;
    rho[[1, 0, 0]] = RHO_AIR;

    let band = band_with(3, RelaxationTimePlacement::Optimized);
    let fit = fit_power_law_fields(&alpha, &gamma, &c, &rho, F_REF, &band)
        .expect("air/tissue field fit converges");

    assert!(
        fit.max_relative_error() < 1.0e-2,
        "worst voxel across the air/tissue contrast: {:.4}",
        fit.max_relative_error()
    );

    // Each phase keeps its own prescribed speed and law on the shared grid.
    for (voxel, density, speed, exponent, alpha0) in [
        (
            0usize,
            RHO_TISSUE,
            C_WATER_LIKE,
            GAMMA_TISSUE,
            alpha_np_m(0.5),
        ),
        (1usize, RHO_AIR, C_AIR, GAMMA_AIR, ALPHA_AIR_NP_M),
    ] {
        let weights: Vec<f64> = fit.weights().iter().map(|w| w[[voxel, 0, 0]]).collect();
        let m_inf = fit.equilibrium_modulus()[[voxel, 0, 0]];

        let omega_ref = TWO_PI * F_REF;
        let m = complex_modulus(m_inf, &weights, fit.relaxation_times(), omega_ref);
        let c_p = omega_ref / wavenumber(density, m, omega_ref).re;
        assert!(
            ((c_p - speed) / speed).abs() < 1.0e-9,
            "voxel {voxel}: phase velocity {c_p} != prescribed {speed}"
        );

        for &f in &[0.6e6, 2.0e6, 4.5e6] {
            let omega = TWO_PI * f;
            let m = complex_modulus(m_inf, &weights, fit.relaxation_times(), omega);
            let got = wavenumber(density, m, omega).im.abs();
            let want = alpha0 * (f / F_REF).powf(exponent);
            assert!(
                ((got - want) / want).abs() < 1.0e-2,
                "voxel {voxel} at {f:e} Hz: {got:e} vs {want:e}"
            );
        }

        // The unrelaxed (high-frequency) speed sets the solver's CFL. It must
        // stay close to the prescribed speed: a relaxation spectrum that
        // reproduced alpha by inflating the instantaneous modulus would silently
        // force a far smaller time step on the whole simulation.
        let unrelaxed: f64 = ((m_inf + weights.iter().sum::<f64>()) / density).sqrt();
        assert!(
            unrelaxed > speed && unrelaxed < 1.10 * speed,
            "voxel {voxel}: unrelaxed speed {unrelaxed:.1} against prescribed {speed}"
        );
    }
}

/// **The distributed per-voxel solve reproduces the serial one bit for bit.**
///
/// Not "to a tolerance": each distinct voxel's Lawson-Hanson solve is
/// independent and deterministic, and the parallel path writes every result to
/// its own index, so no reduction order can vary. Anything short of bitwise
/// equality would mean the distribution changed the mathematics rather than
/// just its schedule (KW-MED-071).
///
/// The oracle is computed here by calling `fit_at_taus` in a plain serial loop
/// on the same shared relaxation times, rather than by comparing the function
/// against itself under two policies — a self-comparison would pass even if
/// both paths were wrong together.
#[test]
fn parallel_field_fit_matches_the_serial_solve() {
    // 6³ = 216 distinct tuples, comfortably past PARALLEL_FIT_THRESHOLD, so the
    // distributed path is the one under test rather than the serial fallback.
    let n = 6usize;
    let shape = [n, n, n];
    let smooth = |i: usize, j: usize, k: usize| {
        (i as f64 + 1.0) * 0.37 + (j as f64) * 0.11 + (k as f64) * 0.023
    };
    let alpha = Array3::from_shape_fn(shape, |[i, j, k]| 4.0 + smooth(i, j, k));
    let gamma = Array3::from_shape_fn(shape, |[i, j, k]| 1.05 + 0.02 * smooth(i, j, k));
    let speed = Array3::from_elem(shape, 1540.0);
    let density = Array3::from_elem(shape, 1000.0);

    let band = band_with(3, RelaxationTimePlacement::Optimized);
    let fit =
        fit_power_law_fields(&alpha, &gamma, &speed, &density, 1.0e6, &band).expect("field fit");

    // Serial oracle on the times the field fit selected.
    let freqs = band.frequencies();
    let omegas: Vec<f64> = freqs.iter().map(|f| std::f64::consts::TAU * f).collect();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let target = PowerLawTarget {
                    alpha_ref_np_m: alpha[[i, j, k]],
                    exponent: gamma[[i, j, k]],
                    f_ref: 1.0e6,
                    sound_speed: speed[[i, j, k]],
                    density: density[[i, j, k]],
                };
                let expected = fit_at_taus(&target, &omegas, &freqs, fit.relaxation_times())
                    .expect("serial fit");
                assert_eq!(
                    fit.equilibrium_modulus()[[i, j, k]],
                    expected.equilibrium_modulus(),
                    "voxel ({i},{j},{k}): equilibrium modulus differs from the serial solve"
                );
                for (arm, weight) in expected.weights().iter().enumerate() {
                    assert_eq!(
                        fit.weights()[arm][[i, j, k]],
                        *weight,
                        "voxel ({i},{j},{k}) arm {arm}: weight differs from the serial solve"
                    );
                }
            }
        }
    }
}
