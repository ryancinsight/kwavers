//! Value-semantic tests for the FWI misfit selector wiring (FWI-1).
//!
//! Two contracts are verified:
//!
//! 1. **Wiring.** `FwiProcessor::with_misfit(t)` routes both the objective
//!    evaluation (`compute_misfit_objective`) and the adjoint source
//!    (`compute_adjoint_source`) through the canonical `MisfitFunction`
//!    dispatcher for the selected type `t`, and the L2 default preserves the
//!    `(dt/2)‖r‖²` least-squares objective exactly.
//!
//! 2. **Cycle-skipping contrast.** For a band-limited wavelet shifted in time,
//!    the L2 misfit is *non-monotonic* in the shift — a full-period offset looks
//!    *better* than a half-period offset (the cycle-skip signature, Virieux &
//!    Operto 2009 §4) — whereas the envelope misfit (Bozdağ et al. 2011) and the
//!    1-Wasserstein optimal-transport misfit (Métivier et al. 2016) are
//!    monotone in the shift magnitude, which is the property that makes them
//!    robust to a poor starting model.

use super::super::FwiProcessor;
use crate::inverse::fwi::time_domain::l2_objective;
use crate::inverse::reconstruction::seismic::{MisfitFunction, MisfitType};
use crate::inverse::seismic::parameters::FwiParameters;
use leto::Array2;

/// Carrier period in samples for the synthetic wavelet.
const PERIOD_SAMPLES: usize = 16;
/// Gaussian envelope half-width in samples (≫ period → broad band-limited burst).
const ENVELOPE_SIGMA: f64 = 32.0;
const N_SAMPLES: usize = 160;
const CENTER: f64 = 80.0;

/// Gaussian-windowed cosine (Morlet-like) wavelet centred at `center` samples,
/// returned as a single-trace `(1, N_SAMPLES)` matrix. Oscillatory: this is the
/// data class on which L2 cycle-skips.
fn morlet_trace(center: f64) -> Array2<f64> {
    let two_pi = std::f64::consts::TAU;
    let mut data = Array2::zeros((1, N_SAMPLES));
    for j in 0..N_SAMPLES {
        let u = j as f64 - center;
        let envelope = (-(u / ENVELOPE_SIGMA).powi(2)).exp();
        let carrier = (two_pi * u / PERIOD_SAMPLES as f64).cos();
        data[[0, j]] = envelope * carrier;
    }
    data
}

/// Strictly positive Gaussian bump centred at `center` samples. This is the
/// data class for which the 1-Wasserstein optimal-transport distance is convex
/// in the time shift (Engquist & Froese 2014): the OT preprocessing
/// (non-negativity + normalisation to a probability density) is exact here, so
/// the transport cost grows monotonically with the displacement.
fn gaussian_bump(center: f64) -> Array2<f64> {
    let mut data = Array2::zeros((1, N_SAMPLES));
    for j in 0..N_SAMPLES {
        let u = j as f64 - center;
        data[[0, j]] = (-(u / ENVELOPE_SIGMA).powi(2)).exp();
    }
    data
}

#[test]
fn default_processor_uses_l2_least_squares() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: N_SAMPLES,
        dt: 0.5,
        ..FwiParameters::default()
    });
    assert!(matches!(processor.misfit_type(), MisfitType::L2Norm));

    let observed = morlet_trace(CENTER);
    let synthetic = morlet_trace(CENTER + 3.0);

    let objective = processor
        .compute_misfit_objective(&observed, &synthetic)
        .expect("objective");
    let expected = l2_objective(0.5, &observed, &synthetic).expect("l2 reference");
    assert_eq!(objective, expected);

    let adjoint = processor
        .compute_adjoint_source(&observed, &synthetic)
        .expect("adjoint");
    assert_eq!(adjoint, &synthetic - &observed);
}

#[test]
fn with_misfit_routes_objective_and_adjoint_through_dispatcher() {
    let observed = morlet_trace(CENTER);
    let synthetic = morlet_trace(CENTER + 5.0);

    for misfit in [
        MisfitType::Envelope,
        MisfitType::Phase,
        MisfitType::Wasserstein,
        MisfitType::Correlation,
        MisfitType::L1Norm,
    ] {
        let processor = FwiProcessor::default().with_misfit(misfit);
        assert_eq!(
            std::mem::discriminant(&processor.misfit_type()),
            std::mem::discriminant(&misfit),
            "with_misfit must store the selected type"
        );

        let reference = MisfitFunction::new(misfit);
        let objective = processor
            .compute_misfit_objective(&observed, &synthetic)
            .expect("objective");
        let expected_objective = reference
            .compute(&observed, &synthetic)
            .expect("ref objective");
        assert_eq!(
            objective, expected_objective,
            "objective must match dispatcher for {misfit:?}"
        );

        let adjoint = processor
            .compute_adjoint_source(&observed, &synthetic)
            .expect("adjoint");
        let expected_adjoint = reference
            .compute_adjoint_source(&observed, &synthetic)
            .expect("ref adjoint");
        assert_eq!(
            adjoint, expected_adjoint,
            "adjoint source must match dispatcher for {misfit:?}"
        );
    }
}

#[test]
fn l2_cycle_skips_while_envelope_stays_monotone_on_oscillatory_data() {
    let observed = morlet_trace(CENTER);
    let half_period = PERIOD_SAMPLES as f64 / 2.0;
    let full_period = PERIOD_SAMPLES as f64;

    let synthetic_half = morlet_trace(CENTER + half_period);
    let synthetic_full = morlet_trace(CENTER + full_period);

    let l2 = MisfitFunction::new(MisfitType::L2Norm);
    let envelope = MisfitFunction::new(MisfitType::Envelope);

    let l2_half = l2.compute(&observed, &synthetic_half).expect("l2 half");
    let l2_full = l2.compute(&observed, &synthetic_full).expect("l2 full");

    let env_half = envelope
        .compute(&observed, &synthetic_half)
        .expect("env half");
    let env_full = envelope
        .compute(&observed, &synthetic_full)
        .expect("env full");

    // Cycle-skip signature: the full-period offset (carrier back in phase) has a
    // *smaller* L2 misfit than the half-period offset (carrier anti-phase),
    // even though it is a larger time shift. A descent method on L2 from the
    // full-period side is therefore pulled toward the wrong cycle.
    assert!(
        l2_full < l2_half,
        "L2 must be non-monotonic in shift (cycle skip): l2_full={l2_full:.6e} l2_half={l2_half:.6e}"
    );

    // The envelope misfit removes the carrier, so its value increases
    // monotonically with the shift magnitude — the larger shift is correctly
    // scored as the worse fit, defeating the cycle skip.
    assert!(
        env_full > env_half && env_half > 0.0,
        "envelope misfit must increase with shift: env_full={env_full:.6e} env_half={env_half:.6e}"
    );
}

#[test]
fn wasserstein_is_convex_in_shift_on_positive_distribution() {
    // The 1-Wasserstein OT distance is convex in the time shift only for
    // positive distributions; on raw oscillatory traces the non-negativity
    // transform destroys that property (a known limitation — Engquist & Froese
    // 2014; Métivier et al. 2016). Validate the implementation on its valid
    // domain: a strictly positive Gaussian bump.
    let observed = gaussian_bump(CENTER);
    let wasserstein = MisfitFunction::new(MisfitType::Wasserstein);

    let mut previous = 0.0;
    for shift in [0.0_f64, 4.0, 8.0, 16.0, 24.0] {
        let synthetic = gaussian_bump(CENTER + shift);
        let distance = wasserstein
            .compute(&observed, &synthetic)
            .expect("wasserstein");
        if shift == 0.0 {
            assert!(distance < 1e-9, "self-distance must vanish: {distance:.3e}");
        } else {
            assert!(
                distance > previous,
                "Wasserstein must increase with shift {shift}: {distance:.6e} <= {previous:.6e}"
            );
        }
        previous = distance;
    }
}
