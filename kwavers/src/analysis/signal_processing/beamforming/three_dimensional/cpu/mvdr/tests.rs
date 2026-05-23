use super::*;
use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use ndarray::Array4;

fn make_config(
    nel: (usize, usize, usize),
    elem_spacing: (f64, f64, f64),
    vol: (usize, usize, usize),
    sound_speed: f64,
    sampling_frequency: f64,
) -> BeamformingConfig3D {
    BeamformingConfig3D {
        num_elements_3d: nel,
        element_spacing_3d: elem_spacing,
        volume_dims: vol,
        voxel_spacing: (1e-3, 1e-3, 1e-3),
        sound_speed,
        sampling_frequency,
        ..BeamformingConfig3D::default()
    }
}

/// ## Theorem: L=1 MVDR is the identity weighting
///
/// For a single-element, single-subarray (L = 1), the MVDR weight vector
/// reduces to the scalar identity so that P · u = 1 for any loading factor δ.
/// The output equals |x̄[0]| = |rf_get(0, 0, τ₀)|.
///
/// ## Proof
/// R = (1/N) Σ_n x₀[n]²  (1×1 positive scalar, call it σ²).
/// R_δ = σ²(1 + δ)   (loading multiplies σ² by (1+δ)).
/// u = 1/R_δ  (1×1 inverse).
/// denom = 1ᵀu = 1/R_δ.
/// P = R_δ.
/// P · u = R_δ · (1/R_δ) = 1  ← independent of δ.
/// output = |P · u · x̄[0]| = |x̄[0]|.
/// With voxel and element both at origin, τ₀ = 0, so x̄[0] = rf[0,0,0,0].
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn mvdr_single_element_equals_signal_at_delay() {
    let config = make_config(
        (1, 1, 1),
        (1e-3, 1e-3, 1e-3),
        (1, 1, 1),
        SOUND_SPEED_WATER_SIM,
        1_000_000.0,
    );
    let mut rf = Array4::<f32>::zeros((1, 1, 6, 1));
    rf[[0, 0, 0, 0]] = 3.0;
    rf[[0, 0, 1, 0]] = 1.0;
    rf[[0, 0, 2, 0]] = 2.0;

    let vol = mvdr_cpu(&rf, &config, 0.1, [1, 1, 1]).unwrap();
    // output = |rf[0,0,0,0]| = 3.0 by the proof above.
    assert!(
        (vol[[0, 0, 0]] - 3.0_f32).abs() < 1e-4_f32,
        "MVDR L=1 identity: expected 3.0, got {}",
        vol[[0, 0, 0]]
    );
}

/// ## Theorem: L=1 MVDR is invariant to diagonal loading
///
/// Corollary of the L=1 identity proof: because P · u = 1 regardless of δ,
/// the output |x̄[0]| is the same for any δ > 0.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn mvdr_single_element_invariant_to_diagonal_loading() {
    let config = make_config(
        (1, 1, 1),
        (1e-3, 1e-3, 1e-3),
        (1, 1, 1),
        SOUND_SPEED_WATER_SIM,
        1_000_000.0,
    );
    let mut rf = Array4::<f32>::zeros((1, 1, 6, 1));
    rf[[0, 0, 0, 0]] = 4.0;

    let reference = mvdr_cpu(&rf, &config, 0.01, [1, 1, 1]).unwrap()[[0, 0, 0]];
    for &delta in &[0.1_f32, 1.0, 10.0, 100.0] {
        let out = mvdr_cpu(&rf, &config, delta, [1, 1, 1]).unwrap()[[0, 0, 0]];
        assert!(
            (out - reference).abs() < 1e-4_f32,
            "L=1 MVDR invariance: δ={delta} gave {out}, expected {reference}"
        );
    }
}

/// ## Theorem: Channel-count mismatch rejected
///
/// `mvdr_cpu` returns `KwaversError::InvalidInput` when the RF channel axis
/// ≠ nel_x × nel_y × nel_z.
/// # Panics
/// - Panics with `"expected InvalidInput, got {other:?}"`.
///
#[test]
fn mvdr_channel_mismatch_returns_error() {
    let config = make_config(
        (1, 1, 1), // expects 1 channel
        (1e-3, 1e-3, 1e-3),
        (1, 1, 1),
        SOUND_SPEED_WATER_SIM,
        1_000_000.0,
    );
    let rf = Array4::<f32>::zeros((1, 7, 4, 1));
    match mvdr_cpu(&rf, &config, 0.01, [1, 1, 1]).unwrap_err() {
        KwaversError::InvalidInput(msg) => {
            assert!(
                msg.contains("channel") || msg.contains("element"),
                "error must reference channel mismatch; got: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// ## Theorem: Subarray exceeding array rejected
///
/// `mvdr_cpu` returns `KwaversError::InvalidInput` when any component of
/// `subarray_size` exceeds the corresponding `num_elements_3d` dimension.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn mvdr_subarray_exceeds_array_returns_error() {
    let config = make_config(
        (1, 2, 1), // nel_y = 2
        (1e-3, 1e-3, 1e-3),
        (1, 1, 1),
        SOUND_SPEED_WATER_SIM,
        1_000_000.0,
    );
    let rf = Array4::<f32>::zeros((1, 2, 4, 1));
    // ly = 3 > nel_y = 2 → must be rejected.
    let result = mvdr_cpu(&rf, &config, 0.01, [1, 3, 1]);
    assert!(
        result.is_err(),
        "MVDR must reject subarray dimension exceeding array"
    );
}

/// ## Theorem: Diagonal loading guarantees Cholesky success and P > 0
///
/// With δ > 0, R_δ = R + δ·(tr(R)/L)·I is strictly positive-definite
/// because the loading term δ·(tr(R)/L) > 0 whenever R has at least one
/// non-zero diagonal entry.  Cholesky therefore always succeeds, and the
/// output power P = 1/(1ᵀ R_δ⁻¹ 1) > 0 because 1ᵀ R_δ⁻¹ 1 is a
/// positive-definite quadratic form.
///
/// Tested on a 2-element co-located array with uncorrelated signals
/// (off-diagonal R entries are small), verifying that the result is finite
/// and non-negative for a range of δ values spanning four orders of magnitude.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn mvdr_diagonal_loading_ensures_finite_positive_output() {
    // 2 elements at the same position (spacing=0) → delay=0 for origin voxel.
    let config = make_config(
        (1, 2, 1),
        (0.0, 0.0, 0.0),
        (1, 1, 1),
        SOUND_SPEED_WATER_SIM,
        1_000_000.0,
    );
    // Uncorrelated pulses: ch0 fires at n=0, ch1 fires at n=2.
    let mut rf = Array4::<f32>::zeros((1, 2, 5, 1));
    rf[[0, 0, 0, 0]] = 2.0;
    rf[[0, 1, 2, 0]] = 2.0;

    for &delta in &[0.001_f32, 0.1, 1.0, 10.0] {
        let vol = mvdr_cpu(&rf, &config, delta, [1, 2, 1]).unwrap();
        let out = vol[[0, 0, 0]];
        assert!(
            out.is_finite() && out >= 0.0,
            "MVDR output must be finite and non-negative for δ={delta}; got {out}"
        );
    }
}
