use super::time::compute_time_derivative;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

/// Validates ∂u₀/∂t = 2t for u₀ = t² + x + y against the autodiff result.
///
/// Analytical: ∂u₀/∂t = 2t.
/// At t = 0.2: expected = 0.4; at t = 0.7: expected = 1.4.
/// Tolerance: 1e-4 (float32 autodiff accumulation).
/// # Panics
/// - Panics if an internal precondition is violated.
///
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
#[test]
fn test_time_derivative_matches_analytic() -> kwavers_core::error::KwaversResult<()> {
    type B = Autodiff<NdArray<f32>>;

    let device = Default::default();

    // input rows: [t, x, y] = [0.2, 0.4, 0.6] and [0.7, 0.1, 0.9]
    let input = Tensor::<B, 2>::from_floats([[0.2, 0.4, 0.6], [0.7, 0.1, 0.9]], &device);

    // u₀ = t² + x + y   →  ∂u₀/∂t = 2t
    // u₁ = 2t + 3x − y  →  ∂u₁/∂t = 2 (constant, not tested here)
    let forward = |input: Tensor<B, 2>| {
        let batch = input.dims()[0];
        let t = input.clone().slice([0..batch, 0..1]);
        let x = input.clone().slice([0..batch, 1..2]);
        let y = input.slice([0..batch, 2..3]);

        let u0 = t.clone().powf_scalar(2.0) + x.clone() + y.clone();
        let u1 = t.clone() * 2.0 + x * 3.0 - y;

        Tensor::cat(vec![u0, u1], 1)
    };

    let du0_dt = compute_time_derivative::<B, _>(forward, &input, 0)?;
    let data = du0_dt.into_data();
    let values = data.as_slice::<f32>().map_err(|e| {
        kwavers_core::error::KwaversError::System(
            kwavers_core::error::SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            },
        )
    })?;

    // ∂u₀/∂t = 2t: [2×0.2, 2×0.7] = [0.4, 1.4]
    let expected = [0.4_f32, 1.4_f32];
    for (got, exp) in values.iter().copied().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-4,
            "∂u₀/∂t mismatch: got {got:.6}, expected {exp:.6}"
        );
    }

    Ok(())
}
