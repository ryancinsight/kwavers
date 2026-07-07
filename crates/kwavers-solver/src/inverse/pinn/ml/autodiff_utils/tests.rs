use super::time::compute_time_derivative;
use coeus_autograd::Var;

type B = coeus_core::MoiraiBackend;

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
#[test]
fn test_time_derivative_matches_analytic() -> kwavers_core::error::KwaversResult<()> {
    let backend = B::default();

    // input rows: [t, x, y] = [0.2, 0.4, 0.6] and [0.7, 0.1, 0.9]
    let input = coeus_tensor::Tensor::from_slice_on(
        vec![2, 3],
        &[0.2_f32, 0.4, 0.6, 0.7, 0.1, 0.9],
        &backend,
    );

    // u₀ = t² + x + y   →  ∂u₀/∂t = 2t
    // u₁ = 2t + 3x − y  →  ∂u₁/∂t = 2 (constant, not tested here)
    let forward = |input: &Var<f32, B>| -> Var<f32, B> {
        let batch = input.tensor.shape()[0];
        let t = coeus_autograd::slice(input, &[(0, batch), (0, 1)]);
        let x = coeus_autograd::slice(input, &[(0, batch), (1, 2)]);
        let y = coeus_autograd::slice(input, &[(0, batch), (2, 3)]);

        let u0 = coeus_autograd::add(&coeus_autograd::add(&coeus_autograd::mul(&t, &t), &x), &y);
        let u1 = coeus_autograd::sub(
            &coeus_autograd::add(&coeus_autograd::scalar_mul(&t, 2.0), &coeus_autograd::scalar_mul(&x, 3.0)),
            &y,
        );

        coeus_autograd::cat(&[&u0, &u1], 1)
    };

    let du0_dt = compute_time_derivative::<B, _>(forward, &input, 0)?;
    let values = du0_dt.as_slice();

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
