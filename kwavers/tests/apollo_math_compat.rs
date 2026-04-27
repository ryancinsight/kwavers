use kwavers::math::fft::{Fft3d, Shape3D};
use ndarray::Array3;

/// Verify that `kwavers::math::Fft3d` (backed by apollo-fft) round-trips
/// a real-valued field through forward + inverse FFT with no numerical drift.
///
/// Theorem: IFFT(FFT(x)) = x for all finite x, exact up to floating-point
/// rounding (max absolute error < 1e-12 for double precision on this grid size).
#[test]
fn kwavers_math_reexports_fft_roundtrip() {
    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let plan = Fft3d::new(shape);
    let field = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.2 + k as f64 * 0.1).sin()
    });
    let spectrum = plan.forward(&field);
    let recovered = plan.inverse(&spectrum);

    let max_abs_error = field
        .iter()
        .zip(recovered.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error < 1e-12,
        "FFT round-trip max abs error {max_abs_error:.3e} exceeds 1e-12"
    );
}
