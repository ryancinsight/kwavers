//! Steering vector generator for uniform linear arrays.

use kwavers_core::constants::numerical::TWO_PI;
use ndarray::Array1;
use num_complex::Complex64;

/// Create a steering vector for a uniform linear array (ULA).
///
/// # Mathematical Definition
///
/// For a linear array with half-wavelength spacing (d = λ/2, k = 2π):
///
/// ```text
/// a(θ) = [1, e^{jkd·sin(θ)}, e^{j2kd·sin(θ)}, ..., e^{j(N-1)kd·sin(θ)}]^T
/// ```
///
/// # Parameters
///
/// - `n`: Number of array elements
/// - `angle_rad`: Steering angle from broadside (radians)
pub fn create_steering_vector(n: usize, angle_rad: f64) -> Array1<Complex64> {
    let k = TWO_PI; // Normalized wavenumber (λ = 1, d = 0.5λ)
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let phase = k * (i as f64) * angle_rad.sin();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect(),
    )
}
