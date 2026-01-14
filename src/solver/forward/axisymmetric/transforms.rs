//! Discrete Hankel Transform for axisymmetric solver
//!
//! Implements the discrete Hankel transform (DHT) for handling radial
//! derivatives in cylindrical coordinates. The DHT is the analog of the
//! Fourier transform for radially symmetric functions.
//!
//! # Theory
//!
//! The Hankel transform of order 0 is defined as:
//!
//! $$
//! F(k) = \int_0^\infty f(r) J_0(kr) r \, dr
//! $$
//!
//! where $J_0$ is the zeroth-order Bessel function of the first kind.
//!
//! # References
//!
//! - Guizar-Sicairos, M., & Gutiérrez-Vega, J. C. (2004). "Computation of
//!   quasi-discrete Hankel transforms of integer order for propagating optical
//!   wave fields." JOSA A, 21(1), 53-58.

use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

/// Discrete Hankel Transform (DHT) implementation
///
/// Uses the quasi-discrete Hankel transform (QDHT) algorithm for
/// efficient computation of radial derivatives.
#[derive(Debug, Clone)]
pub struct DiscreteHankelTransform {
    /// Number of radial points
    nr: usize,
    /// Maximum radial extent (R)
    r_max: f64,
    /// Maximum k-space extent (K)
    k_max: f64,
    /// Radial sample points
    r: Array1<f64>,
    /// k-space sample points
    k: Array1<f64>,
    /// Transform matrix
    transform_matrix: Array2<f64>,
    /// Inverse transform matrix
    inverse_matrix: Array2<f64>,
}

impl DiscreteHankelTransform {
    /// Create a new discrete Hankel transform
    ///
    /// # Arguments
    ///
    /// * `nr` - Number of radial sample points
    /// * `r_max` - Maximum radial extent (m)
    pub fn new(nr: usize, r_max: f64) -> Self {
        // Zeros of J0 Bessel function (approximation)
        let j0_zeros = Self::bessel_j0_zeros(nr + 1);

        // Maximum k-space extent
        let j_nr = j0_zeros[nr];
        let k_max = j_nr / r_max;

        // Radial sample points: r_m = j_{0,m} * R / j_{0,N}
        let r = Array1::from_shape_fn(nr, |m| {
            if m == 0 {
                0.0 // Handle r=0 specially
            } else {
                j0_zeros[m] * r_max / j_nr
            }
        });

        // k-space sample points: k_n = j_{0,n} / R
        let k = Array1::from_shape_fn(nr, |n| if n == 0 { 0.0 } else { j0_zeros[n] / r_max });

        // Compute transform matrix T_mn = 2 * J0(j_{0,m} * j_{0,n} / j_{0,N}) / (j_{0,N} * |J1(j_{0,m})|^2)
        let transform_matrix = Self::compute_transform_matrix(nr, &j0_zeros);
        let inverse_matrix = transform_matrix.clone(); // DHT is self-inverse up to scaling

        Self {
            nr,
            r_max,
            k_max,
            r,
            k,
            transform_matrix,
            inverse_matrix,
        }
    }

    /// Compute zeros of J0 Bessel function using McMahon's approximation
    fn bessel_j0_zeros(n: usize) -> Vec<f64> {
        let mut zeros = vec![0.0]; // TODO: j_{0,0} = 0 (not actually a zero, placeholder)

        for m in 1..=n {
            // McMahon's asymptotic approximation for large m
            let beta = (m as f64 - 0.25) * PI;
            let approx = beta + 1.0 / (8.0 * beta) - 4.0 / (3.0 * (8.0 * beta).powi(3));
            zeros.push(approx);
        }

        zeros
    }

    /// Compute the transform matrix
    fn compute_transform_matrix(nr: usize, j0_zeros: &[f64]) -> Array2<f64> {
        let j_nr = j0_zeros[nr];
        let mut matrix = Array2::zeros((nr, nr));

        for m in 0..nr {
            for n in 0..nr {
                if m == 0 || n == 0 {
                    matrix[[m, n]] = if m == n { 1.0 } else { 0.0 };
                } else {
                    let j_m = j0_zeros[m];
                    let j_n = j0_zeros[n];
                    let arg = j_m * j_n / j_nr;

                    // J0(arg) using series approximation for small args
                    let j0_val = Self::bessel_j0(arg);

                    // J1(j_m) for normalization
                    let j1_m = Self::bessel_j1(j_m);

                    matrix[[m, n]] = 2.0 * j0_val / (j_nr * j1_m * j1_m);
                }
            }
        }

        matrix
    }

    /// Bessel J0 function approximation
    fn bessel_j0(x: f64) -> f64 {
        if x.abs() < 8.0 {
            // Polynomial approximation for |x| < 8
            let y = x * x;
            let ans1 = 57568490574.0
                + y * (-13362590354.0
                    + y * (651619640.7
                        + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
            let ans2 = 57568490411.0
                + y * (1029532985.0
                    + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
            ans1 / ans2
        } else {
            // Asymptotic approximation for large |x|
            let ax = x.abs();
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 0.785398164;
            let ans1 = 1.0
                + y * (-0.1098628627e-2
                    + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            let ans2 = -0.1562499995e-1
                + y * (0.1430488765e-3
                    + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
            (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
        }
    }

    /// Bessel J1 function approximation
    fn bessel_j1(x: f64) -> f64 {
        if x.abs() < 8.0 {
            let y = x * x;
            let ans1 = x
                * (72362614232.0
                    + y * (-7895059235.0
                        + y * (242396853.1
                            + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
            let ans2 = 144725228442.0
                + y * (2300535178.0
                    + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
            ans1 / ans2
        } else {
            let ax = x.abs();
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 2.356194491;
            let ans1 = 1.0
                + y * (0.183105e-2
                    + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
            let ans2 = 0.04687499995
                + y * (-0.2002690873e-3
                    + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
            let ans =
                (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
            if x < 0.0 {
                -ans
            } else {
                ans
            }
        }
    }

    /// Forward Hankel transform
    ///
    /// Transforms a radial function f(r) to its Hankel domain F(k)
    pub fn forward(&self, f: &Array1<f64>) -> Array1<f64> {
        assert_eq!(f.len(), self.nr, "Input size must match transform size");
        self.transform_matrix.dot(f)
    }

    /// Inverse Hankel transform
    ///
    /// Transforms from Hankel domain F(k) back to radial domain f(r)
    pub fn inverse(&self, f_k: &Array1<f64>) -> Array1<f64> {
        assert_eq!(f_k.len(), self.nr, "Input size must match transform size");
        self.inverse_matrix.dot(f_k)
    }

    /// Forward transform for 2D field (radial dimension is axis 1)
    pub fn forward_2d(&self, field: &Array2<f64>) -> Array2<f64> {
        let (nz, nr) = field.dim();
        assert_eq!(nr, self.nr, "Radial dimension must match transform size");

        let mut result = Array2::zeros((nz, nr));
        for i in 0..nz {
            let row = field.index_axis(Axis(0), i);
            let transformed = self.forward(&row.to_owned());
            for j in 0..nr {
                result[[i, j]] = transformed[j];
            }
        }
        result
    }

    /// Inverse transform for 2D field
    pub fn inverse_2d(&self, field: &Array2<f64>) -> Array2<f64> {
        let (nz, nr) = field.dim();
        assert_eq!(nr, self.nr, "Radial dimension must match transform size");

        let mut result = Array2::zeros((nz, nr));
        for i in 0..nz {
            let row = field.index_axis(Axis(0), i);
            let transformed = self.inverse(&row.to_owned());
            for j in 0..nr {
                result[[i, j]] = transformed[j];
            }
        }
        result
    }

    /// Get radial sample points
    pub fn r(&self) -> &Array1<f64> {
        &self.r
    }

    /// Get k-space sample points
    pub fn k(&self) -> &Array1<f64> {
        &self.k
    }

    /// Maximum radial extent
    pub fn r_max(&self) -> f64 {
        self.r_max
    }

    /// Maximum k-space extent
    pub fn k_max(&self) -> f64 {
        self.k_max
    }
}

/// Apply radial derivative in k-space
///
/// For axisymmetric problems, the radial Laplacian term:
/// (1/r) d/dr (r df/dr) = d²f/dr² + (1/r) df/dr
///
/// In k-space this becomes: -k² F(k)
pub fn apply_radial_laplacian(f_k: &Array1<f64>, k: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(f_k.len());
    for i in 0..f_k.len() {
        result[i] = -k[i] * k[i] * f_k[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dht_creation() {
        let dht = DiscreteHankelTransform::new(32, 0.01);
        assert_eq!(dht.nr, 32);
        assert!((dht.r_max - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_bessel_j0() {
        // J0(0) = 1
        assert_relative_eq!(DiscreteHankelTransform::bessel_j0(0.0), 1.0, epsilon = 1e-6);

        // J0(2.4048) ≈ 0 (first zero)
        let j0_zero = DiscreteHankelTransform::bessel_j0(2.4048);
        assert!(j0_zero.abs() < 0.001);
    }

    #[test]
    fn test_forward_inverse_identity() {
        let dht = DiscreteHankelTransform::new(16, 0.01);

        // Create a simple radial function
        let f = Array1::from_shape_fn(16, |i| {
            let r = i as f64 * 0.01 / 16.0;
            (-r * r / 0.001).exp()
        });

        // Forward then inverse should give original (approximately)
        let f_k = dht.forward(&f);
        let f_recovered = dht.inverse(&f_k);

        for i in 1..16 {
            // Skip r=0 which can have numerical issues
            assert_relative_eq!(f[i], f_recovered[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_2d_transform() {
        let dht = DiscreteHankelTransform::new(8, 0.01);

        let field = Array2::from_shape_fn((4, 8), |(i, j)| (i as f64 + j as f64) / 10.0);

        let result = dht.forward_2d(&field);
        assert_eq!(result.dim(), (4, 8));
    }
}
