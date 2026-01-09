//! Covariance Matrix Tapering for improved resolution and robustness
//!
//! Applies spatial tapering to the covariance matrix to reduce sidelobe levels
//! and improve robustness to model errors.
//!
//! # References
//! - Guerci (1999), "Theory and application of covariance matrix tapers for robust adaptive beamforming"
//! - Mailloux (1994), "Covariance matrix augmentation to produce adaptive array pattern troughs"

use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Covariance Matrix Tapering
#[derive(Debug)]
pub struct CovarianceTaper {
    taper_type: TaperType,
}

/// Tapering window type
#[derive(Debug, Clone, Copy)]
pub enum TaperType {
    /// Kaiser window with parameter beta
    Kaiser { beta: f64 },
    /// Blackman window
    Blackman,
    /// Hamming window
    Hamming,
    /// Adaptive - data-dependent selection
    Adaptive,
}

impl CovarianceTaper {
    /// Create Kaiser taper with shape parameter beta
    ///
    /// Typical values: beta = 2.5 to 4.0
    /// Higher beta = narrower mainlobe, higher sidelobes
    pub fn kaiser(beta: f64) -> Self {
        Self {
            taper_type: TaperType::Kaiser { beta },
        }
    }

    /// Create Blackman taper
    pub fn blackman() -> Self {
        Self {
            taper_type: TaperType::Blackman,
        }
    }

    /// Create Hamming taper
    pub fn hamming() -> Self {
        Self {
            taper_type: TaperType::Hamming,
        }
    }

    /// Create adaptive taper (data-dependent selection)
    ///
    /// Automatically selects taper based on covariance matrix condition number
    /// and eigenvalue spread
    pub fn adaptive() -> Self {
        Self {
            taper_type: TaperType::Adaptive,
        }
    }

    /// Apply tapering to covariance matrix
    ///
    /// Returns tapered covariance matrix R_tapered = T ⊙ R
    /// where ⊙ denotes element-wise (Hadamard) product
    pub fn apply(&self, covariance: &Array2<Complex64>) -> Array2<Complex64> {
        let n = covariance.nrows();

        // For adaptive tapering, select best taper based on data characteristics
        let effective_taper = match self.taper_type {
            TaperType::Adaptive => self.select_taper(covariance),
            _ => self.taper_type,
        };

        let mut tapered = covariance.clone();

        // Compute taper weights for each lag
        for i in 0..n {
            for j in 0..n {
                let lag = (i as i32 - j as i32).unsigned_abs() as usize;
                let weight = Self::compute_weight_for_type(effective_taper, lag, n);
                tapered[(i, j)] *= weight;
            }
        }

        tapered
    }

    /// Select optimal taper based on covariance matrix characteristics
    ///
    /// Uses eigenvalue spread and condition number to determine best taper:
    /// - High condition number (>100): Kaiser with high beta for robustness
    /// - Medium condition number (10-100): Blackman for balanced performance
    /// - Low condition number (<10): Hamming for minimal distortion
    fn select_taper(&self, covariance: &Array2<Complex64>) -> TaperType {
        let n = covariance.nrows();

        // Estimate condition number via diagonal elements and trace
        let mut diag_min = f64::INFINITY;
        let mut diag_max = 0.0f64;

        for i in 0..n {
            let val = covariance[(i, i)].norm();
            diag_min = diag_min.min(val);
            diag_max = diag_max.max(val);
        }

        // Rough condition number estimate
        let cond = if diag_min > 1e-12 {
            diag_max / diag_min
        } else {
            1e12
        };

        // Eigenvalue spread estimate via power iteration (quick, approximate)
        let eig_spread = self.estimate_eigenvalue_spread(covariance);

        // Decision logic based on matrix characteristics
        if cond > 100.0 || eig_spread > 100.0 {
            // Ill-conditioned: use strong Kaiser tapering
            TaperType::Kaiser { beta: 4.0 }
        } else if cond > 10.0 || eig_spread > 10.0 {
            // Moderately conditioned: use Blackman
            TaperType::Blackman
        } else {
            // Well-conditioned: use gentle Hamming
            TaperType::Hamming
        }
    }

    /// Estimate eigenvalue spread using power iteration
    fn estimate_eigenvalue_spread(&self, covariance: &Array2<Complex64>) -> f64 {
        let n = covariance.nrows();
        if n == 0 {
            return 1.0;
        }

        // Quick power iteration for largest eigenvalue
        let mut v = Array1::<Complex64>::from_elem(n, Complex64::new(1.0, 0.0));
        for _ in 0..5 {
            // Just 5 iterations for quick estimate
            let mut v_new = Array1::<Complex64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += covariance[(i, j)] * v[j];
                }
            }
            let norm = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-12 {
                v = v_new.mapv(|x| x / norm);
            }
        }

        // Compute Rayleigh quotient for largest eigenvalue
        let mut lambda_max = 0.0;
        for i in 0..n {
            for j in 0..n {
                lambda_max += (v[i].conj() * covariance[(i, j)] * v[j]).re;
            }
        }

        // Estimate smallest eigenvalue from diagonal minimum
        let lambda_min = (0..n)
            .map(|i| covariance[(i, i)].norm())
            .fold(f64::INFINITY, |a, b| a.min(b))
            .max(1e-12);

        lambda_max.abs() / lambda_min
    }

    /// Compute taper weight for given lag and array size
    #[allow(dead_code)]
    fn compute_weight(&self, lag: usize, n: usize) -> f64 {
        Self::compute_weight_for_type(self.taper_type, lag, n)
    }

    /// Static method to compute weight for a specific taper type
    fn compute_weight_for_type(taper_type: TaperType, lag: usize, n: usize) -> f64 {
        match taper_type {
            TaperType::Kaiser { beta } => {
                // Kaiser window: I_0(beta * sqrt(1 - (lag/n)^2)) / I_0(beta)
                let x = lag as f64 / n as f64;
                let arg = beta * (1.0 - x * x).max(0.0).sqrt();
                Self::bessel_i0_static(arg) / Self::bessel_i0_static(beta)
            }
            TaperType::Blackman => {
                // Blackman window
                let x = lag as f64 / (n - 1) as f64;
                0.42 - 0.5 * (std::f64::consts::PI * x).cos()
                    + 0.08 * (2.0 * std::f64::consts::PI * x).cos()
            }
            TaperType::Hamming => {
                // Hamming window
                let x = lag as f64 / (n - 1) as f64;
                0.54 - 0.46 * (std::f64::consts::PI * x).cos()
            }
            TaperType::Adaptive => {
                // This should never be called directly
                // Adaptive types are resolved in apply()
                1.0
            }
        }
    }

    /// Modified Bessel function of the first kind, order 0
    /// Using series approximation
    #[allow(dead_code)]
    fn bessel_i0(&self, x: f64) -> f64 {
        Self::bessel_i0_static(x)
    }

    /// Static version of Bessel I0
    fn bessel_i0_static(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x2 = x * x / 4.0;

        for k in 1..50 {
            term *= x2 / (k * k) as f64;
            sum += term;
            if term < 1e-12 * sum {
                break;
            }
        }
        sum
    }
}
