//! Wall Filter for Clutter Rejection
//!
//! Removes slow-moving clutter from vessel walls and tissue while preserving
//! blood flow signals. Essential for clean Doppler velocity estimation.

use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array2, Array3, ArrayView3};

/// Wall filter types
#[derive(Debug, Clone, Copy)]
pub enum WallFilterType {
    /// Simple high-pass filter (remove DC component)
    HighPass,
    /// Polynomial regression filter (Hoeks et al.)
    Polynomial { order: usize },
    /// IIR filter (infinite impulse response)
    IIR { cutoff_frequency: f64 },
}

/// Wall filter configuration
#[derive(Debug, Clone)]
pub struct WallFilterConfig {
    pub filter_type: WallFilterType,
    pub prf: f64,
}

impl Default for WallFilterConfig {
    fn default() -> Self {
        Self {
            filter_type: WallFilterType::Polynomial { order: 2 },
            prf: 4e3,
        }
    }
}

/// Wall filter for clutter rejection
#[derive(Debug, Clone)]
pub struct WallFilter {
    config: WallFilterConfig,
}

impl WallFilter {
    /// New.
    #[must_use]
    pub fn new(config: WallFilterConfig) -> Self {
        Self { config }
    }

    /// Apply wall filter to I/Q data
    ///
    /// Removes slow-moving clutter (tissue, vessel walls) while preserving
    /// blood flow signals.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply(&self, iq_data: &ArrayView3<Complex64>) -> KwaversResult<Array3<Complex64>> {
        let [ensemble_size, n_depths, n_beams] = iq_data.shape();
        let mut filtered = Array3::<Complex64>::zeros((ensemble_size, n_depths, n_beams));

        match self.config.filter_type {
            WallFilterType::HighPass => {
                // Simple DC removal: subtract mean from each ensemble
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mean = (0..ensemble_size)
                            .map(|n| iq_data[[n, depth, beam]])
                            .sum::<Complex64>()
                            / (ensemble_size as f64);

                        for n in 0..ensemble_size {
                            filtered[[n, depth, beam]] = iq_data[[n, depth, beam]] - mean;
                        }
                    }
                }
            }
            WallFilterType::Polynomial { order } => {
                // Polynomial regression filter (Hoeks et al. 1991, IEEE TUFFC 38(2)):
                // fit p(n) = a₀ + a₁·t + ... + aₖ·tᵏ to the slow-time ensemble
                // at each (depth, beam) and subtract the fitted polynomial.
                // t ∈ [0, 1] is the normalized sample index. Order 0 reduces to
                // mean subtraction; order 1 removes linear drift; order 2 removes
                // quadratic wall acceleration; etc.
                let basis = polynomial_basis(ensemble_size, order);
                let projector = polynomial_projector(&basis);
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mut signal = Vec::<Complex64>::with_capacity(ensemble_size);
                        for n in 0..ensemble_size {
                            signal.push(iq_data[[n, depth, beam]]);
                        }
                        let residual = project_out(&signal, &basis, &projector);
                        for n in 0..ensemble_size {
                            filtered[[n, depth, beam]] = residual[n];
                        }
                    }
                }
            }
            WallFilterType::IIR { cutoff_frequency } => {
                // First-order Butterworth-equivalent high-pass IIR (one-pole,
                // bilinear-transformed) applied along the slow-time axis:
                //   y[n] = α (y[n-1] + x[n] - x[n-1])
                //   α = exp(-2π fc / PRF)
                // The recursion preserves the linearity required to operate
                // independently on the real and imaginary parts of the IQ
                // signal. Initial conditions y[-1] = x[-1] = 0 (zero-state).
                let prf = self.config.prf.max(f64::EPSILON);
                let omega_c = 2.0 * std::f64::consts::PI * cutoff_frequency / prf;
                let alpha = (-omega_c).exp();
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mut y_prev = Complex64::new(0.0, 0.0);
                        let mut x_prev = Complex64::new(0.0, 0.0);
                        for n in 0..ensemble_size {
                            let x = iq_data[[n, depth, beam]];
                            let y = alpha * (y_prev + x - x_prev);
                            filtered[[n, depth, beam]] = y;
                            y_prev = y;
                            x_prev = x;
                        }
                    }
                }
            }
        }

        Ok(filtered)
    }
}

/// Build the Vandermonde basis V[n, k] = t_n^k for t_n = n/(N-1) ∈ [0, 1].
///
/// Returns an `ensemble_size × (order + 1)` row-major [`Array2`] (contiguous
/// storage indexed `basis[[n, i]]`). Used by the polynomial wall filter
/// (Hoeks 1991) to construct the polynomial model subspace independently of
/// the IQ signal.
fn polynomial_basis(ensemble_size: usize, order: usize) -> Array2<f64> {
    let k = order + 1;
    let denom = (ensemble_size.saturating_sub(1)).max(1) as f64;
    let mut basis = Array2::<f64>::zeros((ensemble_size, k));
    for n in 0..ensemble_size {
        let t = n as f64 / denom;
        let mut pow = 1.0;
        for i in 0..k {
            basis[[n, i]] = pow;
            pow *= t;
        }
    }
    basis
}

/// Inverse of the Gram matrix `(VᵀV)⁻¹` for the polynomial basis.
///
/// Solves the small `(order+1) × (order+1)` linear system once per filter
/// invocation; the result is reused across every (depth, beam) signal. Uses
/// Gaussian elimination with partial pivoting. Returns a `k × k` row-major
/// [`Array2`] (contiguous storage indexed `projector[[i, j]]`).
fn polynomial_projector(basis: &Array2<f64>) -> Array2<f64> {
    let [n, k] = basis.shape();
    if k == 0 {
        return Array2::zeros((0, 0));
    }
    // Gram matrix G = V^T V
    let mut gram = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            gram[[i, j]] = (0..n).map(|t| basis[[t, i]] * basis[[t, j]]).sum();
        }
    }
    // Augmented [G | I] for inversion via row-reduction
    let two_k = 2 * k;
    let mut aug = Array2::<f64>::zeros((k, two_k));
    for i in 0..k {
        for j in 0..k {
            aug[[i, j]] = gram[[i, j]];
        }
        aug[[i, k + i]] = 1.0;
    }
    for col in 0..k {
        // Partial pivot
        let pivot = (col..k)
            .max_by(|&a, &b| aug[[a, col]].abs().total_cmp(&aug[[b, col]].abs()))
            .unwrap_or(col);
        if pivot != col {
            for c in 0..two_k {
                let tmp = aug[[col, c]];
                aug[[col, c]] = aug[[pivot, c]];
                aug[[pivot, c]] = tmp;
            }
        }
        let diag = aug[[col, col]];
        if diag.abs() < 1e-300 {
            // Singular — return identity sentinel (filter degenerates to identity)
            return Array2::eye(k);
        }
        for c in 0..two_k {
            aug[[col, c]] /= diag;
        }
        // The pivot row (row `col`) is finalized above and untouched by the
        // `row != col` branches below, so reading it directly per column
        // needs no snapshot (unlike a nested-Vec row, a flat array's pivot
        // row is not borrowed by the outer mutation loop).
        for row in 0..k {
            if row != col {
                let factor = aug[[row, col]];
                for c in 0..two_k {
                    aug[[row, c]] -= factor * aug[[col, c]];
                }
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            inv[[i, j]] = aug[[i, k + j]];
        }
    }
    inv
}

/// Project a complex signal out of the polynomial subspace.
///
/// Returns `x - V (VᵀV)⁻¹ Vᵀ x` — the residual after subtracting the best
/// least-squares polynomial fit. The projection is linear so it operates
/// independently on the real and imaginary parts of the IQ signal.
fn project_out(
    signal: &[Complex64],
    basis: &Array2<f64>,
    projector: &Array2<f64>,
) -> Vec<Complex64> {
    let n = signal.len();
    let [k, _] = projector.shape();
    if k == 0 || n == 0 {
        return signal.to_vec();
    }
    // Vᵀ x  (k-vector of complex coefficients)
    let mut vt_x = vec![Complex64::new(0.0, 0.0); k];
    for i in 0..k {
        for t in 0..n {
            vt_x[i] += signal[t] * basis[[t, i]];
        }
    }
    // a = (VᵀV)⁻¹ Vᵀ x
    let mut coeffs = vec![Complex64::new(0.0, 0.0); k];
    for i in 0..k {
        for j in 0..k {
            coeffs[i] += vt_x[j] * projector[[i, j]];
        }
    }
    // residual = x - V a
    let mut residual = Vec::with_capacity(n);
    for t in 0..n {
        let mut fit = Complex64::new(0.0, 0.0);
        for i in 0..k {
            fit += coeffs[i] * basis[[t, i]];
        }
        residual.push(signal[t] - fit);
    }
    residual
}

#[cfg(test)]
mod tests;
