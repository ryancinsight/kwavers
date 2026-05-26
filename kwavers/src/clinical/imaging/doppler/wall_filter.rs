//! Wall Filter for Clutter Rejection
//!
//! Removes slow-moving clutter from vessel walls and tissue while preserving
//! blood flow signals. Essential for clean Doppler velocity estimation.

use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3};
use num_complex::Complex64;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
        let (ensemble_size, n_depths, n_beams) = iq_data.dim();
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
/// Returns a `Vec<Vec<f64>>` with `ensemble_size` rows and `order + 1` columns.
/// Used by the polynomial wall filter (Hoeks 1991) to construct the polynomial
/// model subspace independently of the IQ signal.
fn polynomial_basis(ensemble_size: usize, order: usize) -> Vec<Vec<f64>> {
    let k = order + 1;
    let denom = (ensemble_size.saturating_sub(1)).max(1) as f64;
    (0..ensemble_size)
        .map(|n| {
            let t = n as f64 / denom;
            let mut row = Vec::with_capacity(k);
            let mut pow = 1.0;
            for _ in 0..k {
                row.push(pow);
                pow *= t;
            }
            row
        })
        .collect()
}

/// Inverse of the Gram matrix `(VᵀV)⁻¹` for the polynomial basis.
///
/// Solves the small `(order+1) × (order+1)` linear system once per filter
/// invocation; the result is reused across every (depth, beam) signal. Uses
/// Gaussian elimination with partial pivoting.
fn polynomial_projector(basis: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = basis.len();
    let k = basis.first().map_or(0, Vec::len);
    if k == 0 {
        return Vec::new();
    }
    // Gram matrix G = V^T V
    let mut gram = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            gram[i][j] = (0..n).map(|t| basis[t][i] * basis[t][j]).sum();
        }
    }
    // Augmented [G | I] for inversion via row-reduction
    let mut aug = vec![vec![0.0_f64; 2 * k]; k];
    for i in 0..k {
        for j in 0..k {
            aug[i][j] = gram[i][j];
        }
        aug[i][k + i] = 1.0;
    }
    for col in 0..k {
        // Partial pivot
        let pivot = (col..k)
            .max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs()))
            .unwrap_or(col);
        aug.swap(col, pivot);
        let diag = aug[col][col];
        if diag.abs() < 1e-300 {
            // Singular — return identity sentinel (filter degenerates to identity)
            let mut id = vec![vec![0.0_f64; k]; k];
            for i in 0..k {
                id[i][i] = 1.0;
            }
            return id;
        }
        for j in 0..2 * k {
            aug[col][j] /= diag;
        }
        for row in 0..k {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * k {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            inv[i][j] = aug[i][k + j];
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
    basis: &[Vec<f64>],
    projector: &[Vec<f64>],
) -> Vec<Complex64> {
    let n = signal.len();
    let k = projector.len();
    if k == 0 || n == 0 {
        return signal.to_vec();
    }
    // Vᵀ x  (k-vector of complex coefficients)
    let mut vt_x = vec![Complex64::new(0.0, 0.0); k];
    for i in 0..k {
        for t in 0..n {
            vt_x[i] += signal[t] * basis[t][i];
        }
    }
    // a = (VᵀV)⁻¹ Vᵀ x
    let mut coeffs = vec![Complex64::new(0.0, 0.0); k];
    for i in 0..k {
        for j in 0..k {
            coeffs[i] += vt_x[j] * projector[i][j];
        }
    }
    // residual = x - V a
    let mut residual = Vec::with_capacity(n);
    for t in 0..n {
        let mut fit = Complex64::new(0.0, 0.0);
        for i in 0..k {
            fit += coeffs[i] * basis[t][i];
        }
        residual.push(signal[t] - fit);
    }
    residual
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex64;

    // ─── HighPass: exact mathematical properties ─────────────────────────────

    /// Constant ensemble after HighPass is identically zero.
    ///
    /// For ensemble [c, c, …, c] (N copies):
    ///   mean = c
    ///   filtered[n] = c − c = 0 for every n.
    #[test]
    fn wall_filter_highpass_constant_ensemble_outputs_zero() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::HighPass,
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let c = Complex64::new(3.5, -2.1);
        // shape: (ensemble=4, depths=3, beams=2)
        let iq = Array3::from_elem((4, 3, 2), c);
        let out = wf.apply(&iq.view()).unwrap();
        for v in out.iter() {
            assert!(
                v.norm() < 1e-12,
                "HighPass on constant ensemble: expected 0+0i, got {v}"
            );
        }
    }

    /// Alternating ensemble [+A, −A, +A, −A] has mean = 0, so HighPass preserves it exactly.
    ///
    /// mean = (A − A + A − A) / 4 = 0
    /// filtered[n] = s[n] − 0 = s[n]
    #[test]
    fn wall_filter_highpass_zero_mean_ensemble_is_unchanged() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::HighPass,
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let a = Complex64::new(1.0, 0.5);
        // (ensemble=4, depths=2, beams=2): alternating +a / −a
        let mut iq = Array3::zeros((4, 2, 2));
        for depth in 0..2 {
            for beam in 0..2 {
                iq[[0, depth, beam]] = a;
                iq[[1, depth, beam]] = -a;
                iq[[2, depth, beam]] = a;
                iq[[3, depth, beam]] = -a;
            }
        }
        let out = wf.apply(&iq.view()).unwrap();
        for (in_val, out_val) in iq.iter().zip(out.iter()) {
            assert!(
                (in_val - out_val).norm() < 1e-12,
                "HighPass on zero-mean ensemble: expected {in_val}, got {out_val}"
            );
        }
    }

    /// After HighPass the ensemble sum at every (depth, beam) is zero.
    ///
    /// Algebraic identity: Σ(xₙ − mean) = Σxₙ − N · mean = 0.
    /// Holds for any input, including non-uniform complex values.
    #[test]
    fn wall_filter_highpass_ensemble_sum_is_zero_for_arbitrary_input() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::HighPass,
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let ensemble_size = 6;
        let n_depths = 3;
        let n_beams = 2;
        let mut iq = Array3::zeros((ensemble_size, n_depths, n_beams));
        // Non-uniform values to ensure a nontrivial mean.
        for n in 0..ensemble_size {
            for d in 0..n_depths {
                for b in 0..n_beams {
                    iq[[n, d, b]] = Complex64::new((n + d + b) as f64, (n * 2) as f64);
                }
            }
        }
        let out = wf.apply(&iq.view()).unwrap();
        for depth in 0..n_depths {
            for beam in 0..n_beams {
                let sum: Complex64 = (0..ensemble_size).map(|n| out[[n, depth, beam]]).sum();
                assert!(
                    sum.norm() < 1e-10,
                    "ensemble sum at ({depth},{beam}) = {sum:.2e}, expected 0"
                );
            }
        }
    }

    /// Polynomial order-2 filter zeroes a constant ensemble.
    ///
    /// The constant signal lies in the polynomial subspace span{1, t, t²},
    /// so the residual after orthogonal projection is exactly zero.
    #[test]
    fn wall_filter_polynomial_constant_ensemble_outputs_zero() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::Polynomial { order: 2 },
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let c = Complex64::new(-7.0, 4.2);
        let iq = Array3::from_elem((5, 2, 3), c);
        let out = wf.apply(&iq.view()).unwrap();
        for v in out.iter() {
            assert!(
                v.norm() < 1e-10,
                "Polynomial filter on constant ensemble: expected 0+0i, got {v}"
            );
        }
    }

    /// Polynomial order-1 filter zeroes a linear ramp ensemble.
    ///
    /// A linear signal x[n] = a + b·t lies in span{1, t}, so order-1 polynomial
    /// regression removes it exactly. This validates that the polynomial filter
    /// actually uses the `order` parameter rather than reducing to DC removal.
    #[test]
    fn wall_filter_polynomial_linear_ramp_outputs_zero() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::Polynomial { order: 1 },
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let ensemble = 6;
        let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
        for n in 0..ensemble {
            // x[n] = (2.0 + 3.0·n) + i·(−1.0 + 0.5·n)
            iq[[n, 0, 0]] =
                Complex64::new(2.0 + 3.0 * n as f64, -1.0 + 0.5 * n as f64);
        }
        let out = wf.apply(&iq.view()).unwrap();
        for v in out.iter() {
            assert!(
                v.norm() < 1e-10,
                "Polynomial order-1 on linear ramp: expected 0+0i, got {v}"
            );
        }
    }

    /// Polynomial order-2 filter zeroes a quadratic ensemble.
    ///
    /// A quadratic signal x[n] = a + b·t + c·t² lies in span{1, t, t²}, so
    /// order-2 polynomial regression removes it exactly. Validates that the
    /// projector handles each order correctly.
    #[test]
    fn wall_filter_polynomial_quadratic_outputs_zero() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::Polynomial { order: 2 },
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let ensemble = 8;
        let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
        for n in 0..ensemble {
            let nf = n as f64;
            iq[[n, 0, 0]] = Complex64::new(
                1.0 + 2.0 * nf + 0.5 * nf * nf,
                -2.0 - nf + 0.25 * nf * nf,
            );
        }
        let out = wf.apply(&iq.view()).unwrap();
        for v in out.iter() {
            assert!(
                v.norm() < 1e-10,
                "Polynomial order-2 on quadratic: expected 0+0i, got {v}"
            );
        }
    }

    /// Polynomial order-1 filter does NOT zero a quadratic ensemble.
    ///
    /// A quadratic signal is not in span{1, t}; the residual after order-1
    /// regression must be non-zero. This validates that the polynomial filter
    /// is genuinely order-dependent (not collapsing to a higher-order projection).
    #[test]
    fn wall_filter_polynomial_order1_leaves_quadratic_residual() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::Polynomial { order: 1 },
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let ensemble = 8;
        let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
        for n in 0..ensemble {
            let nf = n as f64;
            iq[[n, 0, 0]] = Complex64::new(nf * nf, 0.0);
        }
        let out = wf.apply(&iq.view()).unwrap();
        let total_energy: f64 = out.iter().map(|v| v.norm_sqr()).sum();
        assert!(
            total_energy > 1e-3,
            "Order-1 on quadratic should leave residual energy; got {total_energy}"
        );
    }

    /// IIR high-pass: constant DC input produces a transient that decays to zero.
    ///
    /// For x[n] = c and one-pole HPF y[n] = α(y[n-1] + x[n] - x[n-1]) with
    /// y[-1] = x[-1] = 0:
    ///   y[0] = α·c
    ///   y[1] = α²·c
    ///   y[n] = α^(n+1)·c
    /// The steady-state response to DC is zero, but the transient is non-zero
    /// — this is the correct high-pass behavior (Oppenheim & Schafer §8.3).
    #[test]
    fn wall_filter_iir_dc_input_decays_geometrically() {
        let prf = 4.0e3_f64;
        let cutoff = 100.0_f64;
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::IIR {
                cutoff_frequency: cutoff,
            },
            prf,
        };
        let wf = WallFilter::new(cfg);
        let c = Complex64::new(2.0, -1.0);
        let ensemble = 8;
        let iq = Array3::from_elem((ensemble, 1, 1), c);
        let out = wf.apply(&iq.view()).unwrap();

        let alpha = (-2.0 * std::f64::consts::PI * cutoff / prf).exp();
        for n in 0..ensemble {
            let expected = alpha.powi((n as i32) + 1) * c;
            let actual = out[[n, 0, 0]];
            assert!(
                (actual - expected).norm() < 1e-12,
                "IIR DC transient sample {n}: expected {expected}, got {actual}"
            );
        }
    }

    /// IIR high-pass: alternating Nyquist-frequency input passes through with gain.
    ///
    /// For x[n] = (-1)^n · c, the difference x[n] − x[n-1] alternates with
    /// magnitude 2|c|, producing a response near the HPF passband.
    #[test]
    fn wall_filter_iir_alternating_input_is_passed() {
        let prf = 4.0e3_f64;
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::IIR {
                cutoff_frequency: 100.0,
            },
            prf,
        };
        let wf = WallFilter::new(cfg);
        let a = Complex64::new(1.0, 0.0);
        let ensemble = 16;
        let mut iq = Array3::<Complex64>::zeros((ensemble, 1, 1));
        for n in 0..ensemble {
            iq[[n, 0, 0]] = if n.is_multiple_of(2) { a } else { -a };
        }
        let out = wf.apply(&iq.view()).unwrap();
        let dc_input_energy: f64 = iq.iter().map(|v| v.norm_sqr()).sum();
        let out_energy: f64 = out.iter().map(|v| v.norm_sqr()).sum();
        // For a Nyquist-frequency input through a HPF the steady-state gain
        // is large (>0.5 of input energy). The transient samples may be even
        // larger because of the leading edge.
        assert!(
            out_energy > 0.5 * dc_input_energy,
            "IIR should pass Nyquist input: in={dc_input_energy:.3}, out={out_energy:.3}"
        );
    }
}
