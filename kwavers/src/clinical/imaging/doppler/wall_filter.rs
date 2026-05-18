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
            WallFilterType::Polynomial { order: _ } => {
                // Polynomial regression filter (to be implemented)
                // For now, use simple high-pass
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
            WallFilterType::IIR { .. } => {
                // IIR filter (to be implemented)
                // For now, use simple high-pass
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
        }

        Ok(filtered)
    }
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

    /// Polynomial variant (currently delegates to HighPass) zeroes a constant ensemble.
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
                v.norm() < 1e-12,
                "Polynomial filter on constant ensemble: expected 0+0i, got {v}"
            );
        }
    }

    /// IIR variant (currently delegates to HighPass) zeroes a constant ensemble.
    #[test]
    fn wall_filter_iir_constant_ensemble_outputs_zero() {
        let cfg = WallFilterConfig {
            filter_type: WallFilterType::IIR {
                cutoff_frequency: 100.0,
            },
            prf: 4e3,
        };
        let wf = WallFilter::new(cfg);
        let c = Complex64::new(2.0, -1.0);
        let iq = Array3::from_elem((8, 4, 2), c);
        let out = wf.apply(&iq.view()).unwrap();
        for v in out.iter() {
            assert!(
                v.norm() < 1e-12,
                "IIR filter on constant ensemble: expected 0+0i, got {v}"
            );
        }
    }
}
