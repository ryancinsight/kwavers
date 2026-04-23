//! Time-reversal aberration correction
//!
//! ## Mathematical Foundation
//!
//! ### Wave Equation Time-Reversal Symmetry
//!
//! The lossless wave equation `∂²u/∂t² − c²∇²u = 0` is invariant under
//! `t → −t`: if `u(r, t)` is a solution, so is `u(r, −t)`.
//!
//! **Consequence** (Fink 1992): A wave that has been aberrated by propagating
//! through an inhomogeneous medium (e.g. skull) can be perfectly refocused at
//! its source by time-reversing the recorded signals at an aperture and
//! retransmitting them.
//!
//! ### CW Phase Conjugation (single-frequency)
//!
//! For a continuous-wave field at frequency `f`, time-reversal reduces to
//! phase conjugation.  If the complex pressure at transducer element `i` is
//! ```text
//!   p_i = |p_i| · e^{i φ_i}
//! ```
//! then the time-reversed (corrective) transmission is proportional to
//! ```text
//!   p_i* = |p_i| · e^{−i φ_i}
//! ```
//! which exactly cancels the accumulated phase aberration `φ_i`.
//!
//! ### Quality Metric
//!
//! After phase conjugation the residual phases are all zero by construction,
//! so the circular coherence of the corrected phases equals 1.  The quality
//! metric therefore quantifies how well the sampled field was resolved from
//! the grid:
//! ```text
//!   Q = circular_coherence(−arg(p_i) + correction_i)
//! ```
//! For ideal, fully resolved fields Q = 1.  Aliasing or near-zero field
//! magnitudes reduce Q below 1.
//!
//! ## Algorithm
//!
//! 1. For each element position `r_i`, sample the complex measured field via
//!    trilinear interpolation → `p_i`.
//! 2. Correction phase: `φ_corr,i = −arg(p_i)`.
//! 3. Correction amplitude: uniform 1.0 (element amplitude modulation is a
//!    separate transmit-beam-former concern).
//! 4. `focal_gain_db`: use the circular coherence of uncorrected phases `arg(p_i)`.
//! 5. `quality_metric`: circular coherence of `(arg(p_i) + φ_corr,i)` = coherence of zeros = 1
//!    when sampling is exact; deviates from 1 for poorly resolved positions.
//!
//! ## References
//!
//! - Fink M (1992). "Time reversal of ultrasonic fields — Part I: Basic
//!   principles." IEEE Trans. UFFC 39(5):555–566. DOI:10.1109/58.156174
//! - Aubry J-F et al. (2003). "Experimental demonstration of noninvasive
//!   transskull adaptive focusing based on prior CT scans."
//!   J. Acoust. Soc. Am. 113(1):84–93. DOI:10.1121/1.1529663
//! - Thomas J-L, Fink M (1996). "Ultrasonic beam focusing through tissue
//!   inhomogeneities with a time reversal mirror: application to transskull
//!   therapy." IEEE Trans. UFFC 43(6):1122–1129.

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use crate::core::error::KwaversResult;
use log::info;
use ndarray::Array3;
use num_complex::Complex;

impl TranscranialAberrationCorrection {
    /// Apply CW time-reversal (phase conjugation) aberration correction.
    ///
    /// ## Parameters
    /// - `measured_field`: The complex pressure field recorded from a pilot
    ///   source placed at (or simulated at) the target.  This is the forward-
    ///   propagated field from the target through the skull to the transducer
    ///   plane.  Dimensions must match `self.grid`.
    /// - `transducer_positions`: Physical coordinates [x, y, z] (metres) of
    ///   each transducer element.
    ///
    /// ## Returns
    /// Phase conjugate correction `phases[i] = −arg(p(r_i))` for each element.
    ///
    /// ## Panics / Errors
    /// Returns `Err` if the `measured_field` dimensions do not match `self.grid`.
    pub fn apply_time_reversal_correction(
        &self,
        measured_field: &Array3<Complex<f64>>,
        transducer_positions: &[[f64; 3]],
    ) -> KwaversResult<PhaseCorrection> {
        info!(
            "Applying time-reversal aberration correction for {} elements",
            transducer_positions.len()
        );

        let n = transducer_positions.len();
        let mut correction_phases = Vec::with_capacity(n);
        let mut forward_phases = Vec::with_capacity(n); // arg(p_i) before correction

        for &pos in transducer_positions {
            // Convert physical position to fractional grid index.
            let (mx, my, mz) = measured_field.dim();
            let xi = (pos[0] / self.grid.dx).clamp(0.0, mx.saturating_sub(2) as f64);
            let yj = (pos[1] / self.grid.dy).clamp(0.0, my.saturating_sub(2) as f64);
            let zk = (pos[2] / self.grid.dz).clamp(0.0, mz.saturating_sub(2) as f64);

            // Sample complex field at element position via trilinear interpolation.
            let p = Self::trilinear_interpolate_complex(measured_field, xi, yj, zk);

            let phi_fwd = p.arg(); // forward aberration phase φ_i
            forward_phases.push(phi_fwd);
            correction_phases.push(-phi_fwd); // phase conjugation: −φ_i
        }

        // Uniform amplitude weights; tapering is a transmit-array concern.
        let amplitudes = vec![1.0_f64; n];

        // Focal gain improvement from the pre-correction aberration.
        let focal_gain_db =
            TranscranialAberrationCorrection::focal_gain_improvement_db(&forward_phases);

        // Quality: coherence of (forward_phase + correction_phase) = coherence of zeros.
        // Deviates from 1 only when near-zero-magnitude field samples introduce arg() noise.
        let residual_phases: Vec<f64> = forward_phases
            .iter()
            .zip(correction_phases.iter())
            .map(|(phi, corr)| phi + corr)
            .collect();
        let quality_metric = TranscranialAberrationCorrection::circular_coherence(&residual_phases);

        Ok(PhaseCorrection {
            phases: correction_phases,
            amplitudes,
            focal_gain_db,
            quality_metric,
        })
    }

    /// Trilinear interpolation of a complex field at fractional grid position (xi, yj, zk).
    ///
    /// ## Precondition
    /// `0 ≤ xi ≤ nx−2`, `0 ≤ yj ≤ ny−2`, `0 ≤ zk ≤ nz−2` (caller must clamp).
    ///
    /// ## Algorithm
    /// Independently interpolates real and imaginary parts using the standard
    /// trilinear formula (Press et al. 2007, §3.6).
    ///
    /// **References**: Press WH et al. (2007). *Numerical Recipes* 3rd ed. §3.6.
    pub(crate) fn trilinear_interpolate_complex(
        field: &Array3<Complex<f64>>,
        xi: f64,
        yj: f64,
        zk: f64,
    ) -> Complex<f64> {
        let i0 = xi.floor() as usize;
        let j0 = yj.floor() as usize;
        let k0 = zk.floor() as usize;
        let tx = xi - xi.floor();
        let ty = yj - yj.floor();
        let tz = zk - zk.floor();

        let f000 = field[[i0, j0, k0]];
        let f100 = field[[i0 + 1, j0, k0]];
        let f010 = field[[i0, j0 + 1, k0]];
        let f001 = field[[i0, j0, k0 + 1]];
        let f110 = field[[i0 + 1, j0 + 1, k0]];
        let f101 = field[[i0 + 1, j0, k0 + 1]];
        let f011 = field[[i0, j0 + 1, k0 + 1]];
        let f111 = field[[i0 + 1, j0 + 1, k0 + 1]];

        let fx00 = f000 * (1.0 - tx) + f100 * tx;
        let fx10 = f010 * (1.0 - tx) + f110 * tx;
        let fx01 = f001 * (1.0 - tx) + f101 * tx;
        let fx11 = f011 * (1.0 - tx) + f111 * tx;

        let fxy0 = fx00 * (1.0 - ty) + fx10 * ty;
        let fxy1 = fx01 * (1.0 - ty) + fx11 * ty;

        fxy0 * (1.0 - tz) + fxy1 * tz
    }
}
