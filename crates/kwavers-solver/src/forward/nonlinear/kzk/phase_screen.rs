//! Phase-screen correction for inhomogeneous media in the KZK solver.
//!
//! ## Inhomogeneous KZK model
//!
//! At each axial step z_n, a multiplicative phase correction accounts for
//! the local sound-speed variation c(x,y,z_n) ≠ c₀:
//!
//! ```text
//! p(x,y,τ) *= exp(i k₀ δ(x,y,z_n) Δz)
//! δ(x,y,z) = c₀/c(x,y,z) − 1       (relative speed perturbation)
//! k₀ = 2π f₀/c₀
//! ```
//!
//! For homogeneous media δ = 0 and the correction is identity.
//! For soft tissue, |δ| < 0.05 (within 5% of reference speed).
//!
//! ## Operator splitting placement
//!
//! Applied once per full z-step (not per half-step) after the nonlinear
//! sub-step: D(Δz/2)·A(Δz/2)·N(Δz)·A(Δz/2)·D(Δz/2)·**P(Δz)**.
//!
//! ## References
//!
//! - Pinton GF, Dahl J, Rosenzweig S, Trahey GE (2009). "A heterogeneous
//!   nonlinear attenuating full-wave model of ultrasound." IEEE Trans. UFFC
//!   56(3), 474–488. DOI: 10.1109/TUFFC.2009.1066
//! - Zemp RJ, Tavakkoli J, Cobbold RS (2003). J. Acoust. Soc. Am. 113(1), 139–152.

use kwavers_math::fft::Complex64;
use leto::{Array2, Array3, ArrayView2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

/// Phase-screen corrector for axial sound-speed perturbations.
#[derive(Debug)]
pub struct PhaseScreenOperator {
    k0: f64,
    dz: f64,
    /// Pre-computed phase correction exp(i k0 delta dz) for current z-step.
    /// Updated by calling update(z_step).
    phase_correction: Array2<Complex64>,
    nx: usize,
    ny: usize,
}

impl PhaseScreenOperator {
    /// Create a phase-screen operator initialised to the identity screen.
    #[must_use]
    pub fn new(k0: f64, dz: f64, nx: usize, ny: usize) -> Self {
        Self {
            k0,
            dz,
            phase_correction: Array2::from_elem((nx, ny), Complex64::new(1.0, 0.0)),
            nx,
            ny,
        }
    }

    /// Update the stored axial step size.
    pub fn set_step_size(&mut self, dz: f64) {
        self.dz = dz;
    }

    /// Precompute the phase screen for the supplied z-plane.
    pub fn update(&mut self, delta_slice: ArrayView2<f64>) {
        debug_assert_eq!(delta_slice.shape(), [self.nx, self.ny]);
        let screen = self
            .phase_correction
            .as_slice_mut()
            .expect("invariant: phase-screen correction is standard-layout");
        let delta = delta_slice
            .as_slice()
            .expect("invariant: phase-screen input slice is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(screen, |idx, value| {
            let phase = self.k0 * delta[idx] * self.dz;
            *value = Complex64::from_polar(1.0, phase);
        });
    }

    /// Apply the current phase screen to every retarded-time sample.
    pub fn apply(&self, field: &mut Array3<Complex64>) {
        let nt = field.shape()[2];
        let screen = self
            .phase_correction
            .as_slice()
            .expect("invariant: phase-screen correction is standard-layout");
        let field_values = field
            .as_slice_mut()
            .expect("invariant: phase-screen field is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(field_values, |idx, value| {
            *value *= screen[idx / nt];
        });
    }
}
