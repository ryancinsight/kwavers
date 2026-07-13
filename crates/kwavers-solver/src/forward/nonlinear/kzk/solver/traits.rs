//! Bridge from the physics-layer `KZKSolverTrait` contract to the concrete
//! [`super::KZKSolver`] struct.
//!
//! ## API mapping
//!
//! | Trait method          | Solver implementation                              |
//! |-----------------------|----------------------------------------------------|
//! | `step_forward(dz)`    | Sets `config.dz = dz`, then calls `self.step()`    |
//! | `current_field()`     | RMS pressure over retarded time at current z-plane |
//! | `peak_pressure()`     | Delegates to `self.get_peak_pressure()`            |
//!
//! ## `current_field` semantics
//!
//! The solver stores `p(x, y, τ)` as a 3D array.  The trait returns
//! a 2D `Array2<f64>` defined as the RMS pressure over τ:
//!
//! ```text
//! p_rms(i, j) = √( (1/nt) Σ_t p[i,j,t]² )      (Pa)
//! ```
//!
//! This is the most physically relevant single-slice summary for HIFU
//! intensity calculations, where I ∝ p_rms².

use leto::Array2;
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::KZKSolver;
use kwavers_physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolverTrait;

impl KZKSolverTrait for KZKSolver {
    /// Advance the pressure field by axial increment `dz` (m).
    ///
    /// Overrides `config.dz` for this step, then applies the full Strang-split
    /// `D(dz/2)·A(dz/2)·N(dz)·A(dz/2)·D(dz/2)` sequence.
    fn step_forward(&mut self, dz: f64) {
        self.config.dz = dz;
        self.step();
    }

    /// Return the RMS pressure field (Pa) at the current axial z-plane.
    ///
    /// Shape: `(nx, ny)` — transverse grid.
    ///
    /// ## Theorem (RMS as L² norm)
    ///
    /// `p_rms(i,j) = ‖Re[p[i,j,·]]‖_{L²} / √nt`.
    ///
    /// This is proportional to the time-averaged acoustic intensity:
    /// `I(i,j) = p_rms(i,j)² / (ρ₀c₀)` [W/m²].
    ///
    /// ## Theorem (race-freedom)
    ///
    /// Each output element `rms[i,j]` is computed as a sequential reduction
    /// over `self.pressure[[i,j,0..nt]]`, a disjoint slice from every other
    /// transverse cell.  No two Moirai workers share mutable output memory.
    fn current_field(&self) -> Array2<f64> {
        let nt_f64 = self.config.nt as f64;
        let ny = self.config.ny;
        let nt = self.config.nt;
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: KZK pressure is standard-layout");
        let mut rms = Array2::zeros((self.config.nx, self.config.ny));
        let rms_slice = rms
            .as_slice_mut()
            .expect("invariant: KZK RMS output is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(rms_slice, |idx, r| {
            let i = idx / ny;
            let j = idx % ny;
            let base = (i * ny + j) * nt;
            let sum_sq: f64 = (0..nt).map(|t| pressure[base + t].re.powi(2)).sum();
            *r = (sum_sq / nt_f64).sqrt();
        });
        rms
    }

    /// Return the peak positive pressure field (Pa) at the current z-plane.
    fn peak_pressure(&self) -> Array2<f64> {
        self.get_peak_pressure()
    }
}
