//! Real-valued physical observables derived from the internal complex
//! pressure field: full pressure volume, on-axis time signal, time-averaged
//! intensity, and peak positive pressure.

use leto::{Array2, Array3};
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::KZKSolver;

impl KZKSolver {
    /// Return the physical pressure field Re[p(x,y,τ)] as a real array.
    ///
    /// Extracts the real part of the internal complex-field representation.
    /// The imaginary part (diffraction phase accumulator) is discarded.
    #[must_use]
    pub fn get_pressure(&self) -> Array3<f64> {
        self.pressure.mapv(|c| c.re)
    }

    /// Return the physical pressure waveform p(τ) (Pa) at transverse point (x, y).
    ///
    /// Returns `Re[pressure[x, y, 0..nt]]`.
    #[must_use]
    pub fn get_time_signal(&self, x: usize, y: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(self.config.nt);
        for t in 0..self.config.nt {
            signal.push(self.pressure[[x, y, t]].re);
        }
        signal
    }

    /// Calculate time-averaged acoustic intensity I = p²_rms / (ρ₀c₀) (W/m²).
    ///
    /// Uses the physical (real) pressure: `I(i,j) = ⟨Re[p]²⟩_τ / (ρ₀c₀)`.
    ///
    /// ## Theorem (race-freedom)
    ///
    /// Output element `intensity[i,j]` is a sequential reduction over the
    /// disjoint slice `self.pressure[[i,j,0..nt]]`.  No two Moirai workers
    /// share mutable output memory.
    #[must_use]
    pub fn get_intensity(&self) -> Array2<f64> {
        let factor = 1.0 / (self.config.rho0 * self.config.c0 * self.config.nt as f64);
        let ny = self.config.ny;
        let nt = self.config.nt;
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: KZK pressure is standard-layout");
        let mut intensity = Array2::zeros((self.config.nx, self.config.ny));
        let intensity_slice = intensity
            .as_slice_mut()
            .expect("invariant: KZK intensity output is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(intensity_slice, |idx, out| {
            let i = idx / ny;
            let j = idx % ny;
            let base = (i * ny + j) * nt;
            let sum: f64 = (0..nt)
                .map(|t| {
                    let p = pressure[base + t].re;
                    p * p
                })
                .sum();
            *out = sum * factor;
        });
        intensity
    }

    /// Calculate peak positive pressure field max_τ |Re[p(x,y,τ)]| (Pa).
    ///
    /// ## Theorem (race-freedom)
    ///
    /// Output element `peak[i,j]` is computed from the disjoint slice
    /// `self.pressure[[i,j,0..nt]]`.  No two Moirai workers share mutable
    /// output memory.
    #[must_use]
    pub fn get_peak_pressure(&self) -> Array2<f64> {
        let ny = self.config.ny;
        let nt = self.config.nt;
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: KZK pressure is standard-layout");
        let mut peak = Array2::zeros((self.config.nx, self.config.ny));
        let peak_slice = peak
            .as_slice_mut()
            .expect("invariant: KZK peak-pressure output is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(peak_slice, |idx, out| {
            let i = idx / ny;
            let j = idx % ny;
            let base = (i * ny + j) * nt;
            *out = (0..nt)
                .map(|t| pressure[base + t].re.abs())
                .fold(0.0_f64, f64::max);
        });
        peak
    }
}
