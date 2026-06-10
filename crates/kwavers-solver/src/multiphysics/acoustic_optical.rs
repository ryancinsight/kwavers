//! Acoustic-optical coupling module
//!
//! This module provides specialized coupling between acoustic and optical fields.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::Array3;

/// Acoustic-optical solver for coupled simulations
#[derive(Debug)]
pub struct AcousticOpticalSolver {
    /// Photoelastic coefficient
    photoelastic_coefficient: f64,
    /// Grid reference
    _grid: Grid,
}

impl AcousticOpticalSolver {
    /// Create a new acoustic-optical solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: Grid, photoelastic_coefficient: f64) -> Self {
        Self {
            photoelastic_coefficient,
            _grid: grid,
        }
    }

    /// Couple acoustic pressure to optical intensity
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn couple_fields(
        &self,
        pressure: &Array3<f64>,
        intensity: &mut Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Photoelastic effect: pressure changes refractive index
        // which affects optical intensity
        for ((i, j, k), &p) in pressure.indexed_iter() {
            let delta_n = self.photoelastic_coefficient * p;
            let modulation = delta_n.mul_add(dt, 1.0);
            intensity[[i, j, k]] *= modulation;
        }

        Ok(())
    }

    /// Peak photoelastic refractive-index modulation `Δn = p_e · p_peak`.
    #[must_use]
    pub fn index_modulation(&self, peak_pressure_pa: f64) -> f64 {
        self.photoelastic_coefficient * peak_pressure_pa
    }

    /// Acousto-optic **diffraction-order intensities** for light of vacuum
    /// wavelength `optical_wavelength_m` crossing a sound column of width
    /// `interaction_length_m` carrying a wave of peak pressure
    /// `peak_pressure_pa` and acoustic wavelength `acoustic_wavelength_m`, in a
    /// medium of index `refractive_index`.
    ///
    /// Delegates to the complete Klein–Cook coupled-wave model
    /// ([`kwavers_physics::analytical::acousto_optics::solve_coupled_orders`]),
    /// which automatically spans the Raman–Nath (thin-grating) and Bragg
    /// (thick-grating) regimes. `incidence_alpha` is the normalised incidence
    /// parameter (`0` normal, `−½` Bragg). Returns `|Eₗ|²` for
    /// `l = −max_order ..= max_order` (index `l + max_order`).
    #[must_use]
    #[allow(clippy::too_many_arguments)] // physical parameters of the AO interaction; bundling would obscure the optics
    pub fn diffraction_orders(
        &self,
        peak_pressure_pa: f64,
        interaction_length_m: f64,
        optical_wavelength_m: f64,
        refractive_index: f64,
        acoustic_wavelength_m: f64,
        incidence_alpha: f64,
        max_order: u32,
        n_steps: usize,
    ) -> Vec<f64> {
        use kwavers_physics::analytical::acousto_optics::{
            klein_cook_parameter, raman_nath_parameter, solve_coupled_orders,
        };
        let delta_n = self.index_modulation(peak_pressure_pa);
        let nu = raman_nath_parameter(delta_n, interaction_length_m, optical_wavelength_m);
        let q = klein_cook_parameter(
            optical_wavelength_m,
            interaction_length_m,
            refractive_index,
            acoustic_wavelength_m,
        );
        solve_coupled_orders(nu, q, incidence_alpha, max_order, n_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver() -> AcousticOpticalSolver {
        // Photoelastic coefficient chosen so a 1 MPa wave gives a small Δn.
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid");
        AcousticOpticalSolver::new(grid, 1.0e-10) // Δn = 1e-10 · p
    }

    #[test]
    fn index_modulation_is_photoelastic_product() {
        let s = solver();
        assert!((s.index_modulation(1.0e6) - 1.0e-4).abs() < 1e-18);
    }

    /// The solver's diffraction computation conserves energy and, in the
    /// thin-grating (Raman–Nath) regime, spreads light symmetrically about the
    /// undeviated order.
    #[test]
    fn diffraction_orders_conserve_energy_raman_nath() {
        let s = solver();
        let max = 10u32;
        // λ₀=633nm, L=1mm, Λ=300µm, n=1.33 → Q≈0.01 (thin grating); a weak
        // 0.1 MPa wave gives ν≈0.1 (Δn=1e-5), so most light stays undeviated.
        let orders = s.diffraction_orders(1.0e5, 1e-3, 633e-9, 1.33, 300e-6, 0.0, max, 2000);
        assert_eq!(orders.len(), 2 * max as usize + 1);
        let total: f64 = orders.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "energy = {total}");
        // Symmetric about the zeroth order (normal incidence, thin grating).
        let c = max as usize;
        assert!((orders[c + 1] - orders[c - 1]).abs() < 1e-6, "±1 orders symmetric");
        // In the thin-grating limit the solver matches the analytical Raman–Nath
        // result Jₘ²(ν) (cross-check of the coupled solver against the closed form).
        let expected = kwavers_physics::analytical::acousto_optics::raman_nath_order_intensities(
            kwavers_physics::analytical::acousto_optics::raman_nath_parameter(
                s.index_modulation(1.0e5),
                1e-3,
                633e-9,
            ),
            max,
        );
        for (got, want) in orders.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-3, "{got} vs Raman–Nath {want}");
        }
        assert!(orders[c] > 0.95, "weak grating: zeroth dominates ({})", orders[c]);
    }
}
