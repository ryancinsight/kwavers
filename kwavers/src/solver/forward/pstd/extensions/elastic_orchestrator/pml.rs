//! Real-space exponential PML for the elastic PSTD orchestrator.
//!
//! # Theorem (PML attenuation)
//!
//! Let `σ_α(x_α) ≥ 0` be the per-axis damping profile (zero in the
//! interior, monotonically increasing through a thickness `L_α` at the
//! boundary). The PML modification of the elastic equations
//!
//! ```text
//!   ∂_t v_α + σ_α v_α = (1/ρ) (∇·σ)_α
//!   ∂_t σ_αβ + (σ_α + σ_β)/2 · σ_αβ = C_αβγδ ε̇_γδ
//! ```
//!
//! admits the asymptotic solution `v_α(x_α) ∝ exp(−∫₀^x_α σ_α(x') dx')`
//! for an outgoing wave. Hence a wave that traverses a PML of thickness
//! `L_α` and integrated absorption `Σ_α = ∫₀^L_α σ_α(x) dx` is attenuated
//! by `exp(−Σ_α)` in amplitude (twice in round-trip, so reflectivity
//! `R ≈ exp(−2 Σ_α)`).
//!
//! For a polynomial profile `σ_α(x) = σ_max · (x/L_α)^p` with `p = 4` and
//! `σ_max` chosen to give theoretical reflection coefficient `R₀`,
//!
//! ```text
//!   σ_max = − (p + 1) c_max ln(R₀) / (2 L_α)
//! ```
//!
//! follows from Roden & Gedney (2000) eq. 25 with the standard
//! cell-thickness normalisation. For `R₀ = 10⁻⁴`, `p = 4`,
//! `c_max = 1500 m/s`, `L_α = 10·dx`, this gives
//! `σ_max ≈ 4.6 · 10⁵ s⁻¹` in air-water acoustic units.
//!
//! # Implementation strategy
//!
//! The orchestrator applies the PML in real space immediately after the
//! IFFT-back-to-velocity step:
//!
//! ```text
//!   v_α(x, t+dt)  ←  v_α(x, t+dt) · exp(−σ_α(x_α)·dt)
//!   σ_αβ(x, t+dt) ←  σ_αβ(x, t+dt) · exp(−(σ_α(x_α) + σ_β(x_β))/2 · dt)
//! ```
//!
//! The exponential factor is precomputed once at PML construction
//! (`damping_x[i] = exp(−σ_α(x_i)·dt)`) and applied each step as a
//! per-cell multiply. This is simpler than Berenger's split-field PML —
//! it doesn't require splitting velocity and stress into directional
//! sub-fields — but achieves the same exponential absorption at the cost
//! of slightly higher residual reflection. For the dominant
//! pykwavers/KWave.jl parity use cases (short propagation, sensor
//! recording within ≥1 wavelength of the boundary) this is sufficient;
//! Berenger split-field PML is on the backlog for long-range elastic.
//!
//! # Stability
//!
//! Real-space exponential damping is unconditionally stable for any
//! `σ_α · dt ≥ 0` since the multiplier `exp(−σ_α·dt) ∈ (0, 1]`. No
//! additional CFL constraint is introduced.

use ndarray::{Array1, Array3, Zip};
use std::f64::consts::PI;

/// Per-axis exponential damping coefficients precomputed for the
/// orchestrator's grid. All arrays are real-valued and shaped `(N_α,)`
/// since the damping depends on the boundary-distance index alone.
#[derive(Debug, Clone)]
pub struct ElasticPml {
    /// Number of grid cells in the absorbing layer per side along each
    /// axis. `0` ⇒ no PML on that axis.
    thickness_cells: (usize, usize, usize),
    /// `damping_x[i] = exp(−σ_x(i)·dt)` ∈ (0, 1].  Outside the absorbing
    /// layer this is exactly 1.0.
    damping_x: Array1<f64>,
    damping_y: Array1<f64>,
    damping_z: Array1<f64>,
}

impl ElasticPml {
    /// Build a PML for a `(nx, ny, nz)` grid with the given per-axis
    /// thickness in cells.
    ///
    /// `c_max` is the maximum sound speed in the medium (used to set
    /// `σ_max` for a target theoretical reflection coefficient `R0`).
    /// `dt` is the simulation time step. `dx, dy, dz` are the grid
    /// spacings.
    ///
    /// `r0` is the target theoretical reflection coefficient at normal
    /// incidence (Roden & Gedney 2000); `1e-4` is a standard choice.
    /// The polynomial order `p = 4` is hard-coded as the literature
    /// optimum for spectral solvers (Treeby & Cox 2010 Appendix B).
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        thickness_cells: (usize, usize, usize),
        dx: f64,
        dy: f64,
        dz: f64,
        c_max: f64,
        dt: f64,
        r0: f64,
    ) -> Self {
        const P: f64 = 4.0;
        let damping_x = build_axis_damping(nx, thickness_cells.0, dx, c_max, dt, r0, P);
        let damping_y = build_axis_damping(ny, thickness_cells.1, dy, c_max, dt, r0, P);
        let damping_z = build_axis_damping(nz, thickness_cells.2, dz, c_max, dt, r0, P);
        Self {
            thickness_cells,
            damping_x,
            damping_y,
            damping_z,
        }
    }

    /// Apply the PML damping to a real-space velocity / displacement
    /// field, in place.  `damping_field[i, j, k] *= damping_x[i] *
    /// damping_y[j] * damping_z[k]`.
    pub fn apply_to_field(&self, field: &mut Array3<f64>) {
        let dx_s = self.damping_x.as_slice().expect("damping_x contiguous");
        let dy_s = self.damping_y.as_slice().expect("damping_y contiguous");
        let dz_s = self.damping_z.as_slice().expect("damping_z contiguous");
        Zip::indexed(field).par_for_each(|(i, j, k), f| {
            *f *= dx_s[i] * dy_s[j] * dz_s[k];
        });
    }

    /// Borrow the per-axis damping coefficients (for tests / diagnostics).
    #[must_use]
    pub fn damping_axes(&self) -> (&Array1<f64>, &Array1<f64>, &Array1<f64>) {
        (&self.damping_x, &self.damping_y, &self.damping_z)
    }

    /// Borrow the per-axis thickness in cells.
    #[must_use]
    pub fn thickness_cells(&self) -> (usize, usize, usize) {
        self.thickness_cells
    }
}

fn build_axis_damping(
    n: usize,
    thickness: usize,
    dx: f64,
    c_max: f64,
    dt: f64,
    r0: f64,
    p: f64,
) -> Array1<f64> {
    let mut damping = Array1::<f64>::ones(n);
    if thickness == 0 || n < 2 * thickness + 1 {
        return damping;
    }
    let layer_len = thickness as f64 * dx;
    // Roden & Gedney (2000) eq. 25 — optimal σ_max for a polynomial profile
    // of order p giving theoretical reflection coefficient r0.
    let sigma_max = -(p + 1.0) * c_max * r0.ln() / (2.0 * layer_len);
    debug_assert!(
        sigma_max.is_finite() && sigma_max > 0.0,
        "PML σ_max must be positive and finite (r0 = {r0}, c_max = {c_max}, \
         layer_len = {layer_len})"
    );
    // Lower-side absorbing layer: indices 0..thickness, normalised distance
    // from the inner boundary x = 0 outward.
    for i in 0..thickness {
        let x = (thickness - i) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        damping[i] = (-sigma * dt).exp();
    }
    // Upper-side absorbing layer: indices n-thickness..n.
    for i in (n - thickness)..n {
        let x = (i - (n - thickness - 1)) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        damping[i] = (-sigma * dt).exp();
    }
    damping
}

// `PI` is referenced indirectly through future cosine-tapered profiles;
// silence the unused-import warning until that variant lands.
#[allow(dead_code)]
const _PI_REFERENCE: f64 = PI;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_thickness_means_unit_damping() {
        let pml = ElasticPml::new(16, 16, 16, (0, 0, 0), 1e-3, 1e-3, 1e-3, 1500.0, 1e-7, 1e-4);
        let (dx, dy, dz) = pml.damping_axes();
        for v in dx.iter().chain(dy.iter()).chain(dz.iter()) {
            assert_eq!(*v, 1.0, "no-PML damping must be exactly 1.0 everywhere");
        }
    }

    #[test]
    fn damping_is_monotonic_and_in_unit_interval() {
        let pml = ElasticPml::new(32, 32, 32, (8, 8, 8), 1e-3, 1e-3, 1e-3, 1500.0, 1e-7, 1e-4);
        let (dx, _, _) = pml.damping_axes();
        // Inside the absorbing layer at index 0, damping is strongest
        // (smallest value); at the inner edge of the layer (index = thickness)
        // damping is 1.0 (no absorption).
        for v in dx.iter() {
            assert!(*v > 0.0 && *v <= 1.0, "damping = {v} must be in (0, 1]");
        }
        // Innermost cell of left layer must absorb strictly more than the
        // outermost cell of the inner region.
        assert!(dx[0] < dx[8], "damping must increase outward (left side)");
        // Symmetric on the right.
        let n = dx.len();
        assert!(
            dx[n - 1] < dx[n - 9],
            "damping must increase outward (right side)"
        );
    }

    /// Apply the PML to a unit field across `n_passes` passes and verify
    /// that the cumulative attenuation in the absorbing layer matches
    /// `(damping[i])^n_passes` exactly — i.e., the per-step multiplier
    /// commutes with itself, as required for stable absorption.
    #[test]
    fn cumulative_attenuation_matches_per_step_multiplier_to_n() {
        let nx = 16usize;
        let ny = 4usize;
        let nz = 4usize;
        let thickness = 4usize;
        let pml = ElasticPml::new(
            nx,
            ny,
            nz,
            (thickness, 0, 0),
            1e-3,
            1e-3,
            1e-3,
            1500.0,
            1e-7,
            1e-4,
        );

        let mut field = Array3::<f64>::ones((nx, ny, nz));
        let n_passes = 50usize;
        for _ in 0..n_passes {
            pml.apply_to_field(&mut field);
        }

        let (dx, _, _) = pml.damping_axes();
        for i in 0..nx {
            let expected = dx[i].powi(n_passes as i32);
            let actual = field[[i, 0, 0]];
            let rel_err = (actual - expected).abs() / expected.max(1e-300);
            assert!(
                rel_err < 1e-9,
                "i={i}: actual = {actual:.3e}, expected = {expected:.3e}, \
                 rel_err = {rel_err:.3e}"
            );
        }
    }

    /// Roden-Gedney σ_max calibration check.
    ///
    /// For the canonical `r0 = 1e-4`, `p = 4`, `c_max = 1500 m/s`,
    /// `L = 10·dx` with `dx = 1e-3`:
    ///
    /// * `σ_max = −(p+1)·c·ln(r0) / (2L) ≈ 6.91 × 10⁵ s⁻¹`
    /// * Per-step damping at the strongest (outermost) cell:
    ///   `exp(−σ_max·dt) = exp(−6.91·10⁵ · 1·10⁻⁷) ≈ 0.933`.
    /// * Per-step damping at the innermost cell (boundary edge):
    ///   `exp(−σ_max · (1/thickness)⁴ · dt) ≈ exp(−0.0069·10⁻⁴) ≈ 1.0`.
    ///
    /// The PML's design absorption arrives from **sustained** application
    /// over the wave's many-step transit through the layer, NOT from a
    /// single-step single-traversal product. A pulse traveling at `c_max`
    /// crosses one cell in `dx / (c_max · dt) ≈ 6.67` steps; over the
    /// full 10-cell layer that's ~67 steps, during which each cell
    /// applies its damping ~6.67 times to the moving pulse. Cumulative
    /// attenuation experienced by the outgoing wave then ≈
    /// `prod_i exp(−σ_i · (dx/(c_max·dt)) · dt)` = `exp(−L·⟨σ⟩/c_max)`,
    /// recovering the calibrated `r0`.
    ///
    /// This test verifies the per-cell damping is in the expected range
    /// (the building block); the long-propagation absorption test
    /// (separate file) verifies the cumulative behaviour end-to-end.
    #[test]
    fn outermost_damping_matches_roden_gedney_calibration() {
        let nx = 64usize;
        let thickness = 10usize;
        let dx = 1e-3_f64;
        let c_max = 1500.0_f64;
        let dt = 1e-7_f64;
        let r0 = 1e-4_f64;
        let pml = ElasticPml::new(nx, 4, 4, (thickness, 0, 0), dx, 1e-3, 1e-3, c_max, dt, r0);

        let (dx_axis, _, _) = pml.damping_axes();

        // Theoretical σ_max from Roden & Gedney 2000 eq. 25.
        const P: f64 = 4.0;
        let l = thickness as f64 * dx;
        let sigma_max = -(P + 1.0) * c_max * r0.ln() / (2.0 * l);
        let expected_outermost = (-sigma_max * dt).exp();

        // The outermost cell on the LEFT side is index 0; the orchestrator
        // builds a polynomial profile with normalised distance `(thickness − i)
        // / thickness`, which equals 1.0 at i=0.
        let actual_outermost = dx_axis[0];
        let rel_err = (actual_outermost - expected_outermost).abs() / expected_outermost;
        assert!(
            rel_err < 1e-9,
            "outermost cell damping = {actual_outermost:.6e}, \
             expected exp(−σ_max·dt) = {expected_outermost:.6e}, rel_err = {rel_err:.3e}"
        );

        // The innermost layer cell (just outside the PML, index = thickness)
        // is in the interior and must have damping = 1.0.
        assert_eq!(
            dx_axis[thickness], 1.0,
            "first interior cell must have unity damping (no absorption)"
        );

        // Sanity: σ_max is positive and the per-step damping is bounded
        // away from 1 at the strongest cell (otherwise the PML does
        // nothing). For the canonical parameters, expect damping ≤ 0.99.
        assert!(
            actual_outermost < 0.99,
            "outermost damping {actual_outermost:.6e} ≥ 0.99 — PML σ_max \
             too small to absorb meaningfully at the strongest cell"
        );
    }
}
