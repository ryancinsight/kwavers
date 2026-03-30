//! CPML profile computation and management.
//!
//! # Physics: Convolutional Perfectly Matched Layer
//!
//! ## Background
//! The CPML (Convolutional PML) extends Berenger's original split-field PML with a
//! recursive convolution scheme that maintains accuracy for evanescent and oblique waves.
//! For acoustic simulations, CPML modifies the gradient operator in PML regions by
//! introducing a frequency-dependent complex stretching function in the Laplace domain.
//!
//! ## Theorem (CPML Profile Grading, Roden & Gedney 2000)
//! For a PML of thickness d [m] with polynomial grading order m and target reflection
//! coefficient R₀ at normal incidence:
//!
//! ```text
//! σ_max = −(m + 1) · c₀ · ln(R₀) / (2 · d)          [optimal conductivity]
//! σ(ξ)  = σ_max · (ξ / d)^m                           [polynomial grading]
//! κ(ξ)  = 1 + (κ_max − 1) · (ξ / d)^m               [real stretch, κ_max ≥ 1]
//! α(ξ)  = α_max · (1 − ξ / d)                         [CFS parameter]
//! ```
//!
//! where ξ ∈ [0, d] is the depth into the PML (0 at the domain/PML interface, d at the wall).
//!
//! ## Theorem (Recursive Convolution Coefficients)
//! For the Roden & Gedney (2000) formulation with complex-frequency shift (CFS) parameters
//! σ, κ, α, the time-discrete recursive convolution coefficients are:
//!
//! ```text
//! b = exp(−(σ/κ + α) · Δt)
//! a = (σ/κ) · (b − 1) / (σ/κ + α)   [general CFS form]
//!
//! Special case α = 0, κ = 1 (used here):
//!   b = exp(−σ · Δt)
//!   a = b − 1 = exp(−σ · Δt) − 1    [note: a < 0 since 0 < b < 1]
//! ```
//!
//! The CPML memory variable update at each step is:
//! ```text
//! ψ^{n+1}_k = b_k · ψ^n_k + a_k · (∂f/∂x)^n_k
//! ```
//!
//! The effective gradient in PML cells is then:
//! ```text
//! (∂f/∂x)_eff = (∂f/∂x) / κ + ψ
//! ```
//!
//! At the first step (ψ⁰ = 0):
//! ```text
//! ψ¹ = b·ψ⁰ + a·∂f/∂x = (b − 1)·∂f/∂x
//! (∂f/∂x)_eff = ∂f/∂x + (b − 1)·∂f/∂x = b·∂f/∂x   [attenuated by b = exp(−σΔt) ✓]
//! ```
//!
//! ## Implementation Notes
//! - k-Wave uses polynomial order m = 4 (hardcoded here for compatibility)
//! - σ_max matches k-Wave's `pml_alpha * (c/dx) * 1.0^4` at the wall (ξ = d)
//! - The `alpha=0, kappa=1` specialization matches k-Wave's default CPML configuration
//! - `a_coeff = b_coeff − 1.0`; setting a_coeff=0 would disable convolutional memory,
//!   reducing CPML to a simple split-field PML (worse oblique-incidence absorption)
//!
//! ## References
//! - Roden & Gedney (2000). Microwave Opt. Tech. Lett. 27(5), 334–339.
//!   DOI: 10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A
//! - Collino & Tsogka (2001). Geophysics 66(1), 294–307.
//!   DOI: 10.1190/1.1444908
//! - Berenger (1994). J. Comput. Phys. 114(2), 185–200 (original PML).
//!   DOI: 10.1006/jcph.1994.1159

use super::config::CPMLConfig;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array1;

/// CPML absorption and stretching profiles
///
/// Two sets of sigma arrays are maintained:
/// - `sigma_*`: collocated (non-staggered) profiles used for density components (ρx, ρy, ρz).
/// - `sigma_*_sg*`: staggered profiles used for velocity components (ux, uy, uz).
///
/// The staggered profiles follow k-Wave's `get_pml(..., staggered=True)` formula:
///   σ_sgz(k) = pml_alpha · (c/dz) · ((pml_size − k − 0.5) / pml_size)⁴  [left]
///   σ_sgz(k) = pml_alpha · (c/dz) · ((j + 1.5) / pml_size)⁴            [right, j=0-based offset]
///
/// Ref: Treeby & Cox (2010), J. Biomed. Opt. 15(2) 021314; k-Wave pml.py `get_pml`.
#[derive(Debug, Clone)]
pub struct CPMLProfiles {
    pub sigma_x: Array1<f64>,
    pub sigma_y: Array1<f64>,
    pub sigma_z: Array1<f64>,
    /// Staggered sigma for ux (staggered at i+0.5 in x)
    pub sigma_x_sgx: Array1<f64>,
    /// Staggered sigma for uy (staggered at j+0.5 in y)
    pub sigma_y_sgy: Array1<f64>,
    /// Staggered sigma for uz (staggered at k+0.5 in z)
    pub sigma_z_sgz: Array1<f64>,
    pub kappa_x: Array1<f64>,
    pub kappa_y: Array1<f64>,
    pub kappa_z: Array1<f64>,
    pub alpha_x: Array1<f64>,
    pub alpha_y: Array1<f64>,
    pub alpha_z: Array1<f64>,

    // Recursive convolution coefficients
    // b = exp(-(sigma/kappa + alpha) * dt)
    // a = (sigma / (kappa * (sigma + kappa * alpha))) * (b - 1)
    pub a_x: Array1<f64>,
    pub a_y: Array1<f64>,
    pub a_z: Array1<f64>,
    pub b_x: Array1<f64>,
    pub b_y: Array1<f64>,
    pub b_z: Array1<f64>,
}

impl CPMLProfiles {
    /// Create new CPML profiles based on configuration
    pub fn new(config: &CPMLConfig, grid: &Grid, sound_speed: f64, dt: f64) -> KwaversResult<Self> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        // Initialize profile arrays
        let mut profiles = Self {
            sigma_x: Array1::zeros(nx),
            sigma_y: Array1::zeros(ny),
            sigma_z: Array1::zeros(nz),
            sigma_x_sgx: Array1::zeros(nx),
            sigma_y_sgy: Array1::zeros(ny),
            sigma_z_sgz: Array1::zeros(nz),
            kappa_x: Array1::ones(nx),
            kappa_y: Array1::ones(ny),
            kappa_z: Array1::ones(nz),
            alpha_x: Array1::zeros(nx),
            alpha_y: Array1::zeros(ny),
            alpha_z: Array1::zeros(nz),
            a_x: Array1::zeros(nx),
            a_y: Array1::zeros(ny),
            a_z: Array1::zeros(nz),
            b_x: Array1::ones(nx),
            b_y: Array1::ones(ny),
            b_z: Array1::ones(nz),
        };

        profiles.compute_profiles(config, grid, sound_speed, dt)?;
        Ok(profiles)
    }

    fn compute_profiles(
        &mut self,
        config: &CPMLConfig,
        grid: &Grid,
        sound_speed: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute collocated (density) profiles for each direction.
        // Pass per-dimension sigma factor so that different absorption strengths
        // per axis are honoured (k-Wave pml_alpha vector support).
        let alpha_x = config.sigma_factor_for_dimension(0);
        let alpha_y = config.sigma_factor_for_dimension(1);
        let alpha_z = config.sigma_factor_for_dimension(2);

        Self::compute_profile_1d(
            &mut self.sigma_x,
            &mut self.kappa_x,
            &mut self.alpha_x,
            &mut self.a_x,
            &mut self.b_x,
            grid.nx,
            grid.dx,
            config.per_dimension.x,
            config,
            alpha_x,
            sound_speed,
            dt,
        );
        Self::compute_profile_1d(
            &mut self.sigma_y,
            &mut self.kappa_y,
            &mut self.alpha_y,
            &mut self.a_y,
            &mut self.b_y,
            grid.ny,
            grid.dy,
            config.per_dimension.y,
            config,
            alpha_y,
            sound_speed,
            dt,
        );
        Self::compute_profile_1d(
            &mut self.sigma_z,
            &mut self.kappa_z,
            &mut self.alpha_z,
            &mut self.a_z,
            &mut self.b_z,
            grid.nz,
            grid.dz,
            config.per_dimension.z,
            config,
            alpha_z,
            sound_speed,
            dt,
        );

        // Compute staggered (velocity) profiles matching k-Wave's pml_*_sg* arrays
        Self::compute_profile_1d_staggered(
            &mut self.sigma_x_sgx,
            grid.nx,
            grid.dx,
            config.per_dimension.x,
            alpha_x,
            config,
            sound_speed,
        );
        Self::compute_profile_1d_staggered(
            &mut self.sigma_y_sgy,
            grid.ny,
            grid.dy,
            config.per_dimension.y,
            alpha_y,
            config,
            sound_speed,
        );
        Self::compute_profile_1d_staggered(
            &mut self.sigma_z_sgz,
            grid.nz,
            grid.dz,
            config.per_dimension.z,
            alpha_z,
            config,
            sound_speed,
        );
        Ok(())
    }

    /// Staggered K-Wave PML profile for velocity components.
    ///
    /// Matches k-Wave's `get_pml(..., staggered=True)` which shifts the grid position
    /// by +0.5 cells to account for the half-cell offset of staggered velocity fields.
    ///
    /// Left PML at 0-based index i (x = i+1 in 1-based):
    ///   σ_sg(i) = pml_alpha · (c/dx) · ((pml_size − i − 0.5) / pml_size)⁴
    ///
    /// Right PML at 0-based index i (j = i − right_start, j=0..thickness-1):
    ///   σ_sg(i) = pml_alpha · (c/dx) · ((j + 1.5) / pml_size)⁴
    ///
    /// Note: the rightmost staggered cell can exceed σ_max (k-Wave extrapolates beyond
    /// the domain boundary), which is intentional and matches k-Wave behavior.
    ///
    /// Ref: k-Wave pml.py `get_pml(..., staggered=True)`; Treeby & Cox (2010) §2.2.
    fn compute_profile_1d_staggered(
        sigma_sg: &mut Array1<f64>,
        n: usize,
        dx: f64,
        thickness: usize,
        pml_alpha: f64,
        _config: &CPMLConfig,
        sound_speed: f64,
    ) {
        let pml_order: f64 = 4.0;
        let t = thickness as f64;

        // Left staggered: position i+0.5 shifted, nearest PML interface = pml_size
        for i in 0..thickness.min(n) {
            // k-Wave formula: ((x+0.5 - pml_size - 1) / (0 - pml_size))^4
            // With x = i+1 (1-based): ((i + 1.5 - pml_size - 1) / (-pml_size))^4
            //   = ((i + 0.5 - pml_size) / (-pml_size))^4
            //   = ((pml_size - i - 0.5) / pml_size)^4
            let d = (t - i as f64 - 0.5) / t;
            sigma_sg[i] = pml_alpha * (sound_speed / dx) * d.abs().powf(pml_order);
        }

        // Right staggered: position i+0.5 measured from right_start+1
        let right_start = n.saturating_sub(thickness);
        for i in right_start..n {
            let j = (i - right_start) as f64; // 0-based offset within right PML
            // k-Wave formula: ((x+0.5) / pml_size)^4, x=j+1 → (j+1.5)/pml_size
            let d = (j + 1.5) / t;
            sigma_sg[i] = pml_alpha * (sound_speed / dx) * d.powf(pml_order);
        }
    }

    /// Exact K-Wave PML profile computation.
    ///
    /// K-Wave `get_pml()` formula (from `kwave/utils/pml.py`):
    ///   `sigma = pml_alpha * (c / dx) * ((x / N)^4)`
    /// where `pml_alpha` is the user-configurable absorption coefficient (default 2.0)
    /// and `x` runs from 1..N within the PML region.
    ///
    /// For staggered grids, K-Wave uses `(x + 0.5)` instead of `x`,
    /// but the difference is negligible for `N >= 10`.
    ///
    /// The stored sigma values are meant to be applied as `exp(-sigma * dt / 2)`
    /// to both velocity and density fields (split-field PML), once each per step.
    fn compute_profile_1d(
        sigma: &mut Array1<f64>,
        kappa: &mut Array1<f64>,
        alpha: &mut Array1<f64>,
        a_coeff: &mut Array1<f64>,
        b_coeff: &mut Array1<f64>,
        n: usize,
        dx: f64,
        thickness: usize,
        _config: &CPMLConfig,
        pml_alpha: f64,
        sound_speed: f64,
        dt: f64,
    ) {
        // K-Wave hard-codes polynomial order 4 for PML profiles
        let pml_order: f64 = 4.0;

        let mut compute_at = |idx: usize, sigma_val: f64| {
            sigma[idx] = sigma_val;
            // kappa and alpha are not used in K-Wave's simple multiplicative PML,
            // but we keep them for potential CPML convolutional use
            kappa[idx] = 1.0;
            alpha[idx] = 0.0;
            // Roden & Gedney (2000) recursive convolution coefficients.
            // With κ=1, α=0:  b = exp(−σ·Δt),  a = b − 1.
            // Memory update: ψ^{n+1} = b·ψ^n + a·∇f^n
            // Effective gradient: ∇f_eff = ∇f/κ + ψ^{n+1}
            // At step 0 (ψ=0): ∇f_eff = ∇f + (b−1)·∇f = b·∇f  [correctly attenuated].
            // Setting a=0 would make ψ stay 0, reducing CPML to simple split-field PML.
            let b = (-(sigma_val) * dt).exp();
            b_coeff[idx] = b;
            a_coeff[idx] = b - 1.0; // Roden & Gedney 2000, Eq. (9): a = b − 1 (α=0, κ=1)
        };

        // Left boundary: x runs from 1 to thickness
        // K-Wave formula: pml_left = pml_alpha * (c/dx) * (((x - pml_size - 1) / (0 - pml_size))^4)
        // Simplified: distance from interface = (thickness - i) / thickness (1 at boundary, 0 at interface)
        for i in 0..thickness.min(n) {
            // i=0 is deepest PML cell, i=thickness-1 is at interface
            // x = i+1 in MATLAB 1-based terms
            // K-Wave left profile: ((x - pml_size - 1) / (0 - pml_size))^4
            //   = ((i+1 - thickness - 1) / (-thickness))^4
            //   = ((i - thickness) / (-thickness))^4
            //   = ((thickness - i) / thickness)^4
            let d = (thickness - i) as f64 / thickness as f64;
            let s = pml_alpha * (sound_speed / dx) * d.powf(pml_order);
            compute_at(i, s);
        }

        // Right boundary: x runs from 1 to thickness
        // K-Wave formula: pml_right = pml_alpha * (c/dx) * ((x / pml_size)^4)
        let right_start = n.saturating_sub(thickness);
        for i in right_start..n {
            let x = (i - right_start + 1) as f64;
            let d = x / thickness as f64;
            let s = pml_alpha * (sound_speed / dx) * d.powf(pml_order);
            compute_at(i, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::boundary::cpml::config::CPMLConfig;
    use crate::domain::grid::Grid;

    /// The k-Wave σ_max formula at the PML wall (d_normalized = 1.0) is:
    ///   σ_max = pml_alpha × (c₀ / dx)
    ///
    /// For default pml_alpha = 2.0, c₀ = 1500 m/s, dx = 1e-3 m:
    ///   σ_max = 2.0 × (1500 / 1e-3) = 3 000 000 s⁻¹
    ///
    /// Verify the last cell in the PML (highest sigma) matches this formula.
    ///
    /// ## Reference
    /// Treeby & Cox (2010), J. Biomed. Opt. 15(2) 021314, Eq. (10); k-Wave pml.py `get_pml`.
    #[test]
    fn test_cpml_sigma_max_formula() {
        let c0 = 1500.0_f64;
        let dx = 1e-3_f64;
        let pml_size = 10_usize;
        let dt = 1e-7_f64;
        let pml_alpha = 2.0_f64;

        let nx = 32;
        let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
        let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

        let profiles = CPMLProfiles::new(&config, &grid, c0, dt)
            .expect("CPMLProfiles::new should succeed");

        // Left PML: d = (thickness - i) / thickness.
        // At i=0 (the outer wall): d = thickness/thickness = 1.0 → σ = σ_max
        // At i=pml_size-1 (domain interface): d = 1/thickness → σ ≈ 0
        // k-Wave σ_max = pml_alpha × (c₀/dx)
        let expected_sigma_max = pml_alpha * (c0 / dx);
        let actual = profiles.sigma_x[0]; // wall: maximum σ
        assert!(
            (actual - expected_sigma_max).abs() / expected_sigma_max < 0.01,
            "σ_max = {actual:.1} should match k-Wave formula {expected_sigma_max:.1}"
        );
    }

    /// b-coefficient at the PML wall must equal exp(−σ_max · dt).
    ///
    /// ## Reference
    /// Roden & Gedney (2000), Eq. (9): b = exp(−σ·Δt), a = b − 1.
    #[test]
    fn test_cpml_recursive_convolution_coefficients() {
        let c0 = 1500.0_f64;
        let dx = 1e-3_f64;
        let pml_size = 10_usize;
        let dt = 1e-7_f64;
        let pml_alpha = 2.0_f64;

        let nx = 32;
        let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
        let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

        let profiles = CPMLProfiles::new(&config, &grid, c0, dt)
            .expect("CPMLProfiles::new should succeed");

        // At the PML wall (index 0: outer wall, maximum σ)
        let sigma_max = profiles.sigma_x[0];
        let expected_b = (-sigma_max * dt).exp();
        let expected_a = expected_b - 1.0;

        let actual_b = profiles.b_x[0];
        let actual_a = profiles.a_x[0];

        assert!(
            (actual_b - expected_b).abs() < 1e-12,
            "b_x at PML wall: expected {expected_b:.10}, got {actual_b:.10}"
        );
        assert!(
            (actual_a - expected_a).abs() < 1e-12,
            "a_x at PML wall: expected {expected_a:.10}, got {actual_a:.10}"
        );
        // Interior cells must have b=1, a=0 (no damping outside PML)
        let mid = nx / 2;
        assert!(
            (profiles.b_x[mid] - 1.0).abs() < 1e-14,
            "b_x at interior must be 1.0 (no PML), got {}",
            profiles.b_x[mid]
        );
        assert!(
            profiles.a_x[mid].abs() < 1e-14,
            "a_x at interior must be 0.0, got {}",
            profiles.a_x[mid]
        );
    }
}
