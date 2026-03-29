//! CPML field update implementation
//!
//! # Theorem: CPML Recursive Convolution (Roden & Gedney 2000)
//!
//! The Convolutional PML (CPML) replaces the standard PML gradient `∂f/∂x` with
//! an effective gradient that includes a memory (auxiliary field) ψ:
//!
//! ```text
//!   ∂f/∂x_eff = (1/κ) · ∂f/∂x + ψ
//! ```
//!
//! The memory field ψ satisfies the recursive convolution update:
//! ```text
//!   ψ^{n+1} = b · ψ^n + a · (∂f/∂x)^n
//! ```
//!
//! where:
//! - `b = exp(−σ·Δt)` (decay factor, `b ∈ (0,1)` for σ > 0)
//! - `a = b − 1 = exp(−σ·Δt) − 1` (amplitude coefficient, always ≤ 0)
//! - `κ` = stretch factor (κ = 1 in the basic CPML formulation used here)
//!
//! *Interpretation:* At first step (ψ⁰ = 0):
//! ```text
//!   ψ¹ = a · ∂f/∂x = (b−1) · ∂f/∂x
//!   ∂f/∂x_eff = ∂f/∂x + ψ¹ = b · ∂f/∂x
//! ```
//! The effective gradient is attenuated by `b = exp(−σΔt)` immediately — correct absorption.
//!
//! ## Index Mapping (Split PML Memory)
//!
//! Each `psi_p_x` / `psi_v_x` array has shape `(2·thickness, ny, nz)`:
//! - Indices `[0..thickness]` → left boundary cells (i = 0..thickness)
//! - Indices `[thickness..2·thickness]` → right boundary cells (i = nx−thickness..nx)
//!
//! This layout avoids allocating full-grid memory for ψ fields (only PML cells need memory).
//!
//! ## Parallelism
//!
//! All update and correction loops are parallelized via `ndarray::Zip::par_for_each`.
//! Each boundary region (left / right) is processed as a contiguous slice, enabling
//! rayon to distribute work across threads with no data races.
//!
//! ## References
//! - Roden, J.A. & Gedney, S.D. (2000). Convolution PML (CPML): An efficient FDTD
//!   implementation of the CFS-PML for arbitrary media. Microwave Opt. Tech. Lett. 27(5), 334–339.
//! - Collino, F. & Tsogka, C. (2001). Application of the PML absorbing layer model to the linear
//!   elastodynamic problem in anisotropic hetereogeneous media. Geophysics 66(1), 294–307.
//! - Komatitsch, D. & Martin, R. (2007). An unsplit convolutional perfectly matched layer improved
//!   at grazing incidence for the seismic wave equation. Geophysics 72(5), SM155–SM167.

use super::memory::CPMLMemory;
use super::profiles::CPMLProfiles;

use ndarray::{s, Array3, Zip};

/// CPML field updater
#[derive(Debug, Clone)]
pub struct CPMLUpdater {}

impl CPMLUpdater {
    /// Create new updater
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Update memory component for acoustic gradients (used in velocity update)
    pub fn update_p_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.update_p_x_memory(memory, gradient, profiles),
            1 => self.update_p_y_memory(memory, gradient, profiles),
            2 => self.update_p_z_memory(memory, gradient, profiles),
            _ => {}
        }
    }

    /// Update memory component for velocity gradients (used in pressure update)
    pub fn update_v_memory(
        &self,
        memory: &mut CPMLMemory,
        v_gradient: &Array3<f64>,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.update_v_x_memory(memory, v_gradient, profiles),
            1 => self.update_v_y_memory(memory, v_gradient, profiles),
            2 => self.update_v_z_memory(memory, v_gradient, profiles),
            _ => {}
        }
    }

    /// Apply gradient correction from CPML memory
    pub fn apply_p_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.apply_p_x_correction(gradient, memory, profiles),
            1 => self.apply_p_y_correction(gradient, memory, profiles),
            2 => self.apply_p_z_correction(gradient, memory, profiles),
            _ => {}
        }
    }

    /// Apply velocity gradient correction from CPML memory
    pub fn apply_v_correction(
        &self,
        v_gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.apply_v_x_correction(v_gradient, memory, profiles),
            1 => self.apply_v_y_correction(v_gradient, memory, profiles),
            2 => self.apply_v_z_correction(v_gradient, memory, profiles),
            _ => {}
        }
    }

    // --- P Memory (Pressure Gradients) ---

    fn update_p_x_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, _ny, _nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0 / 2;
        let left_count = thickness.min(nx);

        // Left boundary
        Zip::indexed(memory.psi_p_x.slice_mut(s![..left_count, .., ..]))
            .and(gradient.slice(s![..left_count, .., ..]))
            .par_for_each(|(i, _j, _k), psi, &g| {
                *psi = profiles.b_x[i] * *psi + profiles.a_x[i] * g;
            });

        // Right boundary
        if nx > thickness {
            let right_start = nx - thickness;
            Zip::indexed(memory.psi_p_x.slice_mut(s![thickness.., .., ..]))
                .and(gradient.slice(s![right_start.., .., ..]))
                .par_for_each(|(i, _j, _k), psi, &g| {
                    let grid_i = right_start + i;
                    *psi = profiles.b_x[grid_i] * *psi + profiles.a_x[grid_i] * g;
                });
        }
    }

    fn apply_p_x_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, _ny, _nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0 / 2;
        let left_count = thickness.min(nx);

        // Left boundary
        Zip::indexed(gradient.slice_mut(s![..left_count, .., ..]))
            .and(memory.psi_p_x.slice(s![..left_count, .., ..]))
            .par_for_each(|(i, _j, _k), g, &psi| {
                *g = *g / profiles.kappa_x[i] + psi;
            });

        // Right boundary
        if nx > thickness {
            let right_start = nx - thickness;
            Zip::indexed(gradient.slice_mut(s![right_start.., .., ..]))
                .and(memory.psi_p_x.slice(s![thickness.., .., ..]))
                .par_for_each(|(i, _j, _k), g, &psi| {
                    *g = *g / profiles.kappa_x[right_start + i] + psi;
                });
        }
    }

    // --- V Memory (Velocity Gradients) ---

    fn update_v_x_memory(
        &self,
        memory: &mut CPMLMemory,
        v_gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, _ny, _nz) = v_gradient.dim();
        let thickness = memory.psi_v_x.dim().0 / 2;
        let left_count = thickness.min(nx);

        // Left boundary
        Zip::indexed(memory.psi_v_x.slice_mut(s![..left_count, .., ..]))
            .and(v_gradient.slice(s![..left_count, .., ..]))
            .par_for_each(|(i, _j, _k), psi, &g| {
                *psi = profiles.b_x[i] * *psi + profiles.a_x[i] * g;
            });

        // Right boundary
        if nx > thickness {
            let right_start = nx - thickness;
            Zip::indexed(memory.psi_v_x.slice_mut(s![thickness.., .., ..]))
                .and(v_gradient.slice(s![right_start.., .., ..]))
                .par_for_each(|(i, _j, _k), psi, &g| {
                    let grid_i = right_start + i;
                    *psi = profiles.b_x[grid_i] * *psi + profiles.a_x[grid_i] * g;
                });
        }
    }

    fn apply_v_x_correction(
        &self,
        v_gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, _ny, _nz) = v_gradient.dim();
        let thickness = memory.psi_v_x.dim().0 / 2;
        let left_count = thickness.min(nx);

        // Left boundary
        Zip::indexed(v_gradient.slice_mut(s![..left_count, .., ..]))
            .and(memory.psi_v_x.slice(s![..left_count, .., ..]))
            .par_for_each(|(i, _j, _k), g, &psi| {
                *g = *g / profiles.kappa_x[i] + psi;
            });

        // Right boundary
        if nx > thickness {
            let right_start = nx - thickness;
            Zip::indexed(v_gradient.slice_mut(s![right_start.., .., ..]))
                .and(memory.psi_v_x.slice(s![thickness.., .., ..]))
                .par_for_each(|(i, _j, _k), g, &psi| {
                    *g = *g / profiles.kappa_x[right_start + i] + psi;
                });
        }
    }

    // --- Y and Z Components (P and V) ---

    fn update_p_y_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, ny, _nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1 / 2;
        let left_count = thickness.min(ny);

        // Left boundary
        Zip::indexed(memory.psi_p_y.slice_mut(s![.., ..left_count, ..]))
            .and(gradient.slice(s![.., ..left_count, ..]))
            .par_for_each(|(_i, j, _k), psi, &g| {
                *psi = profiles.b_y[j] * *psi + profiles.a_y[j] * g;
            });

        // Right boundary
        if ny > thickness {
            let right_start = ny - thickness;
            Zip::indexed(memory.psi_p_y.slice_mut(s![.., thickness.., ..]))
                .and(gradient.slice(s![.., right_start.., ..]))
                .par_for_each(|(_i, j, _k), psi, &g| {
                    let grid_j = right_start + j;
                    *psi = profiles.b_y[grid_j] * *psi + profiles.a_y[grid_j] * g;
                });
        }
    }

    fn apply_p_y_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, ny, _nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1 / 2;
        let left_count = thickness.min(ny);

        // Left boundary
        Zip::indexed(gradient.slice_mut(s![.., ..left_count, ..]))
            .and(memory.psi_p_y.slice(s![.., ..left_count, ..]))
            .par_for_each(|(_i, j, _k), g, &psi| {
                *g = *g / profiles.kappa_y[j] + psi;
            });

        // Right boundary
        if ny > thickness {
            let right_start = ny - thickness;
            Zip::indexed(gradient.slice_mut(s![.., right_start.., ..]))
                .and(memory.psi_p_y.slice(s![.., thickness.., ..]))
                .par_for_each(|(_i, j, _k), g, &psi| {
                    *g = *g / profiles.kappa_y[right_start + j] + psi;
                });
        }
    }

    fn update_v_y_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, ny, _nz) = gradient.dim();
        let thickness = memory.psi_v_y.dim().1 / 2;
        let left_count = thickness.min(ny);

        // Left boundary
        Zip::indexed(memory.psi_v_y.slice_mut(s![.., ..left_count, ..]))
            .and(gradient.slice(s![.., ..left_count, ..]))
            .par_for_each(|(_i, j, _k), psi, &g| {
                *psi = profiles.b_y[j] * *psi + profiles.a_y[j] * g;
            });

        // Right boundary
        if ny > thickness {
            let right_start = ny - thickness;
            Zip::indexed(memory.psi_v_y.slice_mut(s![.., thickness.., ..]))
                .and(gradient.slice(s![.., right_start.., ..]))
                .par_for_each(|(_i, j, _k), psi, &g| {
                    let grid_j = right_start + j;
                    *psi = profiles.b_y[grid_j] * *psi + profiles.a_y[grid_j] * g;
                });
        }
    }

    fn apply_v_y_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, ny, _nz) = gradient.dim();
        let thickness = memory.psi_v_y.dim().1 / 2;
        let left_count = thickness.min(ny);

        // Left boundary
        Zip::indexed(gradient.slice_mut(s![.., ..left_count, ..]))
            .and(memory.psi_v_y.slice(s![.., ..left_count, ..]))
            .par_for_each(|(_i, j, _k), g, &psi| {
                *g = *g / profiles.kappa_y[j] + psi;
            });

        // Right boundary
        if ny > thickness {
            let right_start = ny - thickness;
            Zip::indexed(gradient.slice_mut(s![.., right_start.., ..]))
                .and(memory.psi_v_y.slice(s![.., thickness.., ..]))
                .par_for_each(|(_i, j, _k), g, &psi| {
                    *g = *g / profiles.kappa_y[right_start + j] + psi;
                });
        }
    }

    fn update_p_z_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, _ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2 / 2;
        let left_count = thickness.min(nz);

        // Left boundary
        Zip::indexed(memory.psi_p_z.slice_mut(s![.., .., ..left_count]))
            .and(gradient.slice(s![.., .., ..left_count]))
            .par_for_each(|(_i, _j, k), psi, &g| {
                *psi = profiles.b_z[k] * *psi + profiles.a_z[k] * g;
            });

        // Right boundary
        if nz > thickness {
            let right_start = nz - thickness;
            Zip::indexed(memory.psi_p_z.slice_mut(s![.., .., thickness..]))
                .and(gradient.slice(s![.., .., right_start..]))
                .par_for_each(|(_i, _j, k), psi, &g| {
                    let grid_k = right_start + k;
                    *psi = profiles.b_z[grid_k] * *psi + profiles.a_z[grid_k] * g;
                });
        }
    }

    fn apply_p_z_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, _ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2 / 2;
        let left_count = thickness.min(nz);

        // Left boundary
        Zip::indexed(gradient.slice_mut(s![.., .., ..left_count]))
            .and(memory.psi_p_z.slice(s![.., .., ..left_count]))
            .par_for_each(|(_i, _j, k), g, &psi| {
                *g = *g / profiles.kappa_z[k] + psi;
            });

        // Right boundary
        if nz > thickness {
            let right_start = nz - thickness;
            Zip::indexed(gradient.slice_mut(s![.., .., right_start..]))
                .and(memory.psi_p_z.slice(s![.., .., thickness..]))
                .par_for_each(|(_i, _j, k), g, &psi| {
                    *g = *g / profiles.kappa_z[right_start + k] + psi;
                });
        }
    }

    fn update_v_z_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, _ny, nz) = gradient.dim();
        let thickness = memory.psi_v_z.dim().2 / 2;
        let left_count = thickness.min(nz);

        // Left boundary
        Zip::indexed(memory.psi_v_z.slice_mut(s![.., .., ..left_count]))
            .and(gradient.slice(s![.., .., ..left_count]))
            .par_for_each(|(_i, _j, k), psi, &g| {
                *psi = profiles.b_z[k] * *psi + profiles.a_z[k] * g;
            });

        // Right boundary
        if nz > thickness {
            let right_start = nz - thickness;
            Zip::indexed(memory.psi_v_z.slice_mut(s![.., .., thickness..]))
                .and(gradient.slice(s![.., .., right_start..]))
                .par_for_each(|(_i, _j, k), psi, &g| {
                    let grid_k = right_start + k;
                    *psi = profiles.b_z[grid_k] * *psi + profiles.a_z[grid_k] * g;
                });
        }
    }

    fn apply_v_z_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (_nx, _ny, nz) = gradient.dim();
        let thickness = memory.psi_v_z.dim().2 / 2;
        let left_count = thickness.min(nz);

        // Left boundary
        Zip::indexed(gradient.slice_mut(s![.., .., ..left_count]))
            .and(memory.psi_v_z.slice(s![.., .., ..left_count]))
            .par_for_each(|(_i, _j, k), g, &psi| {
                *g = *g / profiles.kappa_z[k] + psi;
            });

        // Right boundary
        if nz > thickness {
            let right_start = nz - thickness;
            Zip::indexed(gradient.slice_mut(s![.., .., right_start..]))
                .and(memory.psi_v_z.slice(s![.., .., thickness..]))
                .par_for_each(|(_i, _j, k), g, &psi| {
                    *g = *g / profiles.kappa_z[right_start + k] + psi;
                });
        }
    }
}

impl Default for CPMLUpdater {
    fn default() -> Self {
        Self::new()
    }
}
