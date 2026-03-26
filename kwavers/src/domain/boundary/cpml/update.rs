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
//! ## References
//! - Roden, J.A. & Gedney, S.D. (2000). Convolution PML (CPML): An efficient FDTD
//!   implementation of the CFS-PML for arbitrary media. Microwave Opt. Tech. Lett. 27(5), 334–339.
//! - Collino, F. & Tsogka, C. (2001). Application of the PML absorbing layer model to the linear
//!   elastodynamic problem in anisotropic hetereogeneous media. Geophysics 66(1), 294–307.
//! - Komatitsch, D. & Martin, R. (2007). An unsplit convolutional perfectly matched layer improved
//!   at grazing incidence for the seismic wave equation. Geophysics 72(5), SM155–SM167.

use super::memory::CPMLMemory;
use super::profiles::CPMLProfiles;
use ndarray::Array3;

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
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0 / 2;

        for i in 0..thickness.min(nx) {
            let a = profiles.a_x[i];
            let b = profiles.b_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    memory.psi_p_x[[i, j, k]] =
                        b * memory.psi_p_x[[i, j, k]] + a * gradient[[i, j, k]];
                }
            }
        }

        if nx > thickness {
            for i in (nx - thickness)..nx {
                let a = profiles.a_x[i];
                let b = profiles.b_x[i];
                let mem_idx = i - (nx - thickness) + thickness;
                for j in 0..ny {
                    for k in 0..nz {
                        memory.psi_p_x[[mem_idx, j, k]] =
                            b * memory.psi_p_x[[mem_idx, j, k]] + a * gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_p_x_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0 / 2;

        for i in 0..thickness.min(nx) {
            let inv_k = 1.0 / profiles.kappa_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    gradient[[i, j, k]] = gradient[[i, j, k]] * inv_k + memory.psi_p_x[[i, j, k]];
                }
            }
        }

        if nx > thickness {
            for i in (nx - thickness)..nx {
                let inv_k = 1.0 / profiles.kappa_x[i];
                let mem_idx = i - (nx - thickness) + thickness;
                for j in 0..ny {
                    for k in 0..nz {
                        gradient[[i, j, k]] =
                            gradient[[i, j, k]] * inv_k + memory.psi_p_x[[mem_idx, j, k]];
                    }
                }
            }
        }
    }

    // --- V Memory (Velocity Gradients) ---

    fn update_v_x_memory(
        &self,
        memory: &mut CPMLMemory,
        v_gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = v_gradient.dim();
        let thickness = memory.psi_v_x.dim().0 / 2;

        for i in 0..thickness.min(nx) {
            let a = profiles.a_x[i];
            let b = profiles.b_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    memory.psi_v_x[[i, j, k]] =
                        b * memory.psi_v_x[[i, j, k]] + a * v_gradient[[i, j, k]];
                }
            }
        }

        if nx > thickness {
            for i in (nx - thickness)..nx {
                let a = profiles.a_x[i];
                let b = profiles.b_x[i];
                let mem_idx = i - (nx - thickness) + thickness;
                for j in 0..ny {
                    for k in 0..nz {
                        memory.psi_v_x[[mem_idx, j, k]] =
                            b * memory.psi_v_x[[mem_idx, j, k]] + a * v_gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_v_x_correction(
        &self,
        v_gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = v_gradient.dim();
        let thickness = memory.psi_v_x.dim().0 / 2;

        for i in 0..thickness.min(nx) {
            let inv_k = 1.0 / profiles.kappa_x[i];
            for j in 0..ny {
                for k in 0..nz {
                    v_gradient[[i, j, k]] =
                        v_gradient[[i, j, k]] * inv_k + memory.psi_v_x[[i, j, k]];
                }
            }
        }

        if nx > thickness {
            for i in (nx - thickness)..nx {
                let inv_k = 1.0 / profiles.kappa_x[i];
                let mem_idx = i - (nx - thickness) + thickness;
                for j in 0..ny {
                    for k in 0..nz {
                        v_gradient[[i, j, k]] =
                            v_gradient[[i, j, k]] * inv_k + memory.psi_v_x[[mem_idx, j, k]];
                    }
                }
            }
        }
    }

    // --- Y and Z Components (P and V) ---
    // Similar blocks for Y and Z...

    fn update_p_y_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1 / 2;
        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                let a = profiles.a_y[j];
                let b = profiles.b_y[j];
                for k in 0..nz {
                    memory.psi_p_y[[i, j, k]] =
                        b * memory.psi_p_y[[i, j, k]] + a * gradient[[i, j, k]];
                }
            }
            if ny > thickness {
                for j in (ny - thickness)..ny {
                    let a = profiles.a_y[j];
                    let b = profiles.b_y[j];
                    let mem_idx = j - (ny - thickness) + thickness;
                    for k in 0..nz {
                        memory.psi_p_y[[i, mem_idx, k]] =
                            b * memory.psi_p_y[[i, mem_idx, k]] + a * gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_p_y_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1 / 2;
        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                let inv_k = 1.0 / profiles.kappa_y[j];
                for k in 0..nz {
                    gradient[[i, j, k]] = gradient[[i, j, k]] * inv_k + memory.psi_p_y[[i, j, k]];
                }
            }
            if ny > thickness {
                for j in (ny - thickness)..ny {
                    let inv_k = 1.0 / profiles.kappa_y[j];
                    let mem_idx = j - (ny - thickness) + thickness;
                    for k in 0..nz {
                        gradient[[i, j, k]] =
                            gradient[[i, j, k]] * inv_k + memory.psi_p_y[[i, mem_idx, k]];
                    }
                }
            }
        }
    }

    fn update_v_y_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_v_y.dim().1 / 2;
        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                let a = profiles.a_y[j];
                let b = profiles.b_y[j];
                for k in 0..nz {
                    memory.psi_v_y[[i, j, k]] =
                        b * memory.psi_v_y[[i, j, k]] + a * gradient[[i, j, k]];
                }
            }
            if ny > thickness {
                for j in (ny - thickness)..ny {
                    let a = profiles.a_y[j];
                    let b = profiles.b_y[j];
                    let mem_idx = j - (ny - thickness) + thickness;
                    for k in 0..nz {
                        memory.psi_v_y[[i, mem_idx, k]] =
                            b * memory.psi_v_y[[i, mem_idx, k]] + a * gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_v_y_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_v_y.dim().1 / 2;
        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                let inv_k = 1.0 / profiles.kappa_y[j];
                for k in 0..nz {
                    gradient[[i, j, k]] = gradient[[i, j, k]] * inv_k + memory.psi_v_y[[i, j, k]];
                }
            }
            if ny > thickness {
                for j in (ny - thickness)..ny {
                    let inv_k = 1.0 / profiles.kappa_y[j];
                    let mem_idx = j - (ny - thickness) + thickness;
                    for k in 0..nz {
                        gradient[[i, j, k]] =
                            gradient[[i, j, k]] * inv_k + memory.psi_v_y[[i, mem_idx, k]];
                    }
                }
            }
        }
    }

    fn update_p_z_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2 / 2;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    let a = profiles.a_z[k];
                    let b = profiles.b_z[k];
                    memory.psi_p_z[[i, j, k]] =
                        b * memory.psi_p_z[[i, j, k]] + a * gradient[[i, j, k]];
                }
                if nz > thickness {
                    for k in (nz - thickness)..nz {
                        let a = profiles.a_z[k];
                        let b = profiles.b_z[k];
                        let mem_idx = k - (nz - thickness) + thickness;
                        memory.psi_p_z[[i, j, mem_idx]] =
                            b * memory.psi_p_z[[i, j, mem_idx]] + a * gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_p_z_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2 / 2;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    let inv_k = 1.0 / profiles.kappa_z[k];
                    gradient[[i, j, k]] = gradient[[i, j, k]] * inv_k + memory.psi_p_z[[i, j, k]];
                }
                if nz > thickness {
                    for k in (nz - thickness)..nz {
                        let inv_k = 1.0 / profiles.kappa_z[k];
                        let mem_idx = k - (nz - thickness) + thickness;
                        gradient[[i, j, k]] =
                            gradient[[i, j, k]] * inv_k + memory.psi_p_z[[i, j, mem_idx]];
                    }
                }
            }
        }
    }

    fn update_v_z_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_v_z.dim().2 / 2;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    let a = profiles.a_z[k];
                    let b = profiles.b_z[k];
                    memory.psi_v_z[[i, j, k]] =
                        b * memory.psi_v_z[[i, j, k]] + a * gradient[[i, j, k]];
                }
                if nz > thickness {
                    for k in (nz - thickness)..nz {
                        let a = profiles.a_z[k];
                        let b = profiles.b_z[k];
                        let mem_idx = k - (nz - thickness) + thickness;
                        memory.psi_v_z[[i, j, mem_idx]] =
                            b * memory.psi_v_z[[i, j, mem_idx]] + a * gradient[[i, j, k]];
                    }
                }
            }
        }
    }

    fn apply_v_z_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_v_z.dim().2 / 2;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    let inv_k = 1.0 / profiles.kappa_z[k];
                    gradient[[i, j, k]] = gradient[[i, j, k]] * inv_k + memory.psi_v_z[[i, j, k]];
                }
                if nz > thickness {
                    for k in (nz - thickness)..nz {
                        let inv_k = 1.0 / profiles.kappa_z[k];
                        let mem_idx = k - (nz - thickness) + thickness;
                        gradient[[i, j, k]] =
                            gradient[[i, j, k]] * inv_k + memory.psi_v_z[[i, j, mem_idx]];
                    }
                }
            }
        }
    }
}

impl Default for CPMLUpdater {
    fn default() -> Self {
        Self::new()
    }
}
