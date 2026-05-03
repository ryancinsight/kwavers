//! CPML Z-component memory update and correction methods.

use super::super::memory::CPMLMemory;
use super::super::profiles::CPMLProfiles;
use super::CPMLUpdater;
use ndarray::{s, Array3, Zip};

impl CPMLUpdater {
    pub(super) fn update_p_z_memory(
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

    pub(super) fn apply_p_z_correction(
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

    pub(super) fn update_v_z_memory(
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

    pub(super) fn apply_v_z_correction(
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
