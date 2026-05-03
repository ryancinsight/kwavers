//! CPML X-component memory update and correction methods.

use super::super::memory::CPMLMemory;
use super::super::profiles::CPMLProfiles;
use super::CPMLUpdater;
use ndarray::{s, Array3, Zip};

impl CPMLUpdater {
    pub(super) fn update_p_x_memory(
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

    pub(super) fn apply_p_x_correction(
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

    pub(super) fn update_v_x_memory(
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

    pub(super) fn apply_v_x_correction(
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
}
