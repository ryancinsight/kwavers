//! CPML Y-component memory update and correction methods.

use super::super::memory::CPMLMemory;
use super::super::profiles::CPMLProfiles;
use super::CPMLUpdater;
use ndarray::{s, Array3, Zip};

impl CPMLUpdater {
    pub(super) fn update_p_y_memory(
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

    pub(super) fn apply_p_y_correction(
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

    pub(super) fn update_v_y_memory(
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

    pub(super) fn apply_v_y_correction(
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
}
