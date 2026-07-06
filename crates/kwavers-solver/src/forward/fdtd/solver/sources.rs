//! Dynamic source injection: per-step Dirichlet/additive pressure paths,
//! velocity-component injection, and `add_source_arc` injection-mode
//! classification by mask geometry.

use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{s, Array3, Zip};
use std::sync::Arc;

use super::GenericFdtdSolver;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_source::{Source, SourceField, SourceInjectionMode};

fn apply_boundary_pressure_mask(pressure: &mut Array3<f64>, mask: &Array3<f64>, amplitude: f64) {
    assert_eq!(
        pressure.shape(),
        mask.shape(),
        "invariant: FDTD pressure source mask shape matches pressure field"
    );

    if let (Some(pressure_values), Some(mask_values)) = (
        pressure.as_slice_memory_order_mut(),
        mask.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            if mask_values[idx] > 0.0 {
                *pressure_value = amplitude;
            }
        });
    } else {
        Zip::from(pressure)
            .and(mask)
            .for_each(|pressure_value, &mask_value| {
                if mask_value > 0.0 {
                    *pressure_value = amplitude;
                }
            });
    }
}

fn apply_additive_pressure_mask(pressure: &mut Array3<f64>, mask: &Array3<f64>, amplitude: f64) {
    assert_eq!(
        pressure.shape(),
        mask.shape(),
        "invariant: FDTD pressure source mask shape matches pressure field"
    );

    if let (Some(pressure_values), Some(mask_values)) = (
        pressure.as_slice_memory_order_mut(),
        mask.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            *pressure_value += mask_values[idx] * amplitude;
        });
    } else {
        Zip::from(pressure)
            .and(mask)
            .for_each(|pressure_value, &mask_value| *pressure_value += mask_value * amplitude);
    }
}

impl GenericFdtdSolver<Array3<f64>> {
    pub(super) fn apply_dynamic_pressure_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let Self {
            ref dynamic_sources,
            ref mut fields,
            ref grid,
            ref materials,
            source_injection_modes: _,
            ..
        } = self;

        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let _c0_ref = materials.c0[[nx / 2, ny / 2, nz / 2]];
        let _dx = grid.dx;

        for (idx, (source, mask)) in dynamic_sources.iter().enumerate() {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::Pressure => {
                    let mode = self.source_injection_modes[idx];
                    match mode {
                        SourceInjectionMode::Boundary => {
                            // Dirichlet: enforce p = amplitude at boundary
                            apply_boundary_pressure_mask(&mut fields.p, mask, amp);
                        }
                        SourceInjectionMode::Additive { .. } => {
                            // Additive: p += mask * amplitude
                            // For parity with k-Wave's additive mass sources, we do not normalize by mask sum
                            // and we expect the physical scaling to be handled by the caller or precomputed.
                            apply_additive_pressure_mask(&mut fields.p, mask, amp);
                        }
                    }
                }
                SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {}
            }
        }
    }

    pub(super) fn apply_dynamic_pressure_dirichlet(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let Self {
            ref dynamic_sources,
            ref mut fields,
            ..
        } = self;

        for (idx, (source, mask)) in dynamic_sources.iter().enumerate() {
            if source.source_type() != SourceField::Pressure {
                continue;
            }
            if self.source_injection_modes[idx] != SourceInjectionMode::Boundary {
                continue;
            }
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }
            apply_boundary_pressure_mask(&mut fields.p, mask, amp);
        }
    }

    pub(super) fn apply_dynamic_velocity_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let Self {
            ref dynamic_sources,
            ref mut fields,
            ..
        } = self;

        for (source, mask) in dynamic_sources {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::Pressure => {}
                SourceField::VelocityX => {
                    Zip::from(&mut fields.ux)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityY => {
                    Zip::from(&mut fields.uy)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityZ => {
                    Zip::from(&mut fields.uz)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
            }
        }
    }
    /// Add source arc.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);

        // Determine injection mode once and cache it
        let mode = Self::determine_injection_mode(&mask, &self.grid);

        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        Ok(())
    }

    /// Determine injection mode based on source mask spatial distribution
    ///
    /// # Mathematical Specification
    /// - **Boundary Plane Source**: Mask is non-zero only on a single grid plane
    ///   (x=0, x=Nx-1, y=0, y=Ny-1, z=0, or z=Nz-1)
    ///   → Use Dirichlet enforcement: p(boundary) = amplitude(t)
    /// - **Interior Source**: Mask is non-zero in interior or distributed volume
    ///   → Use additive injection: p += (mask / ||mask||) * amplitude(t)
    ///   where ||mask|| is the L1 norm to preserve energy scaling
    fn determine_injection_mode(mask: &Array3<f64>, _grid: &Grid) -> SourceInjectionMode {
        let shape = mask.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        // Count non-zero mask elements
        let mut mask_sum = 0.0;
        let mut nonzero_count = 0;

        // Check if mask is concentrated on a single boundary plane
        let mut is_boundary_plane = false;

        // X boundaries (planes at x=0 or x=nx-1)
        let x0_count = mask
            .slice(s![0, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let xn_count = mask
            .slice(s![nx - 1, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Y boundaries (planes at y=0 or y=ny-1)
        let y0_count = mask
            .slice(s![.., 0, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let yn_count = mask
            .slice(s![.., ny - 1, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Z boundaries (planes at z=0 or z=nz-1)
        let z0_count = mask
            .slice(s![.., .., 0])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let zn_count = mask
            .slice(s![.., .., nz - 1])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Compute total mask statistics
        for &val in mask {
            if val > 0.0 {
                nonzero_count += 1;
                mask_sum += val;
            }
        }

        // If all non-zero elements are on a single boundary plane, use Boundary mode
        if nonzero_count > 0
            && (x0_count == nonzero_count
                || xn_count == nonzero_count
                || y0_count == nonzero_count
                || yn_count == nonzero_count
                || z0_count == nonzero_count
                || zn_count == nonzero_count)
        {
            is_boundary_plane = true;
        }

        if is_boundary_plane {
            SourceInjectionMode::Boundary
        } else {
            // Additive mode: normalize by mask L1 norm to preserve energy
            let scale = if mask_sum > 0.0 { 1.0 / mask_sum } else { 1.0 };
            SourceInjectionMode::Additive { scale }
        }
    }
}
