//! Dynamic source injection: per-step Dirichlet/additive pressure paths,
//! velocity-component injection, and `add_source_arc` injection-mode
//! classification by mask geometry.

use leto::Array3;
use std::sync::Arc;

use super::GenericFdtdSolver;
use crate::forward::lanes::for_each_z_lane;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_source::{Source, SourceField, SourceInjectionMode};

fn clone_mask(mask: &Array3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = mask.shape();
    Array3::from_shape_vec([nx, ny, nz], mask.iter().copied().collect())
        .expect("FDTD source mask shape must match contiguous ndarray storage")
}

fn apply_boundary_pressure_mask(pressure: &mut Array3<f64>, mask: &Array3<f64>, amplitude: f64) {
    assert_eq!(
        pressure.shape(),
        mask.shape(),
        "invariant: FDTD pressure source mask shape matches pressure field"
    );

    let [_, ny, nz] = pressure.shape();
    if let (Some(pressure_values), Some(mask_values)) = (pressure.as_slice_mut(), mask.as_slice()) {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (pressure_value, &mask_value) in
                    lane.iter_mut().zip(&mask_values[start..start + nz])
                {
                    if mask_value > 0.0 {
                        *pressure_value = amplitude;
                    }
                }
            },
        );
    } else {
        for (pressure_value, &mask_value) in pressure.iter_mut().zip(mask.iter()) {
            if mask_value > 0.0 {
                *pressure_value = amplitude;
            }
        }
    }
}

fn apply_additive_pressure_mask(pressure: &mut Array3<f64>, mask: &Array3<f64>, amplitude: f64) {
    assert_eq!(
        pressure.shape(),
        mask.shape(),
        "invariant: FDTD pressure source mask shape matches pressure field"
    );

    let [_, ny, nz] = pressure.shape();
    if let (Some(pressure_values), Some(mask_values)) = (pressure.as_slice_mut(), mask.as_slice()) {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (pressure_value, &mask_value) in
                    lane.iter_mut().zip(&mask_values[start..start + nz])
                {
                    *pressure_value += mask_value * amplitude;
                }
            },
        );
    } else {
        for (pressure_value, &mask_value) in pressure.iter_mut().zip(mask.iter()) {
            *pressure_value += mask_value * amplitude;
        }
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
                    for (u, &m) in fields.ux.iter_mut().zip(mask.iter()) {
                        *u += m * amp;
                    }
                }
                SourceField::VelocityY => {
                    for (u, &m) in fields.uy.iter_mut().zip(mask.iter()) {
                        *u += m * amp;
                    }
                }
                SourceField::VelocityZ => {
                    for (u, &m) in fields.uz.iter_mut().zip(mask.iter()) {
                        *u += m * amp;
                    }
                }
            }
        }
    }
    /// Add source arc.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = clone_mask(&source.create_mask(&self.grid));

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
            .slice_with::<2>(&s![0, .., ..])
            .expect("invariant: x=0 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let xn_count = mask
            .slice_with::<2>(&s![nx - 1, .., ..])
            .expect("invariant: x=nx-1 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Y boundaries (planes at y=0 or y=ny-1)
        let y0_count = mask
            .slice_with::<2>(&s![.., 0, ..])
            .expect("invariant: y=0 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let yn_count = mask
            .slice_with::<2>(&s![.., ny - 1, ..])
            .expect("invariant: y=ny-1 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Z boundaries (planes at z=0 or z=nz-1)
        let z0_count = mask
            .slice_with::<2>(&s![.., .., 0])
            .expect("invariant: z=0 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let zn_count = mask
            .slice_with::<2>(&s![.., .., nz - 1])
            .expect("invariant: z=nz-1 boundary plane slice in range")
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Compute total mask statistics
        for &val in mask.iter() {
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

#[cfg(test)]
mod tests {
    use super::{apply_additive_pressure_mask, apply_boundary_pressure_mask};

    /// One volume below the lane walker's parallel floor and one above it.
    const KERNEL_SHAPES: [[usize; 3]; 2] = [[5, 3, 7], [37, 29, 31]];

    fn kernel_field(shape: [usize; 3], seed: f64) -> leto::Array3<f64> {
        let values = (0..shape.iter().product::<usize>())
            .map(|index| (index as f64).mul_add(0.754_8, seed).sin())
            .collect();
        leto::Array3::from_shape_vec(shape, values).expect("values match the shape")
    }

    fn kernel_at(field: &leto::Array3<f64>, index: usize) -> f64 {
        field.as_slice().expect("owned arrays are contiguous")[index]
    }

    /// A mask whose every third value is non-positive, so both mask kernels
    /// meet elements they leave alone.
    fn kernel_mask(shape: [usize; 3]) -> leto::Array3<f64> {
        let mut mask = kernel_field(shape, 2.0);
        for (index, value) in mask.iter_mut().enumerate() {
            if index % 3 == 0 {
                *value = -value.abs();
            }
        }
        mask
    }

    #[test]
    fn boundary_pressure_mask_is_the_per_element_assignment_to_the_bit() {
        const AMPLITUDE: f64 = 1.75;
        for shape in KERNEL_SHAPES {
            let mask = kernel_mask(shape);
            let initial = kernel_field(shape, 3.0);
            let mut pressure = initial.clone();
            apply_boundary_pressure_mask(&mut pressure, &mask, AMPLITUDE);
            let same = (0..pressure.len()).all(|i| {
                let expected = if kernel_at(&mask, i) > 0.0 {
                    AMPLITUDE
                } else {
                    kernel_at(&initial, i)
                };
                kernel_at(&pressure, i).to_bits() == expected.to_bits()
            });
            assert!(same, "boundary pressure mask diverges at {shape:?}");
        }
    }

    #[test]
    fn additive_pressure_mask_is_the_per_element_sum_to_the_bit() {
        const AMPLITUDE: f64 = 1.75;
        for shape in KERNEL_SHAPES {
            let mask = kernel_mask(shape);
            let initial = kernel_field(shape, 3.0);
            let mut pressure = initial.clone();
            apply_additive_pressure_mask(&mut pressure, &mask, AMPLITUDE);
            let same = (0..pressure.len()).all(|i| {
                let expected = kernel_at(&initial, i) + kernel_at(&mask, i) * AMPLITUDE;
                kernel_at(&pressure, i).to_bits() == expected.to_bits()
            });
            assert!(same, "additive pressure mask diverges at {shape:?}");
        }
    }
}
