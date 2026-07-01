use ndarray::Array3;

use super::super::{ElementShape, KWaveArray};

impl KWaveArray {
    /// Generate a binary mask on the computational grid.
    ///
    /// Returns a 3D boolean array where `true` indicates grid points that are
    /// part of the transducer array.
    pub fn get_array_binary_mask(&self, grid: &kwavers_grid::Grid) -> Array3<bool> {
        let mut mask = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
        for element in &self.elements {
            match element {
                ElementShape::Arc {
                    position,
                    radius,
                    diameter,
                    start_angle,
                    end_angle,
                } => {
                    self.rasterize_arc(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *diameter,
                        *start_angle,
                        *end_angle,
                    );
                }
                ElementShape::Rect {
                    position,
                    width,
                    height,
                    length,
                    euler_xyz_deg,
                } => {
                    let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                    self.rasterize_rect(
                        &mut mask, grid, pos_eff, *width, *height, *length, euler_eff,
                    );
                }
                ElementShape::Disc {
                    position,
                    diameter,
                    focus_position,
                } => {
                    self.rasterize_disc(&mut mask, grid, *position, *diameter, *focus_position);
                }
                ElementShape::ProfiledDisc {
                    position,
                    diameter,
                    focus_position,
                    ..
                } => {
                    self.rasterize_disc(&mut mask, grid, *position, *diameter, *focus_position);
                }
                ElementShape::Bowl {
                    position,
                    radius,
                    diameter,
                } => {
                    self.rasterize_bowl(&mut mask, grid, *position, *radius, *diameter);
                }
                ElementShape::Annulus {
                    position,
                    radius,
                    inner_diameter,
                    outer_diameter,
                } => {
                    self.rasterize_annulus(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *inner_diameter,
                        *outer_diameter,
                    );
                }
            }
        }
        mask
    }

    /// Generate a float-weighted mask on the computational grid.
    ///
    /// Uses k-Wave-compatible surface sampling and local BLI stencils to
    /// produce per-cell weights that sum to each element's geometric measure
    /// expressed in grid cells.
    pub fn get_array_weighted_mask(&self, grid: &kwavers_grid::Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for element in &self.elements {
            match element {
                ElementShape::Bowl {
                    position,
                    radius,
                    diameter,
                } => {
                    self.rasterize_bowl_weighted(&mut mask, grid, *position, *radius, *diameter);
                }
                ElementShape::Disc {
                    position,
                    diameter,
                    focus_position,
                } => {
                    self.rasterize_disc_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *diameter,
                        *focus_position,
                    );
                }
                ElementShape::ProfiledDisc {
                    position,
                    diameter,
                    focus_position,
                    profile,
                } => {
                    self.rasterize_profiled_disc_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *diameter,
                        *focus_position,
                        *profile,
                    );
                }
                ElementShape::Arc {
                    position,
                    radius,
                    diameter,
                    start_angle,
                    end_angle,
                } => {
                    self.rasterize_arc_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *diameter,
                        *start_angle,
                        *end_angle,
                    );
                }
                ElementShape::Rect {
                    position,
                    width,
                    height,
                    length,
                    euler_xyz_deg,
                } => {
                    let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                    self.rasterize_rect_weighted(
                        &mut mask, grid, pos_eff, *width, *height, *length, euler_eff,
                    );
                }
                ElementShape::Annulus {
                    position,
                    radius,
                    inner_diameter,
                    outer_diameter,
                } => {
                    self.rasterize_annulus_weighted(
                        &mut mask,
                        grid,
                        *position,
                        *radius,
                        *inner_diameter,
                        *outer_diameter,
                    );
                }
            }
        }
        mask
    }
}
