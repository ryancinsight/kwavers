//! KWaveArray - Custom transducer geometry builder for k-Wave compatibility
//!
//! This module provides a builder-pattern API for creating custom transducer arrays
//! with mixed geometries (arcs, discs, rectangles), matching k-wave-python's KWaveArray.

use ndarray::{Array2, Array3};

/// Element shape types for custom transducer arrays
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementShape {
    /// Arc-shaped element (curved line in 3D)
    Arc {
        /// Arc center position [x, y, z]
        position: (f64, f64, f64),
        /// Arc radius [m]
        radius: f64,
        /// Arc diameter [m]
        diameter: f64,
        /// Start angle [degrees]
        start_angle: f64,
        /// End angle [degrees]
        end_angle: f64,
    },
    /// Rectangular element
    Rect {
        /// Center position [x, y, z]
        position: (f64, f64, f64),
        /// Width in x [m]
        width: f64,
        /// Height in y [m]
        height: f64,
        /// Length in z [m]
        length: f64,
    },
    /// Disc/circular element
    Disc {
        /// Center position [x, y, z]
        position: (f64, f64, f64),
        /// Disc diameter [m]
        diameter: f64,
    },
    /// Bowl/spherical cap element
    Bowl {
        /// Bowl center position [x, y, z]
        position: (f64, f64, f64),
        /// Bowl radius of curvature [m]
        radius: f64,
        /// Bowl diameter [m]
        diameter: f64,
    },
}

/// Custom transducer array with mixed element geometries
///
/// This type allows building arbitrary transducer arrays by adding elements
/// of different shapes, matching k-wave-python's KWaveArray functionality.
#[derive(Debug, Clone)]
pub struct KWaveArray {
    /// Collection of element shapes
    elements: Vec<ElementShape>,
    /// Operating frequency [Hz]
    frequency: f64,
    /// Sound speed [m/s]
    sound_speed: f64,
    /// Element width for arc elements [m]
    element_width: f64,
}

impl KWaveArray {
    /// Create a new empty KWaveArray
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            frequency: 1e6,
            sound_speed: 1540.0,
            element_width: 0.5e-3,
        }
    }

    /// Get the operating frequency [Hz]
    #[must_use]
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Create a new array with specified frequency and sound speed
    #[must_use]
    pub fn with_params(frequency: f64, sound_speed: f64) -> Self {
        Self {
            elements: Vec::new(),
            frequency,
            sound_speed,
            element_width: 0.5e-3,
        }
    }

    /// Set the element width (for arc discretization)
    #[must_use]
    pub fn with_element_width(mut self, width: f64) -> Self {
        self.element_width = width;
        self
    }

    /// Add an arc-shaped element
    ///
    /// # Arguments
    /// * `position` - Arc center [x, y, z] in meters
    /// * `radius` - Arc radius in meters
    /// * `diameter` - Element diameter in meters
    pub fn add_arc_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Arc {
            position,
            radius,
            diameter,
            start_angle: -45.0,
            end_angle: 45.0,
        });
        self
    }

    /// Add an arc-shaped element with custom angles
    ///
    /// # Arguments
    /// * `position` - Arc center [x, y, z] in meters
    /// * `radius` - Arc radius in meters
    /// * `diameter` - Element diameter in meters
    /// * `start_angle` - Start angle in degrees
    /// * `end_angle` - End angle in degrees
    pub fn add_arc_element_with_angles(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Arc {
            position,
            radius,
            diameter,
            start_angle,
            end_angle,
        });
        self
    }

    /// Add a rectangular element
    ///
    /// # Arguments
    /// * `position` - Center position [x, y, z] in meters
    /// * `width` - Width in x-direction [m]
    /// * `height` - Height in y-direction [m]
    /// * `length` - Length in z-direction [m]
    pub fn add_rect_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Rect {
            position,
            width,
            height,
            length,
        });
        self
    }

    /// Add a disc-shaped element
    ///
    /// # Arguments
    /// * `position` - Center position [x, y, z] in meters
    /// * `diameter` - Disc diameter in meters
    pub fn add_disc_element(&mut self, position: (f64, f64, f64), diameter: f64) -> &mut Self {
        self.elements
            .push(ElementShape::Disc { position, diameter });
        self
    }

    /// Add a bowl-shaped element (focused transducer)
    ///
    /// # Arguments
    /// * `position` - Bowl center position [x, y, z] in meters
    /// * `radius` - Radius of curvature [m]
    /// * `diameter` - Bowl aperture diameter [m]
    pub fn add_bowl_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Bowl {
            position,
            radius,
            diameter,
        });
        self
    }

    /// Get the number of elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Get all element positions (centroids)
    #[must_use]
    pub fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .map(|e| match e {
                ElementShape::Arc { position, .. } => *position,
                ElementShape::Rect { position, .. } => *position,
                ElementShape::Disc { position, .. } => *position,
                ElementShape::Bowl { position, .. } => *position,
            })
            .collect()
    }

    /// Generate binary mask on computational grid
    ///
    /// Returns a 3D boolean array where true indicates grid points
    /// that are part of the transducer array.
    ///
    /// # Arguments
    /// * `grid` - Computational grid defining the domain
    pub fn get_array_binary_mask(&self, grid: &crate::domain::grid::Grid) -> Array3<bool> {
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
                } => {
                    self.rasterize_rect(&mut mask, grid, *position, *width, *height, *length);
                }
                ElementShape::Disc { position, diameter } => {
                    self.rasterize_disc(&mut mask, grid, *position, *diameter);
                }
                ElementShape::Bowl {
                    position,
                    radius,
                    diameter,
                } => {
                    self.rasterize_bowl(&mut mask, grid, *position, *radius, *diameter);
                }
            }
        }

        mask
    }

    /// Generate distributed source signal matrix
    ///
    /// Returns a 2D array [n_elements, time_steps] where each row
    /// is the time signal for one element.
    ///
    /// # Arguments
    /// * `signal` - Input time signal (1D array)
    pub fn get_distributed_source_signal(&self, signal: &ndarray::Array1<f64>) -> Array2<f64> {
        let n_elements = self.elements.len();
        let n_times = signal.len();

        // Broadcast signal to all elements
        let mut distributed = Array2::zeros((n_elements, n_times));
        for i in 0..n_elements {
            for t in 0..n_times {
                distributed[[i, t]] = signal[t];
            }
        }

        distributed
    }

    /// Calculate focus delays for electronic focusing
    ///
    /// Returns time delays [s] for each element to focus at a point.
    ///
    /// # Arguments
    /// * `focus_point` - Focus position [x, y, z] in meters
    #[must_use]
    pub fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let positions = self.get_element_positions();
        let c = self.sound_speed;

        positions
            .iter()
            .map(|(ex, ey, ez)| {
                let dist = ((ex - focus_point.0).powi(2)
                    + (ey - focus_point.1).powi(2)
                    + (ez - focus_point.2).powi(2))
                .sqrt();
                dist / c
            })
            .collect()
    }

    /// Rasterize an arc element onto the grid
    fn rasterize_arc(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        let nx = grid.nx as isize;
        let ny = grid.ny as isize;
        let nz = grid.nz as isize;
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;

        // Discretize arc into points
        let num_points =
            ((end_angle - start_angle).abs() * radius / self.element_width).ceil() as usize;
        let num_points = num_points.max(10);

        for i in 0..=num_points {
            let angle_deg =
                start_angle + (end_angle - start_angle) * (i as f64 / num_points as f64);
            let angle_rad = angle_deg.to_radians();

            // Calculate arc point
            let ax = center.0 + radius * angle_rad.cos();
            let ay = center.1 + radius * angle_rad.sin();
            let az = center.2;

            // Mark grid cells within element diameter
            let radius_cells = (diameter / 2.0 / dx).ceil() as isize;

            let ix = ((ax - grid.origin[0]) / dx).round() as isize;
            let iy = ((ay - grid.origin[1]) / dy).round() as isize;
            let iz = ((az - grid.origin[2]) / dz).round() as isize;

            for di in -radius_cells..=radius_cells {
                for dj in -radius_cells..=radius_cells {
                    for dk in -radius_cells..=radius_cells {
                        let xi = ix + di;
                        let yi = iy + dj;
                        let zi = iz + dk;

                        if xi >= 0 && xi < nx && yi >= 0 && yi < ny && zi >= 0 && zi < nz {
                            // Check if within circular element area
                            let dist = ((di as f64 * dx).powi(2)
                                + (dj as f64 * dy).powi(2)
                                + (dk as f64 * dz).powi(2))
                            .sqrt();
                            if dist <= diameter / 2.0 {
                                mask[[xi as usize, yi as usize, zi as usize]] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rasterize a rectangular element onto the grid
    fn rasterize_rect(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
    ) {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        // Calculate grid bounds for rectangle
        let x_min = ((center.0 - width / 2.0 - grid.origin[0]) / grid.dx).floor() as usize;
        let x_max = ((center.0 + width / 2.0 - grid.origin[0]) / grid.dx).ceil() as usize;
        let y_min = ((center.1 - height / 2.0 - grid.origin[1]) / grid.dy).floor() as usize;
        let y_max = ((center.1 + height / 2.0 - grid.origin[1]) / grid.dy).ceil() as usize;
        let z_min = ((center.2 - length / 2.0 - grid.origin[2]) / grid.dz).floor() as usize;
        let z_max = ((center.2 + length / 2.0 - grid.origin[2]) / grid.dz).ceil() as usize;

        // Clamp to grid bounds
        let x_min = x_min.min(nx);
        let x_max = x_max.min(nx);
        let y_min = y_min.min(ny);
        let y_max = y_max.min(ny);
        let z_min = z_min.min(nz);
        let z_max = z_max.min(nz);

        for i in x_min..x_max {
            for j in y_min..y_max {
                for k in z_min..z_max {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    /// Rasterize a disc element onto the grid
    fn rasterize_disc(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
    ) {
        let nx = grid.nx as isize;
        let ny = grid.ny as isize;
        let nz = grid.nz as isize;

        let ix = ((center.0 - grid.origin[0]) / grid.dx).round() as isize;
        let iy = ((center.1 - grid.origin[1]) / grid.dy).round() as isize;
        let iz = ((center.2 - grid.origin[2]) / grid.dz).round() as isize;

        let radius_cells = (diameter / 2.0 / grid.dx).ceil() as isize;

        for di in -radius_cells..=radius_cells {
            for dj in -radius_cells..=radius_cells {
                for dk in -radius_cells..=radius_cells {
                    let xi = ix + di;
                    let yi = iy + dj;
                    let zi = iz + dk;

                    if xi >= 0 && xi < nx && yi >= 0 && yi < ny && zi >= 0 && zi < nz {
                        let dist = ((di as f64 * grid.dx).powi(2)
                            + (dj as f64 * grid.dy).powi(2)
                            + (dk as f64 * grid.dz).powi(2))
                        .sqrt();
                        if dist <= diameter / 2.0 {
                            mask[[xi as usize, yi as usize, zi as usize]] = true;
                        }
                    }
                }
            }
        }
    }

    /// Rasterize a bowl element onto the grid
    ///
    /// Uses the radius of curvature to determine which grid points lie on
    /// the bowl surface (spherical cap with given diameter aperture).
    fn rasterize_bowl(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) {
        let half_aperture = diameter / 2.0;
        let nx = grid.nx as isize;
        let ny = grid.ny as isize;
        let nz = grid.nz as isize;

        // Bowl is a spherical cap: points at distance `radius` from the
        // geometric focus, within the aperture diameter
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = grid.origin[0] + (ix as f64 + 0.5) * grid.dx;
                    let y = grid.origin[1] + (iy as f64 + 0.5) * grid.dy;
                    let z = grid.origin[2] + (iz as f64 + 0.5) * grid.dz;

                    let dx_c = x - center.0;
                    let dy_c = y - center.1;
                    let dz_c = z - center.2;
                    let lateral = (dy_c * dy_c + dz_c * dz_c).sqrt();

                    // Check if within aperture
                    if lateral > half_aperture {
                        continue;
                    }

                    // Distance from this grid point to the bowl center
                    let dist = (dx_c * dx_c + dy_c * dy_c + dz_c * dz_c).sqrt();

                    // Point is on the bowl if its distance from the center
                    // equals the radius (within half a voxel diagonal)
                    let voxel_diag =
                        (grid.dx * grid.dx + grid.dy * grid.dy + grid.dz * grid.dz).sqrt();
                    if (dist - radius).abs() < voxel_diag {
                        mask[[ix as usize, iy as usize, iz as usize]] = true;
                    }
                }
            }
        }
    }
}

impl Default for KWaveArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kwave_array_creation() {
        let mut array = KWaveArray::new();
        array.add_disc_element((0.0, 0.0, 0.0), 0.01);
        array.add_rect_element((0.01, 0.0, 0.0), 0.005, 0.005, 0.001);

        assert_eq!(array.num_elements(), 2);
    }

    #[test]
    fn test_kwave_array_binary_mask() {
        let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let mut array = KWaveArray::new();
        array.add_disc_element((0.016, 0.016, 0.016), 0.005);

        let mask = array.get_array_binary_mask(&grid);
        let active_count = mask.iter().filter(|&&v| v).count();
        assert!(active_count > 0);
    }

    #[test]
    fn test_focus_delays() {
        let mut array = KWaveArray::with_params(1e6, 1500.0);
        array.add_disc_element((0.0, 0.0, 0.0), 0.005);
        array.add_disc_element((0.01, 0.0, 0.0), 0.005);

        let delays = array.get_focus_delays((0.005, 0.0, 0.02));
        assert_eq!(delays.len(), 2);
        assert!(delays[0] > 0.0);
        assert!(delays[1] > 0.0);
    }
}
