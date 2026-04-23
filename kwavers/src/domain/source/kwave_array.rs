//! KWaveArray - Custom transducer geometry builder for k-Wave compatibility
//!
//! This module provides a builder-pattern API for creating custom transducer arrays
//! with mixed geometries (arcs, discs, rectangles), matching k-wave-python's KWaveArray.

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use ndarray::{Array2, Array3};

const DISC_SAMPLE_UPSAMPLING_RATE: f64 = 10.0;
const DISC_PACKING_NUMBER: f64 = 7.0;
const DISC_BLI_TOLERANCE: f64 = 0.05;
const DISC_AXIS_EPSILON: f64 = 1.0e-12;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_5_f64;

/// Apodization window for per-element amplitude weighting
///
/// # Theorem — Window Functions (Harris 1978)
///
/// Window functions reduce sidelobes in the array beam pattern at the cost of broadening
/// the main lobe. For an N-element array with uniform spacing:
/// - Rectangular: unity weights → minimum main-lobe width, maximum sidelobes (-13 dB)
/// - Hann: `wᵢ = 0.5(1 - cos(2πi/(N-1)))` → -31 dB sidelobes, 1.5× wider main lobe
/// - Hamming: `wᵢ = 0.54 - 0.46·cos(2πi/(N-1)))` → -43 dB sidelobes
///
/// Reference: Harris, F.J. (1978). "On the use of windows for harmonic analysis with
/// the discrete Fourier transform." Proc. IEEE 66(1):51–83.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApodizationWindow {
    /// All weights = 1.0 (no apodization)
    Rectangular,
    /// Hann window — -31 dB sidelobes
    Hann,
    /// Hamming window — -43 dB sidelobes
    Hamming,
}

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
        /// Intrinsic X-Y-Z Euler rotation in degrees applied about the element
        /// center. `(0.0, 0.0, 0.0)` keeps the rectangle axis-aligned and
        /// dispatches the fast AABB rasterizer.
        euler_xyz_deg: (f64, f64, f64),
    },
    /// Disc/circular element
    Disc {
        /// Center position [x, y, z]
        position: (f64, f64, f64),
        /// Disc diameter [m]
        diameter: f64,
        /// Optional focus point defining the disc normal [x, y, z]
        focus_position: Option<(f64, f64, f64)>,
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
    _element_width: f64,
}

impl KWaveArray {
    /// Create a new empty KWaveArray
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            frequency: 1e6,
            sound_speed: SOUND_SPEED_TISSUE,
            _element_width: 0.5e-3,
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
            _element_width: 0.5e-3,
        }
    }

    /// Update the operating frequency while preserving existing elements.
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
    }

    /// Update the sound speed while preserving existing elements.
    pub fn set_sound_speed(&mut self, sound_speed: f64) {
        self.sound_speed = sound_speed;
    }

    /// Set the element width (for arc discretization)
    #[must_use]
    pub fn with_element_width(mut self, width: f64) -> Self {
        self._element_width = width;
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
            euler_xyz_deg: (0.0, 0.0, 0.0),
        });
        self
    }

    /// Add a rectangular element rotated about its center by intrinsic X-Y-Z
    /// Euler angles (degrees). Matches the upstream k-wave-python
    /// `KWaveArray.add_rect_element` rotation contract used by the linear
    /// array transducer example.
    pub fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) -> &mut Self {
        self.elements.push(ElementShape::Rect {
            position,
            width,
            height,
            length,
            euler_xyz_deg,
        });
        self
    }

    /// Add a disc-shaped element
    ///
    /// # Arguments
    /// * `position` - Center position [x, y, z] in meters
    /// * `diameter` - Disc diameter in meters
    pub fn add_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) -> &mut Self {
        self.elements.push(ElementShape::Disc {
            position,
            diameter,
            focus_position,
        });
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
                    euler_xyz_deg,
                } => {
                    self.rasterize_rect(
                        &mut mask,
                        grid,
                        *position,
                        *width,
                        *height,
                        *length,
                        *euler_xyz_deg,
                    );
                }
                ElementShape::Disc {
                    position,
                    diameter,
                    focus_position,
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

    /// Calculate per-element time delays [s] for electronic focusing at a point.
    ///
    /// # Algorithm — Time-Delay Focusing (Selfridge et al. 1980)
    ///
    /// For element `i` at position `pᵢ` and focus point `f`, the delay that causes
    /// all wavefronts to arrive at the focus simultaneously is:
    /// ```text
    /// τᵢ = (d_max - dᵢ) / c
    /// ```
    /// where `dᵢ = |pᵢ - f|` (element-to-focus distance) and `d_max = max(dᵢ)`.
    ///
    /// This ensures the element farthest from the focus fires first (zero delay) and
    /// elements closer to the focus are delayed so all signals add coherently at `f`.
    ///
    /// # Arguments
    /// * `focus_point` - Focus position `(x, y, z)` in metres
    ///
    /// # Returns
    /// `Vec<f64>` of delays in seconds, one per element. All values ≥ 0.
    ///
    /// # Reference
    /// Selfridge, A.R., Kino, G.S. & Khuri-Yakub, B.T. (1980). "A theory for the
    /// radiation pattern of a narrow-strip acoustic transducer." Appl. Phys. Lett.
    /// 37(1):35–36.
    #[must_use]
    pub fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        let positions = self.get_element_positions();
        let c = self.sound_speed;

        // Compute distance from each element to the focus point
        let distances: Vec<f64> = positions
            .iter()
            .map(|(ex, ey, ez)| {
                ((ex - focus_point.0).powi(2)
                    + (ey - focus_point.1).powi(2)
                    + (ez - focus_point.2).powi(2))
                .sqrt()
            })
            .collect();

        // Maximum distance — the element at d_max fires first (zero delay)
        let d_max = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        distances.iter().map(|&d| (d_max - d) / c).collect()
    }

    /// Compute per-element amplitude weights for beam apodization.
    ///
    /// # Algorithm — Discrete Window Functions (Harris 1978)
    ///
    /// For N elements indexed `i = 0, …, N-1`:
    /// - `Rectangular`: `wᵢ = 1.0`  (uniform, maximum sidelobes)
    /// - `Hann`:        `wᵢ = 0.5 · (1 − cos(2π·i/(N−1)))` if N > 1, else 1.0
    /// - `Hamming`:     `wᵢ = 0.54 − 0.46·cos(2π·i/(N−1))` if N > 1, else 1.0
    ///
    /// # Arguments
    /// * `window` - Window type selecting the apodization profile
    ///
    /// # Returns
    /// `Vec<f64>` of weights in `[0, 1]`, length = number of elements.
    ///
    /// # Reference
    /// Harris, F.J. (1978). "On the use of windows for harmonic analysis with the
    /// discrete Fourier transform." Proc. IEEE 66(1):51–83.
    #[must_use]
    pub fn get_apodization(&self, window: ApodizationWindow) -> Vec<f64> {
        let n = self.elements.len();
        if n == 0 {
            return Vec::new();
        }
        match window {
            ApodizationWindow::Rectangular => vec![1.0; n],
            ApodizationWindow::Hann => {
                if n == 1 {
                    return vec![1.0];
                }
                (0..n)
                    .map(|i| {
                        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
                    })
                    .collect()
            }
            ApodizationWindow::Hamming => {
                if n == 1 {
                    return vec![1.0];
                }
                (0..n)
                    .map(|i| {
                        0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                    })
                    .collect()
            }
        }
    }

    /// Rasterize an arc element onto the grid.
    ///
    /// # Algorithm - Line-Sampled Arc with BLI Stencil
    ///
    /// The arc is treated as a 1D curved line segment with geometric measure
    /// `L = r · |Δθ|`.  The segment is sampled at half-step offsets along the
    /// angular span and each sample is mapped through the same local BLI
    /// stencil as the disc and bowl elements.
    fn rasterize_arc(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        _diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_arc_points(
            grid,
            center,
            radius,
            start_angle,
            end_angle,
            |point, scale| {
                self.map_surface_sample(
                    grid,
                    &x_vec,
                    &y_vec,
                    &z_vec,
                    point,
                    scale,
                    true,
                    |ix, iy, iz, _| {
                        mask[[ix, iy, iz]] = true;
                    },
                );
            },
        );
    }

    /// Rasterize an arc element onto the grid with area-conserving weights.
    fn rasterize_arc_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        _diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_arc_points(
            grid,
            center,
            radius,
            start_angle,
            end_angle,
            |point, scale| {
                self.map_surface_sample(
                    grid,
                    &x_vec,
                    &y_vec,
                    &z_vec,
                    point,
                    scale,
                    false,
                    |ix, iy, iz, weight| {
                        mask[[ix, iy, iz]] += weight;
                    },
                );
            },
        );
    }

    /// Rasterize a rectangular element onto the grid.
    ///
    /// When `euler_xyz_deg == (0, 0, 0)` the rectangle is axis-aligned and we
    /// use the fast AABB marker. Otherwise we sample the rectangle surface on
    /// a sub-grid lattice (upsampled by `DISC_SAMPLE_UPSAMPLING_RATE`) and
    /// apply the intrinsic X-Y-Z rotation before marking the containing cell
    /// of each sample — matching the upstream k-wave-python rotated-rect
    /// contract used by `at_linear_array_transducer`.
    fn rasterize_rect(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        if euler_xyz_deg.0 == 0.0 && euler_xyz_deg.1 == 0.0 && euler_xyz_deg.2 == 0.0 {
            let x_min = ((center.0 - width / 2.0 - grid.origin[0]) / grid.dx).floor() as usize;
            let x_max = ((center.0 + width / 2.0 - grid.origin[0]) / grid.dx).ceil() as usize;
            let y_min = ((center.1 - height / 2.0 - grid.origin[1]) / grid.dy).floor() as usize;
            let y_max = ((center.1 + height / 2.0 - grid.origin[1]) / grid.dy).ceil() as usize;
            let z_min = ((center.2 - length / 2.0 - grid.origin[2]) / grid.dz).floor() as usize;
            let z_max = ((center.2 + length / 2.0 - grid.origin[2]) / grid.dz).ceil() as usize;

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
            return;
        }

        let rot = euler_xyz_rotation_matrix(euler_xyz_deg);
        let dmin = grid.dx.min(grid.dy).min(grid.dz);
        let sample = |extent: f64| {
            ((extent / dmin).ceil() * DISC_SAMPLE_UPSAMPLING_RATE)
                .max(1.0) as usize
                + 1
        };
        let nu = sample(width);
        let nv = sample(height);
        let nw = sample(length.max(dmin));

        let step = |n: usize, extent: f64| -> f64 {
            if n <= 1 {
                0.0
            } else {
                extent / (n - 1) as f64
            }
        };
        let du = step(nu, width);
        let dv = step(nv, height);
        let dw = step(nw, length);

        for iu in 0..nu {
            let lu = -width / 2.0 + du * iu as f64;
            for iv in 0..nv {
                let lv = -height / 2.0 + dv * iv as f64;
                for iw in 0..nw {
                    let lw = if nw <= 1 {
                        0.0
                    } else {
                        -length / 2.0 + dw * iw as f64
                    };
                    let (rx, ry, rz) = apply_matrix(&rot, (lu, lv, lw));
                    let gx = ((center.0 + rx - grid.origin[0]) / grid.dx).round();
                    let gy = ((center.1 + ry - grid.origin[1]) / grid.dy).round();
                    let gz = ((center.2 + rz - grid.origin[2]) / grid.dz).round();
                    if gx < 0.0 || gy < 0.0 || gz < 0.0 {
                        continue;
                    }
                    let ix = gx as usize;
                    let iy = gy as usize;
                    let iz = gz as usize;
                    if ix < nx && iy < ny && iz < nz {
                        mask[[ix, iy, iz]] = true;
                    }
                }
            }
        }
    }

    /// Rasterize a disc element onto the grid.
    ///
    /// # Algorithm - Oriented Disc Sampling
    ///
    /// Let `c` be the disc center and `f` an optional focus point. When `f`
    /// is provided, the unit normal is `n = (f - c) / ||f - c||`; otherwise
    /// the canonical disc normal is `e_z`. The disc is sampled on an
    /// orthonormal basis `(u, v)` spanning the plane orthogonal to `n`.
    ///
    /// Each sample point is
    ///
    /// `p(r, θ) = c + r cos(θ) u + r sin(θ) v`
    ///
    /// and contributes one unit of occupancy to the containing grid cell.
    /// The weighted variant uses the same sample set with a constant scale so
    /// the total discrete mass remains `A / dx²` up to roundoff.
    fn rasterize_disc(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_disc_points(grid, center, diameter, focus_position, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                true,
                |ix, iy, iz, _| {
                    mask[[ix, iy, iz]] = true;
                },
            );
        });
    }

    /// Rasterize a disc element onto the grid with area-conserving weights.
    fn rasterize_disc_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_disc_points(grid, center, diameter, focus_position, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                false,
                |ix, iy, iz, weight| {
                    mask[[ix, iy, iz]] += weight;
                },
            );
        });
    }

    fn rasterize_disc_points<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let radius = diameter / 2.0;
        let area = std::f64::consts::PI * radius * radius;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let num_radial =
            ((target_points as f64 / std::f64::consts::PI).sqrt().ceil() as usize).max(1);
        let num_points = self.disc_sample_count(num_radial);
        let scale = m_grid / num_points as f64;
        let (u, v, _) = self.disc_basis(center, focus_position);

        let mut emit = |x_local: f64, y_local: f64| {
            let point = [
                center.0 + x_local * u[0] + y_local * v[0],
                center.1 + x_local * u[1] + y_local * v[1],
                center.2 + x_local * u[2] + y_local * v[2],
            ];
            visit(point, scale);
        };

        emit(0.0, 0.0);
        if num_radial == 1 {
            return;
        }

        let radial_denom = (num_radial - 1) as f64;
        let radial_step = (radius - radius / (2.0 * radial_denom)) / radial_denom;

        for ring_idx in 1..num_radial {
            let ring_radius = ring_idx as f64 * radial_step;
            let num_theta = ((ring_idx as f64) * DISC_PACKING_NUMBER).round().max(1.0) as usize;
            for theta_idx in 0..num_theta {
                let theta = 2.0 * std::f64::consts::PI * theta_idx as f64 / num_theta as f64;
                emit(ring_radius * theta.cos(), ring_radius * theta.sin());
            }
        }
    }

    fn map_surface_sample<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        x_vec: &ndarray::Array1<f64>,
        y_vec: &ndarray::Array1<f64>,
        z_vec: &ndarray::Array1<f64>,
        point: [f64; 3],
        scale: f64,
        mask_only: bool,
        mut visit: F,
    ) where
        F: FnMut(usize, usize, usize, f64),
    {
        let decay_subs = (1.0 / (std::f64::consts::PI * DISC_BLI_TOLERANCE)).ceil() as isize;
        let ongrid_threshold = grid.dx * 1.0e-3;

        let (ix0, x_closest) = Self::nearest_coordinate_index(x_vec, point[0]);
        let (iy0, y_closest) = Self::nearest_coordinate_index(y_vec, point[1]);
        let (iz0, z_closest) = Self::nearest_coordinate_index(z_vec, point[2]);

        let x_on_grid = (x_closest - point[0]).abs() < ongrid_threshold;
        let y_on_grid = (y_closest - point[1]).abs() < ongrid_threshold;
        let z_on_grid = (z_closest - point[2]).abs() < ongrid_threshold;

        if grid.nz == 1 {
            let iz = iz0;
            for di in -decay_subs..=decay_subs {
                for dj in -decay_subs..=decay_subs {
                    if (di * dj).abs() > decay_subs {
                        continue;
                    }
                    if x_on_grid && di != 0 {
                        continue;
                    }
                    if y_on_grid && dj != 0 {
                        continue;
                    }

                    let ix = ix0 as isize + di;
                    let iy = iy0 as isize + dj;
                    if ix < 0 || iy < 0 || ix >= grid.nx as isize || iy >= grid.ny as isize {
                        continue;
                    }

                    let ix = ix as usize;
                    let iy = iy as usize;
                    let wx = Self::sinc(std::f64::consts::PI * (x_vec[ix] - point[0]) / grid.dx);
                    let wy = Self::sinc(std::f64::consts::PI * (y_vec[iy] - point[1]) / grid.dy);
                    let weight = scale * wx * wy;
                    if mask_only || weight != 0.0 {
                        visit(ix, iy, iz, if mask_only { 1.0 } else { weight });
                    }
                }
            }
            return;
        }

        for di in -decay_subs..=decay_subs {
            for dj in -decay_subs..=decay_subs {
                for dk in -decay_subs..=decay_subs {
                    if (di * dj * dk).abs() > decay_subs {
                        continue;
                    }
                    if x_on_grid && di != 0 {
                        continue;
                    }
                    if y_on_grid && dj != 0 {
                        continue;
                    }
                    if z_on_grid && dk != 0 {
                        continue;
                    }

                    let ix = ix0 as isize + di;
                    let iy = iy0 as isize + dj;
                    let iz = iz0 as isize + dk;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix >= grid.nx as isize
                        || iy >= grid.ny as isize
                        || iz >= grid.nz as isize
                    {
                        continue;
                    }

                    let ix = ix as usize;
                    let iy = iy as usize;
                    let iz = iz as usize;
                    let wx = Self::sinc(std::f64::consts::PI * (x_vec[ix] - point[0]) / grid.dx);
                    let wy = Self::sinc(std::f64::consts::PI * (y_vec[iy] - point[1]) / grid.dy);
                    let wz = Self::sinc(std::f64::consts::PI * (z_vec[iz] - point[2]) / grid.dz);
                    let weight = scale * wx * wy * wz;
                    if mask_only || weight != 0.0 {
                        visit(ix, iy, iz, if mask_only { 1.0 } else { weight });
                    }
                }
            }
        }
    }

    fn nearest_coordinate_index(coords: &ndarray::Array1<f64>, value: f64) -> (usize, f64) {
        let mut best_index = 0usize;
        let mut best_value = coords[0];
        let mut best_distance = (best_value - value).abs();

        for (index, coordinate) in coords.iter().enumerate().skip(1) {
            let distance = (*coordinate - value).abs();
            if distance < best_distance {
                best_index = index;
                best_value = *coordinate;
                best_distance = distance;
            }
        }

        (best_index, best_value)
    }

    #[inline]
    fn sinc(x: f64) -> f64 {
        if x.abs() <= f64::EPSILON {
            1.0
        } else {
            x.sin() / x
        }
    }

    fn disc_sample_count(&self, num_radial: usize) -> usize {
        if num_radial == 1 {
            return 1;
        }

        let mut num_points = 1usize;
        for ring_idx in 1..num_radial {
            let num_theta = ((ring_idx as f64) * DISC_PACKING_NUMBER).round().max(1.0) as usize;
            num_points += num_theta;
        }
        num_points
    }

    fn disc_basis(
        &self,
        center: (f64, f64, f64),
        focus_position: Option<(f64, f64, f64)>,
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let normal = if let Some((fx, fy, fz)) = focus_position {
            let nx = fx - center.0;
            let ny = fy - center.1;
            let nz = fz - center.2;
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                norm > DISC_AXIS_EPSILON,
                "focus position must differ from disc position"
            );
            [nx / norm, ny / norm, nz / norm]
        } else {
            [0.0, 0.0, 1.0]
        };

        let reference = if normal[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut u = [
            reference[1] * normal[2] - reference[2] * normal[1],
            reference[2] * normal[0] - reference[0] * normal[2],
            reference[0] * normal[1] - reference[1] * normal[0],
        ];
        let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        if u_norm <= DISC_AXIS_EPSILON {
            u = [1.0, 0.0, 0.0];
        } else {
            u[0] /= u_norm;
            u[1] /= u_norm;
            u[2] /= u_norm;
        }
        let v = [
            normal[1] * u[2] - normal[2] * u[1],
            normal[2] * u[0] - normal[0] * u[2],
            normal[0] * u[1] - normal[1] * u[0],
        ];

        (u, v, normal)
    }

    /// Compute the total geometric measure of all elements in the array.
    ///
    /// # Bowl formula (Treeby & Cox 2010)
    /// Spherical cap area = 2π·R·h  where  h = R − √(R²−(D/2)²)
    /// is the sagittal depth and D is the aperture diameter.
    ///
    /// Disc and bowl elements contribute an area, rectangles contribute an
    /// area, and arc elements contribute a line length.
    #[must_use]
    pub fn compute_total_surface_area(&self) -> f64 {
        let mut total = 0.0_f64;
        for element in &self.elements {
            total += match element {
                ElementShape::Bowl {
                    radius: r,
                    diameter: d,
                    ..
                } => {
                    let half_d = d / 2.0;
                    let h = if *r > half_d {
                        r - (r * r - half_d * half_d).sqrt()
                    } else {
                        *r
                    };
                    2.0 * std::f64::consts::PI * r * h
                }
                ElementShape::Disc { diameter: d, .. } => std::f64::consts::PI * (d / 2.0).powi(2),
                ElementShape::Rect {
                    width: w,
                    height: h,
                    ..
                } => w * h,
                ElementShape::Arc {
                    radius: r,
                    start_angle: s,
                    end_angle: e,
                    ..
                } => Self::arc_line_length(*r, *s, *e),
            };
        }
        total
    }

    /// Generate a float-weighted mask on the computational grid.
    ///
    /// Uses k-wave-compatible surface sampling and local BLI stencils to
    /// produce per-cell weights that sum to each element's geometric measure
    /// expressed in grid cells.
    ///
    /// # Algorithm (NGP-BLI equivalence)
    ///
    /// k-Wave's `get_distributed_source_signal` generates `m_integration` =
    /// `ceil(m_grid × upsampling_rate)` sample points on the transducer measure
    /// and maps each via BLI with `scale = m_grid / m_integration`.
    ///
    /// Disc and bowl elements both use deterministic surface samplers and the
    /// same local BLI stencil as k-wave-python (`bli_tolerance = 0.05`,
    /// `bli_type = "sinc"`). Rect elements remain binary masks with uniform
    /// unit weights per active cell. Arc elements use weighted line sampling
    /// with the same BLI stencil and arc-length normalization as the upstream
    /// k-wave-python `make_cart_arc` path.
    ///
    /// # Arguments
    /// * `grid` – Computational grid defining the domain.
    ///
    /// # Returns
    /// `Array3<f64>` where each element contributes its geometric measure in
    /// grid-cell units.
    pub fn get_array_weighted_mask(&self, grid: &crate::domain::grid::Grid) -> Array3<f64> {
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
                    let mut bool_layer = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
                    self.rasterize_rect(
                        &mut bool_layer,
                        grid,
                        *position,
                        *width,
                        *height,
                        *length,
                        *euler_xyz_deg,
                    );
                    ndarray::Zip::from(&mut mask)
                        .and(&bool_layer)
                        .for_each(|m, &b| {
                            if b {
                                *m += 1.0;
                            }
                        });
                }
            }
        }

        mask
    }

    /// Dense surface-sampling rasterization for a bowl element (float weights).
    ///
    /// # Algorithm - Canonical Spiral Bowl Sampling
    ///
    /// The spherical cap is sampled with the same golden-angle spiral used by
    /// k-wave-python's `make_cart_bowl`. The surface points are then mapped
    /// through the same local BLI stencil used for the disc element.
    fn rasterize_bowl_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_bowl_points(grid, center, radius, diameter, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                false,
                |ix, iy, iz, weight| {
                    mask[[ix, iy, iz]] += weight;
                },
            );
        });
    }

    /// Rasterize arc centerline points with half-step angular offsets.
    fn rasterize_arc_points<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let arc_length = Self::arc_line_length(radius, start_angle, end_angle);
        let m_grid = arc_length / grid.dx;
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;
        let angle_span = end_angle - start_angle;

        for idx in 0..num_points {
            let angle_deg = start_angle + angle_span * ((idx as f64 + 0.5) / num_points as f64);
            let angle_rad = angle_deg.to_radians();
            let point = [
                center.0 + radius * angle_rad.cos(),
                center.1 + radius * angle_rad.sin(),
                center.2,
            ];
            visit(point, scale);
        }
    }

    #[inline]
    fn arc_line_length(radius: f64, start_angle: f64, end_angle: f64) -> f64 {
        radius * (end_angle - start_angle).abs().to_radians()
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
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_bowl_points(grid, center, radius, diameter, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                true,
                |ix, iy, iz, _| {
                    mask[[ix, iy, iz]] = true;
                },
            );
        });
    }

    fn rasterize_bowl_points<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = Self::bowl_surface_area(radius, diameter);
        let m_grid = area / (grid.dx * grid.dx);
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;

        if num_points == 1 {
            visit([center.0 - radius, center.1, center.2], scale);
            return;
        }

        let half_aperture = diameter / 2.0;
        let varphi_max = (half_aperture / radius).clamp(-1.0, 1.0).asin();
        let denom = (num_points - 1) as f64;
        let spiral_scale = 2.0 * std::f64::consts::PI * (1.0 - varphi_max.cos());

        for t in 0..num_points {
            let theta = GOLDEN_ANGLE * t as f64;
            let varphi = (1.0 - spiral_scale * (t as f64) / (denom * 2.0 * std::f64::consts::PI))
                .clamp(-1.0, 1.0)
                .acos();
            let radial = radius * varphi.sin();
            let point = [
                center.0 - radius * varphi.cos(),
                center.1 + radial * theta.cos(),
                center.2 + radial * theta.sin(),
            ];
            visit(point, scale);
        }
    }

    #[inline]
    fn bowl_surface_area(radius: f64, diameter: f64) -> f64 {
        let half_aperture = diameter / 2.0;
        let h = if radius > half_aperture {
            radius - (radius * radius - half_aperture * half_aperture).sqrt()
        } else {
            radius
        };
        2.0 * std::f64::consts::PI * radius * h
    }
}

impl Default for KWaveArray {
    fn default() -> Self {
        Self::new()
    }
}

/// Build an intrinsic X-Y-Z Euler rotation matrix (Rz · Ry · Rx) from angles
/// given in degrees. Matches the upstream k-wave-python
/// `rotation.rotate_rotation_matrix` contract used by `KWaveArray`.
fn euler_xyz_rotation_matrix(euler_deg: (f64, f64, f64)) -> [[f64; 3]; 3] {
    let rx = euler_deg.0.to_radians();
    let ry = euler_deg.1.to_radians();
    let rz = euler_deg.2.to_radians();
    let (cx, sx) = (rx.cos(), rx.sin());
    let (cy, sy) = (ry.cos(), ry.sin());
    let (cz, sz) = (rz.cos(), rz.sin());

    let mx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let my = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let mz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    matmul(&matmul(&mz, &my), &mx)
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn apply_matrix(m: &[[f64; 3]; 3], v: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        m[0][0] * v.0 + m[0][1] * v.1 + m[0][2] * v.2,
        m[1][0] * v.0 + m[1][1] * v.1 + m[1][2] * v.2,
        m[2][0] * v.0 + m[2][1] * v.1 + m[2][2] * v.2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kwave_array_creation() {
        let mut array = KWaveArray::new();
        array.add_disc_element((0.0, 0.0, 0.0), 0.01, None);
        array.add_rect_element((0.01, 0.0, 0.0), 0.005, 0.005, 0.001);

        assert_eq!(array.num_elements(), 2);
    }

    #[test]
    fn test_kwave_array_binary_mask() {
        let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let mut array = KWaveArray::new();
        array.add_disc_element((0.016, 0.016, 0.016), 0.005, None);

        let mask = array.get_array_binary_mask(&grid);
        let active_count = mask.iter().filter(|&&v| v).count();
        assert!(active_count > 0);
    }

    #[test]
    fn test_kwave_array_disc_focus_mask_is_planar_and_matches_kwave_python_reference_mass() {
        let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let mut array = KWaveArray::new();
        array.add_disc_element((0.016, 0.016, 0.016), 0.006, Some((0.016, 0.016, 0.024)));

        let weights = array.get_array_weighted_mask(&grid);
        let expected = 28.302_387_208_098_168_f64;
        assert!((weights.sum() - expected).abs() < 5.0e-6);

        let mut active_plane: Option<usize> = None;
        for ((_, _, k), &value) in weights.indexed_iter() {
            if value > 0.0 {
                match active_plane {
                    Some(plane) => assert_eq!(plane, k, "disc weights must remain planar"),
                    None => active_plane = Some(k),
                }
            }
        }
        assert!(
            active_plane.is_some(),
            "disc weights must activate at least one cell"
        );
    }

    #[test]
    fn test_rect_rotation_90_swaps_width_and_height() {
        use crate::domain::grid::Grid;

        let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4)
            .expect("grid");
        let mut unrot = KWaveArray::new();
        unrot.add_rect_element(
            (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
            8.0e-4,
            2.0e-4,
            1.0e-4,
        );
        let unrot_mask = unrot.get_array_binary_mask(&grid);

        let mut rot = KWaveArray::new();
        rot.add_rect_rot_element(
            (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
            8.0e-4,
            2.0e-4,
            1.0e-4,
            (0.0, 0.0, 90.0),
        );
        let rot_mask = rot.get_array_binary_mask(&grid);

        let unrot_count: usize = unrot_mask.iter().filter(|&&b| b).count();
        let rot_count: usize = rot_mask.iter().filter(|&&b| b).count();
        assert!(
            unrot_count > 0 && rot_count > 0,
            "both masks must be non-empty: unrot={unrot_count}, rot={rot_count}",
        );

        let (nx, ny, _nz) = (grid.nx, grid.ny, grid.nz);
        let mut swapped_hits = 0usize;
        for i in 0..nx {
            for j in 0..ny {
                if unrot_mask[[i, j, 2]] {
                    let mirror_i = j;
                    let mirror_j = nx - 1 - i;
                    if mirror_i < nx && mirror_j < ny && rot_mask[[mirror_i, mirror_j, 2]] {
                        swapped_hits += 1;
                    }
                }
            }
        }
        assert!(
            swapped_hits >= unrot_count / 2,
            "90-deg Z rotation must overlap after axis swap ({swapped_hits}/{unrot_count})",
        );
    }

    #[test]
    fn test_kwave_array_setters_preserve_elements() {
        let mut array = KWaveArray::new();
        array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);

        array.set_frequency(2.0e6);
        array.set_sound_speed(1600.0);

        assert_eq!(array.num_elements(), 1);
        assert!((array.frequency() - 2.0e6).abs() < 1.0e-12);

        let delays = array.get_focus_delays((0.0, 0.0, 1.0));
        assert_eq!(delays.len(), 1);
        assert!((delays[0] - 1.0 / 1600.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_focus_delays() {
        let mut array = KWaveArray::with_params(1e6, 1500.0);
        array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
        array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);

        let delays = array.get_focus_delays((0.005, 0.0, 0.02));
        assert_eq!(delays.len(), 2);
        assert!(delays[0] > 0.0);
        assert!(delays[1] > 0.0);
    }

    /// `get_element_delays` and `get_focus_delays` agree for a symmetric two-element array.
    ///
    /// For two elements equidistant from the focus, `get_focus_delays` returns (d/c, d/c) and
    /// `get_element_delays` returns (0, 0) — both elements fire simultaneously.
    #[test]
    fn test_get_element_delays_symmetric_array() {
        let mut array = KWaveArray::with_params(1e6, 1500.0);
        // Two elements at ±5 mm on x-axis, focus on z-axis
        array.add_disc_element((-0.005, 0.0, 0.0), 0.002, None);
        array.add_disc_element((0.005, 0.0, 0.0), 0.002, None);

        let focus = (0.0, 0.0, 0.02);
        let delays = array.get_element_delays(focus);
        assert_eq!(delays.len(), 2);
        // Symmetric array → both elements at equal distance → both delays = 0
        assert!(
            delays[0].abs() < 1e-12 && delays[1].abs() < 1e-12,
            "symmetric elements should have equal (zero) delays: {:?}",
            delays
        );
    }

    /// `get_element_delays` returns non-negative delays and the minimum delay is 0.
    #[test]
    fn test_get_element_delays_non_negative_min_zero() {
        let mut array = KWaveArray::with_params(1e6, 1500.0);
        array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
        array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);
        array.add_disc_element((0.02, 0.0, 0.0), 0.005, None);

        let focus = (0.01, 0.0, 0.03);
        let delays = array.get_element_delays(focus);
        assert_eq!(delays.len(), 3);
        for &d in &delays {
            assert!(d >= 0.0, "all delays must be non-negative");
        }
        let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_delay.abs() < 1e-15, "minimum delay must be 0");
    }

    /// Rectangular apodization returns all-ones vector.
    #[test]
    fn test_apodization_rectangular_all_ones() {
        let mut array = KWaveArray::new();
        for i in 0..8 {
            array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
        }
        let weights = array.get_apodization(ApodizationWindow::Rectangular);
        assert_eq!(weights.len(), 8);
        for w in &weights {
            assert!((w - 1.0).abs() < 1e-15, "rectangular weight must be 1.0");
        }
    }

    /// Hann window: first and last weights ≈ 0, central weight ≈ 1.
    #[test]
    fn test_apodization_hann_endpoints_near_zero() {
        let mut array = KWaveArray::new();
        for i in 0..9 {
            array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
        }
        let weights = array.get_apodization(ApodizationWindow::Hann);
        assert_eq!(weights.len(), 9);
        // Endpoints must be 0 (Hann formula gives exactly 0 at i=0 and i=N-1)
        assert!(
            weights[0].abs() < 1e-12,
            "Hann first weight must be ~0, got {}",
            weights[0]
        );
        assert!(
            weights[8].abs() < 1e-12,
            "Hann last weight must be ~0, got {}",
            weights[8]
        );
        // Central weight at i=4 of 9: w = 0.5*(1 - cos(π)) = 1.0
        assert!(
            (weights[4] - 1.0).abs() < 1e-12,
            "Hann center weight must be 1.0, got {}",
            weights[4]
        );
    }

    /// Hamming window: all weights in [0.08, 1.0] and symmetric.
    #[test]
    fn test_apodization_hamming_range_and_symmetry() {
        let mut array = KWaveArray::new();
        for i in 0..7 {
            array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
        }
        let weights = array.get_apodization(ApodizationWindow::Hamming);
        assert_eq!(weights.len(), 7);
        for &w in &weights {
            // Hamming minimum = 0.54 - 0.46 = 0.08
            assert!(
                (0.07..=1.01).contains(&w),
                "Hamming weight out of range: {}",
                w
            );
        }
        // Symmetric: w[i] ≈ w[N-1-i]
        for i in 0..7 {
            assert!(
                (weights[i] - weights[6 - i]).abs() < 1e-12,
                "Hamming not symmetric at i={}: w[{}]={} w[{}]={}",
                i,
                i,
                weights[i],
                6 - i,
                weights[6 - i]
            );
        }
    }

    /// Single-element array: all windows return [1.0].
    #[test]
    fn test_apodization_single_element() {
        let mut array = KWaveArray::new();
        array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
        for window in [
            ApodizationWindow::Rectangular,
            ApodizationWindow::Hann,
            ApodizationWindow::Hamming,
        ] {
            let weights = array.get_apodization(window);
            assert_eq!(weights.len(), 1);
            assert!(
                (weights[0] - 1.0).abs() < 1e-12,
                "{:?}: single element weight must be 1.0",
                window
            );
        }
    }

    /// SSOT: `KWaveArray::new()` uses the `SOUND_SPEED_TISSUE` constant.
    #[test]
    fn test_default_sound_speed_is_ssot_constant() {
        use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
        // Focus delays use the stored sound speed; a unit-distance element yields
        // delay = d/c = 1 m / c, giving direct access to the default sound speed.
        let mut arr = KWaveArray::new();
        arr.add_disc_element((0.0, 0.0, 0.0), 0.001, None);
        let delays = arr.get_focus_delays((0.0, 0.0, 1.0));
        // d = 1 m → delay = 1 / sound_speed
        assert!(
            (delays[0] - 1.0 / SOUND_SPEED_TISSUE).abs() < 1e-10,
            "default sound speed must equal SOUND_SPEED_TISSUE"
        );
    }
}
