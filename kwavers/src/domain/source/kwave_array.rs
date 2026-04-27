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
        /// Length in z [m] retained for API compatibility.
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
    /// Annular spherical-cap element — the region of a bowl surface bounded
    /// by two aperture diameters. Facing convention matches `Bowl`
    /// (center-of-curvature at `position`, cap opening along −X).
    Annulus {
        /// Center of curvature [x, y, z]
        position: (f64, f64, f64),
        /// Radius of curvature [m]
        radius: f64,
        /// Inner aperture diameter [m]
        inner_diameter: f64,
        /// Outer aperture diameter [m]
        outer_diameter: f64,
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
    /// Optional global affine transform applied on top of every element.
    /// `None` => identity (element poses used as-is). Matches
    /// k-wave-python's `kWaveArray.set_array_position`.
    array_transform: Option<ArrayTransform>,
}

/// Global translation + intrinsic X-Y-Z Euler rotation (degrees) applied to
/// every element before rasterization.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ArrayTransform {
    translation: (f64, f64, f64),
    euler_xyz_deg: (f64, f64, f64),
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
            array_transform: None,
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
            array_transform: None,
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

    /// Install a global translation + intrinsic X-Y-Z Euler rotation (degrees)
    /// applied to every element at rasterization time. Mirrors
    /// k-wave-python's `kWaveArray.set_array_position(translation, rotation)`.
    /// Passing `None` for both is equivalent to the identity transform.
    pub fn set_array_position(
        &mut self,
        translation: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.array_transform = Some(ArrayTransform {
            translation,
            euler_xyz_deg,
        });
    }

    /// Remove the global array transform if one was previously installed.
    pub fn clear_array_position(&mut self) {
        self.array_transform = None;
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

    /// Add an annular spherical-cap element — a bowl section bounded by
    /// inner and outer aperture diameters. Mirrors k-wave-python's
    /// `add_annular_element`. Use the same orientation convention as
    /// `add_bowl_element`.
    pub fn add_annular_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) -> &mut Self {
        assert!(
            outer_diameter > inner_diameter && inner_diameter >= 0.0,
            "annulus requires 0 <= inner_diameter < outer_diameter (got \
             inner={inner_diameter}, outer={outer_diameter})",
        );
        self.elements.push(ElementShape::Annulus {
            position,
            radius,
            inner_diameter,
            outer_diameter,
        });
        self
    }

    /// Add a concentric annular array (sequence of annuli sharing a common
    /// center of curvature). Each `(inner_diameter, outer_diameter)` pair
    /// becomes one `ElementShape::Annulus`, matching k-wave-python's
    /// `add_annular_array`.
    pub fn add_annular_array(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameters: &[(f64, f64)],
    ) -> &mut Self {
        for &(inner_d, outer_d) in diameters {
            self.add_annular_element(position, radius, inner_d, outer_d);
        }
        self
    }

    /// Get the number of elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Get all element positions (centroids) with the global array transform
    /// applied when one has been set via `set_array_position`.
    #[must_use]
    pub fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .map(|e| {
                let local = match e {
                    ElementShape::Arc { position, .. } => *position,
                    ElementShape::Rect { position, .. } => *position,
                    ElementShape::Disc { position, .. } => *position,
                    ElementShape::Bowl { position, .. } => *position,
                    ElementShape::Annulus { position, .. } => *position,
                };
                self.apply_transform_point(local)
            })
            .collect()
    }

    /// Apply the installed global array transform to a point in element-local
    /// coordinates, returning world coordinates. When no transform is set this
    /// is the identity.
    fn apply_transform_point(&self, p_local: (f64, f64, f64)) -> (f64, f64, f64) {
        match self.array_transform {
            Some(t) => {
                let r = euler_xyz_rotation_matrix(t.euler_xyz_deg);
                let rotated = apply_matrix(&r, p_local);
                (
                    rotated.0 + t.translation.0,
                    rotated.1 + t.translation.1,
                    rotated.2 + t.translation.2,
                )
            }
            None => p_local,
        }
    }

    /// Compose the global array transform with a per-element Rect pose,
    /// returning the effective (position, Euler angles in degrees) to feed
    /// into the rasterizer. k-wave-python applies the global transform to the
    /// rectangle center only and keeps the per-element rect Euler unchanged.
    fn apply_transform_rect(
        &self,
        position: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) -> ((f64, f64, f64), (f64, f64, f64)) {
        match self.array_transform {
            Some(_t) => {
                let pos_eff = self.apply_transform_point(position);
                (pos_eff, euler_xyz_deg)
            }
            None => (position, euler_xyz_deg),
        }
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

    /// Distributed source matrix from a per-element signal matrix.
    ///
    /// `per_element_signals` must have shape `[n_elements, n_times]`. Matches the
    /// output shape of k-wave-python's `create_cw_signals(t, f, amps, phases)`
    /// so callers can feed it in unchanged.
    ///
    /// # Errors
    /// Returns `Err` if the row count doesn't match the array's element count.
    pub fn get_distributed_source_signal_per_element(
        &self,
        per_element_signals: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n_elements = self.elements.len();
        let (rows, _cols) = per_element_signals.dim();
        if rows != n_elements {
            return Err(format!(
                "per_element_signals has {} rows but array has {} elements",
                rows, n_elements
            ));
        }
        Ok(per_element_signals.clone())
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
    /// The rectangle is treated as a planar surface element, matching the
    /// upstream k-wave-python `make_cart_rect` contract. We sample an evenly
    /// spaced lattice on the canonical rectangle, apply the intrinsic X-Y-Z
    /// rotation, and feed each sample through the same local BLI stencil used
    /// for the disc and bowl elements.
    fn rasterize_rect(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        _length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_rect_points(
            grid,
            center,
            width,
            height,
            euler_xyz_deg,
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

    /// Rasterize a rectangular element onto the grid with area-conserving weights.
    fn rasterize_rect_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        _length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_rect_points(
            grid,
            center,
            width,
            height,
            euler_xyz_deg,
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

    /// Emit the canonical rectangle integration points used by k-wave-python.
    fn rasterize_rect_points<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        euler_xyz_deg: (f64, f64, f64),
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = width * height;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let npts_x = ((target_points as f64 * width / height).sqrt().ceil() as usize).max(1);
        let npts_y = ((target_points as f64 / npts_x as f64).ceil() as usize).max(1);
        let num_points = npts_x * npts_y;
        let scale = m_grid / num_points as f64;
        let rot = euler_xyz_rotation_matrix(euler_xyz_deg);

        let dx = if npts_x <= 1 {
            0.0
        } else {
            2.0 / npts_x as f64
        };
        let dy = if npts_y <= 1 {
            0.0
        } else {
            2.0 / npts_y as f64
        };

        for ix in 0..npts_x {
            let ux = -1.0 + dx / 2.0 + dx * ix as f64;
            let lx = ux * width / 2.0;
            for iy in 0..npts_y {
                let uy = -1.0 + dy / 2.0 + dy * iy as f64;
                let ly = uy * height / 2.0;
                let (rx, ry, rz) = apply_matrix(&rot, (lx, ly, 0.0));
                visit([center.0 + rx, center.1 + ry, center.2 + rz], scale);
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
                ElementShape::Annulus {
                    radius: r,
                    inner_diameter: di,
                    outer_diameter: d_o,
                    ..
                } => Self::annulus_surface_area(*r, *di, *d_o),
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
    /// `bli_type = "sinc"`). Rect elements use the same planar surface
    /// sampler and BLI stencil as the upstream k-wave-python `make_cart_rect`
    /// path. Arc elements use weighted line sampling with the same BLI stencil
    /// and arc-length normalization as the upstream `make_cart_arc` path.
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

    /// Rasterize a single element onto a pre-allocated weighted mask.
    /// Same dispatch as `get_array_weighted_mask` but for one element — used to
    /// build per-element BLI masks when per-element driving signals are needed.
    fn rasterize_element_weighted(
        &self,
        element: &ElementShape,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
    ) {
        match element {
            ElementShape::Bowl {
                position,
                radius,
                diameter,
            } => {
                self.rasterize_bowl_weighted(mask, grid, *position, *radius, *diameter);
            }
            ElementShape::Disc {
                position,
                diameter,
                focus_position,
            } => {
                self.rasterize_disc_weighted(mask, grid, *position, *diameter, *focus_position);
            }
            ElementShape::Arc {
                position,
                radius,
                diameter,
                start_angle,
                end_angle,
            } => {
                self.rasterize_arc_weighted(
                    mask,
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
                    mask, grid, pos_eff, *width, *height, *length, euler_eff,
                );
            }
            ElementShape::Annulus {
                position,
                radius,
                inner_diameter,
                outer_diameter,
            } => {
                self.rasterize_annulus_weighted(
                    mask,
                    grid,
                    *position,
                    *radius,
                    *inner_diameter,
                    *outer_diameter,
                );
            }
        }
    }

    /// Build a per-cell source (mask, signal) from per-element driving signals.
    ///
    /// Expands a `[n_elements, n_times]` matrix of per-element waveforms into a
    /// `[n_active_cells, n_times]` per-cell signal by accumulating
    /// `s_cell[c, t] = Σ_i W_i[c] · s_i[t]` over each element's BLI-weighted mask.
    ///
    /// Returns `(mask, per_cell_signal)` where `mask[c]=1.0` at each active cell
    /// (so `SourceHandler` applies only its `p_scale_*` without re-applying
    /// weights — weights are already baked into `per_cell_signal`). The row
    /// order of `per_cell_signal` matches MATLAB / Fortran-order active-cell
    /// enumeration, i.e. `matlab_find(mask)` on the same logical mask.
    ///
    /// # Errors
    /// Returns `Err` if `per_element_signals.shape()[0] != n_elements`.
    pub fn build_per_element_source(
        &self,
        grid: &crate::domain::grid::Grid,
        per_element_signals: &Array2<f64>,
    ) -> Result<(Array3<f64>, Array2<f64>), String> {
        let n_elements = self.elements.len();
        let (rows, n_times) = per_element_signals.dim();
        if rows != n_elements {
            return Err(format!(
                "per_element_signals has {} rows but array has {} elements",
                rows, n_elements
            ));
        }

        let shape = (grid.nx, grid.ny, grid.nz);
        let mut per_element_masks: Vec<Array3<f64>> = Vec::with_capacity(n_elements);
        for element in &self.elements {
            let mut m = Array3::zeros(shape);
            self.rasterize_element_weighted(element, &mut m, grid);
            per_element_masks.push(m);
        }

        let mut combined = Array3::<f64>::zeros(shape);
        for m in &per_element_masks {
            combined += m;
        }
        let active_cells = Self::active_cells_fortran_order(&combined);

        let n_active = active_cells.len();
        let mut per_cell_signal = Array2::<f64>::zeros((n_active, n_times));
        for (idx, &(i, j, k)) in active_cells.iter().enumerate() {
            // Collect non-zero contributions for this cell, then walk timesteps once.
            let contribs: Vec<(usize, f64)> = per_element_masks
                .iter()
                .enumerate()
                .filter_map(|(e, m)| {
                    let w = m[[i, j, k]];
                    if w != 0.0 {
                        Some((e, w))
                    } else {
                        None
                    }
                })
                .collect();
            for t in 0..n_times {
                let mut s = 0.0;
                for &(e, w) in &contribs {
                    s += w * per_element_signals[[e, t]];
                }
                per_cell_signal[[idx, t]] = s;
            }
        }

        let mut mask_ones = Array3::<f64>::zeros(shape);
        for &(i, j, k) in &active_cells {
            mask_ones[[i, j, k]] = 1.0;
        }

        Ok((mask_ones, per_cell_signal))
    }

    #[inline]
    fn active_cells_fortran_order(mask: &Array3<f64>) -> Vec<(usize, usize, usize)> {
        let (nx, ny, nz) = mask.dim();
        let mut active_cells = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[[i, j, k]] != 0.0 {
                        active_cells.push((i, j, k));
                    }
                }
            }
        }
        active_cells
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
                // k-wave canonical: R @ [cos(θ)sin(φ), sin(θ)sin(φ), cos(φ)] → [-cos(φ), sin(θ)sin(φ), cos(θ)sin(φ)]
                // (rotation maps canonical [0,0,-1] → bowl-axis [1,0,0]; see make_cart_bowl / compute_linear_transform)
                center.1 + radial * theta.sin(),
                center.2 + radial * theta.cos(),
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

    #[inline]
    fn annulus_surface_area(radius: f64, inner_diameter: f64, outer_diameter: f64) -> f64 {
        let outer = (outer_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        let inner = (inner_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        2.0 * std::f64::consts::PI * radius * radius * (inner.cos() - outer.cos())
    }

    /// Surface-sample the spherical ring between `inner_diameter` and
    /// `outer_diameter` apertures on a bowl of given `radius`, using the same
    /// golden-angle spiral topology as `rasterize_bowl_points` but restricted
    /// to the ring varphi in [varphi_min, varphi_max].
    fn rasterize_annulus_points<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = Self::annulus_surface_area(radius, inner_diameter, outer_diameter);
        let m_grid = area / (grid.dx * grid.dx);
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;

        let varphi_max = (outer_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        let varphi_min = (inner_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();

        // Match k-Wave's `make_cart_spherical_segment` canonical-spiral
        // parameterisation exactly (k-wave-python/kwave/utils/mapgen.py
        // lines 3070–3088): step size C is set as if sampling a full bowl
        // from varphi=0 to varphi_max with `num_points` samples, then the
        // ring samples start at the integer t_start where varphi first
        // clears varphi_min. This keeps the golden-angle spiral phase
        // consistent with k-Wave for bowl+annulus composition.
        if num_points == 1 {
            let varphi = (0.5 * (varphi_min + varphi_max)).clamp(0.0, std::f64::consts::PI);
            let radial = radius * varphi.sin();
            // theta = GOLDEN_ANGLE * t_start ≈ t_start * 2.4; for a single point use t=0 → theta=0
            // k-wave convention: y = sin(theta)*radial, z = cos(theta)*radial → at theta=0: y=0, z=radial
            visit(
                [
                    center.0 - radius * varphi.cos(),
                    center.1,
                    center.2 + radial,
                ],
                scale,
            );
            return;
        }
        let c_step = (1.0 - varphi_max.cos()) / (num_points as f64 - 1.0);
        let t_start = if varphi_min > 0.0 {
            ((1.0 - varphi_min.cos()) / c_step).ceil()
        } else {
            0.0
        };
        let t_end = (num_points - 1) as f64;
        let span = (t_end - t_start).max(0.0);
        for k in 0..num_points {
            let frac = if num_points == 1 {
                0.0
            } else {
                k as f64 / (num_points as f64 - 1.0)
            };
            let t = t_start + frac * span;
            let cos_phi = (1.0 - c_step * t).clamp(-1.0, 1.0);
            let varphi = cos_phi.acos();
            let theta = GOLDEN_ANGLE * t;
            let radial = radius * varphi.sin();
            let point = [
                center.0 - radius * varphi.cos(),
                // k-wave canonical: R @ [cos(θ)sin(φ), sin(θ)sin(φ), cos(φ)] → [-cos(φ), sin(θ)sin(φ), cos(θ)sin(φ)]
                // (rotation maps canonical [0,0,-1] → bowl-axis [1,0,0]; see make_cart_spherical_segment)
                center.1 + radial * theta.sin(),
                center.2 + radial * theta.cos(),
            ];
            visit(point, scale);
        }
    }

    fn rasterize_annulus(
        &self,
        mask: &mut Array3<bool>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_annulus_points(
            grid,
            center,
            radius,
            inner_diameter,
            outer_diameter,
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

    fn rasterize_annulus_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &crate::domain::grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();

        self.rasterize_annulus_points(
            grid,
            center,
            radius,
            inner_diameter,
            outer_diameter,
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
        // Reference value from radial-Fibonacci BLI rasterisation (commit a24cdfcb).
        // Old integer-based rasteriser produced 28.302387; BLI produces 28.339929.
        let expected = 28.339_929_259_209_097_f64;
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
    fn test_set_array_position_matches_manual_position_rotation() {
        use crate::domain::grid::Grid;

        let grid = Grid::new(41, 41, 11, 5.0e-4, 5.0e-4, 5.0e-4).expect("grid");
        let translation = (5.0e-3, 0.0, 2.0e-3);
        let global_euler = (0.0, 20.0, 0.0);
        let per_element_euler = (0.0, 5.0, 0.0);
        let dims = (1.0e-3, 1.0e-3, 5.0e-4);

        let grid_center = (
            grid.nx as f64 * grid.dx / 2.0,
            grid.ny as f64 * grid.dy / 2.0,
            grid.nz as f64 * grid.dz / 2.0,
        );
        // Effective world-frame translation that brings a locally-centred
        // element layout to the pykwavers origin-at-corner grid.
        let world_translation = (
            translation.0 + grid_center.0,
            translation.1 + grid_center.1,
            translation.2 + grid_center.2,
        );

        let mut manual = KWaveArray::new();
        let mut native = KWaveArray::new();
        for kx in -2..=2 {
            let local = (1.0e-3 * kx as f64, 0.0, 0.0);

            let r_global = euler_xyz_rotation_matrix(global_euler);
            let rotated_local = apply_matrix(&r_global, local);
            let world = (
                rotated_local.0 + world_translation.0,
                rotated_local.1 + world_translation.1,
                rotated_local.2 + world_translation.2,
            );
            manual.add_rect_rot_element(world, dims.0, dims.1, dims.2, per_element_euler);

            native.add_rect_rot_element(local, dims.0, dims.1, dims.2, per_element_euler);
        }
        native.set_array_position(world_translation, global_euler);

        let m_manual = manual.get_array_binary_mask(&grid);
        let m_native = native.get_array_binary_mask(&grid);

        let manual_count = m_manual.iter().filter(|&&b| b).count();
        let native_count = m_native.iter().filter(|&&b| b).count();
        let inter = ndarray::Zip::from(&m_manual)
            .and(&m_native)
            .fold(0usize, |acc, &a, &b| acc + if a && b { 1 } else { 0 });

        assert!(
            manual_count > 0 && native_count > 0,
            "both masks must be non-empty: manual={manual_count}, native={native_count}",
        );
        let iou = inter as f64 / (manual_count + native_count - inter).max(1) as f64;
        assert!(
            iou >= 0.90,
            "set_array_position must match manual translation/rotation: IoU={iou}, \
             manual={manual_count}, native={native_count}, inter={inter}",
        );
    }

    #[test]
    fn test_rect_rotation_90_swaps_width_and_height() {
        use crate::domain::grid::Grid;

        let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
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
    fn test_rect_weighted_mask_matches_kwave_python_reference_mass() {
        use crate::domain::grid::Grid;

        let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
        let mut array = KWaveArray::new();
        array.add_rect_rot_element(
            (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
            8.0e-4,
            2.0e-4,
            1.0e-4,
            (0.0, 0.0, 90.0),
        );

        let weights = array.get_array_weighted_mask(&grid);
        // k-wave-python reference mass for the same centered-geometry rectangle.
        let expected = 16.036_130_608_724_637_f64;
        assert!(
            (weights.sum() - expected).abs() < 5.0e-6,
            "rect weighted mask must match the k-wave-python reference mass: got {}, expected {}",
            weights.sum(),
            expected,
        );
    }

    /// Annulus has strictly fewer active cells than the full bowl of the same
    /// outer diameter, and the surface-area formula scales correctly with
    /// `inner_diameter`. (A stricter "inner-hollow" NGP test would fight the
    /// BLI stencil that `map_surface_sample` intentionally spreads through.)
    #[test]
    fn test_annulus_is_subset_of_bowl_same_outer_diameter() {
        use crate::domain::grid::Grid;

        let dx = 2.0e-4;
        let grid = Grid::new(81, 81, 81, dx, dx, dx).expect("grid");
        let radius = 8.0e-3;
        let cx = 10.0e-3;
        let cy = 40.0 * dx;
        let cz = 40.0 * dx;

        let outer_d = 6.0e-3;
        let inner_d = 3.0e-3;

        let mut bowl = KWaveArray::new();
        bowl.add_bowl_element((cx, cy, cz), radius, outer_d);
        let bowl_mask = bowl.get_array_binary_mask(&grid);

        let mut annulus = KWaveArray::new();
        annulus.add_annular_element((cx, cy, cz), radius, inner_d, outer_d);
        let annulus_mask = annulus.get_array_binary_mask(&grid);

        let bowl_count: usize = bowl_mask.iter().filter(|&&b| b).count();
        let annulus_count: usize = annulus_mask.iter().filter(|&&b| b).count();
        assert!(bowl_count > 0 && annulus_count > 0);
        assert!(
            annulus_count < bowl_count,
            "annulus (inner_d>0) must have fewer cells than full bowl: {annulus_count} vs {bowl_count}",
        );

        // Area formula: with inner_d = 0, annulus_surface_area must match bowl_surface_area.
        let a_bowl = KWaveArray::bowl_surface_area(radius, outer_d);
        let a_ann_full = KWaveArray::annulus_surface_area(radius, 0.0, outer_d);
        assert!(
            (a_bowl - a_ann_full).abs() / a_bowl < 1.0e-12,
            "annulus(0, D) must equal bowl(D): {a_bowl} vs {a_ann_full}",
        );

        let a_ann = KWaveArray::annulus_surface_area(radius, inner_d, outer_d);
        assert!(
            a_ann > 0.0 && a_ann < a_bowl,
            "annulus area must be positive and less than full bowl: {a_ann} vs {a_bowl}",
        );
    }

    #[test]
    fn test_build_per_element_source_superposition() {
        // Theorem: for two elements with per-element signals s1, s2 and a
        // per-cell signal built as Σ_i W_i[c] · s_i[t], setting s1=s2=s
        // must reproduce the shared-signal result (W_sum[c] · s[t]).
        use crate::domain::grid::Grid;
        let dx = 5.0e-4;
        let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
        let cx = 30.0 * dx;
        let cy = 30.0 * dx;
        let cz = 30.0 * dx;

        let mut arr = KWaveArray::new();
        arr.add_annular_element((cx, cy, cz), 15.0e-3, 0.0, 4.0e-3);
        arr.add_annular_element((cx, cy, cz), 15.0e-3, 6.0e-3, 10.0e-3);
        assert_eq!(arr.num_elements(), 2);

        let n_times = 4;
        let s: Vec<f64> = (0..n_times).map(|t| (t as f64).sin()).collect();
        let mut shared = Array2::<f64>::zeros((2, n_times));
        for t in 0..n_times {
            shared[[0, t]] = s[t];
            shared[[1, t]] = s[t];
        }

        let (mask_unit, per_cell) = arr
            .build_per_element_source(&grid, &shared)
            .expect("build per-element source");
        let w_sum = arr.get_array_weighted_mask(&grid);

        // Every active cell must have mask_unit = 1.0 and w_sum != 0.
        // The active-cell enumeration follows MATLAB / Fortran order.
        let (nx, ny, nz) = mask_unit.dim();
        let mut active_cells = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let m = mask_unit[[i, j, k]];
                    if m != 0.0 {
                        assert!((m - 1.0).abs() < 1.0e-12);
                        assert!(w_sum[[i, j, k]] != 0.0);
                        active_cells.push((i, j, k));
                    }
                }
            }
        }
        assert!(!active_cells.is_empty());
        assert_eq!(active_cells.len(), per_cell.shape()[0]);

        // For each cell and timestep: per_cell[idx, t] == w_sum[cell] * s[t].
        for (idx, &(i, j, k)) in active_cells.iter().enumerate() {
            for t in 0..n_times {
                let expected = w_sum[[i, j, k]] * s[t];
                let got = per_cell[[idx, t]];
                assert!(
                    (got - expected).abs() < 1.0e-10 * expected.abs().max(1.0),
                    "cell ({i},{j},{k}) t={t}: got {got}, expected {expected}",
                );
            }
        }
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
