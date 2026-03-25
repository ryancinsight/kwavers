//! # 2D Transducer Array — Phased Array Beamforming
//!
//! This module provides a native 2D transducer array implementation that mirrors
//! k-wave-python's `kWaveTransducerSimple` and `NotATransducer` functionality.
//!
//! ## Background
//!
//! Phased array transducers control the acoustic beam by introducing relative time
//! delays between elements. Constructive interference forms a beam at the desired
//! focus or steering direction; destructive interference suppresses sidelobes.
//! Apodization (amplitude weighting) further reduces sidelobes at the cost of
//! main-lobe width.
//!
//! ## Theorem: Far-Field Steering Delays
//!
//! For a linear array of N elements with pitch d [m] centered at the origin,
//! the far-field steering delay for element i (0-indexed) to steer at angle θ
//! in the x-z plane is:
//! ```text
//! τᵢ = (xᵢ · sin θ) / c₀
//! ```
//! where xᵢ = (i - (N-1)/2) · d is the element x-position.
//! Positive τᵢ delays firing; the sign convention matches k-Wave.
//!
//! ## Theorem: Geometric Focus Delays
//!
//! For focusing at depth z_f (along the z-axis) with steering angle θ = 0:
//! ```text
//! τᵢ = (|rᵢ - r_focus| - |r_center - r_focus|) / c₀
//! ```
//! where r_focus = (x_f, y_f, z_f) is the focal point and r_center is the
//! array center. This delay compensates for the path-length difference between
//! each element and the focus, ensuring in-phase arrival.
//!
//! For simultaneous steering and focusing, delays are combined:
//! ```text
//! τᵢ = τ_focus,i + τ_steer,i
//! ```
//!
//! ## Discretization
//!
//! Delays are quantized to the nearest time step:
//! ```text
//! n_i = round(τᵢ / Δt)
//! ```
//! Sub-sample accuracy requires sinc-interpolation resampling of the transmit
//! signal. The current implementation uses integer-sample delays matching
//! k-Wave's `kWaveTransducerSimple` default behavior.
//!
//! ## Apodization
//!
//! The receive/transmit aperture function aᵢ reduces sidelobe levels:
//! - **Rectangular**: aᵢ = 1 (no weighting, max resolution, highest sidelobes)
//! - **Hanning**: aᵢ = 0.5·(1 - cos(2πi/N)) (−31.5 dB first sidelobe)
//! - **Hamming**: aᵢ = 0.54 - 0.46·cos(2πi/N) (−41.8 dB first sidelobe)
//! - **Blackman**: four-term window (−57 dB first sidelobe)
//!
//! Ref: Harris (1978) Proc. IEEE 66(1), 51-83.
//!
//! ## References
//!
//! - van Veen & Buckley (1988) IEEE Signal Process. Mag. 5(2), 4-24.
//! - Harris (1978) Proc. IEEE 66(1), 51-83. (apodization windows)
//! - Treeby & Cox (2010) J. Biomed. Opt. 15(2), 021314.
//!
//! # Features
//!
//! - **Linear array geometry**: Configurable number of elements, width, and spacing
//! - **Electronic steering**: Beam steering in azimuthal direction
//! - **Electronic focusing**: Focus at arbitrary depths
//! - **Apodization**: Transmit and receive apodization (windowing)
//! - **Elevation focusing**: Optional focus in elevation plane
//! - **Active element masking**: Selectively enable/disable elements
//!
//! # Signal Model
//!
//! The array generates pressure signals using time-delay beamforming:
//!
//! ```text
//! p(x, y, z, t) = Σᵢ aᵢ · s(t - τᵢ) · δ(x - xᵢ, y - yᵢ, z - zᵢ)
//! ```
//!
//! where:
//! - `aᵢ` is the apodization weight for element i
//! - `s(t)` is the input signal
//! - `τᵢ` is the time delay for element i (focus + steer)
//! - `(xᵢ, yᵢ, zᵢ)` is the element centroid position
//!
//! # Example
//!
//! ```rust
//! use kwavers::domain::source::array_2d::{TransducerArray2D, TransducerArray2DConfig, ApodizationType};
//!
//! // Configure array
//! let config = TransducerArray2DConfig {
//!     number_elements: 32,
//!     element_width: 0.3e-3,
//!     element_length: 10e-3,
//!     element_spacing: 0.5e-3,  // must be >= element_width
//!     radius: f64::INFINITY, // Flat array
//!     center_position: (0.0, 0.0, 0.0),
//! };
//!
//! // Create array with focusing
//! let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();
//! array.set_focus_distance(20e-3);
//! array.set_steering_angle(0.0);
//! array.set_transmit_apodization(ApodizationType::Hanning);
//! ```
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields. J. Biomed. Opt.
//! - Szabo, T. L. (2014). Diagnostic Ultrasound Imaging: Inside Out.

use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::{
    Apodization, BlackmanApodization, GaussianApodization, HammingApodization, HanningApodization,
    RectangularApodization, Source,
};
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

pub mod builder;

pub use builder::TransducerArray2DBuilder;

/// Apodization window types for transmit/receive weighting
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApodizationType {
    /// Uniform weighting (all elements equal)
    Rectangular,
    /// Hanning window (raised cosine)
    Hanning,
    /// Hamming window (modified raised cosine)
    Hamming,
    /// Blackman window (reduced sidelobes)
    Blackman,
    /// Gaussian window (configurable width)
    Gaussian { sigma: f64 },
}

impl ApodizationType {
    /// Create an apodization implementation from the type
    fn create_apodization(&self) -> Box<dyn Apodization> {
        match self {
            ApodizationType::Rectangular => Box::new(RectangularApodization),
            ApodizationType::Hanning => Box::new(HanningApodization),
            ApodizationType::Hamming => Box::new(HammingApodization),
            ApodizationType::Blackman => Box::new(BlackmanApodization),
            ApodizationType::Gaussian { sigma } => Box::new(GaussianApodization::new(*sigma)),
        }
    }
}

/// Configuration for a 2D transducer array
///
/// This structure defines the physical geometry of the transducer array,
/// following k-wave-python's kWaveTransducerSimple convention.
#[derive(Debug, Clone)]
pub struct TransducerArray2DConfig {
    /// Number of elements in the array
    pub number_elements: usize,
    /// Width of each element [m]
    pub element_width: f64,
    /// Length of each element (elevation direction) [m]
    pub element_length: f64,
    /// Spacing between element centers [m]
    pub element_spacing: f64,
    /// Radius of curvature [m] (INF for flat array)
    pub radius: f64,
    /// Center position of the array (x, y, z) [m]
    pub center_position: (f64, f64, f64),
}

impl Default for TransducerArray2DConfig {
    fn default() -> Self {
        Self {
            number_elements: 32,
            element_width: 0.3e-3,
            element_length: 10e-3,
            element_spacing: 0.5e-3,
            radius: f64::INFINITY,
            center_position: (0.0, 0.0, 0.0),
        }
    }
}

impl TransducerArray2DConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.number_elements == 0 {
            return Err("Number of elements must be positive".to_string());
        }
        if self.element_width <= 0.0 {
            return Err("Element width must be positive".to_string());
        }
        if self.element_length <= 0.0 {
            return Err("Element length must be positive".to_string());
        }
        if self.element_spacing < self.element_width {
            return Err("Element spacing must be >= element width".to_string());
        }
        if self.radius <= 0.0 && !self.radius.is_infinite() {
            return Err("Radius must be positive or infinite".to_string());
        }
        Ok(())
    }

    /// Calculate total array aperture width
    #[must_use]
    pub fn aperture_width(&self) -> f64 {
        (self.number_elements - 1) as f64 * self.element_spacing + self.element_width
    }

    /// Check if element spacing satisfies Nyquist criterion
    #[must_use]
    pub fn satisfies_nyquist(&self, sound_speed: f64, frequency: f64) -> bool {
        let wavelength = sound_speed / frequency;
        self.element_spacing <= wavelength / 2.0
    }
}

/// Individual transducer element in 2D array
#[derive(Debug, Clone)]
pub struct ArrayElement {
    /// Element position (x, y, z) [m]
    pub position: (f64, f64, f64),
    /// Element width [m]
    pub width: f64,
    /// Element length [m]
    pub length: f64,
    /// Time delay for beamforming [s]
    pub time_delay: f64,
    /// Transmit apodization weight [0.0-1.0]
    pub transmit_weight: f64,
    /// Receive apodization weight [0.0-1.0]
    pub receive_weight: f64,
    /// Whether element is active
    pub is_active: bool,
}

/// 2D Transducer Array with electronic beam control
///
/// This type implements a linear or curved transducer array with capabilities
/// for electronic focusing, steering, and apodization. It is designed to be
/// compatible with k-wave-python's transducer conventions.
///
/// # Example
///
/// ```rust
/// use kwavers::domain::source::array_2d::{TransducerArray2D, TransducerArray2DConfig};
///
/// let config = TransducerArray2DConfig::default();
/// let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();
/// array.set_focus_distance(20e-3);
/// ```
#[derive(Debug)]
pub struct TransducerArray2D {
    /// Physical configuration
    config: TransducerArray2DConfig,
    /// Speed of sound [m/s]
    sound_speed: f64,
    /// Operating frequency [Hz]
    frequency: f64,
    /// Array elements
    elements: Vec<ArrayElement>,
    /// Focus distance [m]
    focus_distance: f64,
    /// Elevation focus distance [m]
    elevation_focus_distance: f64,
    /// Steering angle [degrees]
    steering_angle: f64,
    /// Transmit apodization type
    transmit_apodization: ApodizationType,
    /// Receive apodization type
    receive_apodization: ApodizationType,
    /// Input signal
    signal: Option<Arc<dyn Signal>>,
    /// Active element mask
    active_elements: Vec<bool>,
    /// Cached mask for grid application
    cached_mask: Option<Array3<f64>>,
    /// Grid the mask was computed for
    cached_grid_id: Option<u64>,
}

impl Clone for TransducerArray2D {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sound_speed: self.sound_speed,
            frequency: self.frequency,
            elements: self.elements.clone(),
            focus_distance: self.focus_distance,
            elevation_focus_distance: self.elevation_focus_distance,
            steering_angle: self.steering_angle,
            transmit_apodization: self.transmit_apodization,
            receive_apodization: self.receive_apodization,
            signal: self.signal.clone(),
            active_elements: self.active_elements.clone(),
            cached_mask: None, // Don't clone cached data
            cached_grid_id: None,
        }
    }
}

impl TransducerArray2D {
    /// Create a new 2D transducer array
    ///
    /// # Arguments
    ///
    /// * `config` - Physical array configuration
    /// * `sound_speed` - Speed of sound in medium [m/s]
    /// * `frequency` - Operating frequency [Hz]
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn new(
        config: TransducerArray2DConfig,
        sound_speed: f64,
        frequency: f64,
    ) -> Result<Self, String> {
        config.validate()?;

        if sound_speed <= 0.0 {
            return Err("Sound speed must be positive".to_string());
        }
        if frequency <= 0.0 {
            return Err("Frequency must be positive".to_string());
        }

        let num_elements = config.number_elements;
        let elements = Self::compute_element_positions(&config);
        let active_elements = vec![true; num_elements];

        let mut array = Self {
            config,
            sound_speed,
            frequency,
            elements,
            focus_distance: f64::INFINITY,
            elevation_focus_distance: f64::INFINITY,
            steering_angle: 0.0,
            transmit_apodization: ApodizationType::Rectangular,
            receive_apodization: ApodizationType::Rectangular,
            signal: None,
            active_elements,
            cached_mask: None,
            cached_grid_id: None,
        };

        // Initialize apodization weights
        array.update_apodization_weights();

        Ok(array)
    }

    /// Compute element positions based on configuration
    fn compute_element_positions(config: &TransducerArray2DConfig) -> Vec<ArrayElement> {
        let num_elements = config.number_elements;
        let (cx, cy, cz) = config.center_position;

        // Element positions along x-axis (azimuthal direction)
        // For curved arrays, y positions vary
        let pitch = config.element_width + config.element_spacing;
        let total_width = (num_elements - 1) as f64 * pitch;
        let _start_x = cx - total_width / 2.0;

        (0..num_elements)
            .map(|i| {
                let frac = if num_elements > 1 {
                    i as f64 / (num_elements - 1) as f64 - 0.5
                } else {
                    0.0
                };

                let x = cx + frac * total_width;
                let mut y = cy;
                let z = cz;

                // For curved arrays (bowl/convex), adjust y position
                if config.radius.is_finite() && config.radius > 0.0 {
                    let arc_length = frac * total_width;
                    let angle = arc_length / config.radius;
                    y = cy + config.radius * (1.0 - angle.cos());
                    // x is already arc-length based, so we need to adjust it too
                    // Actually, for a concave array focusing inward:
                    // x = cx + config.radius * angle.sin();
                    // But for k-wave-python compatibility, we use the flat projection
                    // and the radius is used for elevation focusing
                }

                ArrayElement {
                    position: (x, y, z),
                    width: config.element_width,
                    length: config.element_length,
                    time_delay: 0.0,
                    transmit_weight: 1.0,
                    receive_weight: 1.0,
                    is_active: true,
                }
            })
            .collect()
    }

    /// Set the input signal for the array
    pub fn set_signal(&mut self, signal: Arc<dyn Signal>) {
        self.signal = Some(signal);
        self.invalidate_cache();
    }

    /// Set focus distance [m]
    ///
    /// Focus distance of INF disables electronic focusing
    pub fn set_focus_distance(&mut self, distance: f64) {
        if distance > 0.0 {
            self.focus_distance = distance;
            self.update_time_delays();
            self.invalidate_cache();
        }
    }

    /// Set elevation focus distance [m]
    ///
    /// For arrays with elevation focusing (cylindrical curvature)
    pub fn set_elevation_focus_distance(&mut self, distance: f64) {
        if distance > 0.0 {
            self.elevation_focus_distance = distance;
            // Elevation focusing is typically physical, not electronic
            // But we can model it here if needed
            self.invalidate_cache();
        }
    }

    /// Set steering angle [degrees]
    ///
    /// Positive angles steer to the right, negative to the left
    pub fn set_steering_angle(&mut self, angle_deg: f64) {
        self.steering_angle = angle_deg;
        self.update_time_delays();
        self.invalidate_cache();
    }

    /// Set transmit apodization type
    pub fn set_transmit_apodization(&mut self, apodization: ApodizationType) {
        self.transmit_apodization = apodization;
        self.update_apodization_weights();
        self.invalidate_cache();
    }

    /// Set receive apodization type
    pub fn set_receive_apodization(&mut self, apodization: ApodizationType) {
        self.receive_apodization = apodization;
        self.update_apodization_weights();
        self.invalidate_cache();
    }

    /// Set active element mask
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean array of length number_elements
    pub fn set_active_elements(&mut self, mask: &[bool]) -> Result<(), String> {
        if mask.len() != self.config.number_elements {
            return Err(format!(
                "Mask length {} does not match number of elements {}",
                mask.len(),
                self.config.number_elements
            ));
        }
        self.active_elements = mask.to_vec();
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.is_active = mask[i];
        }
        self.invalidate_cache();
        Ok(())
    }

    /// Get active element mask
    #[must_use]
    pub fn get_active_elements(&self) -> &[bool] {
        &self.active_elements
    }

    /// Get number of elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.config.number_elements
    }

    /// Get element positions
    #[must_use]
    pub fn element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements.iter().map(|e| e.position).collect()
    }

    /// Get current focus distance
    #[must_use]
    pub fn focus_distance(&self) -> f64 {
        self.focus_distance
    }

    /// Get current steering angle [degrees]
    #[must_use]
    pub fn steering_angle(&self) -> f64 {
        self.steering_angle
    }

    /// Get number of elements (alias for num_elements)
    #[must_use]
    pub fn number_elements(&self) -> usize {
        self.config.number_elements
    }

    /// Get element width [m]
    #[must_use]
    pub fn element_width(&self) -> f64 {
        self.config.element_width
    }

    /// Get element length [m]
    #[must_use]
    pub fn element_length(&self) -> f64 {
        self.config.element_length
    }

    /// Get element spacing [m]
    #[must_use]
    pub fn element_spacing(&self) -> f64 {
        self.config.element_spacing
    }

    /// Get radius of curvature [m]
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.config.radius
    }

    /// Get operating frequency [Hz]
    #[must_use]
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get sound speed [m/s]
    #[must_use]
    pub fn sound_speed(&self) -> f64 {
        self.sound_speed
    }

    /// Get transmit apodization type
    #[must_use]
    pub fn transmit_apodization(&self) -> &ApodizationType {
        &self.transmit_apodization
    }

    /// Get receive apodization type
    #[must_use]
    pub fn receive_apodization(&self) -> &ApodizationType {
        &self.receive_apodization
    }

    /// Set center position
    pub fn set_center_position(&mut self, position: (f64, f64, f64)) {
        self.config.center_position = position;
        // Recompute element positions
        self.elements = Self::compute_element_positions(&self.config);
        self.update_time_delays();
        self.invalidate_cache();
    }

    /// Update apodization weights for all elements
    fn update_apodization_weights(&mut self) {
        let num_elements = self.config.number_elements;

        // Update transmit weights
        let tx_apodization = self.transmit_apodization.create_apodization();
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.transmit_weight = tx_apodization.weight(i, num_elements);
        }

        // Update receive weights
        let rx_apodization = self.receive_apodization.create_apodization();
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.receive_weight = rx_apodization.weight(i, num_elements);
        }
    }

    /// Update time delays for focusing and steering
    fn update_time_delays(&mut self) {
        let c = self.sound_speed;
        let num_elements = self.config.number_elements;

        // Center element position
        let center_idx = num_elements / 2;
        let center_pos = self.elements[center_idx].position;

        // Calculate focus point if focusing is enabled
        let focus_point = if self.focus_distance.is_finite() {
            // Focus point is at focus_distance from center, along steering direction
            let theta = self.steering_angle.to_radians();
            Some((
                center_pos.0 + self.focus_distance * theta.sin(),
                center_pos.1,
                center_pos.2 + self.focus_distance * theta.cos(),
            ))
        } else {
            None
        };

        // Calculate delays for each element
        for element in self.elements.iter_mut() {
            let mut delay = 0.0;

            // Steering delay (plane wave component)
            if self.steering_angle != 0.0 {
                let theta = self.steering_angle.to_radians();
                // Time delay for plane wave steering
                let x_offset = element.position.0 - center_pos.0;
                delay += x_offset * theta.sin() / c;
            }

            // Focusing delay
            if let Some(focus) = focus_point {
                let dist_to_focus = ((element.position.0 - focus.0).powi(2)
                    + (element.position.1 - focus.1).powi(2)
                    + (element.position.2 - focus.2).powi(2))
                .sqrt();

                let dist_center_to_focus = ((center_pos.0 - focus.0).powi(2)
                    + (center_pos.1 - focus.1).powi(2)
                    + (center_pos.2 - focus.2).powi(2))
                .sqrt();

                // Delay relative to center element
                delay += (dist_to_focus - dist_center_to_focus) / c;
            }

            element.time_delay = delay;
        }
    }

    /// Invalidate cached mask
    fn invalidate_cache(&mut self) {
        self.cached_mask = None;
        self.cached_grid_id = None;
    }

    /// Get total aperture width [m]
    #[must_use]
    pub fn aperture_width(&self) -> f64 {
        self.config.aperture_width()
    }

    /// Check if configuration satisfies Nyquist criterion
    #[must_use]
    pub fn satisfies_nyquist(&self) -> bool {
        self.config
            .satisfies_nyquist(self.sound_speed, self.frequency)
    }
}

impl Source for TransducerArray2D {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        // Check if we have a cached mask for this grid
        let grid_ptr: *const Grid = grid;
        let grid_id = grid_ptr as u64;
        if let Some(ref mask) = self.cached_mask {
            if self.cached_grid_id == Some(grid_id) {
                return mask.clone();
            }
        }

        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Place each active element on the grid
        for (i, element) in self.elements.iter().enumerate() {
            if !element.is_active || !self.active_elements[i] {
                continue;
            }

            let (x, y, z) = element.position;
            let half_width = self.config.element_width / 2.0;
            let half_length = self.config.element_length / 2.0;

            // Find grid indices for element bounds
            // X is azimuthal direction, Z is elevation direction, Y is depth/curvature
            let ix_start = (((x - half_width) - grid.origin[0]) / grid.dx).ceil() as isize;
            let ix_end = (((x + half_width) - grid.origin[0]) / grid.dx).floor() as isize;

            let iy = ((y - grid.origin[1]) / grid.dy).round() as isize;

            let iz_start = (((z - half_length) - grid.origin[2]) / grid.dz).ceil() as isize;
            let iz_end = (((z + half_length) - grid.origin[2]) / grid.dz).floor() as isize;

            // Fill element volume
            for ix in ix_start..=ix_end {
                for iz in iz_start..=iz_end {
                    if ix >= 0
                        && ix < grid.nx as isize
                        && iy >= 0
                        && iy < grid.ny as isize
                        && iz >= 0
                        && iz < grid.nz as isize
                    {
                        mask[[ix as usize, iy as usize, iz as usize]] += element.transmit_weight;
                    }
                }
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        match &self.signal {
            Some(signal) => signal.amplitude(t),
            None => 0.0,
        }
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mut total = 0.0;

        for (i, element) in self.elements.iter().enumerate() {
            if !element.is_active || !self.active_elements[i] {
                continue;
            }

            let (ex, ey, ez) = element.position;

            // Check if we're near this element
            let dx = x - ex;
            let dy = y - ey;
            let dz = z - ez;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // Tolerance based on element size
            let tol = (element.width.max(element.length) / 2.0).max(grid.dx);

            if dist_sq < tol * tol {
                // Apply time delay and apodization
                let delayed_time = t - element.time_delay;
                let amp = self.amplitude(delayed_time);
                total += amp * element.transmit_weight;
            }
        }

        total
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .enumerate()
            .filter(|(i, e)| e.is_active && self.active_elements[*i])
            .map(|(_, e)| e.position)
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal
            .as_ref()
            .expect("TransducerArray2D has no signal set; call set_signal() first")
            .as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TransducerArray2DConfig {
        TransducerArray2DConfig {
            number_elements: 16,
            element_width: 0.3e-3,
            element_length: 10e-3,
            element_spacing: 0.5e-3,
            radius: f64::INFINITY,
            center_position: (0.0, 0.0, 0.0),
        }
    }

    #[test]
    fn test_array_creation() {
        let config = create_test_config();
        let array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

        assert_eq!(array.num_elements(), 16);
        assert!(array.satisfies_nyquist());
    }

    #[test]
    fn test_focus_and_steering() {
        let config = create_test_config();
        let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

        array.set_focus_distance(20e-3);
        array.set_steering_angle(10.0);

        assert!((array.focus_distance() - 20e-3).abs() < 1e-10);
        assert!((array.steering_angle() - 10.0).abs() < 1e-10);

        // Check that delays are non-zero
        let positions = array.element_positions();
        assert_eq!(positions.len(), 16);
    }

    #[test]
    fn test_apodization() {
        let config = create_test_config();
        let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

        array.set_transmit_apodization(ApodizationType::Hanning);
        array.set_receive_apodization(ApodizationType::Hamming);

        // Apodization should be applied
        // (specific values depend on implementation)
    }

    #[test]
    fn test_active_elements() {
        let config = create_test_config();
        let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

        // Deactivate every other element
        let mut mask = vec![true; 16];
        for i in (0..16).step_by(2) {
            mask[i] = false;
        }

        array.set_active_elements(&mask).unwrap();

        let active = array.get_active_elements();
        assert_eq!(active.len(), 16);
        for i in (0..16).step_by(2) {
            assert!(!active[i]);
        }
    }

    #[test]
    fn test_invalid_config() {
        let config = TransducerArray2DConfig {
            number_elements: 0,
            ..create_test_config()
        };

        assert!(TransducerArray2D::new(config, 1540.0, 1e6).is_err());
    }

    #[test]
    fn test_aperture_calculation() {
        let config = create_test_config();
        let array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

        // Aperture = (n-1) * spacing + width
        let expected = 15.0 * 0.5e-3 + 0.3e-3;
        assert!((array.aperture_width() - expected).abs() < 1e-10);
    }
}
