//! `TransducerArray2D` struct and impl.

use super::types::{ApodizationType, Array2dElement, TransducerArray2DConfig};
use crate::domain::signal::Signal;
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// 2D Transducer Array with electronic beam control
#[derive(Debug)]
pub struct TransducerArray2D {
    pub(super) config: TransducerArray2DConfig,
    pub(super) sound_speed: f64,
    pub(super) frequency: f64,
    pub(super) elements: Vec<Array2dElement>,
    pub(super) focus_distance: f64,
    pub(super) elevation_focus_distance: f64,
    pub(super) steering_angle: f64,
    pub(super) transmit_apodization: ApodizationType,
    pub(super) receive_apodization: ApodizationType,
    pub(super) signal: Option<Arc<dyn Signal>>,
    pub(super) active_elements: Vec<bool>,
    pub(super) cached_mask: Option<Array3<f64>>,
    pub(super) cached_grid_id: Option<u64>,
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
            cached_mask: None,
            cached_grid_id: None,
        }
    }
}

impl TransducerArray2D {
    /// Create a new 2D transducer array
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: TransducerArray2DConfig,
        sound_speed: f64,
        frequency: f64,
    ) -> Result<Self, String> {
        config.validate()?;

        if sound_speed <= 0.0 {
            return Err("Sound speed must be positive".to_owned());
        }
        if frequency <= 0.0 {
            return Err("Frequency must be positive".to_owned());
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
            transmit_apodization: ApodizationType::Uniform,
            receive_apodization: ApodizationType::Uniform,
            signal: None,
            active_elements,
            cached_mask: None,
            cached_grid_id: None,
        };

        array.update_apodization_weights();

        Ok(array)
    }

    pub(super) fn compute_element_positions(
        config: &TransducerArray2DConfig,
    ) -> Vec<Array2dElement> {
        let num_elements = config.number_elements;
        let (cx, cy, cz) = config.center_position;

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

                if config.radius.is_finite() && config.radius > 0.0 {
                    let arc_length = frac * total_width;
                    let angle = arc_length / config.radius;
                    y = config.radius.mul_add(1.0 - angle.cos(), cy);
                }

                Array2dElement {
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

    /// Set focus distance (m)
    pub fn set_focus_distance(&mut self, distance: f64) {
        if distance > 0.0 {
            self.focus_distance = distance;
            self.update_time_delays();
            self.invalidate_cache();
        }
    }

    /// Set elevation focus distance (m)
    pub fn set_elevation_focus_distance(&mut self, distance: f64) {
        if distance > 0.0 {
            self.elevation_focus_distance = distance;
            self.invalidate_cache();
        }
    }

    /// Set steering angle (degrees)
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_receive_apodization(&mut self, apodization: ApodizationType) {
        self.receive_apodization = apodization;
        self.update_apodization_weights();
        self.invalidate_cache();
    }

    /// Set active element mask
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Get current steering angle (degrees)
    #[must_use]
    pub fn steering_angle(&self) -> f64 {
        self.steering_angle
    }

    /// Get number of elements (alias for num_elements)
    #[must_use]
    pub fn number_elements(&self) -> usize {
        self.config.number_elements
    }

    /// Get element width (m)
    #[must_use]
    pub fn element_width(&self) -> f64 {
        self.config.element_width
    }

    /// Get element length (m)
    #[must_use]
    pub fn element_length(&self) -> f64 {
        self.config.element_length
    }

    /// Get element spacing (m)
    #[must_use]
    pub fn element_spacing(&self) -> f64 {
        self.config.element_spacing
    }

    /// Get radius of curvature (m)
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.config.radius
    }

    /// Get operating frequency (Hz)
    #[must_use]
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get sound speed (m/s)
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
        self.elements = Self::compute_element_positions(&self.config);
        self.update_time_delays();
        self.invalidate_cache();
    }

    pub(super) fn update_apodization_weights(&mut self) {
        let num_elements = self.config.number_elements;

        let tx_apodization = super::types::create_apodization(&self.transmit_apodization);
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.transmit_weight = tx_apodization.weight(i, num_elements);
        }

        let rx_apodization = super::types::create_apodization(&self.receive_apodization);
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.receive_weight = rx_apodization.weight(i, num_elements);
        }
    }

    pub(super) fn update_time_delays(&mut self) {
        let c = self.sound_speed;
        let num_elements = self.config.number_elements;

        let center_idx = num_elements / 2;
        let center_pos = self.elements[center_idx].position;

        let focus_point = if self.focus_distance.is_finite() {
            let theta = self.steering_angle.to_radians();
            Some((
                self.focus_distance.mul_add(theta.sin(), center_pos.0),
                center_pos.1,
                self.focus_distance.mul_add(theta.cos(), center_pos.2),
            ))
        } else {
            None
        };

        for element in &mut self.elements {
            let mut delay = 0.0;

            if self.steering_angle != 0.0 {
                let theta = self.steering_angle.to_radians();
                let x_offset = element.position.0 - center_pos.0;
                delay += x_offset * theta.sin() / c;
            }

            if let Some(focus) = focus_point {
                let dist_to_focus = (element.position.2 - focus.2)
                    .mul_add(
                        element.position.2 - focus.2,
                        (element.position.1 - focus.1).mul_add(
                            element.position.1 - focus.1,
                            (element.position.0 - focus.0).powi(2),
                        ),
                    )
                    .sqrt();

                let dist_center_to_focus = (center_pos.2 - focus.2)
                    .mul_add(
                        center_pos.2 - focus.2,
                        (center_pos.1 - focus.1)
                            .mul_add(center_pos.1 - focus.1, (center_pos.0 - focus.0).powi(2)),
                    )
                    .sqrt();

                delay += (dist_to_focus - dist_center_to_focus) / c;
            }

            element.time_delay = delay;
        }
    }

    pub(super) fn invalidate_cache(&mut self) {
        self.cached_mask = None;
        self.cached_grid_id = None;
    }

    /// Get total aperture width (m)
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
