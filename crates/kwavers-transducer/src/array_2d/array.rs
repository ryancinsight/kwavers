//! `TransducerArray2D` struct and impl.

use super::types::{ApodizationType, Array2dElement, TransducerArray2DConfig};
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Time, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, Radian, Second};
use kwavers_signal::Signal;
use leto::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// 2D Transducer Array with electronic beam control
#[derive(Debug)]
pub struct TransducerArray2D {
    pub(super) config: TransducerArray2DConfig,
    pub(super) sound_speed: Velocity<f64>,
    pub(super) frequency: Frequency<f64>,
    pub(super) elements: Vec<Array2dElement>,
    pub(super) focus_distance: Option<Length<f64>>,
    pub(super) elevation_focus_distance: Option<Length<f64>>,
    pub(super) steering_angle: Angle<f64>,
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
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: TransducerArray2DConfig,
        sound_speed: Velocity<f64>,
        frequency: Frequency<f64>,
    ) -> Result<Self, String> {
        config.validate()?;

        let sound_speed_m_s = sound_speed.in_unit::<MeterPerSecond>();
        if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
            return Err(format!(
                "Sound speed must be finite and positive, got {sound_speed_m_s}"
            ));
        }
        let frequency_hz = frequency.in_unit::<Hertz>();
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(format!(
                "Frequency must be finite and positive, got {frequency_hz}"
            ));
        }

        let num_elements = config.number_elements;
        let elements = Self::compute_element_positions(&config);
        let active_elements = vec![true; num_elements];

        let mut array = Self {
            config,
            sound_speed,
            frequency,
            elements,
            focus_distance: None,
            elevation_focus_distance: None,
            steering_angle: Angle::from_unit::<Radian>(0.0),
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
        let [cx, cy, cz] = config
            .center_position
            .map(|coordinate| coordinate.in_unit::<Meter>());

        let pitch = config.element_spacing.in_unit::<Meter>();
        let total_width = (num_elements - 1) as f64 * pitch;

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

                if let Some(radius) = config.curvature.radius() {
                    let radius_m = radius.in_unit::<Meter>();
                    let arc_length = frac * total_width;
                    let angle = arc_length / radius_m;
                    y = radius_m.mul_add(1.0 - angle.cos(), cy);
                }

                Array2dElement {
                    position: [
                        Length::from_unit::<Meter>(x),
                        Length::from_unit::<Meter>(y),
                        Length::from_unit::<Meter>(z),
                    ],
                    width: config.element_width,
                    length: config.element_length,
                    time_delay: Time::from_unit::<Second>(0.0),
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

    /// Set the azimuthal focus distance.
    pub fn set_focus_distance(&mut self, distance: Length<f64>) -> Result<(), String> {
        let distance_m = distance.in_unit::<Meter>();
        if !distance_m.is_finite() || distance_m <= 0.0 {
            return Err(format!(
                "Focus distance must be finite and positive, got {distance_m}"
            ));
        }
        self.focus_distance = Some(distance);
        self.update_time_delays();
        self.invalidate_cache();
        Ok(())
    }

    /// Clear the azimuthal focus and use an unfocused wavefront.
    pub fn clear_focus_distance(&mut self) {
        self.focus_distance = None;
        self.update_time_delays();
        self.invalidate_cache();
    }

    /// Set the elevation focus distance.
    pub fn set_elevation_focus_distance(&mut self, distance: Length<f64>) -> Result<(), String> {
        let distance_m = distance.in_unit::<Meter>();
        if !distance_m.is_finite() || distance_m <= 0.0 {
            return Err(format!(
                "Elevation focus distance must be finite and positive, got {distance_m}"
            ));
        }
        self.elevation_focus_distance = Some(distance);
        self.invalidate_cache();
        Ok(())
    }

    /// Clear the elevation focus and use an unfocused elevation wavefront.
    pub fn clear_elevation_focus_distance(&mut self) {
        self.elevation_focus_distance = None;
        self.invalidate_cache();
    }

    /// Set the steering angle in the coherent SI radian unit.
    pub fn set_steering_angle(&mut self, angle: Angle<f64>) -> Result<(), String> {
        let angle_rad = angle.in_unit::<Radian>();
        if !angle_rad.is_finite() {
            return Err(format!("Steering angle must be finite, got {angle_rad}"));
        }
        self.steering_angle = angle;
        self.update_time_delays();
        self.invalidate_cache();
        Ok(())
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
    pub fn element_positions(&self) -> Vec<[Length<f64>; 3]> {
        self.elements.iter().map(|e| e.position).collect()
    }

    /// Get the current azimuthal focus distance, if focusing is enabled.
    #[must_use]
    pub fn focus_distance(&self) -> Option<Length<f64>> {
        self.focus_distance
    }

    /// Get the current elevation focus distance, if focusing is enabled.
    #[must_use]
    pub fn elevation_focus_distance(&self) -> Option<Length<f64>> {
        self.elevation_focus_distance
    }

    /// Get the current steering angle in radians.
    #[must_use]
    pub fn steering_angle(&self) -> Angle<f64> {
        self.steering_angle
    }

    /// Get number of elements (alias for num_elements)
    #[must_use]
    pub fn number_elements(&self) -> usize {
        self.config.number_elements
    }

    /// Get element width.
    #[must_use]
    pub fn element_width(&self) -> Length<f64> {
        self.config.element_width
    }

    /// Get element length.
    #[must_use]
    pub fn element_length(&self) -> Length<f64> {
        self.config.element_length
    }

    /// Get center-to-center element spacing.
    #[must_use]
    pub fn element_spacing(&self) -> Length<f64> {
        self.config.element_spacing
    }

    /// Get the array surface geometry.
    #[must_use]
    pub fn curvature(&self) -> super::types::ArrayCurvature {
        self.config.curvature
    }

    /// Get operating frequency.
    #[must_use]
    pub fn frequency(&self) -> Frequency<f64> {
        self.frequency
    }

    /// Get sound speed.
    #[must_use]
    pub fn sound_speed(&self) -> Velocity<f64> {
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

    /// Set center position.
    pub fn set_center_position(&mut self, position: [Length<f64>; 3]) -> Result<(), String> {
        if position
            .iter()
            .any(|coordinate| !coordinate.in_unit::<Meter>().is_finite())
        {
            return Err("Center position coordinates must be finite".to_owned());
        }
        self.config.center_position = position;
        self.elements = Self::compute_element_positions(&self.config);
        self.update_time_delays();
        self.invalidate_cache();
        Ok(())
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
        let c = self.sound_speed.in_unit::<MeterPerSecond>();
        let num_elements = self.config.number_elements;

        let center_idx = num_elements / 2;
        let center_pos = self.elements[center_idx]
            .position
            .map(|coordinate| coordinate.in_unit::<Meter>());
        let steering_angle = self.steering_angle.in_unit::<Radian>();

        let focus_point = self.focus_distance.map(|focus_distance| {
            (
                focus_distance
                    .in_unit::<Meter>()
                    .mul_add(steering_angle.sin(), center_pos[0]),
                center_pos[1],
                focus_distance
                    .in_unit::<Meter>()
                    .mul_add(steering_angle.cos(), center_pos[2]),
            )
        });

        for element in &mut self.elements {
            let position = element
                .position
                .map(|coordinate| coordinate.in_unit::<Meter>());
            let mut delay = 0.0;

            if steering_angle != 0.0 {
                let x_offset = position[0] - center_pos[0];
                delay += x_offset * steering_angle.sin() / c;
            }

            if let Some(focus) = focus_point {
                let dist_to_focus = (position[2] - focus.2)
                    .mul_add(
                        position[2] - focus.2,
                        (position[1] - focus.1)
                            .mul_add(position[1] - focus.1, (position[0] - focus.0).powi(2)),
                    )
                    .sqrt();

                let dist_center_to_focus = (center_pos[2] - focus.2)
                    .mul_add(
                        center_pos[2] - focus.2,
                        (center_pos[1] - focus.1)
                            .mul_add(center_pos[1] - focus.1, (center_pos[0] - focus.0).powi(2)),
                    )
                    .sqrt();

                delay += (dist_to_focus - dist_center_to_focus) / c;
            }

            element.time_delay = Time::from_unit::<Second>(delay);
        }
    }

    pub(super) fn invalidate_cache(&mut self) {
        self.cached_mask = None;
        self.cached_grid_id = None;
    }

    /// Get total aperture width.
    #[must_use]
    pub fn aperture_width(&self) -> Length<f64> {
        self.config.aperture_width()
    }

    /// Check if configuration satisfies Nyquist criterion
    #[must_use]
    pub fn satisfies_nyquist(&self) -> bool {
        self.config
            .satisfies_nyquist(self.sound_speed, self.frequency)
    }
}
