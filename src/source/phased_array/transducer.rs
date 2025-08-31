//! Main phased array transducer implementation

use super::beamforming::{BeamformingCalculator, BeamformingMode};
use super::config::PhasedArrayConfig;
use super::crosstalk::CrosstalkModel;
use super::element::TransducerElement;
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::Signal;
use crate::source::Source;
use ndarray::{Array1, Array3, Zip};
use std::sync::Arc;

/// Phased array transducer with electronic beam control
#[derive(Debug)]
pub struct PhasedArrayTransducer {
    /// Array configuration
    config: PhasedArrayConfig,
    /// Individual elements
    elements: Vec<TransducerElement>,
    /// Signal generator
    signal: Arc<dyn Signal>,
    /// Beamforming mode
    beamforming_mode: BeamformingMode,
    /// Sound speed in medium [m/s]
    sound_speed: f64,
    /// Cross-talk model
    crosstalk_model: Option<CrosstalkModel>,
    /// Beamforming calculator
    beamformer: BeamformingCalculator,
}

impl PhasedArrayTransducer {
    /// Create phased array transducer
    pub fn create(
        config: PhasedArrayConfig,
        signal: Arc<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Self> {
        // Validate configuration
        config.validate().map_err(|e| {
            crate::error::KwaversError::Validation(crate::error::ValidationError::FieldValidation {
                field: "phased_array_config".to_string(),
                value: format!("{:?}", config),
                constraint: e,
            })
        })?;

        // Get sound speed at array center
        let (cx, cy, cz) = config.center_position;
        let sound_speed = medium.sound_speed(cx, cy, cz, grid);

        // Create elements
        let elements = Self::create_elements(&config);

        // Create cross-talk model if enabled
        let crosstalk_model = if config.enable_crosstalk {
            Some(CrosstalkModel::create(
                config.num_elements,
                config.crosstalk_coefficient,
            ))
        } else {
            None
        };

        // Create beamforming calculator
        let beamformer = BeamformingCalculator::with_medium(sound_speed, config.frequency);

        let mut transducer = Self {
            config,
            elements,
            signal,
            beamforming_mode: BeamformingMode::PlaneWave {
                direction: (0.0, 0.0, 1.0),
            },
            sound_speed,
            crosstalk_model,
            beamformer,
        };

        // Apply default beamforming
        transducer.update_delays();

        Ok(transducer)
    }

    /// Create element array
    fn create_elements(config: &PhasedArrayConfig) -> Vec<TransducerElement> {
        let mut elements = Vec::with_capacity(config.num_elements);
        let half_array = (config.num_elements as f64 - 1.0) / 2.0;

        for i in 0..config.num_elements {
            let offset = (i as f64 - half_array) * config.element_spacing;
            let position = (
                config.center_position.0 + offset,
                config.center_position.1,
                config.center_position.2,
            );

            elements.push(TransducerElement::at_position(
                i,
                position,
                config.element_width,
                config.element_height,
            ));
        }

        elements
    }

    /// Set beamforming mode
    pub fn set_beamforming(&mut self, mode: BeamformingMode) {
        self.beamforming_mode = mode;
        self.update_delays();
    }

    /// Update element delays based on beamforming mode
    fn update_delays(&mut self) {
        let positions: Vec<_> = self.elements.iter().map(|e| e.position).collect();

        let delays = match &self.beamforming_mode {
            BeamformingMode::Focus { target } => {
                self.beamformer.calculate_focus_delays(&positions, *target)
            }
            BeamformingMode::Steer { theta, phi } => self
                .beamformer
                .calculate_steering_delays(&positions, *theta, *phi),
            BeamformingMode::PlaneWave { direction } => self
                .beamformer
                .calculate_plane_wave_delays(&positions, *direction),
            BeamformingMode::Custom { delays } => delays.clone(),
        };

        // Apply delays to elements
        for (element, delay) in self.elements.iter_mut().zip(delays.iter()) {
            element.phase_delay = *delay;
        }
    }

    /// Calculate pressure field at given time
    pub fn calculate_field(&self, grid: &Grid, time: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Generate element signals
        let mut element_signals = Array1::zeros(self.config.num_elements);
        for (i, element) in self.elements.iter().enumerate() {
            let signal_value = self.signal.amplitude(time);
            element_signals[i] = element.apply_modulation(signal_value, time);
        }

        // Apply cross-talk if enabled
        if let Some(ref crosstalk) = self.crosstalk_model {
            element_signals = crosstalk.apply(&element_signals);
        }

        // Calculate field contribution from each element
        Zip::indexed(&mut field).for_each(|(i, j, k), pressure| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            for (element, &signal) in self.elements.iter().zip(element_signals.iter()) {
                let distance = Self::distance_to_element(element, x, y, z);

                if distance > 0.0 {
                    // Calculate propagation delay
                    let propagation_time = distance / self.sound_speed;

                    // Calculate directivity
                    let theta = ((x - element.position.0).powi(2)
                        + (y - element.position.1).powi(2))
                    .sqrt()
                    .atan2(z - element.position.2);
                    let directivity =
                        element.directivity(theta, self.config.frequency, self.sound_speed);

                    // Add contribution with spherical spreading
                    *pressure += signal * directivity / (4.0 * std::f64::consts::PI * distance);
                }
            }
        });

        field
    }

    /// Calculate distance from element to point
    fn distance_to_element(element: &TransducerElement, x: f64, y: f64, z: f64) -> f64 {
        let dx = x - element.position.0;
        let dy = y - element.position.1;
        let dz = z - element.position.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get beam width at focal distance
    pub fn beam_width(&self) -> f64 {
        self.beamformer
            .calculate_beam_width(self.config.aperture_size())
    }

    /// Get focal zone depth
    pub fn focal_zone(&self, focal_distance: f64) -> f64 {
        self.beamformer
            .calculate_focal_zone(self.config.aperture_size(), focal_distance)
    }

    /// Get the delays for each element
    pub fn element_delays(&self) -> Vec<f64> {
        self.elements.iter().map(|e| e.phase_delay).collect()
    }
}

impl Source for PhasedArrayTransducer {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        // Create mask with source at element positions
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for element in &self.elements {
            if let Some((i, j, k)) =
                grid.position_to_indices(element.position.0, element.position.1, element.position.2)
            {
                mask[[i, j, k]] = element.amplitude_weight;
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements.iter().map(|e| e.position).collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        // Calculate contribution from all elements
        let mut total_pressure = 0.0;

        for element in &self.elements {
            let distance = Self::distance_to_element(element, x, y, z);

            if distance > 0.0 {
                let propagation_time = distance / self.sound_speed;
                let retarded_time = t - propagation_time;

                if retarded_time >= 0.0 {
                    let signal_value = self.signal.amplitude(retarded_time);
                    let modulated = element.apply_modulation(signal_value, retarded_time);

                    // Calculate directivity
                    let theta = ((x - element.position.0).powi(2)
                        + (y - element.position.1).powi(2))
                    .sqrt()
                    .atan2(z - element.position.2);
                    let directivity =
                        element.directivity(theta, self.config.frequency, self.sound_speed);

                    // Add with spherical spreading
                    total_pressure +=
                        modulated * directivity / (4.0 * std::f64::consts::PI * distance);
                }
            }
        }

        total_pressure
    }
}
