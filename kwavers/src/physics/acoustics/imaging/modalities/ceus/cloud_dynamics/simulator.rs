//! Cloud dynamics simulation engine
//!
//! Core simulation loop: initialize cloud → iterate dynamics → capture state snapshots.

use super::config::{CloudBubble, CloudConfig};
use super::incident_field::{CloudResponse, CloudState, IncidentField};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::imaging::modalities::ceus::microbubble::{
    BubbleDynamics, Microbubble,
};
use log::{debug, info};

/// Microbubble cloud dynamics simulator
#[derive(Debug)]
pub struct CloudDynamics {
    /// Configuration
    pub(crate) config: CloudConfig,
    /// Bubble population
    pub(crate) bubbles: Vec<CloudBubble>,
    /// Computational grid for field calculations
    pub(crate) _grid: Option<Grid>,
    /// Microbubble dynamics solver for individual bubbles
    pub(crate) bubble_solver: BubbleDynamics,
    /// Acoustic field incident on the cloud
    pub(crate) incident_field: Option<IncidentField>,
}

impl CloudDynamics {
    /// Create new cloud dynamics simulator
    pub fn new(config: CloudConfig) -> KwaversResult<Self> {
        let bubble_solver = BubbleDynamics::new();

        Ok(Self {
            config,
            bubbles: Vec::new(),
            _grid: None,
            bubble_solver,
            incident_field: None,
        })
    }

    /// Initialize bubble cloud with random positions
    pub fn initialize_cloud(&mut self) -> KwaversResult<()> {
        self.bubbles.clear();

        let volume =
            self.config.dimensions[0] * self.config.dimensions[1] * self.config.dimensions[2];
        let actual_num_bubbles = (self.config.concentration * volume) as usize;
        let num_bubbles = actual_num_bubbles.min(self.config.num_bubbles);

        info!(
            "Initializing cloud with {} bubbles in {:.2e} m³ volume",
            num_bubbles, volume
        );

        for i in 0..num_bubbles {
            let x = self.config.dimensions[0] * rand::random::<f64>();
            let y = self.config.dimensions[1] * rand::random::<f64>();
            let z = self.config.dimensions[2] * rand::random::<f64>();

            let mut properties = Microbubble::sono_vue();
            let size_factor = 0.5 + rand::random::<f64>();
            properties.radius_eq *= size_factor;
            let current_radius = properties.radius_eq;

            let bubble = CloudBubble {
                properties,
                position: [x, y, z],
                velocity: [0.0, 0.0, 0.0],
                current_radius,
                id: i,
                active: true,
            };

            self.bubbles.push(bubble);
        }

        Ok(())
    }

    /// Set incident acoustic field
    pub fn set_incident_field(&mut self, field: IncidentField) {
        self.incident_field = Some(field);
    }

    /// Simulate cloud dynamics
    pub fn simulate(&mut self) -> KwaversResult<CloudResponse> {
        let n_steps = (self.config.duration / self.config.dt) as usize;
        let mut responses = Vec::new();

        info!(
            "Simulating cloud dynamics: {} steps, {} bubbles",
            n_steps,
            self.bubbles.len()
        );

        responses.push(self.capture_cloud_state());

        for step in 0..n_steps {
            self.update_bubbles()?;

            if self.config.enable_interactions {
                self.handle_interactions()?;
            }

            if self.config.enable_dissolution {
                self.handle_dissolution()?;
            }

            self.update_positions();

            if step % 1000 == 0 {
                responses.push(self.capture_cloud_state());
            }

            if step % 10000 == 0 {
                let active_bubbles = self.bubbles.iter().filter(|b| b.active).count();
                debug!(
                    "Step {}/{}, active bubbles: {}",
                    step, n_steps, active_bubbles
                );
            }
        }

        Ok(CloudResponse {
            time_steps: responses,
            config: self.config.clone(),
        })
    }

    /// Update individual bubble dynamics
    fn update_bubbles(&mut self) -> KwaversResult<()> {
        if let Some(field) = &self.incident_field {
            let mut bubble_data = Vec::new();
            for bubble in &self.bubbles {
                if !bubble.active {
                    continue;
                }

                let pressure = field.pressure_at(bubble.position, 0.0);

                let response = self.bubble_solver.simulate_oscillation(
                    &bubble.properties,
                    pressure,
                    field.frequency,
                    self.config.dt,
                )?;

                bubble_data.push((bubble.id, response));
            }

            for (bubble_id, response) in bubble_data {
                if let Some(bubble_idx) = self.bubbles.iter().position(|b| b.id == bubble_id) {
                    if let Some(&last_radius) = response.radius.last() {
                        self.bubbles[bubble_idx].current_radius = last_radius;
                    }

                    let radiation_force =
                        self.calculate_radiation_force(&self.bubbles[bubble_idx], &response);

                    let mass = (4.0 / 3.0)
                        * std::f64::consts::PI
                        * self.bubbles[bubble_idx].current_radius.powi(3)
                        * 1000.0;
                    for (i, force) in radiation_force.iter().enumerate() {
                        self.bubbles[bubble_idx].velocity[i] += (force / mass) * self.config.dt;
                    }
                }
            }
        }

        Ok(())
    }

    /// Update bubble positions based on velocities
    fn update_positions(&mut self) {
        for bubble in &mut self.bubbles {
            if bubble.active {
                for i in 0..3 {
                    bubble.position[i] += bubble.velocity[i] * self.config.dt;
                }
            }
        }
    }

    /// Capture current cloud state
    fn capture_cloud_state(&self) -> CloudState {
        let active_bubbles: Vec<_> = self.bubbles.iter().filter(|b| b.active).cloned().collect();

        CloudState {
            bubbles: active_bubbles,
            time: 0.0,
        }
    }
}
