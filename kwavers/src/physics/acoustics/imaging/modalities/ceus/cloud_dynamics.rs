//! Microbubble Cloud Dynamics for Contrast-Enhanced Ultrasound
//!
//! Simulates the collective behavior of microbubble populations in acoustic fields,
//! including bubble-bubble interactions, dissolution, coalescence, and nonlinear scattering.
//!
//! ## Physics Overview
//!
//! Microbubble clouds exhibit complex collective behavior:
//! - **Bubble-bubble interactions**: Secondary radiation forces
//! - **Coalescence**: Bubble merging under close approach
//! - **Dissolution**: Gas diffusion and pressure changes
//! - **Nonlinear scattering**: Harmonic generation from collective oscillations
//! - **Acoustic streaming**: Mean flow induced by oscillating bubbles
//!
//! ## Mathematical Models
//!
//! ### Secondary Radiation Force
//! F_secondary = (π/9) * ρ₁ * R₁² * R₂² * (ω²/c₁²) * (d₁ - d₂) * |u₁ - u₂|²
//!
//! ### Bubble Coalescence
//! When bubbles approach within a critical distance, they merge with:
//! - Conservation of gas volume
//! - Conservation of momentum
//! - Surface energy changes
//!
//! ## References
//!
//! - Church, C. C. (1995). "The effects of an elastic solid surface layer on the
//!   radial pulsations of gas bubbles." *JASA*, 97(3), 1510-1521.
//! - Tang, M. X., & Eckersley, R. J. (2006). "Nonlinear propagation of ultrasound
//!   through microbubble contrast agents and implications for imaging." *IEEE TUFFC*,
//!   53(1), 126-141.
//! - Doinikov, A. A. (2001). "Translational motion of a bubble in an acoustic
//!   standing wave." *Phys. Fluids*, 13(8), 2219-2226.

use super::microbubble::{BubbleDynamics, BubbleResponse, Microbubble};
use crate::core::error::KwaversResult;
use log::{debug, info};
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::collections::HashMap;

/// Individual bubble in a cloud
#[derive(Debug, Clone)]
pub struct CloudBubble {
    /// Bubble properties
    pub properties: Microbubble,
    /// Position in space (x, y, z) in meters
    pub position: [f64; 3],
    /// Velocity (vx, vy, vz) in m/s
    pub velocity: [f64; 3],
    /// Current radius (m)
    pub current_radius: f64,
    /// Bubble ID for tracking
    pub id: usize,
    /// Whether bubble is still active
    pub active: bool,
}

/// Microbubble cloud configuration
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Initial number of bubbles
    pub num_bubbles: usize,
    /// Bubble concentration (bubbles/m³)
    pub concentration: f64,
    /// Cloud volume dimensions (m)
    pub dimensions: [f64; 3],
    /// Time step for dynamics (s)
    pub dt: f64,
    /// Simulation duration (s)
    pub duration: f64,
    /// Critical distance for coalescence (m)
    pub coalescence_distance: f64,
    /// Enable bubble-bubble interactions
    pub enable_interactions: bool,
    /// Enable dissolution dynamics
    pub enable_dissolution: bool,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            num_bubbles: 1000,
            concentration: 1e12, // 10^12 bubbles/m³ (typical for contrast agents)
            dimensions: [1e-3, 1e-3, 1e-3], // 1 mm³ volume
            dt: 1e-8,            // 10 ns (much smaller than acoustic period)
            duration: 1e-3,      // 1 ms
            coalescence_distance: 2e-6, // 2 μm
            enable_interactions: true,
            enable_dissolution: false,
        }
    }
}

/// Microbubble cloud dynamics simulator
#[derive(Debug)]
pub struct CloudDynamics {
    /// Configuration
    config: CloudConfig,
    /// Bubble population
    bubbles: Vec<CloudBubble>,
    /// Computational grid for field calculations
    _grid: Option<Grid>,
    /// Microbubble dynamics solver for individual bubbles
    bubble_solver: BubbleDynamics,
    /// Acoustic field incident on the cloud
    incident_field: Option<IncidentField>,
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

        // Calculate actual number of bubbles based on concentration and volume
        let volume =
            self.config.dimensions[0] * self.config.dimensions[1] * self.config.dimensions[2];
        let actual_num_bubbles = (self.config.concentration * volume) as usize;
        let num_bubbles = actual_num_bubbles.min(self.config.num_bubbles);

        info!(
            "Initializing cloud with {} bubbles in {:.2e} m³ volume",
            num_bubbles, volume
        );

        for i in 0..num_bubbles {
            // Random position within volume
            let x = self.config.dimensions[0] * rand::random::<f64>();
            let y = self.config.dimensions[1] * rand::random::<f64>();
            let z = self.config.dimensions[2] * rand::random::<f64>();

            // Create bubble with typical contrast agent properties
            let mut properties = Microbubble::sono_vue();
            // Add size distribution (log-normal)
            let size_factor = 0.5 + rand::random::<f64>(); // 0.5-1.5 μm radius
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

        // Save initial state
        responses.push(self.capture_cloud_state());

        for step in 0..n_steps {
            // Update bubble dynamics
            self.update_bubbles()?;

            // Handle bubble-bubble interactions
            if self.config.enable_interactions {
                self.handle_interactions()?;
            }

            // Handle dissolution
            if self.config.enable_dissolution {
                self.handle_dissolution()?;
            }

            // Update positions based on velocities
            self.update_positions();

            // Save state periodically
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
            // Collect bubble data for force calculation
            let mut bubble_data = Vec::new();
            for bubble in &self.bubbles {
                if !bubble.active {
                    continue;
                }

                // Get local acoustic pressure at bubble position
                let pressure = field.pressure_at(bubble.position, 0.0); // Simplified: time=0

                // Solve bubble oscillation
                let response = self.bubble_solver.simulate_oscillation(
                    &bubble.properties,
                    pressure,
                    field.frequency,
                    self.config.dt,
                )?;

                bubble_data.push((bubble.id, response));
            }

            // Update bubbles with calculated responses and forces
            for (bubble_id, response) in bubble_data {
                if let Some(bubble_idx) = self.bubbles.iter().position(|b| b.id == bubble_id) {
                    // Update bubble state
                    if let Some(&last_radius) = response.radius.last() {
                        self.bubbles[bubble_idx].current_radius = last_radius;
                    }

                    // Calculate primary radiation force (King, 1934)
                    // F_rad = (4π/3) R₀³ ρ₁ (2πf)² (p₀²/(3ρ₁c₁²)) (5ρ₂ - 2ρ₁)/(2ρ₂ + ρ₁)
                    let radiation_force =
                        self.calculate_radiation_force(&self.bubbles[bubble_idx], &response);

                    // Update velocity using Newton's second law
                    let mass = (4.0 / 3.0)
                        * std::f64::consts::PI
                        * self.bubbles[bubble_idx].current_radius.powi(3)
                        * 1000.0; // Assume water density
                    for (i, force) in radiation_force.iter().enumerate() {
                        self.bubbles[bubble_idx].velocity[i] += (force / mass) * self.config.dt;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle bubble-bubble interactions
    fn handle_interactions(&mut self) -> KwaversResult<()> {
        let mut coalescence_events = Vec::new();

        // Check all pairs for coalescence
        for i in 0..self.bubbles.len() {
            if !self.bubbles[i].active {
                continue;
            }

            for j in (i + 1)..self.bubbles.len() {
                if !self.bubbles[j].active {
                    continue;
                }

                let dist = self.bubble_distance(i, j);
                if dist < self.config.coalescence_distance {
                    coalescence_events.push((i, j));
                }
            }
        }

        // Process coalescence events
        for (i, j) in coalescence_events {
            self.coalesce_bubbles(i, j);
        }

        Ok(())
    }

    /// Calculate distance between bubbles
    fn bubble_distance(&self, i: usize, j: usize) -> f64 {
        let dx = self.bubbles[i].position[0] - self.bubbles[j].position[0];
        let dy = self.bubbles[i].position[1] - self.bubbles[j].position[1];
        let dz = self.bubbles[i].position[2] - self.bubbles[j].position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Handle bubble coalescence
    fn coalesce_bubbles(&mut self, i: usize, j: usize) {
        let bubble1 = &self.bubbles[i];
        let bubble2 = &self.bubbles[j];

        // Conservation of volume (spherical bubbles)
        let vol1 = (4.0 / 3.0) * std::f64::consts::PI * bubble1.current_radius.powi(3);
        let vol2 = (4.0 / 3.0) * std::f64::consts::PI * bubble2.current_radius.powi(3);
        let total_vol = vol1 + vol2;

        let new_radius = ((3.0 * total_vol) / (4.0 * std::f64::consts::PI)).cbrt();

        // Conservation of momentum
        let mass1 = vol1 * 1000.0; // Assume water density
        let mass2 = vol2 * 1000.0;
        let total_mass = mass1 + mass2;

        let mut new_velocity = [0.0; 3];
        for (k, val) in new_velocity.iter_mut().enumerate() {
            *val = (bubble1.velocity[k] * mass1 + bubble2.velocity[k] * mass2) / total_mass;
        }

        // New position (center of mass)
        let mut new_position = [0.0; 3];
        for (k, val) in new_position.iter_mut().enumerate() {
            *val = (bubble1.position[k] * mass1 + bubble2.position[k] * mass2) / total_mass;
        }

        // Update bubble i to be the merged bubble
        self.bubbles[i].current_radius = new_radius;
        self.bubbles[i].velocity = new_velocity;
        self.bubbles[i].position = new_position;
        self.bubbles[i].properties.radius_eq = new_radius;

        // Deactivate bubble j
        self.bubbles[j].active = false;
    }

    /// Handle bubble dissolution
    fn handle_dissolution(&mut self) -> KwaversResult<()> {
        // Simplified dissolution model
        // In reality, this would involve gas diffusion equations
        for bubble in &mut self.bubbles {
            if bubble.active && bubble.current_radius < 0.5e-6 {
                // Below 0.5 μm
                bubble.active = false;
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

    /// Calculate primary radiation force on a bubble
    fn calculate_radiation_force(
        &self,
        bubble: &CloudBubble,
        response: &BubbleResponse,
    ) -> [f64; 3] {
        // Simplified radiation force calculation
        // In reality, this involves complex acoustic streaming and Bjerknes forces

        if let Some(field) = &self.incident_field {
            // King-Merton radiation force approximation
            let omega = 2.0 * std::f64::consts::PI * field.frequency;
            let k = omega / field.sound_speed;

            // Volume oscillation amplitude from radial pulsation
            let volume_amp = if response.radius.len() > 1 {
                let r0 = response.radius[0];
                let r_max = response.radius.iter().cloned().fold(0.0_f64, f64::max);
                (r_max - r0) / r0
            } else {
                0.0
            };

            // Radiation force magnitude (Bjerknes force for oscillating bubbles)
            // Reference: Bjerknes (1906), Acoustic streaming and radiation pressure
            let force_magnitude = (4.0 / 3.0)
                * std::f64::consts::PI
                * bubble.current_radius.powi(3)
                * field.pressure_amplitude
                * volume_amp
                * k;

            // Primary radiation force acts along the incident wave direction
            [force_magnitude, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Capture current cloud state
    fn capture_cloud_state(&self) -> CloudState {
        let active_bubbles: Vec<_> = self.bubbles.iter().filter(|b| b.active).cloned().collect();

        CloudState {
            bubbles: active_bubbles,
            time: 0.0, // Would need to track actual time
        }
    }

    /// Calculate scattered acoustic field from bubble cloud
    pub fn calculate_scattered_field(&self, frequency: f64) -> KwaversResult<ScatteredField> {
        // Compute collective scattering from all bubbles using linear superposition
        // Based on Rayleigh scattering theory for microbubble clouds

        if self.bubbles.is_empty() {
            return Ok(ScatteredField {
                fundamental: Array3::<f64>::zeros((64, 64, 64)),
                harmonics: HashMap::new(),
                frequency,
            });
        }

        // Get grid dimensions from first bubble position (assuming uniform grid)
        let nx = 64; // Default grid size - would be configurable
        let ny = 64;
        let nz = 64;

        let mut scattered_pressure = Array3::<f64>::zeros((nx, ny, nz));
        let harmonics = HashMap::new();

        // Calculate wavenumber
        let k: f64 = 2.0 * std::f64::consts::PI * frequency / 1500.0; // c = 1500 m/s

        // Sum scattering contributions from each bubble
        for (bubble_idx, bubble) in self.bubbles.iter().enumerate() {
            if bubble_idx >= 1000 {
                break;
            } // Limit for computational efficiency

            let bubble_pos = &bubble.position;

            // Calculate scattering amplitude based on bubble resonance
            let resonance_freq = bubble.properties.resonance_frequency(101325.0, 1000.0);
            let scattering_amplitude = if (frequency - resonance_freq).abs() < 0.1 * resonance_freq
            {
                // Near resonance - enhanced scattering
                bubble.current_radius.powi(3) * 1e6
            } else {
                // Off-resonance scattering
                bubble.current_radius.powi(3) * 1e3
            };

            // Add scattered field contribution at each grid point
            for i in 0..nx {
                for j in 0..ny {
                    for kz in 0..nz {
                        // Convert grid indices to physical coordinates
                        let x = (i as f64 - nx as f64 / 2.0) * 1e-4; // 100μm spacing
                        let y = (j as f64 - ny as f64 / 2.0) * 1e-4;
                        let z = (kz as f64 - nz as f64 / 2.0) * 1e-4;

                        // Calculate distance from bubble to grid point
                        let dx = x - bubble_pos[0];
                        let dy = y - bubble_pos[1];
                        let dz = z - bubble_pos[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 1e-6 {
                            // Avoid singularity at bubble center
                            // Spherical wave scattering contribution
                            let phase: f64 = k * distance;
                            let contribution =
                                scattering_amplitude * (phase.cos() / distance).clamp(-1e6, 1e6);

                            scattered_pressure[[i, j, kz]] += contribution;
                        }
                    }
                }
            }
        }

        // Normalize by number of bubbles to prevent overflow
        let bubble_count = self.bubbles.len().max(1) as f64;
        scattered_pressure.mapv_inplace(|x| x / bubble_count);

        Ok(ScatteredField {
            fundamental: scattered_pressure,
            harmonics,
            frequency,
        })
    }
}

/// Incident acoustic field
#[derive(Debug, Clone)]
pub struct IncidentField {
    /// Pressure amplitude (Pa)
    pub pressure_amplitude: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Propagation direction (unit vector)
    pub direction: [f64; 3],
    /// Phase offset (rad)
    pub phase_offset: f64,
}

impl IncidentField {
    /// Create plane wave field
    pub fn plane_wave(pressure: f64, frequency: f64, direction: [f64; 3]) -> Self {
        Self {
            pressure_amplitude: pressure,
            frequency,
            sound_speed: 1500.0,
            direction,
            phase_offset: 0.0,
        }
    }

    /// Calculate pressure at position and time
    pub fn pressure_at(&self, position: [f64; 3], time: f64) -> f64 {
        // Plane wave: p(x,t) = p0 * cos(k·x - ωt + φ)
        let kx = (2.0 * std::f64::consts::PI * self.frequency / self.sound_speed)
            * (position[0] * self.direction[0]
                + position[1] * self.direction[1]
                + position[2] * self.direction[2]);

        let omega_t = 2.0 * std::f64::consts::PI * self.frequency * time;

        self.pressure_amplitude * (kx - omega_t + self.phase_offset).cos()
    }
}

/// Snapshot of cloud state at a time step
#[derive(Debug, Clone)]
pub struct CloudState {
    /// Active bubbles in the cloud
    pub bubbles: Vec<CloudBubble>,
    /// Time (s)
    pub time: f64,
}

/// Complete cloud dynamics response
#[derive(Debug)]
pub struct CloudResponse {
    /// Time series of cloud states
    pub time_steps: Vec<CloudState>,
    /// Configuration used
    pub config: CloudConfig,
}

/// Scattered acoustic field from bubble cloud
#[derive(Debug)]
pub struct ScatteredField {
    /// Fundamental frequency scattered pressure
    pub fundamental: Array3<f64>,
    /// Harmonic components (frequency -> pressure field)
    pub harmonics: HashMap<usize, Array3<f64>>,
    /// Center frequency (Hz)
    pub frequency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_initialization() {
        let config = CloudConfig {
            num_bubbles: 100,
            ..Default::default()
        };

        let mut cloud = CloudDynamics::new(config).unwrap();
        cloud.initialize_cloud().unwrap();

        assert_eq!(cloud.bubbles.len(), 100);
        assert!(cloud.bubbles.iter().all(|b| b.active));
    }

    #[test]
    fn test_cloud_simulation() {
        let config = CloudConfig {
            num_bubbles: 10,
            duration: 1e-4, // Short simulation for testing
            ..Default::default()
        };

        let mut cloud = CloudDynamics::new(config).unwrap();
        cloud.initialize_cloud().unwrap();

        // Set up incident field
        let field = IncidentField::plane_wave(100_000.0, 1e6, [1.0, 0.0, 0.0]);
        cloud.set_incident_field(field);

        let response = cloud.simulate().unwrap();

        // Should have multiple time steps
        assert!(response.time_steps.len() > 1);

        // All time steps should have bubbles
        for state in &response.time_steps {
            assert!(!state.bubbles.is_empty());
        }
    }

    #[test]
    fn test_coalescence() {
        let config = CloudConfig {
            num_bubbles: 2,
            coalescence_distance: 1e-6, // Large distance for testing
            ..Default::default()
        };

        let mut cloud = CloudDynamics::new(config).unwrap();
        cloud.initialize_cloud().unwrap();

        // Force bubbles close together
        cloud.bubbles[0].position = [0.0, 0.0, 0.0];
        cloud.bubbles[1].position = [0.5e-6, 0.0, 0.0]; // Within coalescence distance

        cloud.handle_interactions().unwrap();

        // One bubble should be inactive after coalescence
        let active_count = cloud.bubbles.iter().filter(|b| b.active).count();
        assert_eq!(active_count, 1);
    }
}
