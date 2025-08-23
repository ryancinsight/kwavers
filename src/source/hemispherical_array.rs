//! Hemispherical Sparse Array Transducer Implementation
//!
//! Implements hemispherical phased arrays with sparse element control for
//! increased treatment envelope and improved steering efficiency.
//! Based on Insightec ExAblate systems and modern MRgFUS technology.
//!
//! References:
//! - Clement & Hynynen (2002): "A non-invasive method for focusing through the skull"
//! - Pernot et al. (2003): "High power transcranial beam steering for ultrasonic brain therapy"
//! - Aubry et al. (2003): "Experimental demonstration of noninvasive transskull adaptive focusing"
//! - Hertzberg et al. (2010): "Ultrasound focusing using magnetic resonance acoustic radiation force imaging"
//! - Jones et al. (2019): "Transcranial MR-guided focused ultrasound: A review of the technology"

use crate::error::{ConfigError, KwaversError, KwaversResult};
use crate::grid::Grid;
use crate::signal::Signal;
use crate::source::Source;
use ndarray::{Array1, Array3};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

// Hemispherical array constants
/// Typical radius for clinical hemispherical arrays (mm)
const HEMISPHERE_RADIUS: f64 = 150e-3; // 150mm radius (Insightec ExAblate)

/// Half-wavelength element spacing for improved steering (at 650 kHz in water)
const HALF_WAVELENGTH_SPACING: f64 = 1.15e-3; // λ/2 at 650 kHz

/// Maximum steering angle from geometric focus (degrees)
const MAX_STEERING_ANGLE: f64 = 30.0;

/// Minimum element density for sparse arrays (elements per cm²)
const MIN_ELEMENT_DENSITY: f64 = 0.5;

/// Maximum element density for dense packing (elements per cm²)
const MAX_ELEMENT_DENSITY: f64 = 4.0;

/// Grating lobe threshold (dB below main lobe)
const GRATING_LOBE_THRESHOLD: f64 = -30.0;

/// Treatment envelope expansion factor with sparse arrays
const ENVELOPE_EXPANSION_FACTOR: f64 = 1.5;

/// Power efficiency threshold for element selection
const POWER_EFFICIENCY_THRESHOLD: f64 = 0.7;

/// Element configuration for hemispherical arrays
#[derive(Debug, Clone)]
pub struct HemisphereElement {
    /// Element ID
    pub id: usize,
    /// Position in Cartesian coordinates (x, y, z) [m]
    pub position: [f64; 3],
    /// Position in spherical coordinates (r, theta, phi) [rad]
    pub spherical_position: [f64; 3],
    /// Element normal vector (pointing inward)
    pub normal: [f64; 3],
    /// Element area [m²]
    pub area: f64,
    /// Element diameter [m]
    pub diameter: f64,
    /// Phase offset [rad]
    pub phase: f64,
    /// Amplitude weight [0-1]
    pub amplitude: f64,
    /// Is element active in current configuration
    pub active: bool,
    /// Element efficiency for current target
    pub efficiency: f64,
}

/// Sparse array selection strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseSelectionStrategy {
    /// Random sparse sampling
    Random { density: f64 },
    /// Spiral-based sampling (Fermat spiral)
    Spiral { pitch: f64 },
    /// Efficiency-based selection (power delivery)
    EfficiencyBased { threshold: f64 },
    /// Aperture-based selection (geometric)
    ApertureBased { angle: f64 },
    /// Adaptive selection based on target and obstacles
    Adaptive,
    /// Grating lobe suppression
    GratingLobeSuppression,
}

/// Treatment envelope optimization
#[derive(Debug, Clone)]
pub struct TreatmentEnvelope {
    /// Focal zone dimensions (x, y, z) [m]
    pub focal_zone: [f64; 3],
    /// Steering range (±x, ±y, ±z) [m]
    pub steering_range: [f64; 3],
    /// Accessible volume [m³]
    pub accessible_volume: f64,
    /// Power efficiency map
    pub efficiency_map: Array3<f64>,
    /// Grating lobe positions
    pub grating_lobes: Vec<[f64; 3]>,
}

/// Hemispherical sparse array transducer
pub struct HemisphericalArray {
    /// Array elements
    elements: Vec<HemisphereElement>,
    /// Active element indices
    active_elements: HashSet<usize>,
    /// Sparse selection strategy
    selection_strategy: SparseSelectionStrategy,
    /// Array radius [m]
    radius: f64,
    /// Operating frequency [Hz]
    frequency: f64,
    /// Wavelength [m]
    wavelength: f64,
    /// Element spacing [m]
    element_spacing: f64,
    /// Geometric focus position [m]
    geometric_focus: [f64; 3],
    /// Current steering target [m]
    steering_target: [f64; 3],
    /// Treatment envelope
    treatment_envelope: TreatmentEnvelope,
    /// Signal source
    signal: Arc<dyn Signal>,
    /// Element phase delays
    phase_delays: Array1<f64>,
    /// Element amplitude weights
    amplitude_weights: Array1<f64>,
}

impl HemisphericalArray {
    /// Create new hemispherical array with specified parameters
    pub fn new(
        radius: f64,
        frequency: f64,
        element_spacing: f64,
        signal: Arc<dyn Signal>,
    ) -> KwaversResult<Self> {
        // Validate parameters
        if radius <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius".to_string(),
                value: radius.to_string(),
                constraint: "Must be positive".to_string(),
            }));
        }
        if frequency <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }));
        }

        // Calculate wavelength (assuming water/tissue)
        let sound_speed = 1540.0; // m/s in tissue
        let wavelength = sound_speed / frequency;

        // Generate element positions
        let elements = Self::generate_hemisphere_elements(radius, element_spacing, wavelength)?;

        let num_elements = elements.len();
        let active_elements: HashSet<usize> = (0..num_elements).collect();

        // Initialize at geometric focus
        let geometric_focus = [0.0, 0.0, 0.0];
        let steering_target = geometric_focus;

        // Initialize treatment envelope
        let treatment_envelope = TreatmentEnvelope {
            focal_zone: [2e-3, 2e-3, 8e-3], // Typical HIFU focal zone
            steering_range: [
                radius * MAX_STEERING_ANGLE.to_radians().sin(),
                radius * MAX_STEERING_ANGLE.to_radians().sin(),
                radius * 0.2, // ±20% axial steering
            ],
            accessible_volume: 0.0,
            efficiency_map: Array3::ones((10, 10, 10)),
            grating_lobes: Vec::new(),
        };

        let mut array = Self {
            elements,
            active_elements,
            selection_strategy: SparseSelectionStrategy::EfficiencyBased {
                threshold: POWER_EFFICIENCY_THRESHOLD,
            },
            radius,
            frequency,
            wavelength,
            element_spacing,
            geometric_focus,
            steering_target,
            treatment_envelope,
            signal,
            phase_delays: Array1::zeros(num_elements),
            amplitude_weights: Array1::ones(num_elements),
        };

        // Calculate initial phase delays
        array.calculate_phase_delays()?;
        array.calculate_treatment_envelope()?;

        Ok(array)
    }

    /// Generate hemisphere element positions with optimal packing
    fn generate_hemisphere_elements(
        radius: f64,
        spacing: f64,
        wavelength: f64,
    ) -> KwaversResult<Vec<HemisphereElement>> {
        let mut elements = Vec::new();
        let mut id = 0;

        // Use spiral point distribution for uniform coverage
        // Based on Saff & Kuijlaars (1997) algorithm
        let area_per_element = spacing * spacing;
        let hemisphere_area = 2.0 * PI * radius * radius;
        let approx_num_elements = (hemisphere_area / area_per_element) as usize;

        // Golden angle in radians
        let golden_angle = PI * (3.0 - (5.0_f64).sqrt());

        for i in 0..approx_num_elements {
            let y = 1.0 - (i as f64 / (approx_num_elements - 1) as f64);
            let radius_at_y = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;

            // Convert to spherical coordinates
            let phi = theta;
            let theta_spherical = y.acos();

            // Skip elements below hemisphere (theta > π/2)
            if theta_spherical > PI / 2.0 {
                continue;
            }

            // Convert to Cartesian
            let x = radius * theta_spherical.sin() * phi.cos();
            let y_cart = radius * theta_spherical.sin() * phi.sin();
            let z = radius * theta_spherical.cos();

            // Calculate element normal (pointing inward)
            let normal = [-x / radius, -y_cart / radius, -z / radius];

            // Element diameter based on spacing
            let diameter = spacing * 0.9; // 90% fill factor
            let area = PI * (diameter / 2.0).powi(2);

            let element = HemisphereElement {
                id,
                position: [x, y_cart, z],
                spherical_position: [radius, theta_spherical, phi],
                normal,
                area,
                diameter,
                phase: 0.0,
                amplitude: 1.0,
                active: true,
                efficiency: 1.0,
            };

            elements.push(element);
            id += 1;
        }

        Ok(elements)
    }

    /// Apply sparse element selection strategy
    pub fn apply_sparse_selection(
        &mut self,
        strategy: SparseSelectionStrategy,
    ) -> KwaversResult<()> {
        self.selection_strategy = strategy;
        self.active_elements.clear();

        match strategy {
            SparseSelectionStrategy::Random { density } => {
                self.select_random_sparse(density)?;
            }
            SparseSelectionStrategy::Spiral { pitch } => {
                self.select_spiral_sparse(pitch)?;
            }
            SparseSelectionStrategy::EfficiencyBased { threshold } => {
                self.select_efficiency_based(threshold)?;
            }
            SparseSelectionStrategy::ApertureBased { angle } => {
                self.select_aperture_based(angle)?;
            }
            SparseSelectionStrategy::Adaptive => {
                self.select_adaptive()?;
            }
            SparseSelectionStrategy::GratingLobeSuppression => {
                self.select_grating_lobe_suppression()?;
            }
        }

        // Update element states
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.active = self.active_elements.contains(&i);
        }

        // Recalculate phase delays for active elements
        self.calculate_phase_delays()?;

        Ok(())
    }

    /// Random sparse selection
    fn select_random_sparse(&mut self, density: f64) -> KwaversResult<()> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let total_elements = self.elements.len();
        let target_count = (total_elements as f64 * density).round() as usize;

        let mut indices: Vec<usize> = (0..total_elements).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        self.active_elements = indices.iter().take(target_count).cloned().collect();

        Ok(())
    }

    /// Spiral-based sparse selection (Fermat spiral)
    fn select_spiral_sparse(&mut self, pitch: f64) -> KwaversResult<()> {
        // Select elements along a Fermat spiral pattern
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let golden_angle = 2.0 * PI / (golden_ratio * golden_ratio);

        for (i, element) in self.elements.iter().enumerate() {
            let r = element.spherical_position[0];
            let theta = element.spherical_position[1];
            let phi = element.spherical_position[2];

            // Check if element lies near spiral
            let spiral_angle = (r / pitch).sqrt() * golden_angle;
            let angle_diff = (phi - spiral_angle).abs();

            if angle_diff < PI / 10.0 {
                // Within 18 degrees of spiral
                self.active_elements.insert(i);
            }
        }

        Ok(())
    }

    /// Efficiency-based selection for optimal power delivery
    fn select_efficiency_based(&mut self, threshold: f64) -> KwaversResult<()> {
        // Calculate efficiency for each element based on target
        let steering_target = self.steering_target;
        let radius = self.radius;
        for (i, element) in self.elements.iter_mut().enumerate() {
            let efficiency =
                Self::calculate_element_efficiency_static(element, &steering_target, radius);
            element.efficiency = efficiency;

            if efficiency >= threshold {
                self.active_elements.insert(i);
            }
        }

        Ok(())
    }

    /// Aperture-based selection within steering angle
    fn select_aperture_based(&mut self, angle: f64) -> KwaversResult<()> {
        let angle_rad = angle.to_radians();

        for (i, element) in self.elements.iter().enumerate() {
            // Calculate angle between element normal and target direction
            let target_dir = [
                self.steering_target[0] - element.position[0],
                self.steering_target[1] - element.position[1],
                self.steering_target[2] - element.position[2],
            ];

            let target_mag =
                (target_dir[0].powi(2) + target_dir[1].powi(2) + target_dir[2].powi(2)).sqrt();

            let target_unit = [
                target_dir[0] / target_mag,
                target_dir[1] / target_mag,
                target_dir[2] / target_mag,
            ];

            // Dot product for angle
            let cos_angle = element.normal[0] * target_unit[0]
                + element.normal[1] * target_unit[1]
                + element.normal[2] * target_unit[2];

            let element_angle = cos_angle.acos();

            if element_angle <= angle_rad {
                self.active_elements.insert(i);
            }
        }

        Ok(())
    }

    /// Adaptive selection based on target and obstacles
    fn select_adaptive(&mut self) -> KwaversResult<()> {
        // Start with efficiency-based selection
        self.select_efficiency_based(POWER_EFFICIENCY_THRESHOLD)?;

        // Remove elements that would create grating lobes
        self.suppress_grating_lobes()?;

        // Ensure minimum element count for focusing
        let min_elements = (self.elements.len() as f64 * 0.3) as usize;
        if self.active_elements.len() < min_elements {
            // Add back most efficient elements
            let mut efficiencies: Vec<(usize, f64)> = self
                .elements
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.efficiency))
                .collect();
            efficiencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, _) in efficiencies.iter().take(min_elements) {
                self.active_elements.insert(*i);
            }
        }

        Ok(())
    }

    /// Select elements to suppress grating lobes
    fn select_grating_lobe_suppression(&mut self) -> KwaversResult<()> {
        // Use randomized sparse array to break periodicity
        self.select_random_sparse(0.7)?;

        // Add strategic elements to suppress specific grating lobes
        let grating_positions = self.calculate_grating_lobe_positions();

        for grating_pos in grating_positions {
            // Find elements that can destructively interfere
            for (i, element) in self.elements.iter().enumerate() {
                if self.can_suppress_grating_lobe(element, &grating_pos) {
                    self.active_elements.insert(i);
                }
            }
        }

        Ok(())
    }

    /// Calculate element efficiency for power delivery to target
    fn calculate_element_efficiency(&self, element: &HemisphereElement, target: &[f64; 3]) -> f64 {
        Self::calculate_element_efficiency_static(element, target, self.radius)
    }

    fn calculate_element_efficiency_static(
        element: &HemisphereElement,
        target: &[f64; 3],
        radius: f64,
    ) -> f64 {
        // Distance from element to target
        let dx = target[0] - element.position[0];
        let dy = target[1] - element.position[1];
        let dz = target[2] - element.position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Direction to target
        let dir = [dx / distance, dy / distance, dz / distance];

        // Angle between element normal and target direction
        let cos_angle =
            -(element.normal[0] * dir[0] + element.normal[1] * dir[1] + element.normal[2] * dir[2]);

        // Efficiency based on:
        // 1. Geometric factor (cos angle)
        // 2. Distance attenuation (1/r)
        // 3. Directivity pattern (piston source)
        let geometric_factor = cos_angle.max(0.0);
        let distance_factor = (radius / distance).min(1.0);

        // Proper piston source directivity pattern
        // Using sinc function for circular piston radiator
        // Reference: Kino, G. S. (1987). Acoustic waves: devices, imaging, and analog signal processing
        let angle = cos_angle.acos();
        let directivity = if angle.abs() < 1e-10 {
            1.0 // On-axis response
        } else {
            // For piston source: D(θ) = |2J₁(ka·sin(θ))/(ka·sin(θ))|
            // Approximation for typical element size relative to wavelength
            let normalized_angle = angle / std::f64::consts::PI;
            let sinc = if normalized_angle.abs() < 1e-10 {
                1.0
            } else {
                (std::f64::consts::PI * normalized_angle).sin()
                    / (std::f64::consts::PI * normalized_angle)
            };
            sinc.abs()
        };

        geometric_factor * distance_factor * directivity
    }

    /// Calculate element directivity pattern
    fn calculate_directivity(&self, cos_angle: f64) -> f64 {
        // Piston source directivity
        let ka = 2.0 * PI * self.element_spacing / (2.0 * self.wavelength);
        let angle = cos_angle.acos();

        if angle == 0.0 {
            1.0
        } else {
            let x = ka * angle.sin();
            (2.0 * x.j1() / x).abs() // J1 is Bessel function of first kind
        }
    }

    /// Calculate phase delays for focusing
    pub fn calculate_phase_delays(&mut self) -> KwaversResult<()> {
        let sound_speed = 1540.0; // m/s

        for (i, element) in self.elements.iter_mut().enumerate() {
            if !self.active_elements.contains(&i) {
                self.phase_delays[i] = 0.0;
                continue;
            }

            // Calculate distance from element to target
            let dx = self.steering_target[0] - element.position[0];
            let dy = self.steering_target[1] - element.position[1];
            let dz = self.steering_target[2] - element.position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Phase delay for focusing
            let time_delay = distance / sound_speed;
            let phase_delay = -2.0 * PI * self.frequency * time_delay;

            self.phase_delays[i] = phase_delay;
            element.phase = phase_delay;
        }

        // Normalize to minimum phase
        let min_phase = self
            .phase_delays
            .iter()
            .enumerate()
            .filter(|(i, _)| self.active_elements.contains(i))
            .map(|(_, &p)| p)
            .fold(f64::INFINITY, f64::min);

        if min_phase.is_finite() {
            for i in &self.active_elements {
                self.phase_delays[*i] -= min_phase;
            }
        }

        Ok(())
    }

    /// Calculate treatment envelope with current configuration
    pub fn calculate_treatment_envelope(&mut self) -> KwaversResult<()> {
        // Calculate accessible volume
        let volume = 4.0 / 3.0
            * PI
            * self.treatment_envelope.steering_range[0]
            * self.treatment_envelope.steering_range[1]
            * self.treatment_envelope.steering_range[2];

        self.treatment_envelope.accessible_volume = volume * ENVELOPE_EXPANSION_FACTOR;

        // Calculate grating lobe positions
        self.treatment_envelope.grating_lobes = self.calculate_grating_lobe_positions();

        // Update efficiency map
        self.update_efficiency_map()?;

        Ok(())
    }

    /// Calculate grating lobe positions
    fn calculate_grating_lobe_positions(&self) -> Vec<[f64; 3]> {
        let mut grating_lobes = Vec::new();

        // For sparse arrays, grating lobes depend on element distribution
        let avg_spacing = self.calculate_average_element_spacing();

        if avg_spacing > self.wavelength {
            // Grating lobes will appear
            let grating_order_max = (avg_spacing / self.wavelength).floor() as i32;

            for m in -grating_order_max..=grating_order_max {
                if m == 0 {
                    continue;
                } // Skip main lobe

                for n in -grating_order_max..=grating_order_max {
                    let theta_g = (m as f64 * self.wavelength / avg_spacing).asin();
                    let phi_g = (n as f64 * self.wavelength / avg_spacing).asin();

                    if theta_g.is_finite() && phi_g.is_finite() {
                        let x = self.radius * theta_g.sin() * phi_g.cos();
                        let y = self.radius * theta_g.sin() * phi_g.sin();
                        let z = self.radius * theta_g.cos();

                        grating_lobes.push([x, y, z]);
                    }
                }
            }
        }

        grating_lobes
    }

    /// Calculate average element spacing for active elements
    fn calculate_average_element_spacing(&self) -> f64 {
        if self.active_elements.len() < 2 {
            return self.element_spacing;
        }

        let mut total_spacing = 0.0;
        let mut count = 0;

        for &i in &self.active_elements {
            for &j in &self.active_elements {
                if i >= j {
                    continue;
                }

                let e1 = &self.elements[i];
                let e2 = &self.elements[j];

                let dx = e1.position[0] - e2.position[0];
                let dy = e1.position[1] - e2.position[1];
                let dz = e1.position[2] - e2.position[2];
                let spacing = (dx * dx + dy * dy + dz * dz).sqrt();

                total_spacing += spacing;
                count += 1;
            }
        }

        if count > 0 {
            total_spacing / count as f64
        } else {
            self.element_spacing
        }
    }

    /// Update efficiency map for treatment envelope
    fn update_efficiency_map(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.treatment_envelope.efficiency_map.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Map indices to physical coordinates
                    let x = (i as f64 / (nx - 1) as f64 - 0.5)
                        * 2.0
                        * self.treatment_envelope.steering_range[0];
                    let y = (j as f64 / (ny - 1) as f64 - 0.5)
                        * 2.0
                        * self.treatment_envelope.steering_range[1];
                    let z = (k as f64 / (nz - 1) as f64 - 0.5)
                        * 2.0
                        * self.treatment_envelope.steering_range[2];

                    let target = [x, y, z];

                    // Calculate total efficiency at this point
                    let mut total_efficiency = 0.0;
                    for &idx in &self.active_elements {
                        let element = &self.elements[idx];
                        total_efficiency += self.calculate_element_efficiency(element, &target);
                    }

                    self.treatment_envelope.efficiency_map[(i, j, k)] =
                        total_efficiency / self.active_elements.len() as f64;
                }
            }
        }

        Ok(())
    }

    /// Suppress grating lobes by element selection
    fn suppress_grating_lobes(&mut self) -> KwaversResult<()> {
        let grating_positions = self.calculate_grating_lobe_positions();

        // Remove elements that strongly contribute to grating lobes
        let mut elements_to_remove = HashSet::new();

        for grating_pos in &grating_positions {
            for &i in &self.active_elements {
                let element = &self.elements[i];

                // Check if element contributes significantly to this grating lobe
                if self.contributes_to_grating_lobe(element, grating_pos) {
                    elements_to_remove.insert(i);
                }
            }
        }

        // Remove problematic elements
        for i in elements_to_remove {
            self.active_elements.remove(&i);
        }

        Ok(())
    }

    /// Check if element can suppress a grating lobe
    fn can_suppress_grating_lobe(
        &self,
        element: &HemisphereElement,
        grating_pos: &[f64; 3],
    ) -> bool {
        // Element can suppress if it's positioned to create destructive interference
        let dx = grating_pos[0] - element.position[0];
        let dy = grating_pos[1] - element.position[1];
        let dz = grating_pos[2] - element.position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Check if element is at half-wavelength offset
        let phase_diff = (2.0 * PI * distance / self.wavelength) % (2.0 * PI);
        (phase_diff - PI).abs() < PI / 4.0 // Within π/4 of destructive interference
    }

    /// Check if element contributes to grating lobe
    fn contributes_to_grating_lobe(
        &self,
        element: &HemisphereElement,
        grating_pos: &[f64; 3],
    ) -> bool {
        // Calculate element's contribution to grating lobe
        let efficiency = self.calculate_element_efficiency(element, grating_pos);
        efficiency > 0.1 // Significant contribution threshold
    }

    /// Set steering target
    pub fn set_steering_target(&mut self, target: [f64; 3]) -> KwaversResult<()> {
        // Validate steering angle
        let dx = target[0];
        let dy = target[1];
        let dz = target[2];
        let lateral_distance = (dx * dx + dy * dy).sqrt();
        let steering_angle = (lateral_distance / dz).atan().to_degrees();

        if steering_angle.abs() > MAX_STEERING_ANGLE {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "steering_angle".to_string(),
                value: format!("{:.1}°", steering_angle),
                constraint: format!("Must be <= {:.1}°", MAX_STEERING_ANGLE),
            }));
        }

        self.steering_target = target;
        self.calculate_phase_delays()?;

        Ok(())
    }

    /// Get active element count
    pub fn get_active_element_count(&self) -> usize {
        self.active_elements.len()
    }

    /// Get sparsity ratio
    pub fn get_sparsity_ratio(&self) -> f64 {
        self.active_elements.len() as f64 / self.elements.len() as f64
    }

    /// Get treatment envelope
    pub fn get_treatment_envelope(&self) -> &TreatmentEnvelope {
        &self.treatment_envelope
    }
}

// Bessel function approximation for directivity calculation
trait BesselJ1 {
    fn j1(self) -> f64;
}

impl BesselJ1 for f64 {
    fn j1(self) -> f64 {
        // Approximation of Bessel function J1
        let x = self.abs();
        if x < 3.0 {
            let x2 = x * x;
            x * (0.5 - x2 / 8.0 + x2 * x2 / 192.0)
        } else {
            let inv_x = 1.0 / x;
            (2.0 / (PI * x)).sqrt() * ((x - 3.0 * PI / 4.0).cos() * (1.0 - 0.375 * inv_x * inv_x))
        }
    }
}

impl Debug for HemisphericalArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HemisphericalArray {{ radius: {:.1}mm, elements: {} ({} active), frequency: {:.1}kHz, strategy: {:?} }}",
               self.radius * 1000.0,
               self.elements.len(),
               self.active_elements.len(),
               self.frequency / 1000.0,
               self.selection_strategy)
    }
}

impl Source for HemisphericalArray {
    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // For each active element, add its contribution to the mask
        for &idx in &self.active_elements {
            let element = &self.elements[idx];

            // Find nearest grid point to element position (assuming grid starts at origin)
            let i =
                ((element.position[0] + grid.nx as f64 * grid.dx / 2.0) / grid.dx).round() as usize;
            let j =
                ((element.position[1] + grid.ny as f64 * grid.dy / 2.0) / grid.dy).round() as usize;
            let k =
                ((element.position[2] + grid.nz as f64 * grid.dz / 2.0) / grid.dz).round() as usize;

            if i < grid.nx && j < grid.ny && k < grid.nz {
                mask[(i, j, k)] = element.amplitude;
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.active_elements
            .iter()
            .map(|&idx| {
                let e = &self.elements[idx];
                (e.position[0], e.position[1], e.position[2])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::SineWave;

    #[test]
    fn test_hemisphere_generation() {
        let signal = Arc::new(SineWave::new(650e3, 1.0, 0.0));
        let array = HemisphericalArray::new(
            0.15,    // 150mm radius
            650e3,   // 650 kHz
            1.15e-3, // Half wavelength
            signal,
        )
        .unwrap();

        assert!(array.elements.len() > 1000); // Should have many elements
        assert!(array.get_sparsity_ratio() == 1.0); // Initially all active
    }

    #[test]
    fn test_sparse_selection() {
        let signal = Arc::new(SineWave::new(650e3, 1.0, 0.0));
        let mut array = HemisphericalArray::new(0.15, 650e3, 1.15e-3, signal).unwrap();

        // Apply random sparse selection
        array
            .apply_sparse_selection(SparseSelectionStrategy::Random { density: 0.5 })
            .unwrap();

        assert!((array.get_sparsity_ratio() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_steering_limits() {
        let signal = Arc::new(SineWave::new(650e3, 1.0, 0.0));
        let mut array = HemisphericalArray::new(0.15, 650e3, 1.15e-3, signal).unwrap();

        // Test valid steering
        assert!(array.set_steering_target([0.02, 0.0, 0.1]).is_ok());

        // Test excessive steering
        assert!(array.set_steering_target([0.1, 0.0, 0.1]).is_err());
    }

    #[test]
    fn test_efficiency_calculation() {
        let signal = Arc::new(SineWave::new(650e3, 1.0, 0.0));
        let mut array = HemisphericalArray::new(0.15, 650e3, 1.15e-3, signal).unwrap();

        // Apply efficiency-based selection
        array
            .apply_sparse_selection(SparseSelectionStrategy::EfficiencyBased { threshold: 0.7 })
            .unwrap();

        // Should select high-efficiency elements
        assert!(array.get_sparsity_ratio() > 0.3);
        assert!(array.get_sparsity_ratio() < 0.8);
    }
}
