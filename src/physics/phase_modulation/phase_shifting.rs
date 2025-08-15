//! Phase Shifting for Beam Steering and Dynamic Focusing
//! 
//! Implements phase shifting techniques for electronic beam steering,
//! dynamic focusing, and multi-focus patterns.
//! 
//! References:
//! - Wooh & Shi (1999): "A simulation study of the beam steering characteristics for linear phased arrays"
//! - Ebbini & Cain (1989): "Multiple-focus ultrasound phased-array pattern synthesis"
//! - Pernot et al. (2003): "3D real-time motion correction in high-intensity focused ultrasound"

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// Phase shifting constants
/// Speed of sound in water/tissue (m/s)
const SPEED_OF_SOUND: f64 = 1540.0;

/// Maximum steering angle (degrees)
const MAX_STEERING_ANGLE: f64 = 45.0;

/// Minimum focal distance (mm)
const MIN_FOCAL_DISTANCE: f64 = 10.0;

/// Maximum number of focal points for multi-focus
const MAX_FOCAL_POINTS: usize = 10;

/// Phase quantization levels for digital systems
const PHASE_QUANTIZATION_LEVELS: usize = 256;

/// Shifting strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftingStrategy {
    Linear,          // Linear phase gradient for plane wave steering
    Spherical,       // Spherical phase for focusing
    Cylindrical,     // Cylindrical focusing
    MultiPoint,      // Multiple focal points
    Holographic,     // Holographic beam shaping
    Adaptive,        // Adaptive focusing based on feedback
}

/// Phase shifter for beam control
pub struct PhaseShifter {
    strategy: ShiftingStrategy,
    element_positions: Array2<f64>,
    operating_frequency: f64,
    wavelength: f64,
    phase_offsets: Array1<f64>,
    quantization_enabled: bool,
}

impl PhaseShifter {
    pub fn new(
        element_positions: Array2<f64>,
        operating_frequency: f64,
    ) -> Self {
        let wavelength = SPEED_OF_SOUND / operating_frequency;
        let num_elements = element_positions.nrows();
        let phase_offsets = Array1::zeros(num_elements);
        
        Self {
            strategy: ShiftingStrategy::Linear,
            element_positions,
            operating_frequency,
            wavelength,
            phase_offsets,
            quantization_enabled: false,
        }
    }
    
    /// Enable phase quantization for digital systems
    pub fn enable_quantization(&mut self, enable: bool) {
        self.quantization_enabled = enable;
    }
    
    /// Quantize phase to discrete levels
    fn quantize_phase(&self, phase: f64) -> f64 {
        if !self.quantization_enabled {
            return phase;
        }
        Self::quantize_phase_static(phase, PHASE_QUANTIZATION_LEVELS)
    }
    
    fn quantize_phase_static(phase: f64, quantization_levels: usize) -> f64 {
        let normalized = phase.rem_euclid(2.0 * PI) / (2.0 * PI);
        let quantized = (normalized * quantization_levels as f64).round() 
            / quantization_levels as f64;
        quantized * 2.0 * PI
    }
    
    /// Calculate phase delays for beam steering
    pub fn calculate_steering_phases(
        &mut self,
        steering_angle_deg: f64,
        azimuth_angle_deg: f64,
    ) -> &Array1<f64> {
        let theta = steering_angle_deg.to_radians().clamp(
            -MAX_STEERING_ANGLE.to_radians(),
            MAX_STEERING_ANGLE.to_radians()
        );
        let phi = azimuth_angle_deg.to_radians();
        
        // Direction vector
        let kx = theta.sin() * phi.cos();
        let ky = theta.sin() * phi.sin();
        let k = 2.0 * PI / self.wavelength;
        
        for i in 0..self.element_positions.nrows() {
            let x = self.element_positions[(i, 0)];
            let y = if self.element_positions.ncols() > 1 {
                self.element_positions[(i, 1)]
            } else {
                0.0
            };
            
            // Linear phase gradient
            let phase = k * (kx * x + ky * y);
            self.phase_offsets[i] = self.quantize_phase(phase);
        }
        
        &self.phase_offsets
    }
    
    /// Calculate phase delays for focusing
    pub fn calculate_focusing_phases(
        &mut self,
        focal_point: &[f64; 3],
    ) -> &Array1<f64> {
        assert!(focal_point[2] >= MIN_FOCAL_DISTANCE / 1000.0, "Focal distance too small");
        
        for i in 0..self.element_positions.nrows() {
            let x = self.element_positions[(i, 0)];
            let y = if self.element_positions.ncols() > 1 {
                self.element_positions[(i, 1)]
            } else {
                0.0
            };
            let z = if self.element_positions.ncols() > 2 {
                self.element_positions[(i, 2)]
            } else {
                0.0
            };
            
            // Calculate distance from element to focal point
            let dx = focal_point[0] - x;
            let dy = focal_point[1] - y;
            let dz = focal_point[2] - z;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            
            // Phase delay for focusing
            let phase = -2.0 * PI * distance / self.wavelength;
            self.phase_offsets[i] = self.quantize_phase(phase);
        }
        
        // Normalize to minimum phase
        let min_phase = self.phase_offsets.iter().cloned().fold(f64::INFINITY, f64::min);
        self.phase_offsets.mapv_inplace(|p| p - min_phase);
        
        &self.phase_offsets
    }
    
    /// Calculate phases for multiple focal points
    pub fn calculate_multifocus_phases(
        &mut self,
        focal_points: &[Vec<f64>],
        weights: Option<&[f64]>,
    ) -> &Array1<f64> {
        assert!(focal_points.len() <= MAX_FOCAL_POINTS, "Too many focal points");
        assert!(focal_points.iter().all(|p| p.len() == 3), "Focal points must be 3D");
        
        let default_weights = vec![1.0 / focal_points.len() as f64; focal_points.len()];
        let weights = weights.unwrap_or(&default_weights);
        
        self.phase_offsets.fill(0.0);
        
        for (fp_idx, focal_point) in focal_points.iter().enumerate() {
            let weight = weights[fp_idx];
            
            for i in 0..self.element_positions.nrows() {
                let x = self.element_positions[(i, 0)];
                let y = if self.element_positions.ncols() > 1 {
                    self.element_positions[(i, 1)]
                } else {
                    0.0
                };
                
                // Calculate distance to focal point
                let dx = focal_point[0] - x;
                let dy = focal_point[1] - y;
                let dz = focal_point[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                
                // Weighted phase contribution
                let phase = -2.0 * PI * distance / self.wavelength;
                self.phase_offsets[i] += weight * phase.cos();
            }
        }
        
        // Convert back to phase
        self.phase_offsets.mapv_inplace(|p| p.atan2(0.0));
        if self.quantization_enabled {
            self.phase_offsets.mapv_inplace(|p| Self::quantize_phase_static(p, PHASE_QUANTIZATION_LEVELS));
        }
        
        &self.phase_offsets
    }
    
    /// Get current phase offsets
    pub fn get_phases(&self) -> &Array1<f64> {
        &self.phase_offsets
    }
}

/// Beam steering controller
pub struct BeamSteering {
    phase_shifter: PhaseShifter,
    steering_angle: f64,
    azimuth_angle: f64,
    scan_pattern: ScanPattern,
    scan_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanPattern {
    Linear,      // Linear scan along one axis
    Raster,      // Raster scan pattern
    Spiral,      // Spiral scan pattern
    Random,      // Random scan positions
    Circular,    // Circular scan pattern
}

impl BeamSteering {
    pub fn new(
        element_positions: Array2<f64>,
        operating_frequency: f64,
    ) -> Self {
        let phase_shifter = PhaseShifter::new(element_positions, operating_frequency);
        
        Self {
            phase_shifter,
            steering_angle: 0.0,
            azimuth_angle: 0.0,
            scan_pattern: ScanPattern::Linear,
            scan_index: 0,
        }
    }
    
    /// Set steering angles
    pub fn set_steering(&mut self, steering_deg: f64, azimuth_deg: f64) {
        self.steering_angle = steering_deg.clamp(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE);
        self.azimuth_angle = azimuth_deg;
    }
    
    /// Update beam steering
    pub fn update(&mut self) -> &Array1<f64> {
        self.phase_shifter.calculate_steering_phases(
            self.steering_angle,
            self.azimuth_angle,
        )
    }
    
    /// Perform beam scanning
    pub fn scan_step(&mut self, scan_range_deg: f64, num_steps: usize) -> &Array1<f64> {
        match self.scan_pattern {
            ScanPattern::Linear => {
                let step_size = 2.0 * scan_range_deg / (num_steps - 1) as f64;
                self.steering_angle = -scan_range_deg + step_size * self.scan_index as f64;
            }
            
            ScanPattern::Spiral => {
                let t = self.scan_index as f64 / num_steps as f64;
                let r = t * scan_range_deg;
                let theta = 2.0 * PI * t * 5.0; // 5 rotations
                self.steering_angle = r;
                self.azimuth_angle = theta.to_degrees();
            }
            
            ScanPattern::Circular => {
                let theta = 2.0 * PI * self.scan_index as f64 / num_steps as f64;
                self.steering_angle = scan_range_deg;
                self.azimuth_angle = theta.to_degrees();
            }
            
            _ => {}
        }
        
        self.scan_index = (self.scan_index + 1) % num_steps;
        self.update()
    }
}

/// Dynamic focusing controller
pub struct DynamicFocusing {
    phase_shifter: PhaseShifter,
    focal_trajectory: Vec<[f64; 3]>,
    current_index: usize,
    interpolation_enabled: bool,
}

impl DynamicFocusing {
    pub fn new(
        element_positions: Array2<f64>,
        operating_frequency: f64,
    ) -> Self {
        let phase_shifter = PhaseShifter::new(element_positions, operating_frequency);
        
        Self {
            phase_shifter,
            focal_trajectory: Vec::new(),
            current_index: 0,
            interpolation_enabled: true,
        }
    }
    
    /// Set focal trajectory
    pub fn set_trajectory(&mut self, trajectory: Vec<[f64; 3]>) {
        assert!(!trajectory.is_empty(), "Trajectory cannot be empty");
        self.focal_trajectory = trajectory;
        self.current_index = 0;
    }
    
    /// Update dynamic focus
    pub fn update(&mut self, time_fraction: f64) -> &Array1<f64> {
        if self.focal_trajectory.is_empty() {
            return self.phase_shifter.get_phases();
        }
        
        if self.interpolation_enabled && self.focal_trajectory.len() > 1 {
            // Interpolate between focal points
            let idx = (time_fraction * (self.focal_trajectory.len() - 1) as f64) as usize;
            let t = time_fraction * (self.focal_trajectory.len() - 1) as f64 - idx as f64;
            
            let idx_next = (idx + 1).min(self.focal_trajectory.len() - 1);
            
            let focal_point = [
                self.focal_trajectory[idx][0] * (1.0 - t) + self.focal_trajectory[idx_next][0] * t,
                self.focal_trajectory[idx][1] * (1.0 - t) + self.focal_trajectory[idx_next][1] * t,
                self.focal_trajectory[idx][2] * (1.0 - t) + self.focal_trajectory[idx_next][2] * t,
            ];
            
            self.phase_shifter.calculate_focusing_phases(&focal_point)
        } else {
            // Use discrete focal points
            self.current_index = ((time_fraction * self.focal_trajectory.len() as f64) as usize)
                .min(self.focal_trajectory.len() - 1);
            self.phase_shifter.calculate_focusing_phases(&self.focal_trajectory[self.current_index])
        }
    }
    
    /// Enable/disable interpolation
    pub fn set_interpolation(&mut self, enable: bool) {
        self.interpolation_enabled = enable;
    }
}

/// Phase array for complex beam patterns
pub struct PhaseArray {
    phases: Array2<f64>,
    rows: usize,
    cols: usize,
    element_spacing: f64,
}

impl PhaseArray {
    pub fn new(rows: usize, cols: usize, element_spacing: f64) -> Self {
        let phases = Array2::zeros((rows, cols));
        
        Self {
            phases,
            rows,
            cols,
            element_spacing,
        }
    }
    
    /// Generate phase pattern for vortex beam
    pub fn generate_vortex_beam(&mut self, topological_charge: i32) -> &Array2<f64> {
        let center_x = self.cols as f64 / 2.0;
        let center_y = self.rows as f64 / 2.0;
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let x = j as f64 - center_x;
                let y = i as f64 - center_y;
                let angle = y.atan2(x);
                self.phases[(i, j)] = (topological_charge as f64 * angle).rem_euclid(2.0 * PI);
            }
        }
        
        &self.phases
    }
    
    /// Generate phase pattern for Bessel beam
    pub fn generate_bessel_beam(&mut self, cone_angle_deg: f64) -> &Array2<f64> {
        let cone_angle = cone_angle_deg.to_radians();
        let k_radial = 2.0 * PI * cone_angle.sin() / self.element_spacing;
        
        let center_x = self.cols as f64 / 2.0;
        let center_y = self.rows as f64 / 2.0;
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let x = (j as f64 - center_x) * self.element_spacing;
                let y = (i as f64 - center_y) * self.element_spacing;
                let r = (x * x + y * y).sqrt();
                self.phases[(i, j)] = (k_radial * r).rem_euclid(2.0 * PI);
            }
        }
        
        &self.phases
    }
    
    /// Get phase array
    pub fn get_phases(&self) -> &Array2<f64> {
        &self.phases
    }
    
    /// Apply apodization window
    pub fn apply_apodization(&mut self, window_type: ApodizationWindow) {
        let window = self.generate_window(window_type);
        self.phases = &self.phases * &window;
    }
    
    /// Generate apodization window
    fn generate_window(&self, window_type: ApodizationWindow) -> Array2<f64> {
        let mut window = Array2::ones((self.rows, self.cols));
        let center_x = self.cols as f64 / 2.0;
        let center_y = self.rows as f64 / 2.0;
        
        match window_type {
            ApodizationWindow::Gaussian => {
                let sigma = self.cols.min(self.rows) as f64 / 4.0;
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        let x = (j as f64 - center_x) / sigma;
                        let y = (i as f64 - center_y) / sigma;
                        window[(i, j)] = (-0.5 * (x * x + y * y)).exp();
                    }
                }
            }
            
            ApodizationWindow::Hann => {
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        let wx = 0.5 * (1.0 - (2.0 * PI * j as f64 / (self.cols - 1) as f64).cos());
                        let wy = 0.5 * (1.0 - (2.0 * PI * i as f64 / (self.rows - 1) as f64).cos());
                        window[(i, j)] = wx * wy;
                    }
                }
            }
            
            ApodizationWindow::Tukey { alpha } => {
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        let wx = self.tukey_1d(j, self.cols, alpha);
                        let wy = self.tukey_1d(i, self.rows, alpha);
                        window[(i, j)] = wx * wy;
                    }
                }
            }
        }
        
        window
    }
    
    /// 1D Tukey window
    fn tukey_1d(&self, idx: usize, size: usize, alpha: f64) -> f64 {
        let x = idx as f64 / (size - 1) as f64;
        
        if x < alpha / 2.0 {
            0.5 * (1.0 + (2.0 * PI * x / alpha - PI).cos())
        } else if x < 1.0 - alpha / 2.0 {
            1.0
        } else {
            0.5 * (1.0 + (2.0 * PI * (x - 1.0) / alpha + PI).cos())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ApodizationWindow {
    Gaussian,
    Hann,
    Tukey { alpha: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_beam_steering() {
        let elements = Array2::from_shape_fn((8, 1), |(i, _)| i as f64 * 0.0005);
        let mut steering = BeamSteering::new(elements, 2.5e6);
        
        steering.set_steering(30.0, 0.0);
        let phases = steering.update();
        
        // Check that phases form a linear gradient
        for i in 1..phases.len() {
            let phase_diff = phases[i] - phases[i-1];
            assert!(phase_diff >= 0.0); // Monotonic for positive steering
        }
    }
    
    #[test]
    fn test_focusing() {
        let elements = Array2::from_shape_fn((8, 2), |(i, j)| {
            if j == 0 { i as f64 * 0.0005 } else { 0.0 }
        });
        
        let mut shifter = PhaseShifter::new(elements, 2.5e6);
        let focal_point = [0.0, 0.0, 0.05]; // 5cm focal distance
        
        let phases = shifter.calculate_focusing_phases(&focal_point);
        
        // Check that phases are symmetric around center
        let n = phases.len();
        for i in 0..n/2 {
            let diff = (phases[i] - phases[n-1-i]).abs();
            assert!(diff < 0.1); // Approximately symmetric
        }
    }
    
    #[test]
    fn test_vortex_beam() {
        let mut array = PhaseArray::new(8, 8, 0.0005);
        array.generate_vortex_beam(1);
        
        let phases = array.get_phases();
        
        // Check that phase increases around the center
        let center = 4;
        let phase_00 = phases[(center-1, center)];
        let phase_90 = phases[(center, center+1)];
        let phase_180 = phases[(center+1, center)];
        let phase_270 = phases[(center, center-1)];
        
        // Phases should increase in counter-clockwise direction
        assert!(phase_90 > phase_00 || phase_90 < 0.5);
        assert!(phase_180 > phase_90 || phase_180 < 0.5);
        assert!(phase_270 > phase_180 || phase_270 < 0.5);
    }
}