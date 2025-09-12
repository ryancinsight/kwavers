//! Phased array implementation
//!
//! Provides comprehensive phased array functionality combining
//! beam steering, dynamic focusing, and array management.

use crate::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::physics::phase_modulation::phase_shifting::beam::BeamSteering;
use crate::physics::phase_modulation::phase_shifting::core::{
    calculate_wavelength, wrap_phase, SPEED_OF_SOUND,
};
use crate::physics::phase_modulation::phase_shifting::focus::DynamicFocusing;
use crate::physics::phase_modulation::phase_shifting::shifter::PhaseShifter;

/// Phased array system
#[derive(Debug)]
pub struct PhaseArray {
    /// Array element positions
    element_positions: Array2<f64>,
    /// Operating frequency
    frequency: f64,
    /// Phase shifter
    #[allow(dead_code)] // Phase control hardware interface
    phase_shifter: PhaseShifter,
    /// Beam steering controller
    beam_steering: BeamSteering,
    /// Dynamic focusing controller
    dynamic_focusing: DynamicFocusing,
}

impl PhaseArray {
    /// Create a new phased array
    #[must_use]
    pub fn new(element_positions: Array2<f64>, frequency: f64) -> Self {
        let phase_shifter = PhaseShifter::new(element_positions.clone(), frequency);
        let beam_steering = BeamSteering::new(element_positions.clone(), frequency);
        let dynamic_focusing = DynamicFocusing::new(element_positions.clone(), frequency);

        Self {
            element_positions,
            frequency,
            phase_shifter,
            beam_steering,
            dynamic_focusing,
        }
    }

    /// Configure linear array
    #[must_use]
    pub fn configure_linear(num_elements: usize, spacing: f64, frequency: f64) -> Self {
        let mut positions = Array2::zeros((num_elements, 3));
        for i in 0..num_elements {
            positions[[i, 0]] = i as f64 * spacing - (num_elements - 1) as f64 * spacing / 2.0;
        }
        Self::new(positions, frequency)
    }

    /// Configure rectangular array
    #[must_use]
    pub fn configure_rectangular(nx: usize, ny: usize, dx: f64, dy: f64, frequency: f64) -> Self {
        let num_elements = nx * ny;
        let mut positions = Array2::zeros((num_elements, 3));

        let mut idx = 0;
        for j in 0..ny {
            for i in 0..nx {
                positions[[idx, 0]] = i as f64 * dx - (nx - 1) as f64 * dx / 2.0;
                positions[[idx, 1]] = j as f64 * dy - (ny - 1) as f64 * dy / 2.0;
                idx += 1;
            }
        }

        Self::new(positions, frequency)
    }

    /// Configure circular array
    #[must_use]
    pub fn configure_circular(
        num_rings: usize,
        elements_per_ring: usize,
        ring_spacing: f64,
        frequency: f64,
    ) -> Self {
        let num_elements = num_rings * elements_per_ring;
        let mut positions = Array2::zeros((num_elements, 3));

        let mut idx = 0;
        for ring in 0..num_rings {
            let radius = (ring + 1) as f64 * ring_spacing;
            for elem in 0..elements_per_ring {
                let angle = 2.0 * PI * elem as f64 / elements_per_ring as f64;
                positions[[idx, 0]] = radius * angle.cos();
                positions[[idx, 1]] = radius * angle.sin();
                idx += 1;
            }
        }

        Self::new(positions, frequency)
    }

    /// Set beam steering angles
    pub fn steer_beam(&mut self, azimuth: f64, elevation: f64) -> KwaversResult<()> {
        self.beam_steering.set_steering_angles(azimuth, elevation)
    }

    /// Set focal point
    pub fn set_focus(&mut self, x: f64, y: f64, z: f64) -> KwaversResult<()> {
        self.dynamic_focusing.set_focal_point(x, y, z)
    }

    /// Set multiple focal points
    pub fn set_multi_focus(&mut self, points: Vec<[f64; 3]>) -> KwaversResult<()> {
        self.dynamic_focusing.set_multiple_focal_points(points)
    }

    /// Get combined phase distribution
    #[must_use]
    pub fn get_phase_distribution(&self) -> Array1<f64> {
        let steering_phases = self.beam_steering.get_phase_distribution();
        let focusing_phases = self.dynamic_focusing.get_phase_distribution();

        // Combine steering and focusing phases
        let mut combined = Array1::zeros(steering_phases.len());
        for i in 0..combined.len() {
            combined[i] = wrap_phase(steering_phases[i] + focusing_phases[i]);
        }

        combined
    }

    /// Calculate field at a point
    #[must_use]
    pub fn calculate_field(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);
        let k = 2.0 * PI / wavelength;

        let phases = self.get_phase_distribution();
        let amplitudes = self.dynamic_focusing.get_amplitude_weights();

        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for i in 0..self.element_positions.nrows() {
            let pos = self.element_positions.row(i);
            let dx = x - pos[0];
            let dy = y - pos[1];
            let dz = z - pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            if distance > 0.0 {
                let phase = phases[i] + k * distance;
                let amplitude = amplitudes[i] / distance; // Include 1/r decay

                sum_real += amplitude * phase.cos();
                sum_imag += amplitude * phase.sin();
            }
        }

        (sum_real, sum_imag)
    }

    /// Calculate intensity at a point
    #[must_use]
    pub fn calculate_intensity(&self, x: f64, y: f64, z: f64) -> f64 {
        let (real, imag) = self.calculate_field(x, y, z);
        real * real + imag * imag
    }

    /// Check system performance
    #[must_use]
    pub fn check_performance(&self) -> PerformanceMetrics {
        let wavelength = calculate_wavelength(self.frequency, SPEED_OF_SOUND);

        // Find minimum element spacing
        let mut min_spacing = f64::INFINITY;
        for i in 0..self.element_positions.nrows() - 1 {
            for j in i + 1..self.element_positions.nrows() {
                let pos1 = self.element_positions.row(i);
                let pos2 = self.element_positions.row(j);
                let spacing = ((pos2[0] - pos1[0]).powi(2)
                    + (pos2[1] - pos1[1]).powi(2)
                    + (pos2[2] - pos1[2]).powi(2))
                .sqrt();
                if spacing < min_spacing {
                    min_spacing = spacing;
                }
            }
        }

        // Calculate array aperture
        let mut max_extent: [f64; 3] = [0.0, 0.0, 0.0];
        let mut min_extent: [f64; 3] = [f64::INFINITY, f64::INFINITY, f64::INFINITY];
        for i in 0..self.element_positions.nrows() {
            let pos = self.element_positions.row(i);
            for dim in 0..3 {
                max_extent[dim] = max_extent[dim].max(pos[dim]);
                min_extent[dim] = min_extent[dim].min(pos[dim]);
            }
        }

        let aperture_size = ((max_extent[0] - min_extent[0]).powi(2)
            + (max_extent[1] - min_extent[1]).powi(2)
            + (max_extent[2] - min_extent[2]).powi(2))
        .sqrt();

        PerformanceMetrics {
            element_spacing_ratio: min_spacing / wavelength,
            aperture_size,
            directivity: aperture_size / wavelength,
            grating_lobe_free: min_spacing < wavelength / 2.0,
            num_elements: self.element_positions.nrows(),
        }
    }
}

/// Performance metrics for phased array
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Element spacing to wavelength ratio
    pub element_spacing_ratio: f64,
    /// Total aperture size
    pub aperture_size: f64,
    /// Directivity estimate
    pub directivity: f64,
    /// Whether array is free from grating lobes
    pub grating_lobe_free: bool,
    /// Number of elements
    pub num_elements: usize,
}
