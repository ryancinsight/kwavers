//! Phased array implementation
//!
//! Provides comprehensive phased array functionality combining
//! beam steering, dynamic focusing, and array management.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::physics::phase_modulation::phase_shifting::beam::BeamSteering;
use crate::physics::phase_modulation::phase_shifting::core::{
    calculate_wavelength, wrap_phase, SPEED_OF_SOUND,
};
use crate::physics::phase_modulation::phase_shifting::focus::DynamicFocusing;

/// Phased array system
#[derive(Debug)]
pub struct PhaseArray {
    /// Array element positions
    element_positions: Array2<f64>,
    /// Operating frequency
    frequency: f64,
    /// Beam steering controller
    beam_steering: BeamSteering,
    /// Dynamic focusing controller
    dynamic_focusing: DynamicFocusing,
}

impl PhaseArray {
    /// Create a new phased array
    #[must_use]
    pub fn new(element_positions: Array2<f64>, frequency: f64) -> Self {
        let beam_steering = BeamSteering::new(element_positions.clone(), frequency);
        let dynamic_focusing = DynamicFocusing::new(element_positions.clone(), frequency);

        Self {
            element_positions,
            frequency,
            beam_steering,
            dynamic_focusing,
        }
    }

    /// Configure linear array
    #[must_use]
    pub fn configure_linear(num_elements: usize, spacing: f64, frequency: f64) -> Self {
        let mut positions = Array2::zeros((num_elements, 3));
        for i in 0..num_elements {
            positions[[i, 0]] =
                (i as f64).mul_add(spacing, -((num_elements - 1) as f64 * spacing / 2.0));
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
                positions[[idx, 0]] = (i as f64).mul_add(dx, -((nx - 1) as f64 * dx / 2.0));
                positions[[idx, 1]] = (j as f64).mul_add(dy, -((ny - 1) as f64 * dy / 2.0));
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn steer_beam(&mut self, azimuth: f64, elevation: f64) -> KwaversResult<()> {
        self.beam_steering.set_steering_angles(azimuth, elevation)
    }

    /// Set focal point
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_focus(&mut self, x: f64, y: f64, z: f64) -> KwaversResult<()> {
        self.dynamic_focusing.set_focal_point(x, y, z)
    }

    /// Set multiple focal points
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

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
        real.mul_add(real, imag * imag)
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
                let spacing = (pos2[2] - pos1[2])
                    .mul_add(
                        pos2[2] - pos1[2],
                        (pos2[1] - pos1[1]).mul_add(pos2[1] - pos1[1], (pos2[0] - pos1[0]).powi(2)),
                    )
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

        let aperture_size = (max_extent[2] - min_extent[2])
            .mul_add(
                max_extent[2] - min_extent[2],
                (max_extent[1] - min_extent[1]).mul_add(
                    max_extent[1] - min_extent[1],
                    (max_extent[0] - min_extent[0]).powi(2),
                ),
            )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// configure_linear produces N elements at the correct positions.
    ///
    /// 8 elements with spacing λ/2 = 750 µm at 1 MHz in water:
    /// positions span [−3.5·d, 3.5·d] symmetrically.
    #[test]
    fn configure_linear_produces_correct_element_count_and_positions() {
        let n = 8_usize;
        let spacing = 750e-6_f64; // λ/2 at 1 MHz in water
        let freq = 1e6_f64;
        let arr = PhaseArray::configure_linear(n, spacing, freq);

        assert_eq!(
            arr.element_positions.nrows(),
            n,
            "linear array must have {n} elements"
        );

        // Y and Z columns must be zero (linear in X only)
        for i in 0..n {
            assert!((arr.element_positions[[i, 1]]).abs() < 1e-15);
            assert!((arr.element_positions[[i, 2]]).abs() < 1e-15);
        }
        // First element at −(n−1)/2 · spacing
        let expected_first = -((n - 1) as f64 * spacing / 2.0);
        assert!(
            (arr.element_positions[[0, 0]] - expected_first).abs() < 1e-15,
            "first element position: expected {expected_first}, got {}",
            arr.element_positions[[0, 0]]
        );
    }

    /// configure_rectangular produces nx×ny elements.
    #[test]
    fn configure_rectangular_produces_correct_element_count() {
        let arr = PhaseArray::configure_rectangular(4, 3, 500e-6, 500e-6, 1e6);
        assert_eq!(
            arr.element_positions.nrows(),
            12,
            "4×3 array must have 12 elements"
        );
    }

    /// configure_circular produces num_rings × elements_per_ring elements.
    #[test]
    fn configure_circular_produces_correct_element_count() {
        let arr = PhaseArray::configure_circular(3, 8, 1e-3, 1e6);
        assert_eq!(
            arr.element_positions.nrows(),
            24,
            "3 rings × 8 elements = 24"
        );
    }

    /// check_performance reports correct element count and grating-lobe-free status.
    ///
    /// At spacing = 0.4·λ (< λ/2): condition `min_spacing < λ/2` is satisfied → grating_lobe_free = true.
    /// At spacing = 0.6·λ (> λ/2): condition fails → grating_lobe_free = false.
    #[test]
    fn check_performance_grating_lobe_free_at_sub_half_lambda_spacing() {
        // c = 1500 m/s, f = 1 MHz → λ = 1.5 mm; 0.4·λ = 600 µm < 750 µm = λ/2
        let arr = PhaseArray::configure_linear(8, 600e-6, 1e6);
        let metrics = arr.check_performance();
        assert_eq!(metrics.num_elements, 8);
        assert!(
            metrics.grating_lobe_free,
            "0.4λ spacing must be grating-lobe-free (ratio={:.3})",
            metrics.element_spacing_ratio
        );

        // Spacing = 0.6·λ = 900 µm > λ/2 → not grating-lobe-free
        let arr2 = PhaseArray::configure_linear(8, 900e-6, 1e6);
        let metrics2 = arr2.check_performance();
        assert!(
            !metrics2.grating_lobe_free,
            "0.6λ spacing must NOT be grating-lobe-free (ratio={:.3})",
            metrics2.element_spacing_ratio
        );
    }

    /// calculate_intensity at the center of an on-axis focused array is finite and positive.
    #[test]
    fn calculate_intensity_at_origin_is_finite_and_positive() {
        let arr = PhaseArray::configure_linear(8, 750e-6, 1e6);
        let intensity = arr.calculate_intensity(0.0, 0.0, 0.01);
        assert!(
            intensity.is_finite() && intensity > 0.0,
            "intensity must be finite and positive (got {intensity})"
        );
    }
}
