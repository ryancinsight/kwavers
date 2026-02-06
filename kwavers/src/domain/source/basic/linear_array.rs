use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::{Apodization, Source};
use log::debug;
use ndarray::Array3;
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct LinearArray {
    length: f64,
    num_elements: usize,
    x_pos: f64,
    y_pos: f64,
    z_pos: f64,
    signal: Arc<dyn Signal>,
    time_delays: Vec<f64>, // Changed from phase_delays to time_delays
    apodization_weights: Vec<f64>,
}

impl Clone for LinearArray {
    fn clone(&self) -> Self {
        Self {
            length: self.length,
            num_elements: self.num_elements,
            x_pos: self.x_pos,
            y_pos: self.y_pos,
            z_pos: self.z_pos,
            signal: self.signal.clone(),
            time_delays: self.time_delays.clone(), // Changed from phase_delays
            apodization_weights: self.apodization_weights.clone(),
        }
    }
}

impl LinearArray {
    #[allow(clippy::too_many_arguments)]
    pub fn new<A: Apodization>(
        length: f64,
        num_elements: usize,
        position: (f64, f64, f64),
        signal: Arc<dyn Signal>,
        sound_speed: f64,
        frequency: f64,
        apodization: A,
    ) -> Self {
        assert!(length > 0.0 && num_elements > 0);
        let c = sound_speed;
        let wavelength = c / frequency;
        let optimal_spacing = wavelength / 2.0;
        let actual_spacing = length / (num_elements.max(2) - 1) as f64;

        if actual_spacing > optimal_spacing {
            debug!(
                "Warning: Element spacing ({:.3} mm) exceeds optimal spacing ({:.3} mm)",
                actual_spacing * 1000.0,
                optimal_spacing * 1000.0
            );
        }

        let apodization_weights: Vec<f64> = (0..num_elements)
            .map(|i| apodization.weight(i, num_elements))
            .collect();

        Self {
            length,
            num_elements,
            x_pos: position.0,
            y_pos: position.1,
            z_pos: position.2,
            signal,
            time_delays: vec![0.0; num_elements], // Changed from phase_delays
            apodization_weights,
        }
    }

    /// Adjust focus using proper time delays for broadband signals
    /// This replaces the incorrect phase delay approach
    pub fn adjust_focus(&mut self, focus_x: f64, focus_y: f64, focus_z: f64, sound_speed: f64) {
        let c = sound_speed;
        let spacing = self.element_spacing();
        let start_x = self.x_pos - self.length / 2.0;

        // Calculate time delays (not phase delays) for proper broadband focusing
        self.time_delays
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delay)| {
                let x_elem = start_x + i as f64 * spacing;
                let distance = ((x_elem - focus_x).powi(2)
                    + (self.y_pos - focus_y).powi(2)
                    + (self.z_pos - focus_z).powi(2))
                .sqrt();
                *delay = distance / c; // Time delay, not phase delay
            });
        debug!(
            "Adjusted focus to ({}, {}, {}) using time delays",
            focus_x, focus_y, focus_z
        );
    }

    fn element_spacing(&self) -> f64 {
        self.length / (self.num_elements.max(2) - 1) as f64
    }
}

impl Source for LinearArray {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let spacing = self.element_spacing();
        let start_x = self.x_pos - self.length / 2.0;

        for i in 0..self.num_elements {
            let x_elem = start_x + i as f64 * spacing;
            if let Some((ix, iy, iz)) = grid.position_to_indices(x_elem, self.y_pos, self.z_pos) {
                mask[(ix, iy, iz)] = self.apodization_weights[i];
            }
        }
        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        // For arrays, return the base signal amplitude
        // Individual element delays are handled in the mask application
        self.signal.amplitude(t)
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let spacing = self.element_spacing();
        let tolerance = grid.dx.max(grid.dy) * 0.5;
        let start_x = self.x_pos - self.length / 2.0;

        // This is inefficient but correct for focusing.
        // It sums contributions from all elements close to the point.
        let mut sum_val = 0.0;

        // Check if we are close to the array line
        if (y - self.y_pos).abs() < tolerance && (z - self.z_pos).abs() < tolerance {
            // Find closest element(s)
            // i = (x - start_x) / spacing
            let idx_f = (x - start_x) / spacing;
            let idx = idx_f.round() as isize;

            // Check neighboring elements too to be safe with grid discretization
            for i in (idx - 1)..=(idx + 1) {
                if i >= 0 && i < self.num_elements as isize {
                    let i_usize = i as usize;
                    let x_elem = start_x + i as f64 * spacing;
                    if (x - x_elem).abs() < tolerance {
                        let delay = self.time_delays[i_usize];
                        let weight = self.apodization_weights[i_usize];
                        sum_val += weight * self.signal.amplitude(t - delay);
                    }
                }
            }
        }
        sum_val
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        let spacing = self.element_spacing();
        let start_x = self.x_pos - self.length / 2.0;
        (0..self.num_elements)
            .map(|i| (start_x + i as f64 * spacing, self.y_pos, self.z_pos))
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}
