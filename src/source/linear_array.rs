use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::Signal;
use crate::source::{Apodization, Source};
use log::debug;
use rayon::prelude::*;
use std::fmt::Debug;
use ndarray::Array3;

#[derive(Debug)]
pub struct LinearArray {
    length: f64,
    num_elements: usize,
    y_pos: f64,
    z_pos: f64,
    signal: Box<dyn Signal>,
    time_delays: Vec<f64>, // Changed from phase_delays to time_delays
    apodization_weights: Vec<f64>,
}

impl Clone for LinearArray {
    fn clone(&self) -> Self {
        Self {
            length: self.length,
            num_elements: self.num_elements,
            y_pos: self.y_pos,
            z_pos: self.z_pos,
            signal: self.signal.clone_box(),
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
        y_pos: f64,
        z_pos: f64,
        signal: Box<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
        frequency: f64,
        apodization: A,
    ) -> Self {
        assert!(length > 0.0 && num_elements > 0);
        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
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
            .map(|i| {
                apodization.weight(i, num_elements)
            })
            .collect();

        Self {
            length,
            num_elements,
            y_pos,
            z_pos,
            signal,
            time_delays: vec![0.0; num_elements], // Changed from phase_delays
            apodization_weights,
        }
    }

    /// Adjust focus using proper time delays for broadband signals
    /// This replaces the incorrect phase delay approach
    pub fn adjust_focus(
        &mut self,
        focus_x: f64,
        focus_y: f64,
        focus_z: f64,
        medium: &dyn Medium,
        grid: &Grid,
    ) {
        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let spacing = self.element_spacing();
        
        // Calculate time delays (not phase delays) for proper broadband focusing
        self.time_delays
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delay)| {
                let x_elem = i as f64 * spacing;
                let distance = ((x_elem - focus_x).powi(2)
                    + (self.y_pos - focus_y).powi(2)
                    + (self.z_pos - focus_z).powi(2))
                .sqrt();
                *delay = distance / c; // Time delay, not phase delay
            });
        debug!("Adjusted focus to ({}, {}, {}) using time delays", focus_x, focus_y, focus_z);
    }

    fn element_spacing(&self) -> f64 {
        self.length / (self.num_elements.max(2) - 1) as f64
    }
}

impl Source for LinearArray {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let spacing = self.element_spacing();
        
        for i in 0..self.num_elements {
            let x_elem = i as f64 * spacing;
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
    
    /// Legacy method - DEPRECATED for performance reasons
    /// Use create_mask() and amplitude() for better performance
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let spacing = self.element_spacing();
        let tolerance = grid.dx * 0.5;
        (0..self.num_elements)
            .into_par_iter()
            .map(|i| {
                let x_elem = i as f64 * spacing;
                if (x - x_elem).abs() < tolerance
                    && (y - self.y_pos).abs() < tolerance
                    && (z - self.z_pos).abs() < tolerance
                {
                    let time_delay = self.time_delays[i];
                    // Apply time delay by sampling signal at delayed time
                    let temporal_amplitude = self.signal.amplitude(t - time_delay);
                    let spatial_weight = self.apodization_weights[i];
                    temporal_amplitude * spatial_weight
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        let spacing = self.element_spacing();
        (0..self.num_elements)
            .map(|i| (i as f64 * spacing, self.y_pos, self.z_pos))
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        &*self.signal
    }
}
