use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::Signal;
use crate::source::{Apodization, Source};
use log::debug;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Debug)]
pub struct LinearArray {
    length: f64,
    num_elements: usize,
    y_pos: f64,
    z_pos: f64,
    signal: Box<dyn Signal>,
    phase_delays: Vec<f64>,
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
            phase_delays: self.phase_delays.clone(),
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
        let optimal_elements = (length / optimal_spacing).ceil() as usize;
        let num_elements = if num_elements == 0 {
            optimal_elements.max(1)
        } else {
            num_elements
        };
        let apodization_weights = (0..num_elements)
            .map(|i| apodization.weight(i, num_elements))
            .collect();
        debug!(
            "LinearArray: length = {}, num_elements = {}, apodized",
            length, num_elements
        );
        Self {
            length,
            num_elements,
            y_pos,
            z_pos,
            signal,
            phase_delays: vec![0.0; num_elements],
            apodization_weights,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_focus<A: Apodization>(
        length: f64,
        num_elements: usize,
        y_pos: f64,
        z_pos: f64,
        signal: Box<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
        frequency: f64,
        focus_x: f64,
        focus_y: f64,
        focus_z: f64,
        apodization: A,
    ) -> Self {
        let mut array = Self::new(
            length,
            num_elements,
            y_pos,
            z_pos,
            signal,
            medium,
            grid,
            frequency,
            apodization,
        );
        array.adjust_focus(frequency, focus_x, focus_y, focus_z, medium, grid);
        array
    }

    pub fn adjust_focus(
        &mut self,
        frequency: f64,
        focus_x: f64,
        focus_y: f64,
        focus_z: f64,
        medium: &dyn Medium,
        grid: &Grid,
    ) {
        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let spacing = self.element_spacing();
        self.phase_delays
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delay)| {
                let x_elem = i as f64 * spacing;
                let distance = ((x_elem - focus_x).powi(2)
                    + (self.y_pos - focus_y).powi(2)
                    + (self.z_pos - focus_z).powi(2))
                .sqrt();
                *delay = 2.0 * std::f64::consts::PI * frequency * (distance / c);
            });
        debug!("Adjusted focus to ({}, {}, {})", focus_x, focus_y, focus_z);
    }

    fn element_spacing(&self) -> f64 {
        self.length / (self.num_elements.max(2) - 1) as f64
    }
}

impl Source for LinearArray {
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
                    let temporal_amplitude = self.signal.amplitude(t);
                    let temporal_phase = self.signal.phase(t);
                    let spatial_phase = self.phase_delays[i];
                    let spatial_weight = self.apodization_weights[i];
                    temporal_amplitude * (temporal_phase + spatial_phase).cos() * spatial_weight
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
