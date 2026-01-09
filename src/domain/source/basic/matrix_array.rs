use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::signal::Signal;
use crate::domain::source::{Apodization, Source};
use log::debug;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fmt::Debug;

#[derive(Debug)]
pub struct MatrixArray {
    width: f64,
    height: f64,
    num_x: usize,
    num_y: usize,
    z_pos: f64,
    signal: Box<dyn Signal>,
    phase_delays: Vec<f64>,
    apodization_weights: Vec<f64>,
}

impl Clone for MatrixArray {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            num_x: self.num_x,
            num_y: self.num_y,
            z_pos: self.z_pos,
            signal: self.signal.clone_box(),
            phase_delays: self.phase_delays.clone(),
            apodization_weights: self.apodization_weights.clone(),
        }
    }
}

impl MatrixArray {
    #[allow(clippy::too_many_arguments)]
    pub fn new<A: Apodization>(
        width: f64,
        height: f64,
        num_x: usize,
        num_y: usize,
        z_pos: f64,
        signal: Box<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
        frequency: f64,
        apodization: A,
    ) -> Self {
        assert!(width > 0.0 && height > 0.0 && num_x > 0 && num_y > 0);
        let c = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let wavelength = c / frequency;
        let optimal_dx = wavelength / 2.0;
        let optimal_dy = wavelength / 2.0;
        let num_x = if num_x == 0 {
            (width / optimal_dx).ceil() as usize
        } else {
            num_x
        };
        let num_y = if num_y == 0 {
            (height / optimal_dy).ceil() as usize
        } else {
            num_y
        };
        let apodization_weights = (0..num_x * num_y)
            .map(|idx| {
                let ix = idx % num_x;
                let iy = idx / num_x;
                let x_norm = (ix as f64 - (num_x - 1) as f64 / 2.0) / ((num_x - 1) as f64 / 2.0);
                let y_norm = (iy as f64 - (num_y - 1) as f64 / 2.0) / ((num_y - 1) as f64 / 2.0);
                let r = (x_norm * x_norm + y_norm * y_norm).sqrt();
                apodization.weight(
                    (r.min(1.0) * num_x.max(num_y) as f64 / 2.0) as usize,
                    num_x.max(num_y),
                )
            })
            .collect();
        debug!(
            "MatrixArray: width = {}, height = {}, num_x = {}, num_y = {}, apodized",
            width, height, num_x, num_y
        );
        Self {
            width,
            height,
            num_x,
            num_y,
            z_pos,
            signal,
            phase_delays: vec![0.0; num_x * num_y],
            apodization_weights,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_focus<A: Apodization>(
        width: f64,
        height: f64,
        num_x: usize,
        num_y: usize,
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
            width,
            height,
            num_x,
            num_y,
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
        let c = crate::domain::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();
        self.phase_delays
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, delay)| {
                let ix = idx % self.num_x;
                let iy = idx / self.num_x;
                let x_elem = ix as f64 * dx - self.width / 2.0;
                let y_elem = iy as f64 * dy - self.height / 2.0;
                let distance = ((x_elem - focus_x).powi(2)
                    + (y_elem - focus_y).powi(2)
                    + (self.z_pos - focus_z).powi(2))
                .sqrt();
                *delay = 2.0 * PI * frequency * (distance / c);
            });
        debug!("Adjusted focus to ({}, {}, {})", focus_x, focus_y, focus_z);
    }

    fn element_spacing_x(&self) -> f64 {
        self.width / (self.num_x.max(2) - 1) as f64
    }
    fn element_spacing_y(&self) -> f64 {
        self.height / (self.num_y.max(2) - 1) as f64
    }
}

impl Source for MatrixArray {
    fn create_mask(&self, grid: &Grid) -> ndarray::Array3<f64> {
        let mut mask = ndarray::Array3::zeros((grid.nx, grid.ny, grid.nz));
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();

        for iy in 0..self.num_y {
            for ix in 0..self.num_x {
                let x_elem = ix as f64 * dx - self.width / 2.0;
                let y_elem = iy as f64 * dy - self.height / 2.0;
                let idx = iy * self.num_x + ix;

                if let Some((gx, gy, gz)) = grid.position_to_indices(x_elem, y_elem, self.z_pos) {
                    mask[(gx, gy, gz)] = self.apodization_weights[idx];
                }
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();
        let tolerance = grid.dx.max(grid.dy) * 0.5;
        (0..self.num_y * self.num_x)
            .into_par_iter()
            .map(|idx| {
                let ix = idx % self.num_x;
                let iy = idx / self.num_x;
                let x_elem = ix as f64 * dx - self.width / 2.0;
                let y_elem = iy as f64 * dy - self.height / 2.0;
                if (x - x_elem).abs() < tolerance
                    && (y - y_elem).abs() < tolerance
                    && (z - self.z_pos).abs() < tolerance
                {
                    let temporal_amplitude = self.signal.amplitude(t);
                    let temporal_phase = self.signal.phase(t);
                    let spatial_phase = self.phase_delays[idx];
                    let spatial_weight = self.apodization_weights[idx];
                    temporal_amplitude * (temporal_phase + spatial_phase).cos() * spatial_weight
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();
        let mut positions = Vec::with_capacity(self.num_x * self.num_y);
        for iy in 0..self.num_y {
            for ix in 0..self.num_x {
                let x = ix as f64 * dx - self.width / 2.0;
                let y = iy as f64 * dy - self.height / 2.0;
                positions.push((x, y, self.z_pos));
            }
        }
        positions
    }

    fn signal(&self) -> &dyn Signal {
        &*self.signal
    }
}
