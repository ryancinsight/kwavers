use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::{Apodization, Source};
use log::debug;
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct MatrixArray {
    width: f64,
    height: f64,
    num_x: usize,
    num_y: usize,
    x_pos: f64,
    y_pos: f64,
    z_pos: f64,
    signal: Arc<dyn Signal>,
    time_delays: Vec<f64>, // Changed from phase_delays to time_delays
    apodization_weights: Vec<f64>,
}

impl Clone for MatrixArray {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            num_x: self.num_x,
            num_y: self.num_y,
            x_pos: self.x_pos,
            y_pos: self.y_pos,
            z_pos: self.z_pos,
            signal: self.signal.clone(),
            time_delays: self.time_delays.clone(),
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
        position: (f64, f64, f64),
        signal: Arc<dyn Signal>,
        sound_speed: f64,
        frequency: f64,
        apodization: A,
    ) -> Self {
        assert!(width > 0.0 && height > 0.0 && num_x > 0 && num_y > 0);
        let c = sound_speed;
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
            x_pos: position.0,
            y_pos: position.1,
            z_pos: position.2,
            signal,
            time_delays: vec![0.0; num_x * num_y],
            apodization_weights,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_focus<A: Apodization>(
        width: f64,
        height: f64,
        num_x: usize,
        num_y: usize,
        position: (f64, f64, f64),
        signal: Arc<dyn Signal>,
        sound_speed: f64,
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
            position,
            signal,
            sound_speed,
            frequency,
            apodization,
        );
        array.adjust_focus(focus_x, focus_y, focus_z, sound_speed);
        array
    }

    pub fn adjust_focus(
        &mut self,
        focus_x: f64,
        focus_y: f64,
        focus_z: f64,
        sound_speed: f64,
    ) {
        let c = sound_speed;
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();
        let start_x = self.x_pos - self.width / 2.0;
        let start_y = self.y_pos - self.height / 2.0;

        self.time_delays
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, delay)| {
                let ix = idx % self.num_x;
                let iy = idx / self.num_x;
                let x_elem = start_x + ix as f64 * dx;
                let y_elem = start_y + iy as f64 * dy;
                let distance = ((x_elem - focus_x).powi(2)
                    + (y_elem - focus_y).powi(2)
                    + (self.z_pos - focus_z).powi(2))
                .sqrt();
                *delay = distance / c;
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
        let start_x = self.x_pos - self.width / 2.0;
        let start_y = self.y_pos - self.height / 2.0;

        for iy in 0..self.num_y {
            for ix in 0..self.num_x {
                let x_elem = start_x + ix as f64 * dx;
                let y_elem = start_y + iy as f64 * dy;
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
        let start_x = self.x_pos - self.width / 2.0;
        let start_y = self.y_pos - self.height / 2.0;

        // Optimized spatial lookup
        if (z - self.z_pos).abs() >= tolerance {
            return 0.0;
        }

        let mut sum_val = 0.0;

        // Calculate theoretical index of element closest to x, y
        // x = start_x + ix*dx -> ix = (x - start_x)/dx
        let ix_f = (x - start_x) / dx;
        let iy_f = (y - start_y) / dy;

        let ix_center = ix_f.round() as isize;
        let iy_center = iy_f.round() as isize;

        // Check 3x3 neighborhood
        for iy in (iy_center - 1)..=(iy_center + 1) {
            for ix in (ix_center - 1)..=(ix_center + 1) {
                if ix >= 0 && ix < self.num_x as isize && iy >= 0 && iy < self.num_y as isize {
                    let idx = (iy as usize) * self.num_x + (ix as usize);
                    let x_elem = start_x + (ix as f64) * dx;
                    let y_elem = start_y + (iy as f64) * dy;

                    if (x - x_elem).abs() < tolerance && (y - y_elem).abs() < tolerance {
                        let delay = self.time_delays[idx];
                        let weight = self.apodization_weights[idx];
                        sum_val += weight * self.signal.amplitude(t - delay);
                    }
                }
            }
        }
        sum_val
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        let dx = self.element_spacing_x();
        let dy = self.element_spacing_y();
        let start_x = self.x_pos - self.width / 2.0;
        let start_y = self.y_pos - self.height / 2.0;
        let mut positions = Vec::with_capacity(self.num_x * self.num_y);
        for iy in 0..self.num_y {
            for ix in 0..self.num_x {
                let x = start_x + ix as f64 * dx;
                let y = start_y + iy as f64 * dy;
                positions.push((x, y, self.z_pos));
            }
        }
        positions
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}
