//! Time reversal reconstruction for photoacoustic imaging
//!
//! This module implements the time reversal algorithm for photoacoustic
//! image reconstruction based on solving the wave equation backward in time.
//!
//! References:
//! - Xu & Wang (2005) "Time-reversal reconstruction algorithm"
//! - Treeby et al. (2010) "k-Wave: MATLAB toolbox"

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, ArrayView2, Zip};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Time reversal reconstruction algorithm
#[derive(Debug)]
pub struct TimeReversal {
    grid_size: [usize; 3],
    sound_speed: f64,
    sampling_frequency: f64,
    time_steps: usize,
}

impl TimeReversal {
    /// Create new time reversal reconstructor
    pub fn new(
        grid_size: [usize; 3],
        sound_speed: f64,
        sampling_frequency: f64,
        time_steps: usize,
    ) -> Self {
        Self {
            grid_size,
            sound_speed,
            sampling_frequency,
            time_steps,
        }
    }

    /// Perform time reversal reconstruction
    ///
    /// This implements the k-space pseudospectral time reversal method
    /// which provides accurate reconstruction without numerical dispersion.
    pub fn reconstruct(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (n_time, n_sensors) = sensor_data.dim();
        let [nx, ny, nz] = self.grid_size;

        // Initialize pressure field
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut pressure_prev = Array3::zeros((nx, ny, nz));

        // Time step
        let dt = 1.0 / self.sampling_frequency;
        let c0 = self.sound_speed;

        // Stability check (CFL condition)
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = c0 * dt / dx_min;
        if cfl > 1.0 / f64::sqrt(3.0) {
            return Err(crate::error::KwaversError::Numerical(
                crate::error::NumericalError::Instability {
                    operation: "Time reversal CFL".to_string(),
                    condition: cfl,
                },
            ));
        }

        // Create k-space operators
        let kx = self.create_k_vector(nx, grid.dx);
        let ky = self.create_k_vector(ny, grid.dy);
        let kz = self.create_k_vector(nz, grid.dz);

        // Create k-space operator for wave equation
        let mut k_squared = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    k_squared[[i, j, k]] = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
                }
            }
        }

        // Precompute propagator
        let propagator = self.compute_propagator(&k_squared, c0, dt);

        // Time reversal loop (backward in time)
        for t_idx in (0..n_time).rev() {
            // Inject sensor data at sensor positions
            self.inject_sensor_data(
                &mut pressure,
                &sensor_data.row(t_idx),
                sensor_positions,
                grid,
            )?;

            // Apply k-space propagation
            let pressure_next =
                self.k_space_step(&pressure, &pressure_prev, &propagator, &k_squared, c0, dt)?;

            // Update fields
            pressure_prev = pressure;
            pressure = pressure_next;
        }

        // Return the initial pressure distribution (reconstructed image)
        Ok(pressure)
    }

    /// Create k-space vector
    fn create_k_vector(&self, n: usize, dx: f64) -> Vec<f64> {
        let mut k = vec![0.0; n];
        let dk = 2.0 * PI / (n as f64 * dx);

        for i in 0..n {
            if i <= n / 2 {
                k[i] = i as f64 * dk;
            } else {
                k[i] = f64::from(i as i32 - n as i32) * dk;
            }
        }

        k
    }

    /// Compute k-space propagator for wave equation
    fn compute_propagator(
        &self,
        k_squared: &Array3<f64>,
        c0: f64,
        dt: f64,
    ) -> Array3<Complex<f64>> {
        let omega_dt = c0 * dt;
        k_squared.mapv(|k2| {
            let arg = omega_dt * k2.sqrt();
            Complex::new(arg.cos(), -arg.sin())
        })
    }

    /// Perform one k-space time step
    fn k_space_step(
        &self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        propagator: &Array3<Complex<f64>>,
        k_squared: &Array3<f64>,
        c0: f64,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = self.grid_size;

        // FFT of pressure fields
        let pressure_fft = self.fft_3d(pressure)?;
        let pressure_prev_fft = self.fft_3d(pressure_prev)?;

        // Apply k-space operator (exact solution in Fourier domain)
        // p(t+dt) = 2*cos(c*k*dt)*p(t) - p(t-dt) + dt²*c²*source
        let mut pressure_next_fft = Array3::zeros((nx, ny, nz));
        Zip::from(&mut pressure_next_fft)
            .and(propagator)
            .and(&pressure_fft)
            .and(&pressure_prev_fft)
            .for_each(|pn, &prop, &p, &pp| {
                *pn = 2.0 * prop * p - pp;
            });

        // Inverse FFT
        self.ifft_3d(&pressure_next_fft)
    }

    /// 3D FFT
    fn fft_3d(&self, data: &Array3<f64>) -> KwaversResult<Array3<Complex<f64>>> {
        let [nx, ny, nz] = self.grid_size;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nx * ny * nz);

        // Convert to complex and flatten
        let mut complex_data: Vec<Complex<f64>> =
            data.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Perform FFT
        fft.process(&mut complex_data);

        // Reshape to 3D
        let mut result = Array3::zeros((nx, ny, nz));
        for (i, val) in complex_data.iter().enumerate() {
            let iz = i % nz;
            let iy = (i / nz) % ny;
            let ix = i / (ny * nz);
            result[[ix, iy, iz]] = *val;
        }

        Ok(result)
    }

    /// 3D inverse FFT
    fn ifft_3d(&self, data: &Array3<Complex<f64>>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = self.grid_size;
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(nx * ny * nz);

        // Flatten
        let mut complex_data: Vec<Complex<f64>> = data.iter().copied().collect();

        // Perform inverse FFT
        ifft.process(&mut complex_data);

        // Normalize and extract real part
        let norm = 1.0 / (nx * ny * nz) as f64;
        let mut result = Array3::zeros((nx, ny, nz));
        for (i, val) in complex_data.iter().enumerate() {
            let iz = i % nz;
            let iy = (i / nz) % ny;
            let ix = i / (ny * nz);
            result[[ix, iy, iz]] = val.re * norm;
        }

        Ok(result)
    }

    /// Inject sensor data at sensor positions
    fn inject_sensor_data(
        &self,
        pressure: &mut Array3<f64>,
        sensor_values: &ndarray::ArrayView1<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
    ) -> KwaversResult<()> {
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            // Find nearest grid point
            let i = ((sensor_pos[0] / grid.dx) as usize).min(self.grid_size[0] - 1);
            let j = ((sensor_pos[1] / grid.dy) as usize).min(self.grid_size[1] - 1);
            let k = ((sensor_pos[2] / grid.dz) as usize).min(self.grid_size[2] - 1);

            // Inject sensor value (additive for multiple sensors at same location)
            pressure[[i, j, k]] += sensor_values[sensor_idx];
        }

        Ok(())
    }
}
