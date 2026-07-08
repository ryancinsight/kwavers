//! Pressure field statistics for k-Wave parity recording modes.
//!
//! Accumulates per-voxel statistics over the simulation time loop, matching
//! k-Wave's `sensor.record = {'p_max', 'p_min', 'p_rms', 'p_final'}` behaviour.
//!
//! Standard-layout accumulations use Moirai chunk dispatch for O(N^3)
//! parallelism with no intermediate allocations; non-standard layouts retain
//! sequential ndarray semantics.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use ndarray::Array1;

#[doc(hidden)]
pub trait PressureArray3Access {
    fn shape3(&self) -> [usize; 3];
    fn as_slice_opt(&self) -> Option<&[f64]>;
    fn iter_values<'a>(&'a self) -> Box<dyn Iterator<Item = &'a f64> + 'a>;
}

impl PressureArray3Access for ndarray::Array3<f64> {
    fn shape3(&self) -> [usize; 3] {
        let (nx, ny, nz) = self.dim();
        [nx, ny, nz]
    }

    fn as_slice_opt(&self) -> Option<&[f64]> {
        self.as_slice()
    }

    fn iter_values<'a>(&'a self) -> Box<dyn Iterator<Item = &'a f64> + 'a> {
        Box::new(self.iter())
    }
}

impl PressureArray3Access for LetoArray3<f64> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn as_slice_opt(&self) -> Option<&[f64]> {
        self.as_slice()
    }

    fn iter_values<'a>(&'a self) -> Box<dyn Iterator<Item = &'a f64> + 'a> {
        Box::new(self.iter())
    }
}

/// Spatial statistics for pressure field over time
#[derive(Debug, Clone)]
pub struct PressureFieldStatistics {
    /// Maximum pressure at each spatial point (p_max)
    pub p_max: LetoArray3<f64>,
    /// Minimum pressure at each spatial point (p_min)
    pub p_min: LetoArray3<f64>,
    /// Sum of squared pressures for RMS calculation (p_rms)
    pub p_squared_sum: LetoArray3<f64>,
    /// Final pressure field (p_final)
    pub p_final: LetoArray3<f64>,
    /// Number of time steps recorded
    pub time_step_count: usize,
    /// Shape of the grid
    shape: (usize, usize, usize),
}

impl PressureFieldStatistics {
    /// Create new statistics tracker for given grid dimensions
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            p_max: LetoArray3::from_elem([nx, ny, nz], f64::NEG_INFINITY),
            p_min: LetoArray3::from_elem([nx, ny, nz], f64::INFINITY),
            p_squared_sum: LetoArray3::zeros([nx, ny, nz]),
            p_final: LetoArray3::zeros([nx, ny, nz]),
            time_step_count: 0,
            shape: (nx, ny, nz),
        }
    }

    /// Accumulate one time step of pressure data.
    ///
    /// Updates max, min, squared sum, and final field element-wise.
    /// Single-pass, provider-parallelized for standard layouts:
    /// O(N^3), no intermediate allocation.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn update<P: PressureArray3Access>(&mut self, pressure: &P) {
        debug_assert_eq!(
            pressure.shape3(),
            [self.shape.0, self.shape.1, self.shape.2],
            "pressure field shape mismatch in PressureFieldStatistics::update"
        );

        match (
            self.p_max.as_slice_mut(),
            self.p_min.as_slice_mut(),
            self.p_squared_sum.as_slice_mut(),
            self.p_final.as_slice_mut(),
            pressure.as_slice_opt(),
        ) {
            (Some(p_max), Some(p_min), Some(p_squared_sum), Some(p_final), Some(pressure)) => {
                for lane in 0..p_max.len() {
                    let p = pressure[lane];
                    if p > p_max[lane] {
                        p_max[lane] = p;
                    }
                    if p < p_min[lane] {
                        p_min[lane] = p;
                    }
                    p_squared_sum[lane] += p * p;
                    p_final[lane] = p;
                }
            }
            _ => {
                for ((((pmax, pmin), sq), pfin), p) in self
                    .p_max
                    .iter_mut()
                    .zip(self.p_min.iter_mut())
                    .zip(self.p_squared_sum.iter_mut())
                    .zip(self.p_final.iter_mut())
                    .zip(pressure.iter_values())
                {
                    let p = *p;
                    if p > *pmax {
                        *pmax = p;
                    }
                    if p < *pmin {
                        *pmin = p;
                    }
                    *sq += p * p;
                    *pfin = p;
                }
            }
        }

        self.time_step_count += 1;
    }

    /// Calculate RMS pressure field
    #[must_use]
    pub fn p_rms(&self) -> LetoArray3<f64> {
        if self.time_step_count == 0 {
            return LetoArray3::zeros([self.shape.0, self.shape.1, self.shape.2]);
        }
        let mut out = LetoArray3::<f64>::zeros([self.shape.0, self.shape.1, self.shape.2]);
        for (dst, &value) in out.iter_mut().zip(self.p_squared_sum.iter()) {
            *dst = (value / self.time_step_count as f64).sqrt();
        }
        out
    }

    /// Get p_max as array
    #[must_use]
    pub fn get_p_max(&self) -> &LetoArray3<f64> {
        &self.p_max
    }

    /// Get p_min as array
    #[must_use]
    pub fn get_p_min(&self) -> &LetoArray3<f64> {
        &self.p_min
    }

    /// Get p_final as array
    #[must_use]
    pub fn get_p_final(&self) -> &LetoArray3<f64> {
        &self.p_final
    }

    /// Get global maximum pressure value
    #[must_use]
    pub fn global_p_max(&self) -> f64 {
        self.p_max.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Get global minimum pressure value
    #[must_use]
    pub fn global_p_min(&self) -> f64 {
        self.p_min.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Get global RMS pressure value
    #[must_use]
    pub fn global_p_rms(&self) -> f64 {
        if self.time_step_count == 0 {
            return 0.0;
        }
        let sum_squared: f64 = self.p_squared_sum.iter().sum();
        let n = (self.shape.0 * self.shape.1 * self.shape.2) as f64;
        (sum_squared / (self.time_step_count as f64 * n)).sqrt()
    }

    /// Sample statistics at sensor positions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_p_max(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_p_max(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with maximum pressure at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_max(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        fill_field_at_positions(&self.p_max, positions, out)
    }

    /// Sample minimum pressure at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_p_min(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_p_min(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with minimum pressure at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_min(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        fill_field_at_positions(&self.p_min, positions, out)
    }

    /// Sample RMS pressure at sensor positions.
    ///
    /// With no accumulated time steps the RMS is the neutral zero field, matching
    /// [`p_rms`](Self::p_rms) and avoiding undefined `0/0` sampling.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_p_rms(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_p_rms(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with RMS pressure at sensor positions.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn fill_p_rms(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        validate_sample_output_len(positions, out)?;

        if self.time_step_count == 0 {
            out.fill(0.0);
            return Ok(());
        }

        let n = self.time_step_count as f64;
        for (row, &(i, j, k)) in positions.iter().enumerate() {
            out[row] = (self.p_squared_sum[[i, j, k]] / n).sqrt();
        }
        Ok(())
    }

    /// Sample final pressure at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_p_final(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_p_final(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with final pressure at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_final(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        fill_field_at_positions(&self.p_final, positions, out)
    }

    /// Sample all pressure statistics at sensor positions.
    #[must_use]
    pub fn sample_at_positions(&self, positions: &[(usize, usize, usize)]) -> SampledStatistics {
        SampledStatistics {
            p_max: self.sample_p_max(positions),
            p_min: self.sample_p_min(positions),
            p_rms: self.sample_p_rms(positions),
            p_final: self.sample_p_final(positions),
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.p_max.fill(f64::NEG_INFINITY);
        self.p_min.fill(f64::INFINITY);
        self.p_squared_sum.fill(0.0);
        self.p_final.fill(0.0);
        self.time_step_count = 0;
    }
}

/// Statistics sampled at specific sensor positions
#[derive(Debug, Clone)]
pub struct SampledStatistics {
    pub p_max: Array1<f64>,
    pub p_min: Array1<f64>,
    pub p_rms: Array1<f64>,
    pub p_final: Array1<f64>,
}

impl SampledStatistics {
    /// Get number of sensors
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.p_max.len()
    }
}

fn fill_field_at_positions(
    field: &LetoArray3<f64>,
    positions: &[(usize, usize, usize)],
    out: &mut Array1<f64>,
) -> KwaversResult<()> {
    validate_sample_output_len(positions, out)?;
    for (row, &(i, j, k)) in positions.iter().enumerate() {
        out[row] = field[[i, j, k]];
    }
    Ok(())
}

fn validate_sample_output_len(
    positions: &[(usize, usize, usize)],
    out: &Array1<f64>,
) -> KwaversResult<()> {
    if out.len() != positions.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "pressure-stat output length {} != sensor count {}",
            out.len(),
            positions.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_pressure_statistics_basic() {
        let mut stats = PressureFieldStatistics::new(4, 4, 4);

        // Simulate 3 time steps
        for t in 0..3 {
            let pressure = Array3::from_elem((4, 4, 4), (t as f64 + 1.0) * 1000.0);
            stats.update(&pressure);
        }

        assert_eq!(stats.time_step_count, 3);
        assert_eq!(stats.global_p_max(), 3000.0);
        assert_eq!(stats.global_p_min(), 1000.0);

        // RMS should be sqrt((1000² + 2000² + 3000²) / 3) = sqrt(14000000/3)
        let expected_rms =
            ((1000.0f64.powi(2) + 2000.0f64.powi(2) + 3000.0f64.powi(2)) / 3.0).sqrt();
        assert!((stats.global_p_rms() - expected_rms).abs() < 1e-10);
    }

    #[test]
    fn test_sample_at_positions() {
        let mut stats = PressureFieldStatistics::new(10, 10, 10);

        let pressure = Array3::from_elem((10, 10, 10), 5000.0);
        stats.update(&pressure);

        let positions = vec![(0, 0, 0), (5, 5, 5), (9, 9, 9)];
        let sampled = stats.sample_at_positions(&positions);

        assert_eq!(sampled.num_sensors(), 3);
        assert!(sampled.p_max.iter().all(|&v| v == 5000.0));
    }

    #[test]
    fn test_single_field_sampling_and_zero_step_rms() {
        let stats = PressureFieldStatistics::new(3, 1, 1);
        let positions = vec![(0, 0, 0), (2, 0, 0)];

        let rms = stats.sample_p_rms(&positions);
        assert_eq!(rms.len(), 2);
        assert!(rms.iter().all(|&v| v == 0.0));

        let final_pressure = stats.sample_p_final(&positions);
        assert!(final_pressure.iter().all(|&v| v == 0.0));
    }
}
