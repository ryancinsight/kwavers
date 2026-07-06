//! Acoustic propagation and reconstruction methods for PhotoacousticSimulator.

use kwavers_core::error::KwaversResult;
use kwavers_imaging::photoacoustic::{InitialPressure, PhotoacousticResult, PressureFieldSeries};
use kwavers_solver::inverse::reconstruction::photoacoustic::{
    PhotoacousticAlgorithm, PhotoacousticReconstructor, ReconstructionPhotoacousticConfig,
};
use leto::Array3;
use ndarray::Array2;

use super::super::acoustics;
use super::super::reconstruction;
use super::simulator::PhotoacousticSimulator;

impl PhotoacousticSimulator {
    /// Compute initial pressure distribution from optical absorption
    ///
    /// Uses photoacoustic equation: p₀(r) = Γ · μₐ(r) · Φ(r)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_initial_pressure(
        &self,
        fluence: &Array3<f64>,
    ) -> KwaversResult<InitialPressure> {
        acoustics::compute_initial_pressure(
            &self.grid,
            &self.optical_properties,
            fluence,
            &self.parameters.gruneisen_parameters,
            &self.parameters.wavelengths,
        )
    }

    /// Compute multi-wavelength initial pressure distributions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_multi_wavelength_pressure(
        &self,
        fluence_fields: &[Array3<f64>],
    ) -> KwaversResult<Vec<InitialPressure>> {
        acoustics::compute_multi_wavelength_pressure(
            &self.grid,
            &self.optical_properties,
            fluence_fields,
            &self.parameters.gruneisen_parameters,
            &self.parameters.wavelengths,
        )
    }

    /// Run multi-wavelength photoacoustic simulation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn simulate_multi_wavelength(&self) -> KwaversResult<Vec<(Array3<f64>, InitialPressure)>> {
        let fluence_fields = self.compute_multi_wavelength_fluence()?;

        let results: Result<Vec<_>, _> = fluence_fields
            .iter()
            .map(|fluence| {
                self.compute_initial_pressure(fluence)
                    .map(|pressure| (fluence.clone(), pressure))
            })
            .collect();

        results
    }

    /// Run photoacoustic simulation with numerically stable acoustic propagation
    ///
    /// # Algorithm
    ///
    /// 1. Propagate acoustic wave using FDTD time-stepping
    /// 2. Record pressure snapshots at detector positions
    /// 3. Reconstruct initial pressure using universal back-projection
    /// 4. Compute SNR from reconstructed image
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn simulate(
        &mut self,
        initial_pressure: &InitialPressure,
    ) -> KwaversResult<PhotoacousticResult> {
        let num_time_steps = 400;
        let snapshot_interval = 10;

        let (pressure_fields, time_points) = acoustics::propagate_acoustic_wave(
            &self.grid,
            initial_pressure,
            self.parameters.speed_of_sound,
            self.fdtd_solver.config.cfl_factor,
            num_time_steps,
            snapshot_interval,
        )?;

        let reconstructed_image = self.reconstruct_with_solver(&pressure_fields, &time_points)?;

        let signal_power = reconstructed_image.iter().map(|&x| x * x).sum::<f64>()
            / reconstructed_image.size() as f64;
        let noise_power = 1e-12;
        let snr = 10.0 * (signal_power / noise_power).log10();

        Ok(PhotoacousticResult {
            pressure_fields: PressureFieldSeries::new(pressure_fields)?,
            time: time_points,
            reconstructed_image,
            snr,
        })
    }

    /// Reconstruct using the dedicated PhotoacousticReconstructor from the solver module
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn reconstruct_with_solver(
        &self,
        pressure_fields: &[Array3<f64>],
        time_points: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        let n_time = time_points.len();
        let detectors = reconstruction::compute_detector_positions(&self.grid, 72);
        let mut sensor_data = Array2::zeros((n_time, detectors.len()));

        for (d_idx, &(dx, dy, dz)) in detectors.iter().enumerate() {
            for (t_idx, field) in pressure_fields.iter().enumerate() {
                sensor_data[[t_idx, d_idx]] =
                    reconstruction::interpolate_detector_signal(&self.grid, field, dx, dy, dz);
            }
        }

        let detector_positions: Vec<[f64; 3]> = detectors
            .iter()
            .map(|&(x, y, z)| [x * self.grid.dx, y * self.grid.dy, z * self.grid.dz])
            .collect();

        let config = ReconstructionPhotoacousticConfig {
            algorithm: PhotoacousticAlgorithm::UniversalBackProjection,
            sensor_positions: detector_positions.clone(),
            grid_size: [self.grid.nx, self.grid.ny, self.grid.nz],
            grid_spacing: [self.grid.dx, self.grid.dy, self.grid.dz],
            sound_speed: self.parameters.speed_of_sound,
            sampling_frequency: 1.0 / (time_points[1] - time_points[0]),
            envelope_detection: false,
            bandpass_filter: None,
            regularization_parameter: 0.0,
        };

        PhotoacousticReconstructor::new(config).universal_back_projection_leto(
            sensor_data.view(),
            &detector_positions,
            [self.grid.nx, self.grid.ny, self.grid.nz],
            self.parameters.speed_of_sound,
            1.0 / (time_points[1] - time_points[0]),
        )
    }

    /// Time Reversal Reconstruction (Universal Back-Projection)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn time_reversal_reconstruction(
        &self,
        pressure_fields: &[Array3<f64>],
        time_points: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        reconstruction::time_reversal_reconstruction(
            &self.grid,
            pressure_fields,
            time_points,
            self.parameters.speed_of_sound,
            72,
        )
    }
}
