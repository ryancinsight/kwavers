//! Reconstruction Algorithm Module
//!
//! Core time-reversal reconstruction implementation.

use crate::{
    core::error::KwaversResult,
    domain::{
        grid::Grid,
        medium::Medium,
        sensor::recorder::Recorder,
        source::{Source, TimeVaryingSource},
    },
    solver::plugin_based::PluginBasedSolver,
};
use log::{debug, info};
use ndarray::{Array2, Array3};
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    config::TimeReversalConfig,
    processing::{apply_spatial_window, AmplitudeCorrector, FrequencyFilter},
    validation::InputValidator,
};

/// Time-reversal reconstructor
#[derive(Debug)]
pub struct TimeReversalReconstructor {
    config: TimeReversalConfig,
    frequency_filter: FrequencyFilter,
    amplitude_corrector: AmplitudeCorrector,
}

impl TimeReversalReconstructor {
    /// Create a new time-reversal reconstructor
    pub fn new(config: TimeReversalConfig) -> KwaversResult<Self> {
        config.validate()?;

        Ok(Self {
            frequency_filter: FrequencyFilter::new(),
            amplitude_corrector: AmplitudeCorrector::new(config.max_amplification),
            config,
        })
    }

    /// Perform time-reversal reconstruction
    pub fn reconstruct(
        &mut self,
        pressure_data: &Array2<f64>,
        sensor_indices: &[(usize, usize, usize)],
        grid: &Grid,
        solver: &mut PluginBasedSolver,
        recorder: &mut Recorder,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        info!("Starting time-reversal reconstruction");

        // Validate inputs
        InputValidator::validate_sensor_data(pressure_data, sensor_indices, grid)?;
        InputValidator::validate_grid_dimensions(grid)?;

        // Prepare time-reversed signals
        let medium = solver.medium().clone();
        let reversed_signals = self.prepare_reversed_signals(
            pressure_data,
            sensor_indices,
            grid,
            solver.time().dt,
            &medium,
            frequency,
        )?;

        // Initialize reconstruction field
        let mut reconstruction = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Perform reconstruction iterations
        for iteration in 0..self.config.iterations {
            debug!(
                "Time-reversal iteration {}/{}",
                iteration + 1,
                self.config.iterations
            );

            // Apply time-reversed signals as sources
            self.apply_reversed_sources(&reversed_signals, solver, sensor_indices, grid)?;

            // Propagate backwards in time
            let iteration_result =
                self.propagate_backwards(grid, solver, recorder, frequency, &reversed_signals)?;

            // Accumulate reconstruction
            reconstruction += &iteration_result;

            // Check convergence for iterative methods
            if self.config.iterations > 1 {
                let convergence = self.check_convergence(&reconstruction, &iteration_result)?;
                if convergence < self.config.tolerance {
                    info!("Converged after {} iterations", iteration + 1);
                    break;
                }
            }
        }

        // Apply post-processing
        let reconstruction = self.post_process(reconstruction, grid)?;

        info!("Time-reversal reconstruction completed");
        Ok(reconstruction)
    }

    /// Prepare time-reversed signals
    fn prepare_reversed_signals(
        &mut self,
        pressure_data: &Array2<f64>,
        sensor_indices: &[(usize, usize, usize)],
        grid: &Grid,
        dt: f64,
        medium: &Arc<dyn Medium>,
        frequency: f64,
    ) -> KwaversResult<HashMap<usize, Vec<f64>>> {
        let mut reversed_signals = HashMap::new();
        for (sensor_idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            // Get signal for this sensor
            let signal_row = pressure_data.row(sensor_idx);
            let mut signal = signal_row.to_vec();

            // Reverse the signal in time
            signal.reverse();

            // Apply phase conjugation if enabled
            if self.config.phase_conjugation {
                signal = self.apply_phase_conjugation(signal)?;
            }

            // Apply frequency filter if configured
            if self.config.apply_frequency_filter {
                if let Some(freq_range) = self.config.frequency_range {
                    signal =
                        self.frequency_filter
                            .bandpass(&signal, dt, freq_range.0, freq_range.1)?;
                }
            }

            // Apply amplitude correction if configured
            if self.config.amplitude_correction {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let position_meters = (x, y, z);

                signal = self.amplitude_corrector.apply_correction(
                    signal,
                    dt,
                    medium,
                    grid,
                    frequency,
                    [position_meters.0, position_meters.1, position_meters.2],
                    self.config.absorption_compensation,
                )?;
            }

            // Apply dispersion correction if configured
            if self.config.dispersion_correction {
                signal = self.amplitude_corrector.apply_dispersion_correction(
                    signal,
                    dt,
                    medium,
                    grid,
                    self.config.reference_sound_speed,
                )?;
            }

            reversed_signals.insert(sensor_idx, signal);
        }

        debug!("Prepared {} reversed signals", reversed_signals.len());
        Ok(reversed_signals)
    }

    /// Apply phase conjugation to a signal
    fn apply_phase_conjugation(&self, signal: Vec<f64>) -> KwaversResult<Vec<f64>> {
        // For real signals, phase conjugation in time domain is just time reversal
        // which is already done. For complex processing, we would conjugate here.
        Ok(signal)
    }

    /// Apply reversed sources to the solver
    fn apply_reversed_sources(
        &self,
        reversed_signals: &HashMap<usize, Vec<f64>>,
        solver: &mut PluginBasedSolver,
        sensor_indices: &[(usize, usize, usize)],
        _grid: &Grid,
    ) -> KwaversResult<()> {
        // Clear existing sources
        solver.clear_sources();

        // Create a time-varying source for each sensor
        let mut sources: Vec<Box<dyn Source>> = Vec::new();

        for (sensor_idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            if let Some(signal) = reversed_signals.get(&sensor_idx) {
                // Using grid indices as position for TimeVaryingSource
                let position_indices = (i, j, k);

                // Create time-varying source with reversed signal
                let source =
                    TimeVaryingSource::new(position_indices, signal.clone(), solver.time().dt);
                sources.push(Box::new(source));

                debug!(
                    "Added reversed source for sensor {} at indices {:?}",
                    sensor_idx, position_indices
                );
            }
        }

        // Add all sources to the solver
        for source in sources {
            solver.add_source(source)?;
        }

        Ok(())
    }

    /// Propagate backwards in time
    fn propagate_backwards(
        &self,
        _grid: &Grid,
        solver: &mut PluginBasedSolver,
        _recorder: &mut Recorder,
        _frequency: f64,
        reversed_signals: &HashMap<usize, Vec<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        // Determine propagation time from signal length
        let max_signal_length = reversed_signals
            .values()
            .map(std::vec::Vec::len)
            .max()
            .unwrap_or(0);

        let propagation_steps = max_signal_length;

        // Run solver for backward propagation
        for step in 0..propagation_steps {
            solver.step()?;

            // Record at intervals if needed
            if step % 10 == 0 {
                // Recording disabled: Solver API returns Array4<f64> but recorder expects HashMap<UnifiedFieldType, Array3>
                // Future: Unified field interface for solver/recorder compatibility (Sprint 123+)
                // recorder.record(solver.fields(), solver.time().current)?;
            }
        }

        // Extract the reconstruction field (typically maximum pressure)
        let pressure_field = solver
            .get_field(crate::domain::field::mapping::UnifiedFieldType::Pressure)
            .ok_or_else(|| {
                crate::core::error::KwaversError::Validation(
                    crate::core::error::ValidationError::FieldValidation {
                        field: "pressure_field".to_string(),
                        value: "missing".to_string(),
                        constraint: "pressure field not found in solver".to_string(),
                    },
                )
            })?;
        Ok(pressure_field)
    }

    /// Check convergence between iterations
    fn check_convergence(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
    ) -> KwaversResult<f64> {
        let diff = current - previous;
        let norm_diff = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_current = current.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm_current > 0.0 {
            Ok(norm_diff / norm_current)
        } else {
            Ok(0.0)
        }
    }

    /// Post-process the reconstruction
    fn post_process(
        &self,
        mut reconstruction: Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Apply spatial windowing if configured
        if self.config.spatial_windowing {
            reconstruction = apply_spatial_window(reconstruction, grid, 0.1)?;
        }

        // Normalize the reconstruction
        let max_val = reconstruction.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        if max_val > 0.0 {
            reconstruction.mapv_inplace(|x| x / max_val);
        }

        Ok(reconstruction)
    }
}
