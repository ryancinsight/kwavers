//! High-level Simulation API for k-Wave-like experience
//!
//! This module provides the `Simulation` struct which orchestrates
//! the solver, sources, sensors, and medium to run a complete simulation.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::sensor::Sensor;
use crate::solver::kspace::config::KSpaceConfig;
use crate::solver::kspace::solver::KSpaceSolver;
use crate::solver::kspace::sources::{KSpaceSource, SourceMode};
use crate::solver::kspace::sensors::{SensorConfig};
use crate::source::{Source, SourceType};
use ndarray::{Array2, Array3};
use log::info;
use std::collections::HashMap;

/// Main simulation controller
pub struct Simulation<'a, M: Medium> {
    solver: KSpaceSolver,
    grid: Grid,
    sources: Vec<&'a dyn Source>,
    sensors: Vec<&'a mut Sensor>,
    medium: &'a M,
    config: KSpaceConfig,
}

impl<'a, M: Medium> Simulation<'a, M> {
    /// Create a new simulation
    pub fn new(
        config: KSpaceConfig,
        grid: Grid,
        medium: &'a M,
        sources: Vec<&'a dyn Source>,
        sensors: Vec<&'a mut Sensor>,
    ) -> KwaversResult<Self> {
        
        // 1. Prepare Sources
        // Combine generic sources into KSpaceSource
        let nt = config.nt;
        let dt = config.dt;
        
        let mut point_sources: HashMap<(usize, usize, usize), Vec<f64>> = HashMap::new();
        // Accumulate velocity sources: key -> [ux, uy, uz]
        let mut velocity_sources: HashMap<(usize, usize, usize), [Vec<f64>; 3]> = HashMap::new();
        
        // Initial conditions accumulators
        let mut p0_acc = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut ux0_acc = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut uy0_acc = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut uz0_acc = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        let mut has_p0 = false;
        let mut has_u0 = false;

        for source in &sources {
            let mask = source.create_mask(&grid);
            let s_type = source.source_type();
            let init_amp = source.initial_amplitude();
            
            // Iterate mask to find source points
            // Note: indexed_iter gives ((i, j, k), &val)
            for ((i, j, k), &val) in mask.indexed_iter() {
                if val.abs() > 1e-12 {
                    // 1. Time-varying signal processing
                    // Calculate signal for this point for all time steps
                    // We assume amplitude(t) * val
                    let mut signal = Vec::with_capacity(nt);
                    for step in 0..nt {
                        let t = step as f64 * dt;
                        signal.push(source.amplitude(t) * val);
                    }
                    
                    match s_type {
                        SourceType::Pressure => {
                            point_sources.entry((i, j, k))
                                .and_modify(|acc| {
                                    for (t, s) in signal.iter().enumerate() {
                                        acc[t] += s;
                                    }
                                })
                                .or_insert(signal);
                        },
                        SourceType::VelocityX => {
                            velocity_sources.entry((i, j, k))
                                .and_modify(|acc| {
                                    for (t, s) in signal.iter().enumerate() {
                                        acc[0][t] += s;
                                    }
                                })
                                .or_insert_with(|| {
                                    let zeros = vec![0.0; nt];
                                    [signal, zeros.clone(), zeros.clone()]
                                });
                        },
                        SourceType::VelocityY => {
                            velocity_sources.entry((i, j, k))
                                .and_modify(|acc| {
                                    for (t, s) in signal.iter().enumerate() {
                                        acc[1][t] += s;
                                    }
                                })
                                .or_insert_with(|| {
                                    let zeros = vec![0.0; nt];
                                    [zeros.clone(), signal, zeros.clone()]
                                });
                        },
                        SourceType::VelocityZ => {
                            velocity_sources.entry((i, j, k))
                                .and_modify(|acc| {
                                    for (t, s) in signal.iter().enumerate() {
                                        acc[2][t] += s;
                                    }
                                })
                                .or_insert_with(|| {
                                    let zeros = vec![0.0; nt];
                                    [zeros.clone(), zeros.clone(), signal]
                                });
                        },
                    }

                    // 2. Initial conditions processing
                    if init_amp.abs() > 1e-12 {
                        let contribution = val * init_amp;
                        match s_type {
                            SourceType::Pressure => {
                                p0_acc[[i, j, k]] += contribution;
                                has_p0 = true;
                            },
                            SourceType::VelocityX => {
                                ux0_acc[[i, j, k]] += contribution;
                                has_u0 = true;
                            },
                            SourceType::VelocityY => {
                                uy0_acc[[i, j, k]] += contribution;
                                has_u0 = true;
                            },
                            SourceType::VelocityZ => {
                                uz0_acc[[i, j, k]] += contribution;
                                has_u0 = true;
                            },
                        }
                    }
                }
            }
        }
        
        // Convert Pressure Sources
        let mut sorted_indices: Vec<(usize, usize, usize)> = point_sources.keys().cloned().collect();
        sorted_indices.sort(); // Deterministic order (lexicographical)
        
        let num_points = sorted_indices.len();
        let mut p_mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut p_signal = Array2::zeros((num_points, nt));
        
        for (idx, &grid_idx) in sorted_indices.iter().enumerate() {
            p_mask[grid_idx] = 1.0;
            let sig = &point_sources[&grid_idx];
            for t in 0..nt {
                p_signal[[idx, t]] = sig[t];
            }
        }

        // Convert Velocity Sources
        let mut u_indices: Vec<(usize, usize, usize)> = velocity_sources.keys().cloned().collect();
        u_indices.sort();
        
        let num_u_points = u_indices.len();
        let mut u_mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        // [3, num_sources, time_steps]
        let mut u_signal = Array3::zeros((3, num_u_points, nt));
        
        for (idx, &grid_idx) in u_indices.iter().enumerate() {
            u_mask[grid_idx] = 1.0;
            let sigs = &velocity_sources[&grid_idx];
            for t in 0..nt {
                u_signal[[0, idx, t]] = sigs[0][t];
                u_signal[[1, idx, t]] = sigs[1][t];
                u_signal[[2, idx, t]] = sigs[2][t];
            }
        }
        
        let kspace_source = KSpaceSource {
            p0: if has_p0 { Some(p0_acc) } else { None },
            u0: if has_u0 { Some((ux0_acc, uy0_acc, uz0_acc)) } else { None },
            p_mask: if num_points > 0 { Some(p_mask) } else { None },
            p_signal: if num_points > 0 { Some(p_signal) } else { None },
            u_mask: if num_u_points > 0 { Some(u_mask) } else { None },
            u_signal: if num_u_points > 0 { Some(u_signal) } else { None },
            p_mode: SourceMode::Additive, // Default to additive
            u_mode: SourceMode::Additive,
        };

        // 2. Prepare Sensors
        // Combine generic Sensors into KWaveSensorConfig
        let mut sensor_mask_arr = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
        
        for sensor in &sensors {
            for &(i, j, k) in &sensor.positions {
                if i < grid.nx && j < grid.ny && k < grid.nz {
                    sensor_mask_arr[[i, j, k]] = true;
                }
            }
        }
        
        // Update config with sensor mask
        let mut config_clone = config.clone();
        config_clone.sensor_mask = Some(sensor_mask_arr);
        
        let solver = KSpaceSolver::new(config_clone, grid.clone(), medium, kspace_source)?;

        Ok(Self {
            solver,
            grid,
            sources,
            sensors,
            medium,
            config,
        })
    }

    /// Run the simulation
    pub fn run(&mut self) -> KwaversResult<()> {
        info!("Starting simulation with {} steps", self.config.nt);
        
        // 1. Update Sensor Mask in Solver (if it wasn't set in new)
        // KSpaceSolver::new reads config.sensor_mask.
        // So we must set it in `new`.
        
        // 2. Time Loop
        for step in 0..self.config.nt {
            self.solver.step_forward()?;
            
            if step % 100 == 0 {
                info!("Step {}/{} completed", step, self.config.nt);
            }
        }
        
        // 3. Extract Sensor Data
        let data = self.solver.sensor_handler().extract_data();
        
        // 4. Map back to Sensors
        if let Some(p_data) = data.p {
            // p_data is [num_total_sensor_points, nt]
            // We need to map rows back to sensors.
            // We need the solver's sensor_indices.
            let sensor_indices = self.solver.sensor_handler().sensor_indices();
            let index_map: HashMap<(usize, usize, usize), usize> = sensor_indices.iter()
                .enumerate()
                .map(|(i, &pos)| (pos, i))
                .collect();
            
            for sensor in &mut self.sensors {
                let num_points = sensor.positions.len();
                let nt = self.config.nt;
                let mut p_sensor = Array2::zeros((num_points, nt));
                
                for (local_idx, pos) in sensor.positions.iter().enumerate() {
                    if let Some(&global_row_idx) = index_map.get(pos) {
                        // Copy row from p_data to p_sensor
                        // p_data is [total_sensors, nt]
                        for t in 0..nt {
                            p_sensor[[local_idx, t]] = p_data[[global_row_idx, t]];
                        }
                    }
                }
                sensor.pressure_data = p_sensor;
            }
        }

        info!("Simulation completed");
        Ok(())
    }
}
