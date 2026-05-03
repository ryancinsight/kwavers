//! `ElasticWaveSolver` struct definition and construction.

use super::super::super::boundary::{PMLBoundary, PMLConfig};
use super::super::super::types::{ElasticWaveConfig, VolumetricWaveConfig};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use ndarray::Array3;

/// 3D Elastic Wave Solver.
#[derive(Debug)]
pub struct ElasticWaveSolver {
    pub(super) grid: Grid,
    pub(super) density: Array3<f64>,
    pub(super) lambda: Array3<f64>,
    pub(super) mu: Array3<f64>,
    pub(super) pml: PMLBoundary,
    pub(super) config: ElasticWaveConfig,
    pub(super) volumetric_config: VolumetricWaveConfig,
    pub(crate) sensor_recorder: SensorRecorder,
}

impl ElasticWaveSolver {
    pub fn new(grid: &Grid, medium: &dyn Medium, config: ElasticWaveConfig) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let density = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| medium.density(i, j, k));
        let lambda = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
            medium.lame_lambda(x, y, z, grid)
        });
        let mu = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
            medium.lame_mu(x, y, z, grid)
        });
        let pml_config = PMLConfig {
            thickness: config.pml_thickness,
            sigma_max: 100.0,
            profile_order: 2,
            reflection_target: 1e-4,
        };
        let pml = PMLBoundary::new(grid, pml_config);
        let shape = (nx, ny, nz);
        let sensor_recorder = SensorRecorder::new(config.sensor_mask.as_ref(), shape, 0)?;
        Ok(Self {
            grid: grid.clone(),
            density,
            lambda,
            mu,
            pml,
            config,
            volumetric_config: VolumetricWaveConfig::default(),
            sensor_recorder,
        })
    }

    pub fn set_volumetric_config(&mut self, volumetric_config: VolumetricWaveConfig) {
        self.volumetric_config = volumetric_config;
    }
}
