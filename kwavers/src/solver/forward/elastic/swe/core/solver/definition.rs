//! `ElasticWaveSolver` struct definition and construction.

use super::super::super::boundary::{ElasticSwePMLBoundary, SwePmlConfig};
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
    pub(super) pml: ElasticSwePMLBoundary,
    pub(super) config: ElasticWaveConfig,
    pub(super) volumetric_config: VolumetricWaveConfig,
    pub(crate) sensor_recorder: SensorRecorder,
}

impl ElasticWaveSolver {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
        // Compute σ_max from the medium's maximum P-wave speed so the PML
        // provides the target reflection (1e-4) regardless of wave speed.
        // σ_max = −ln(R) · c_max / (2 · L_pml)  (Collino & Tsogka 2001)
        let c_max_p = lambda
            .iter()
            .zip(mu.iter())
            .zip(density.iter())
            .filter_map(|((la, mv), rho)| {
                if *rho > 0.0 {
                    Some((2.0f64.mul_add(*mv, *la) / *rho).sqrt())
                } else {
                    None
                }
            })
            .fold(0.0_f64, f64::max);
        let sigma_max = if c_max_p > 0.0 {
            ElasticSwePMLBoundary::optimize_sigma_max(1e-4, c_max_p, grid, config.pml_thickness)
        } else {
            1e6 // fallback for degenerate media
        };
        let pml_config = SwePmlConfig {
            thickness: config.pml_thickness,
            sigma_max,
            profile_order: 2,
            reflection_target: 1e-4,
        };
        let pml = ElasticSwePMLBoundary::new(grid, pml_config);
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
