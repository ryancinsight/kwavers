//! Physics plugin implementation for hybrid solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState};
use crate::solver::hybrid::{HybridConfig, HybridSolver};
use ndarray::Array4;
use std::collections::HashMap;

/// Hybrid solver plugin for integration with physics pipeline
#[derive(Debug)]
pub struct HybridPlugin {
    solver: HybridSolver,
    metadata: PluginMetadata,
}

impl HybridPlugin {
    /// Create a new hybrid solver plugin
    pub fn new(config: HybridConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = HybridSolver::new(config, grid)?;
        let metadata = PluginMetadata {
            id: "hybrid_solver".to_string(),
            name: "Hybrid PSTD/FDTD Solver".to_string(),
            version: "1.0.0".to_string(),
            author: "Kwavers Team".to_string(),
            description: "Adaptive hybrid solver combining PSTD and FDTD methods".to_string(),
            license: "MIT".to_string(),
        };

        Ok(Self { solver, metadata })
    }
}

impl crate::physics::plugin::Plugin for HybridPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        PluginState::Initialized
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        // Solver is already initialized in new()
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        self.solver.update(fields, medium, dt, t)
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
            UnifiedFieldType::Density,
            UnifiedFieldType::SoundSpeed,
        ]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    fn diagnostics(&self) -> HashMap<String, f64> {
        let mut diagnostics = HashMap::new();
        let metrics = self.solver.metrics();

        diagnostics.insert("pstd_fraction".to_string(), metrics.pstd_fraction());
        diagnostics.insert(
            "total_time_ms".to_string(),
            metrics.total_time().as_millis() as f64,
        );

        diagnostics
    }
}
