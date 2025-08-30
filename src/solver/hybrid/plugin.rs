//! Physics plugin implementation for hybrid solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PluginContext, PluginMetadata, PluginState};
use crate::solver::hybrid::{HybridConfig, HybridSolver};
use ndarray::Array4;

/// Hybrid solver plugin for integration with physics pipeline
#[derive(Debug, Debug))]
pub struct HybridPlugin {
    solver: HybridSolver,
    metadata: PluginMetadata,
    state: PluginState,
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

        Ok(Self {
            solver,
            metadata,
            state: PluginState::Created,
        })
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

    fn diagnostics(&self) -> String {
        let metrics = self.solver.metrics();
        format!(
            "Hybrid Plugin - PSTD fraction: {:.2}%, Total time: {:.2}ms",
            metrics.pstd_fraction() * 100.0,
            metrics.total_time().as_millis()
        )
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
