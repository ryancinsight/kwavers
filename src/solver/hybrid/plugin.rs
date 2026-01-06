//! Physics plugin implementation for hybrid solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PluginContext, PluginMetadata, PluginState};
use crate::solver::hybrid::{HybridConfig, HybridSolver};
use ndarray::Array4;

/// Hybrid solver plugin for integration with physics pipeline
#[derive(Debug)]
pub struct HybridPlugin {
    solver: Option<HybridSolver>,
    config: HybridConfig,
    metadata: PluginMetadata,
    state: PluginState,
}

impl HybridPlugin {
    /// Create a new hybrid solver plugin
    pub fn new(config: HybridConfig, _grid: &Grid) -> KwaversResult<Self> {
        let metadata = PluginMetadata {
            id: "hybrid_solver".to_string(),
            name: "Hybrid PSTD/FDTD Solver".to_string(),
            version: "1.0.0".to_string(),
            author: "Kwavers Team".to_string(),
            description: "Adaptive hybrid solver combining PSTD and FDTD methods".to_string(),
            license: "MIT".to_string(),
        };

        Ok(Self {
            solver: None,
            config,
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
        self.state
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let solver = HybridSolver::new(self.config.clone(), grid, medium)?;
        self.solver = Some(solver);
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            crate::error::KwaversError::InternalError("Hybrid solver not initialized".to_string())
        })?;

        // Hybrid solver requires a single source and boundary for its update method.
        // We use the first source from the context or a NullSource if none are provided.
        // The boundary is taken from the context.
        use crate::source::{NullSource, Source};
        let null_source = NullSource::new();
        let source: &dyn Source = context
            .sources
            .first()
            .map(|s| s.as_ref())
            .unwrap_or(&null_source);
        let boundary = &mut *context.boundary;

        solver.update(fields, medium, source, boundary, dt, t)
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
        if let Some(solver) = &self.solver {
            let metrics = solver.metrics();
            format!(
                "Hybrid Plugin - PSTD fraction: {:.2}%, Total time: {:.2}ms",
                metrics.pstd_fraction() * 100.0,
                metrics.total_time().as_millis()
            )
        } else {
            "Hybrid Plugin - Not initialized".to_string()
        }
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
