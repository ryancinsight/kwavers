//! Physics plugin implementation for hybrid solver

use crate::hybrid::{HybridConfig, HybridSolver};
use crate::plugin::{PluginMetadata, PluginState};
use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array4;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: HybridConfig, _grid: &Grid) -> KwaversResult<Self> {
        let metadata = PluginMetadata {
            id: "hybrid_solver".to_owned(),
            name: "Hybrid PSTD/FDTD Solver".to_owned(),
            version: "1.0.0".to_owned(),
            author: "Kwavers Team".to_owned(),
            description: "Adaptive hybrid solver combining PSTD and FDTD methods".to_owned(),
            license: "MIT".to_owned(),
        };

        Ok(Self {
            solver: None,
            config,
            metadata,
            state: PluginState::Created,
        })
    }
}

impl crate::plugin::Plugin for HybridPlugin {
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
        context: &mut crate::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            kwavers_core::error::KwaversError::InternalError(
                "Hybrid solver not initialized".to_owned(),
            )
        })?;

        // Hybrid solver requires a single source and boundary for its update method.
        // We use the first source from the context or a NullSource if none are provided.
        // The boundary is taken from the context.
        use kwavers_source::{NullSource, Source};
        let null_source = NullSource::new();
        let source: &dyn Source = context.sources.first().map_or(&null_source, |s| s.as_ref());
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
            "Hybrid Plugin - Not initialized".to_owned()
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
