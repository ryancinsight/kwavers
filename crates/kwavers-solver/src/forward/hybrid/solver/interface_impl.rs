use super::HybridSolver;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_domain::source::Source;
use std::sync::Arc;

impl crate::interface::Solver for HybridSolver {
    fn name(&self) -> &str {
        "Hybrid"
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()> {
        let arc_source: Arc<dyn Source> = Arc::from(source);
        self.pstd_solver.add_source_arc(arc_source.clone())?;
        self.fdtd_solver.add_source_arc(arc_source)?;
        Ok(())
    }

    fn add_sensor(&mut self, _sensor: &kwavers_domain::sensor::GridSensorSet) -> KwaversResult<()> {
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        for _ in 0..num_steps {
            self.step_forward()?;
        }
        Ok(())
    }

    fn step_forward(&mut self) -> KwaversResult<()> {
        // Inherent step_forward on HybridSolver; direct call, no run(1) dispatch.
        HybridSolver::step_forward(self)
    }

    fn pressure_field(&self) -> &ndarray::Array3<f64> {
        &self.fields.p
    }

    fn velocity_fields(
        &self,
    ) -> (
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
    ) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    fn statistics(&self) -> crate::interface::SolverStatistics {
        let max_pressure = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_velocity = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));

        crate::interface::SolverStatistics {
            total_steps: self.time_step,
            current_step: self.time_step,
            computation_time: std::time::Duration::default(),
            memory_usage: 0,
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, _feature: crate::feature::SolverFeature) -> bool {
        true
    }

    fn enable_feature(
        &mut self,
        _feature: crate::feature::SolverFeature,
        _enable: bool,
    ) -> KwaversResult<()> {
        Ok(())
    }
}
