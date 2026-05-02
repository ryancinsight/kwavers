use super::HybridSolver;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use std::sync::Arc;

impl crate::solver::interface::Solver for HybridSolver {
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

    fn add_sensor(&mut self, _sensor: &crate::domain::sensor::GridSensorSet) -> KwaversResult<()> {
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        for _ in 0..num_steps {
            self.step_forward()?;
        }
        Ok(())
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

    fn statistics(&self) -> crate::solver::interface::SolverStatistics {
        let max_pressure = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_velocity = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));

        crate::solver::interface::SolverStatistics {
            total_steps: self.time_step,
            current_step: self.time_step,
            computation_time: std::time::Duration::default(),
            memory_usage: 0,
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, _feature: crate::solver::feature::SolverFeature) -> bool {
        true
    }

    fn enable_feature(
        &mut self,
        _feature: crate::solver::feature::SolverFeature,
        _enable: bool,
    ) -> KwaversResult<()> {
        Ok(())
    }
}
