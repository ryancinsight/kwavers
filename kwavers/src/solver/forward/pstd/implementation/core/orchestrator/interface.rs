use super::PSTDSolver;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use std::sync::Arc;

impl crate::solver::interface::Solver for PSTDSolver {
    fn name(&self) -> &str {
        "PSTD"
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    fn add_source(&mut self, source: Box<dyn crate::domain::source::Source>) -> KwaversResult<()> {
        self.add_source_arc(Arc::from(source))
    }

    fn add_sensor(&mut self, _sensor: &crate::domain::sensor::GridSensorSet) -> KwaversResult<()> {
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        self.run_orchestrated(num_steps).map(|_| ())
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
        let max_p = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_v = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        crate::solver::interface::SolverStatistics {
            total_steps: self.time_step_index,
            current_step: self.time_step_index,
            computation_time: std::time::Duration::default(),
            memory_usage: 0,
            max_pressure: max_p,
            max_velocity: max_v,
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
