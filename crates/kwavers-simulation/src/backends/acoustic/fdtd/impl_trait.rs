//! `AcousticSolverBackend` impl for `FdtdBackend`.

use super::super::backend::AcousticSolverBackend;
use super::backend::FdtdBackend;
use kwavers_core::error::KwaversResult;
use kwavers_source::Source;
use ndarray::Array3;
use std::sync::Arc;

impl AcousticSolverBackend for FdtdBackend {
    fn step(&mut self) -> KwaversResult<()> {
        self.solver.step_forward()?;
        self.sync_shadow_fields();
        self.current_time += self.solver.config.dt;
        Ok(())
    }

    fn get_pressure_field(&self) -> &Array3<f64> {
        &self.pressure
    }

    fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    fn get_intensity_field(&self) -> KwaversResult<Array3<f64>> {
        // Plane-wave approximation: I = p² / (ρc) = p² / Z
        let p = &self.pressure;
        let rho = &self.solver.materials.rho0;
        let c = &self.solver.materials.c0;
        let impedance = rho * c;
        Ok(p.mapv(|p_val| p_val * p_val) / &impedance)
    }

    fn get_impedance_field(&self) -> KwaversResult<Array3<f64>> {
        let rho = &self.solver.materials.rho0;
        let c = &self.solver.materials.c0;
        Ok(rho * c)
    }

    fn get_dt(&self) -> f64 {
        self.solver.config.dt
    }

    fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        self.solver.add_source_arc(source)
    }

    fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn get_grid_dimensions(&self) -> (usize, usize, usize) {
        self.grid_dims
    }
}
