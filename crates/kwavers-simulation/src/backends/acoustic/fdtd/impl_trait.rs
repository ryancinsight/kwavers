//! `AcousticSolverBackend` impl for `FdtdBackend`.

use super::super::backend::AcousticSolverBackend;
use super::backend::FdtdBackend;
use kwavers_core::error::KwaversResult;
use kwavers_source::Source;
use leto::Array3;
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
        let mut result = Array3::<f64>::zeros(p.shape());
        for (p_idx, (&p_val, (&rho_val, &c_val))) in
            p.iter().zip(rho.iter().zip(c.iter())).enumerate()
        {
            let shape = p.shape();
            let i = p_idx / (shape[1] * shape[2]);
            let rem = p_idx % (shape[1] * shape[2]);
            let j = rem / shape[2];
            let k = rem % shape[2];
            let impedance = rho_val * c_val;
            let val = if impedance > 0.0 {
                p_val * p_val / impedance
            } else {
                0.0
            };
            result[[i, j, k]] = val;
        }
        Ok(result)
    }

    fn get_impedance_field(&self) -> KwaversResult<Array3<f64>> {
        let rho = &self.solver.materials.rho0;
        let c = &self.solver.materials.c0;
        let mut result = Array3::<f64>::zeros(rho.shape());
        for (idx, (&rho_val, &c_val)) in rho.iter().zip(c.iter()).enumerate() {
            let shape = rho.shape();
            let i = idx / (shape[1] * shape[2]);
            let rem = idx % (shape[1] * shape[2]);
            let j = rem / shape[2];
            let k = rem % shape[2];
            result[[i, j, k]] = rho_val * c_val;
        }
        Ok(result)
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
