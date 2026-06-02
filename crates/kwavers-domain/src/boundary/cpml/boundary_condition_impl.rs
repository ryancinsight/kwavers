//! `BoundaryCondition` and `AbsorbingBoundary` trait impls for `CPMLBoundary`

use super::CPMLBoundary;
use kwavers_core::error::KwaversResult;
use crate::boundary::traits::{AbsorbingBoundary, BoundaryCondition, BoundaryDirections};
use crate::grid::GridTopology;
use ndarray::{Array3, ArrayViewMut3, Zip};

// Implement new BoundaryCondition trait system
impl BoundaryCondition for CPMLBoundary {
    fn name(&self) -> &str {
        "CPML (Convolutional PML)"
    }

    fn active_directions(&self) -> BoundaryDirections {
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        _time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let spacing = grid.spacing();
        let dt = if dt > 0.0 {
            dt
        } else {
            self.estimate_dt_from_spacing(&spacing)
        };

        // Apply damping using sigma profiles
        Zip::indexed(field).par_for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt * 0.5).exp();
            }
        });

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        field: &mut Array3<num_complex::Complex<f64>>,
        grid: &dyn GridTopology,
        _time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let spacing = grid.spacing();
        let dt = if dt > 0.0 {
            dt
        } else {
            self.estimate_dt_from_spacing(&spacing)
        };

        Zip::indexed(field).par_for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                let decay = (-sigma_total * dt * 0.5).exp();
                val.re *= decay;
                val.im *= decay;
            }
        });

        Ok(())
    }

    fn reflection_coefficient(&self, angle_degrees: f64, _frequency: f64, sound_speed: f64) -> f64 {
        // Use typical grid spacing for estimation
        let dx = 1e-4;
        self.estimate_reflection(angle_degrees, dx, sound_speed)
    }

    fn reset(&mut self) {
        self.memory.reset();
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn memory_usage(&self) -> usize {
        // Estimate memory usage from all components
        std::mem::size_of_val(self)
            + self.memory.psi_v_x.len() * std::mem::size_of::<f64>()
            + self.memory.psi_v_y.len() * std::mem::size_of::<f64>()
            + self.memory.psi_v_z.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_x.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_y.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_z.len() * std::mem::size_of::<f64>()
    }
}

impl AbsorbingBoundary for CPMLBoundary {
    fn thickness(&self) -> usize {
        self.config.thickness
    }

    fn absorption_profile(&self, indices: [usize; 3], _grid: &dyn GridTopology) -> f64 {
        let s_x = self.profiles.sigma_x[indices[0]];
        let s_y = self.profiles.sigma_y[indices[1]];
        let s_z = self.profiles.sigma_z[indices[2]];
        s_x + s_y + s_z
    }

    fn target_reflection(&self) -> f64 {
        self.config.target_reflection
    }
}
