//! ElectromagneticWaveEquation trait impl and PEC boundary conditions.

use super::types::ElectromagneticFdtdSolver;
use crate::domain::field::EMFields;
use crate::physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};

impl ElectromagneticWaveEquation for ElectromagneticFdtdSolver {
    fn em_dimension(&self) -> EMDimension {
        EMDimension::Three
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        &self.materials
    }

    fn em_fields(&self) -> &EMFields {
        &self.fields_cache
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.dt = dt;
        self.update_electric_fields();
        self.update_magnetic_fields();
        self.time_step += 1;
        self.update_field_cache();
        Ok(())
    }

    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields) {
        self.apply_pec_boundaries();
        self.update_field_cache();
        *fields = self.fields_cache.clone();
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        let all_finite = self.ex.iter().all(|v| v.is_finite())
            && self.ey.iter().all(|v| v.is_finite())
            && self.ez.iter().all(|v| v.is_finite())
            && self.hx.iter().all(|v| v.is_finite())
            && self.hy.iter().all(|v| v.is_finite())
            && self.hz.iter().all(|v| v.is_finite())
            && fields.electric.iter().all(|v| v.is_finite())
            && fields.magnetic.iter().all(|v| v.is_finite());

        if !all_finite {
            return Err("non-finite values detected in EM fields".to_string());
        }

        Ok(())
    }
}

impl ElectromagneticFdtdSolver {
    /// Apply Perfect Electric Conductor (PEC) boundary conditions.
    pub(super) fn apply_pec_boundaries(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        for j in 0..ny {
            for k in 0..nz {
                self.ey[[0, j, k]] = 0.0;
                self.ey[[nx, j, k]] = 0.0;
                self.ez[[0, j, k]] = 0.0;
                self.ez[[nx, j, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                self.ex[[i, 0, k]] = 0.0;
                self.ex[[i, ny, k]] = 0.0;
                self.ez[[i, 0, k]] = 0.0;
                self.ez[[i, ny, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                self.ex[[i, j, 0]] = 0.0;
                self.ex[[i, j, nz]] = 0.0;
                self.ey[[i, j, 0]] = 0.0;
                self.ey[[i, j, nz]] = 0.0;
            }
        }
    }
}
