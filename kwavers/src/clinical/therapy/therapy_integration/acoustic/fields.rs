use super::AcousticWaveSolver;
use crate::core::constants::numerical::MPA_TO_PA;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::Source;
use ndarray::Array3;
use std::sync::Arc;

impl AcousticWaveSolver {
    /// Get current pressure field (Pa).
    ///
    /// Array dimensions are `[nx, ny, nz]`. Physical position of `field[[i,j,k]]`:
    /// ```text
    /// (x, y, z) = (i*dx, j*dy, k*dz)
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn pressure_field(&self) -> &Array3<f64> {
        self.backend.get_pressure_field()
    }

    /// Get current particle velocity fields (m/s) as `(vx, vy, vz)`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        self.backend.get_velocity_fields()
    }

    /// Get acoustic intensity field (W/m²) using plane wave approximation I = p²/(ρ₀c₀).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn intensity_field(&self) -> KwaversResult<Array3<f64>> {
        self.backend.get_intensity_field()
    }

    /// Get maximum pressure magnitude (MPa).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn max_pressure(&self) -> f64 {
        let p = self.pressure_field();
        let p_max = p.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        p_max / MPA_TO_PA // Pa → MPa
    }

    /// Get spatial peak temporal average intensity (W/cm²).
    ///
    /// Computes I_spta = max[(1/T_avg) ∫ p²/Z dt] over the field.
    ///
    /// # Errors
    ///
    /// Returns error if `averaging_time` ≤ 0 or if impedance field computation fails.
    pub fn spta_intensity(&self, averaging_time: f64) -> KwaversResult<f64> {
        if averaging_time <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Averaging time must be positive".into(),
            ));
        }

        let impedance = self.backend.get_impedance_field()?;
        let dt = self.backend.get_dt();
        let normalization = dt / averaging_time;

        let i_spta = self
            .accumulated_p_squared
            .iter()
            .zip(impedance.iter())
            .fold(0.0_f64, |max_val, (&acc_p2, &z)| {
                let val = (acc_p2 * normalization) / z;
                if val.is_nan() {
                    max_val
                } else {
                    max_val.max(val)
                }
            });

        Ok(i_spta / 1e4)
    }

    /// Register an acoustic source evaluated at each time step.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        self.backend.add_source(source)
    }

    /// Get grid dimensions as `(nx, ny, nz)`.
    pub fn grid_dimensions(&self) -> (usize, usize, usize) {
        (self.grid.nx, self.grid.ny, self.grid.nz)
    }
}
