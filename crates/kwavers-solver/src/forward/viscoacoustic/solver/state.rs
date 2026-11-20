use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Maximum unrelaxed (high-frequency) sound speed `max √(M_U(x)/ρ(x))`
    /// \[m·s⁻¹] over the grid — the CFL reference speed.
    #[must_use]
    pub fn unrelaxed_speed(&self) -> f64 {
        self.max_unrelaxed_speed
    }

    /// Pressure field \\[Pa\].
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.p
    }

    /// Overwrite the pressure field (shape must match the grid); resets velocity
    /// and memory variables to zero. Used to set an initial condition.
    /// # Errors
    /// - Shape mismatch.
    pub fn set_pressure(&mut self, p: &Array3<f64>) -> KwaversResult<()> {
        if p.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(
                "pressure shape must equal the grid".to_owned(),
            ));
        }
        self.p.assign(p);
        if self.axes.x {
            self.vx.fill(0.0);
        }
        if self.axes.y {
            self.vy.fill(0.0);
        }
        if self.axes.z {
            self.vz.fill(0.0);
        }
        for s in &mut self.sigma {
            s.fill(0.0);
        }
        // Restart the simulation clock and sensor traces (sources/sensors kept).
        self.step_count = 0;
        for trace in &mut self.sensor_record {
            trace.clear();
        }
        Ok(())
    }
}
