use crate::core::error::{KwaversError, KwaversResult};
use super::AcousticWaveSolver;

impl AcousticWaveSolver {
    /// Advance simulation by one time step.
    pub fn step(&mut self) -> KwaversResult<()> {
        self.backend.step()?;

        let p = self.backend.get_pressure_field();
        self.accumulated_p_squared
            .zip_mut_with(p, |acc, &val| *acc += val * val);

        Ok(())
    }

    /// Advance simulation by the specified physical time duration (seconds).
    pub fn advance(&mut self, duration: f64) -> KwaversResult<()> {
        if duration < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Duration must be non-negative".into(),
            ));
        }

        let dt = self.backend.get_dt();
        let num_steps = (duration / dt).ceil() as usize;

        for _ in 0..num_steps {
            self.step()?;
        }

        Ok(())
    }

    /// Get simulation time step (s).
    pub fn timestep(&self) -> f64 {
        self.backend.get_dt()
    }

    /// Get current simulation time (s).
    pub fn current_time(&self) -> f64 {
        self.backend.get_current_time()
    }
}
