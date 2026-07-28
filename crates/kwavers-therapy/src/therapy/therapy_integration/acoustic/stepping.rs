use super::AcousticWaveSolver;
use aequitas::systems::si::quantities::Time;
use kwavers_core::error::{KwaversError, KwaversResult};

impl AcousticWaveSolver {
    /// Advance simulation by one time step.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn step(&mut self) -> KwaversResult<()> {
        self.backend.step()?;

        let p = self.backend.get_pressure_field();
        leto_ops::zip_mut_with(
            &mut self.accumulated_p_squared.view_mut(),
            &p.view(),
            |acc, val| *acc += val * val,
        )?;

        Ok(())
    }

    /// Advance simulation by the specified physical time duration (seconds).
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn advance(&mut self, duration: Time<f64>) -> KwaversResult<()> {
        let duration_seconds = duration.into_base();
        if duration_seconds < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Duration must be non-negative".into(),
            ));
        }

        let dt = self.backend.get_dt();
        let num_steps = (duration_seconds / dt).ceil() as usize;

        for _ in 0..num_steps {
            self.step()?;
        }

        Ok(())
    }

    /// Get simulation time step (s).
    pub fn timestep(&self) -> Time<f64> {
        Time::from_base(self.backend.get_dt())
    }

    /// Get current simulation time.
    pub fn current_time(&self) -> Time<f64> {
        Time::from_base(self.backend.get_current_time())
    }
}
