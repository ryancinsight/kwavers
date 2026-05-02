use super::NonlinearElasticWaveSolver;

impl NonlinearElasticWaveSolver {
    /// Calculate stable time step using CFL condition (amplitude = max_strain * 1e-3).
    #[must_use]
    #[allow(dead_code)]
    pub(super) fn calculate_time_step(&self) -> f64 {
        self.calculate_time_step_for_amplitude(self.config.max_strain * 1e-3)
    }

    /// Calculate stable time step for given amplitude.
    ///
    /// ## CFL condition for nonlinear waves
    /// `dt ≤ CFL * Δx / c_max`,  where `c_max = c(1 + β |u| / u_ref)`.
    pub(super) fn calculate_time_step_for_amplitude(&self, max_abs_u: f64) -> f64 {
        let c = self.config.sound_speed();
        let beta = self.config.nonlinearity_parameter.abs();
        let u_ref = 1e-3;
        let cfl = 0.45;

        let max_u_over_ref = (max_abs_u / u_ref).max(0.0);
        let max_speed = c * (1.0 + beta * max_u_over_ref);
        let dt_cfl = cfl * self.grid.dx / max_speed.max(f64::EPSILON);

        dt_cfl.min(self.config.max_dt).max(f64::EPSILON)
    }
}
