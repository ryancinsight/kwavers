// time/mod.rs
use aequitas::systems::si::quantities::{Length, ThermalDiffusivity, Time as Duration, Velocity};
use leto::Array1;
use log::debug;

#[derive(Debug, Clone)]
pub struct Time {
    pub dt: Duration,    // Time step
    pub n_steps: usize,  // Number of time steps
    pub t_max: Duration, // Total duration
}

#[derive(Debug, Clone, Default)]
pub struct StabilityConstraints {
    pub max_dt: Option<Duration>,
    pub cfl_number: Option<f64>,
    pub max_wave_speed: Option<Velocity>,
    pub diffusion_coefficient: Option<ThermalDiffusivity>,
}

impl Time {
    /// New.
    /// # Panics
    /// - Panics if assertion fails: `Time step and number of steps must be positive`.
    ///
    #[must_use]
    pub fn new(dt: Duration, n_steps: usize) -> Self {
        let dt_seconds = dt.into_base();
        assert!(
            dt_seconds > 0.0 && n_steps > 0,
            "Time step and number of steps must be positive"
        );
        let t_max = Duration::from_base(dt_seconds * (n_steps - 1) as f64);
        debug!(
            "Time initialized: dt = {:.6e}, n_steps = {}, t_max = {:.6e}",
            dt_seconds,
            n_steps,
            t_max.into_base()
        );
        Self { dt, n_steps, t_max }
    }

    /// From grid and duration.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn from_grid_and_duration(
        dx: Length,
        dy: Length,
        dz: Length,
        sound_speed: Velocity,
        duration: Duration,
    ) -> Self {
        let dx = dx.into_base();
        let dy = dy.into_base();
        let dz = dz.into_base();
        let sound_speed = sound_speed.into_base();
        let duration = duration.into_base();
        assert!(dx > 0.0 && dy > 0.0 && dz > 0.0 && sound_speed > 0.0 && duration > 0.0);
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414); // Relaxed CFL for k-space
        let dt = max_dt * 0.9;
        let n_steps = (duration / dt).ceil() as usize + 1;
        Self::new(Duration::from_base(dt), n_steps)
    }

    #[must_use]
    pub fn duration(&self) -> Duration {
        self.t_max
    }

    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.n_steps
    }

    #[must_use]
    pub fn time_vector(&self) -> Array1<f64> {
        let n = self.n_steps;
        Array1::from_shape_fn([n], |[i]| {
            if n <= 1 {
                0.0
            } else {
                i as f64 * self.t_max.into_base() / (n - 1) as f64
            }
        })
    }

    #[must_use]
    pub fn is_stable(&self, dx: Length, dy: Length, dz: Length, sound_speed: Velocity) -> bool {
        let dx = dx.into_base();
        let dy = dy.into_base();
        let dz = dz.into_base();
        let sound_speed = sound_speed.into_base();
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414);
        self.dt.into_base() <= max_dt
    }

    pub fn adjust_for_stability(
        &mut self,
        dx: Length,
        dy: Length,
        dz: Length,
        sound_speed: Velocity,
    ) {
        let dx = dx.into_base();
        let dy = dy.into_base();
        let dz = dz.into_base();
        let sound_speed = sound_speed.into_base();
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414);
        if self.dt.into_base() > max_dt {
            self.dt = Duration::from_base(max_dt * 0.9);
            self.n_steps = (self.t_max.into_base() / self.dt.into_base()).ceil() as usize + 1;
            self.t_max = Duration::from_base(self.dt.into_base() * (self.n_steps - 1) as f64);
            debug!(
                "Adjusted time for k-space stability: dt = {:.6e}, n_steps = {}",
                self.dt.into_base(),
                self.n_steps
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(meters: f64) -> Length {
        Length::from_base(meters)
    }

    #[test]
    fn typed_time_preserves_duration_and_stability_contracts() {
        let mut time = Time::new(Duration::from_base(1.0e-7), 3);
        assert_eq!(time.duration().into_base(), 2.0e-7);
        assert_eq!(time.time_vector().into_vec(), vec![0.0, 1.0e-7, 2.0e-7]);
        assert!(time.is_stable(
            length(1.0e-3),
            length(1.0e-3),
            length(1.0e-3),
            Velocity::from_base(1.0e3),
        ));

        time.adjust_for_stability(
            length(1.0e-5),
            length(1.0e-5),
            length(1.0e-5),
            Velocity::from_base(1.0e3),
        );
        assert!(time.is_stable(
            length(1.0e-5),
            length(1.0e-5),
            length(1.0e-5),
            Velocity::from_base(1.0e3),
        ));
    }

    #[test]
    fn grid_duration_constructor_returns_typed_step_and_duration() {
        let time = Time::from_grid_and_duration(
            length(1.0e-3),
            length(1.0e-3),
            length(2.0e-3),
            Velocity::from_base(1.5e3),
            Duration::from_base(1.0e-3),
        );
        assert!(time.dt.into_base() > 0.0);
        assert!(time.duration().into_base() >= 1.0e-3);
    }
}
