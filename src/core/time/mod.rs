// time/mod.rs
use log::debug;
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct Time {
    pub dt: f64,        // Time step (seconds)
    pub n_steps: usize, // Number of time steps
    pub t_max: f64,     // Total duration (seconds)
}

impl Time {
    #[must_use]
    pub fn new(dt: f64, n_steps: usize) -> Self {
        assert!(
            dt > 0.0 && n_steps > 0,
            "Time step and number of steps must be positive"
        );
        let t_max = dt * (n_steps - 1) as f64;
        debug!(
            "Time initialized: dt = {:.6e}, n_steps = {}, t_max = {:.6e}",
            dt, n_steps, t_max
        );
        Self { dt, n_steps, t_max }
    }

    #[must_use]
    pub fn from_grid_and_duration(
        dx: f64,
        dy: f64,
        dz: f64,
        sound_speed: f64,
        duration: f64,
    ) -> Self {
        assert!(dx > 0.0 && dy > 0.0 && dz > 0.0 && sound_speed > 0.0 && duration > 0.0);
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414); // Relaxed CFL for k-space
        let dt = max_dt * 0.9;
        let n_steps = (duration / dt).ceil() as usize;
        Self::new(dt, n_steps)
    }

    #[must_use]
    pub fn duration(&self) -> f64 {
        self.t_max
    }

    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.n_steps
    }

    #[must_use]
    pub fn time_vector(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.t_max, self.n_steps)
    }

    #[must_use]
    pub fn is_stable(&self, dx: f64, dy: f64, dz: f64, sound_speed: f64) -> bool {
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414);
        self.dt <= max_dt
    }

    pub fn adjust_for_stability(&mut self, dx: f64, dy: f64, dz: f64, sound_speed: f64) {
        let min_dx = dx.min(dy).min(dz);
        let max_dt = min_dx / (sound_speed * 1.414);
        if self.dt > max_dt {
            self.dt = max_dt * 0.9;
            self.n_steps = (self.t_max / self.dt).ceil() as usize + 1;
            self.t_max = self.dt * (self.n_steps - 1) as f64;
            debug!(
                "Adjusted time for k-space stability: dt = {:.6e}, n_steps = {}",
                self.dt, self.n_steps
            );
        }
    }
}
