// time/mod.rs
use aequitas::systems::si::quantities::{Length, Time as TimeQuantity, Velocity};
use leto::Array1;
use log::debug;

#[derive(Debug, Clone)]
pub struct Time {
    pub dt: f64,        // Time step (seconds)
    pub n_steps: usize, // Number of time steps
    pub t_max: f64,     // Total duration (seconds)
}

/// Relaxed k-space stability factor.
///
/// The k-space pseudospectral scheme tolerates Courant numbers above the
/// classic finite-difference limit; the fleet convention caps the step at the
/// diagonal-transit time `min_dx / (c * sqrt(2))`, i.e. the time for a wave to
/// cross one cell along a 2D diagonal. Dimensions: `[m] / ([m/s]) = [s]`.
const KSPACE_DIAGONAL_FACTOR: f64 = std::f64::consts::SQRT_2;

/// Largest stable step in seconds for a grid of minimum spacing `min_dx`
/// metres and sound speed `sound_speed` metres per second.
///
/// Centralizes the k-space CFL bound `min_dx / (sound_speed * sqrt(2))` so the
/// three call sites (`from_grid_and_duration`, `is_stable`,
/// `adjust_for_stability`) cannot drift apart.
fn cfl_max_dt(min_dx: f64, sound_speed: f64) -> f64 {
    min_dx / (sound_speed * KSPACE_DIAGONAL_FACTOR)
}

#[derive(Debug, Clone, Default)]
pub struct StabilityConstraints {
    pub max_dt: Option<f64>,
    pub cfl_number: Option<f64>,
    pub max_wave_speed: Option<f64>,
    pub diffusion_coefficient: Option<f64>,
}

impl Time {
    /// New.
    /// # Panics
    /// - Panics if assertion fails: `Time step and number of steps must be positive`.
    ///
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

    /// From grid and duration.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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
        let max_dt = cfl_max_dt(min_dx, sound_speed);
        let dt = max_dt * 0.9;
        let n_steps = (duration / dt).ceil() as usize;
        Self::new(dt, n_steps)
    }

    /// Typed successor of [`Time::from_grid_and_duration`]: grid spacings as
    /// [`Length`], sound speed as [`Velocity`], span as [`TimeQuantity`].
    /// Quantities convert to base units (metres, metres per second, seconds)
    /// at the boundary; the stored clock and the hot loop stay `f64`.
    /// # Panics
    /// - Panics under the same precondition as `from_grid_and_duration`
    ///   (positive spacings, speed, and duration in base units).
    ///
    #[must_use]
    pub fn from_quantities(
        dx: Length,
        dy: Length,
        dz: Length,
        sound_speed: Velocity,
        duration: TimeQuantity,
    ) -> Self {
        Self::from_grid_and_duration(
            *dx.as_base(),
            *dy.as_base(),
            *dz.as_base(),
            *sound_speed.as_base(),
            *duration.as_base(),
        )
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
        let n = self.n_steps;
        Array1::from_shape_fn([n], |[i]| {
            if n <= 1 {
                0.0
            } else {
                i as f64 * self.t_max / (n - 1) as f64
            }
        })
    }

    #[must_use]
    pub fn is_stable(&self, dx: f64, dy: f64, dz: f64, sound_speed: f64) -> bool {
        let min_dx = dx.min(dy).min(dz);
        self.dt <= cfl_max_dt(min_dx, sound_speed)
    }

    pub fn adjust_for_stability(&mut self, dx: f64, dy: f64, dz: f64, sound_speed: f64) {
        let min_dx = dx.min(dy).min(dz);
        let max_dt = cfl_max_dt(min_dx, sound_speed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::F64;

    /// The CFL bound is the diagonal-transit time: `[m] / ([m/s]) = [s]`.
    #[test]
    fn cfl_bound_matches_diagonal_transit_definition() {
        let min_dx = 0.5e-3;
        let sound_speed = 1500.0;
        let expected = min_dx / (sound_speed * std::f64::consts::SQRT_2);
        assert_eq!(cfl_max_dt(min_dx, sound_speed), expected);
    }

    /// `from_grid_and_duration` applies the documented 0.9 safety factor and
    /// the resulting grid is stable by the same bound it was built from.
    #[test]
    fn grid_duration_applies_safety_factor_and_is_stable() {
        let dx = 0.5e-3;
        let sound_speed = 1500.0;
        let time = Time::from_grid_and_duration(dx, dx, dx, sound_speed, 1.0e-3);
        let expected_dt = cfl_max_dt(dx, sound_speed) * 0.9;
        assert_eq!(time.dt, expected_dt);
        assert!(time.is_stable(dx, dx, dx, sound_speed));
    }

    /// `adjust_for_stability` clamps an oversized step back under the bound
    /// while preserving the simulated span.
    #[test]
    fn adjust_for_stability_clamps_and_preserves_span() {
        let dx = 0.5e-3;
        let sound_speed = 1500.0;
        let mut time = Time::new(cfl_max_dt(dx, sound_speed) * 2.0, 100);
        let original_span = time.duration();
        assert_eq!(original_span, cfl_max_dt(dx, sound_speed) * 2.0 * 99.0);
        time.adjust_for_stability(dx, dx, dx, sound_speed);
        assert!(time.is_stable(dx, dx, dx, sound_speed));
        assert!(time.duration() >= original_span);
        // n_steps is hundreds here, far below the 2^53 exactness bound of
        // `F64::from_count`; the pre-push cast gate owns the baseline.
        assert_eq!(time.t_max, time.dt * F64::from_count(time.n_steps - 1).0);
    }

    /// The typed successor builds exactly what the untyped one builds from
    /// the same physical inputs in base units.
    #[test]
    fn from_quantities_matches_untyped_successor() {
        let dx = 0.5e-3;
        let sound_speed = 1500.0;
        let duration = 1.0e-3;
        let expected = Time::from_grid_and_duration(dx, dx, dx, sound_speed, duration);
        let typed = Time::from_quantities(
            Length::from_base(dx),
            Length::from_base(dx),
            Length::from_base(dx),
            Velocity::from_base(sound_speed),
            TimeQuantity::from_base(duration),
        );
        assert_eq!(typed.dt, expected.dt);
        assert_eq!(typed.n_steps, expected.n_steps);
        assert_eq!(typed.t_max, expected.t_max);
        assert_eq!(typed.dt, cfl_max_dt(dx, sound_speed) * 0.9);
        assert!(typed.is_stable(dx, dx, dx, sound_speed));
    }

    /// Unit invariance at the boundary: millimetre/millisecond inputs build
    /// the identical clock to metre/second inputs, including anisotropic
    /// spacings where the minimum is not the first axis.
    #[test]
    fn from_quantities_is_unit_invariant() {
        use aequitas::systems::si::units::{Millimeter, Millisecond};

        let base = Time::from_grid_and_duration(2.0e-3, 1.0e-3, 0.5e-3, 1500.0, 1.0e-3);
        let typed = Time::from_quantities(
            Length::from_unit::<Millimeter>(2.0),
            Length::from_unit::<Millimeter>(1.0),
            Length::from_unit::<Millimeter>(0.5),
            Velocity::from_base(1500.0),
            TimeQuantity::from_unit::<Millisecond>(1.0),
        );
        assert_eq!(typed.dt, base.dt);
        assert_eq!(typed.n_steps, base.n_steps);
        assert_eq!(typed.t_max, base.t_max);
        assert_eq!(typed.dt, cfl_max_dt(0.5e-3, 1500.0) * 0.9);
    }
}
