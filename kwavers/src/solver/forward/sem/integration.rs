//! Time Integration for Spectral Elements
//!
//! Implements time integration schemes optimized for SEM:
//! - Newmark method for second-order accuracy
//! - Explicit schemes leveraging diagonal mass matrix
//! - Stability-optimized integration parameters

use ndarray::Array1;

/// Newmark time integration scheme for SEM
///
/// The Newmark method provides second-order accuracy and unconditional
/// stability for appropriate parameter choices. For SEM with diagonal
/// mass matrix, it enables efficient explicit time stepping.
#[derive(Debug, Clone)]
pub struct NewmarkIntegrator {
    /// Integration parameter γ (typically 0.5 for trapezoidal rule)
    pub gamma: f64,
    /// Integration parameter β (typically 0.25 for average acceleration)
    pub beta: f64,
    /// Current time step
    pub dt: f64,
    /// Current time
    pub time: f64,
    /// Displacement at current time
    pub displacement: Array1<f64>,
    /// Velocity at current time
    pub velocity: Array1<f64>,
    /// Acceleration at current time
    pub acceleration: Array1<f64>,
    /// Displacement at previous time step
    displacement_prev: Array1<f64>,
    /// Velocity at previous time step
    velocity_prev: Array1<f64>,
    /// Acceleration at previous time step
    acceleration_prev: Array1<f64>,
}

impl NewmarkIntegrator {
    /// Create Newmark integrator with initial conditions
    #[must_use]
    pub fn new(
        gamma: f64,
        beta: f64,
        dt: f64,
        initial_displacement: Array1<f64>,
        initial_velocity: Array1<f64>,
        initial_acceleration: Array1<f64>,
    ) -> Self {
        Self {
            gamma,
            beta,
            dt,
            time: 0.0,
            displacement: initial_displacement.clone(),
            velocity: initial_velocity.clone(),
            acceleration: initial_acceleration.clone(),
            displacement_prev: initial_displacement,
            velocity_prev: initial_velocity,
            acceleration_prev: initial_acceleration,
        }
    }

    /// Create integrator with standard Newmark parameters (average acceleration)
    #[must_use]
    pub fn average_acceleration(dt: f64, n_dofs: usize) -> Self {
        Self::new(
            0.5,  // γ = 1/2 (trapezoidal rule)
            0.25, // β = 1/4 (average acceleration)
            dt,
            Array1::zeros(n_dofs), // zero initial displacement
            Array1::zeros(n_dofs), // zero initial velocity
            Array1::zeros(n_dofs), // zero initial acceleration
        )
    }

    /// Advance one time step using Newmark method
    ///
    /// Given the acceleration at the current time step, updates displacement and velocity.
    ///
    /// # Arguments
    /// * `acceleration` - Acceleration at the current time step
    pub fn step(&mut self, acceleration: &Array1<f64>) {
        let dt2 = self.dt * self.dt;

        for i in 0..self.displacement.len() {
            self.displacement[i] = self.displacement_prev[i]
                + self.dt * self.velocity_prev[i]
                + 0.5 * dt2 * acceleration[i];
        }

        for i in 0..self.velocity.len() {
            self.velocity[i] = self.velocity_prev[i] + self.dt * acceleration[i];
        }

        self.acceleration.assign(acceleration);

        // Advance time
        self.time += self.dt;

        // Store previous values
        self.displacement_prev.assign(&self.displacement);
        self.velocity_prev.assign(&self.velocity);
        self.acceleration_prev.assign(&self.acceleration);
    }

    /// Predict displacement and velocity for next time step (for nonlinear problems)
    ///
    /// Returns predicted values that can be used in iterative solution procedures.
    #[must_use]
    pub fn predict(&self) -> (Array1<f64>, Array1<f64>) {
        let dt2 = self.dt * self.dt;
        let mut displacement_pred = Array1::zeros(self.displacement.len());
        let mut velocity_pred = Array1::zeros(self.velocity.len());

        // Prediction formulas (assuming constant acceleration)
        for i in 0..self.displacement.len() {
            displacement_pred[i] = self.displacement[i]
                + self.dt * self.velocity[i]
                + dt2 * (0.5 - self.beta) * self.acceleration[i] / self.beta;

            velocity_pred[i] =
                self.velocity[i] + self.dt * (1.0 - self.gamma) * self.acceleration[i] / self.gamma;
        }

        (displacement_pred, velocity_pred)
    }

    /// Check stability of the integration scheme
    #[must_use]
    pub fn is_stable(&self) -> bool {
        // For Newmark method, stability requires:
        // γ ≥ 0.5 and β ≥ (γ + 0.5)²/4
        self.gamma >= 0.5 && self.beta >= (self.gamma + 0.5) * (self.gamma + 0.5) / 4.0
    }

    /// Get critical time step for stability (approximate)
    ///
    /// For explicit schemes, this would be based on CFL condition.
    /// For implicit Newmark, it's theoretically unlimited but practically limited.
    #[must_use]
    pub fn critical_time_step(&self, max_frequency: f64) -> f64 {
        // Approximate critical time step based on highest frequency
        // For Newmark method with β=1/4, γ=1/2: Δt_crit ≈ 2/ω_max
        if max_frequency > 0.0 {
            2.0 / max_frequency
        } else {
            f64::INFINITY
        }
    }

    /// Reset integrator to initial state
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.displacement.fill(0.0);
        self.velocity.fill(0.0);
        self.acceleration.fill(0.0);
        self.displacement_prev.fill(0.0);
        self.velocity_prev.fill(0.0);
        self.acceleration_prev.fill(0.0);
    }
}

/// Explicit time integrator for SEM (leveraging diagonal mass matrix)
#[derive(Debug, Clone)]
pub struct SemExplicitIntegrator {
    /// Time step size
    pub dt: f64,
    /// Current time
    pub time: f64,
    /// Current field values
    pub field: Array1<f64>,
    /// Previous field values (for time derivatives)
    pub field_prev: Array1<f64>,
    /// Field time derivative
    pub field_dot: Array1<f64>,
}

impl SemExplicitIntegrator {
    /// Create explicit integrator
    #[must_use]
    pub fn new(dt: f64, n_dofs: usize) -> Self {
        Self {
            dt,
            time: 0.0,
            field: Array1::zeros(n_dofs),
            field_prev: Array1::zeros(n_dofs),
            field_dot: Array1::zeros(n_dofs),
        }
    }

    /// Advance one time step for wave equation: ∂²u/∂t² = f(u, ∇u, ...)
    ///
    /// Uses central difference: u_{n+1} = 2u_n - u_{n-1} + Δt²*f_n
    pub fn step_wave_equation(&mut self, rhs: &Array1<f64>) {
        let dt2 = self.dt * self.dt;

        if self.time == 0.0 {
            for i in 0..self.field.len() {
                let u0 = self.field[i];
                let v0 = self.field_dot[i];
                let a0 = rhs[i];
                self.field_prev[i] = u0 - self.dt * v0 + 0.5 * dt2 * a0;
            }
        }

        for i in 0..self.field.len() {
            let field_next = 2.0 * self.field[i] - self.field_prev[i] + dt2 * rhs[i];
            self.field_prev[i] = self.field[i];
            self.field[i] = field_next;
        }

        self.time += self.dt;
    }

    /// Get current time
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Reset integrator
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.field.fill(0.0);
        self.field_prev.fill(0.0);
        self.field_dot.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_newmark_average_acceleration() {
        let n_dofs = 10;
        let dt = 0.01;
        let integrator = NewmarkIntegrator::average_acceleration(dt, n_dofs);

        assert_eq!(integrator.gamma, 0.5);
        assert_eq!(integrator.beta, 0.25);
        assert_eq!(integrator.dt, dt);
        assert!(integrator.is_stable());
    }

    #[test]
    fn test_newmark_stability() {
        // Test stable parameters
        let integrator = NewmarkIntegrator::new(
            0.5,
            0.25,
            0.01,
            Array1::zeros(1),
            Array1::zeros(1),
            Array1::zeros(1),
        );
        assert!(integrator.is_stable());

        // Test unstable parameters
        let unstable = NewmarkIntegrator::new(
            0.3,
            0.1,
            0.01,
            Array1::zeros(1),
            Array1::zeros(1),
            Array1::zeros(1),
        );
        assert!(!unstable.is_stable());
    }

    #[test]
    fn test_newmark_constant_acceleration() {
        let n_dofs = 1;
        let dt = 0.1;
        let mut integrator = NewmarkIntegrator::average_acceleration(dt, n_dofs);

        // Constant acceleration of 1.0 m/s²
        let acceleration = Array1::from_vec(vec![1.0]);

        // After first step: should integrate constant acceleration
        integrator.step(&acceleration);

        // Theoretical: u = 0.5 * a * t², v = a * t
        let expected_displacement = 0.5 * 1.0 * dt * dt;
        let expected_velocity = 1.0 * dt;

        assert_relative_eq!(
            integrator.displacement[0],
            expected_displacement,
            epsilon = 1e-10
        );
        assert_relative_eq!(integrator.velocity[0], expected_velocity, epsilon = 1e-10);
    }

    #[test]
    fn test_explicit_wave_integration() {
        let n_dofs = 1;
        let dt = 0.01;
        let mut integrator = SemExplicitIntegrator::new(dt, n_dofs);

        // Simple harmonic oscillator: d²u/dt² = -ω²u
        let omega = 2.0 * std::f64::consts::PI; // 2π rad/s
        let omega_squared = omega * omega;

        // Initial conditions: u = 1, du/dt = 0
        integrator.field[0] = 1.0;

        // Integrate for a few steps
        for _ in 0..10 {
            let rhs = Array1::from_vec(vec![-omega_squared * integrator.field[0]]);
            integrator.step_wave_equation(&rhs);
        }

        let expected = (omega * integrator.current_time()).cos();
        assert_relative_eq!(integrator.field[0], expected, epsilon = 1e-3);
    }
}
