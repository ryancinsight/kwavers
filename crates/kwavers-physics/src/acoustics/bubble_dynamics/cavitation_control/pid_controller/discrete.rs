use super::core::{PIDGains, DEFAULT_OUTPUT_MAX, DEFAULT_OUTPUT_MIN};

/// PID controller using Tustin's method for discrete-time implementation
/// Suitable for digital control systems
#[derive(Debug)]
pub struct TustinPIDController {
    gains: PIDGains,
    sample_time: f64,
    // State variables for discrete implementation
    proportional_state: f64,
    integral_state: f64,
    derivative_state: f64,
    previous_input: f64,
    output_limits: (f64, f64),
}

impl TustinPIDController {
    #[must_use]
    pub fn new(gains: PIDGains, sample_time: f64) -> Self {
        Self {
            gains,
            sample_time,
            proportional_state: 0.0,
            integral_state: 0.0,
            derivative_state: 0.0,
            previous_input: 0.0,
            output_limits: (DEFAULT_OUTPUT_MIN, DEFAULT_OUTPUT_MAX),
        }
    }

    /// Update using discrete-time equations
    pub fn update(&mut self, setpoint: f64, measurement: f64) -> f64 {
        let error = setpoint - measurement;

        // Discrete proportional
        self.proportional_state = self.gains.kp * error;

        // Discrete integral (trapezoidal rule)
        self.integral_state += self.gains.ki * self.sample_time * error;

        // Discrete derivative (backward difference)
        let derivative_raw = (measurement - self.previous_input) / self.sample_time;
        self.derivative_state = self.gains.kd * derivative_raw;

        // Total output
        let output = self.proportional_state + self.integral_state - self.derivative_state;

        // Apply limits and anti-windup
        let limited_output = output.clamp(self.output_limits.0, self.output_limits.1);

        if (output - limited_output).abs() > 1e-6 {
            // Back-calculation anti-windup
            self.integral_state = limited_output - self.proportional_state + self.derivative_state;
        }

        self.previous_input = measurement;

        limited_output
    }

    pub fn reset(&mut self) {
        self.proportional_state = 0.0;
        self.integral_state = 0.0;
        self.derivative_state = 0.0;
        self.previous_input = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::PIDGains;
    use super::*;

    /// P-only Tustin controller (ki=kd=0): update(setpoint=1, meas=0) returns kp*error=1.0.
    #[test]
    fn tustin_p_only_output_equals_kp_times_error() {
        let gains = PIDGains {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
        };
        let mut ctrl = TustinPIDController::new(gains, 0.001);
        let out = ctrl.update(1.0, 0.0);
        assert!(
            (out - 1.0).abs() < 1e-14,
            "P-only output must equal kp*error=1.0; got {out}"
        );
    }

    /// With ki > 0, integral_state accumulates proportionally to error and sample_time.
    ///
    /// After one step: integral += ki * dt * error = 0.5 * 0.001 * 1.0 = 0.0005.
    /// output = kp*error + integral = 0 + 0.0005 = 0.0005 (within [0,1]).
    #[test]
    fn tustin_i_only_accumulates_after_one_step() {
        let gains = PIDGains {
            kp: 0.0,
            ki: 0.5,
            kd: 0.0,
        };
        let dt = 0.001;
        let mut ctrl = TustinPIDController::new(gains, dt);
        let out = ctrl.update(1.0, 0.0); // error=1.0
        let expected = 0.5 * dt * 1.0; // 0.0005
        assert!(
            (out - expected).abs() < 1e-14,
            "I-only output after one step must be {expected}; got {out}"
        );
    }

    /// reset zeros all state variables.
    ///
    /// Invariant: after `reset()`, the controller must respond identically to a freshly
    /// constructed controller with the same gains. Comparing against a fresh controller
    /// (rather than a hand-computed P-only value) is the correct invariant because any
    /// single `update` step with `ki > 0` legitimately adds `ki*dt*error` to the integral
    /// state on the first call — the integral does not stay at zero after one step.
    #[test]
    fn tustin_reset_clears_all_states() {
        let gains = PIDGains {
            kp: 1.0,
            ki: 1.0,
            kd: 0.0,
        };
        let dt = 0.001;
        let mut ctrl = TustinPIDController::new(gains, dt);
        ctrl.update(1.0, 0.0);
        ctrl.reset();
        let post_reset = ctrl.update(0.5, 0.0);

        let mut fresh = TustinPIDController::new(gains, dt);
        let fresh_out = fresh.update(0.5, 0.0);

        assert!(
            (post_reset - fresh_out).abs() < 1e-14,
            "post-reset response must equal fresh-controller response; \
             post_reset={post_reset}, fresh={fresh_out}"
        );
        // And the fresh response itself must equal kp*error + ki*dt*error - 0
        // = 1.0*0.5 + 1.0*0.001*0.5 = 0.5005, confirming reset cleared the prior accumulation.
        let expected = gains.kp * 0.5 + gains.ki * dt * 0.5;
        assert!(
            (post_reset - expected).abs() < 1e-14,
            "post-reset response must equal kp*error + ki*dt*error = {expected}; got {post_reset}"
        );
    }
}
