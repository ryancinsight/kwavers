use super::continuous::PIDController;
use super::core::{PIDConfig, PIDGains};

#[test]
fn test_pid_step_response() {
    let config = PIDConfig {
        gains: PIDGains {
            kp: 2.0,
            ki: 1.0,
            kd: 0.5,
        },
        sample_time: 0.01,
        ..Default::default()
    };

    let mut controller = PIDController::new(config);
    controller.set_setpoint(1.0);

    let mut measurement = 0.0;
    let dt = 0.01; // Time step

    // Run simulation for sufficient time to reach steady state
    for i in 0..500 {
        // Increased iterations
        let output = controller.update(measurement);

        // Apply control limits (PID output may be limited)
        let control = output.control_signal.clamp(0.0, 10.0);

        // Simple first-order system simulation: dx/dt = u - x
        // Using Euler integration: x(t+dt) = x(t) + dt * (u - x)
        let rate = control - measurement;
        measurement += dt * rate;

        // Check for early convergence
        if i > 100 && (measurement - 1.0).abs() < 0.01 {
            break;
        }
    }

    // Should converge close to setpoint
    assert!(
        (measurement - 1.0).abs() < 0.1,
        "Failed to converge: measurement = {}, expected ~1.0",
        measurement
    );
}

#[test]
fn test_anti_windup() {
    let config = PIDConfig {
        gains: PIDGains {
            kp: 1.0,
            ki: 10.0,
            kd: 0.0,
        },
        integral_limit: 1.0,
        ..Default::default()
    };

    let mut controller = PIDController::new(config);
    controller.set_setpoint(10.0); // Large setpoint

    // Run with measurement stuck at 0
    for _ in 0..100 {
        controller.update(0.0);
    }

    // Integral should be limited
    assert!(controller.integral_value().abs() <= 1.0);
}
