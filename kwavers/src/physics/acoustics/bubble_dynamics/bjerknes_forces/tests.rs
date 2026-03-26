use super::*;

#[test]
fn test_bjerknes_calculator_creation() {
    let config = BjerknesConfig::default();
    let _calc = BjerknesCalculator::new(config);
}

#[test]
fn test_primary_bjerknes_force_calculation() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let radius = 5e-6; // 5 μm
    let pressure = 100e3; // 100 kPa
    let gradient = 1e6; // 1 MPa/m

    let force = calc
        .primary_bjerknes_force(radius, pressure, gradient)
        .unwrap();

    assert!(force.is_finite());
}

#[test]
fn test_secondary_bjerknes_attractive() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let r1 = 5e-6;
    let r2 = 5e-6;
    let v1 = 1e-15;
    let v2 = 1e-15;
    let phase = 0.0; // In-phase oscillations
    let distance = 50e-6;

    let force = calc
        .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
        .unwrap();

    assert_eq!(force.interaction_type, InteractionType::Attractive);
    assert!(!force.coalescing); // Distance > coalescence distance
}

#[test]
fn test_secondary_bjerknes_repulsive() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let r1 = 5e-6;
    let r2 = 5e-6;
    let v1 = 1e-15;
    let v2 = 1e-15;
    let phase = std::f64::consts::PI; // Out-of-phase oscillations
    let distance = 50e-6;

    let force = calc
        .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
        .unwrap();

    assert_eq!(force.interaction_type, InteractionType::Repulsive);
}

#[test]
fn test_coalescence_detection() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let r1 = 5e-6;
    let r2 = 5e-6;
    let v1 = 1e-15;
    let v2 = 1e-15;
    let phase = 0.0;

    // Very close distance (within coalescence range)
    let distance = 0.5e-6; // 0.5 μm
    let force = calc
        .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
        .unwrap();

    assert!(force.coalescing);
}

#[test]
fn test_bubble_motion_prediction() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let radius = 5e-6;
    let force = 1e-12; // 1 pN
    let velocity = 0.1; // 0.1 m/s
    let dt = 1e-6; // 1 μs

    let result = calc
        .predict_bubble_motion(radius, force, velocity, dt)
        .unwrap();

    assert!(result.0.is_finite()); // Displacement
    assert!(result.1.is_finite()); // Velocity
}

#[test]
fn test_coalescence_probability() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    // Close distance, approaching
    let prob_approach = calc.coalescence_probability(0.5e-6, 0.1);
    assert!(prob_approach > 0.0);

    // Far distance
    let prob_far = calc.coalescence_probability(100e-6, 0.1);
    assert_eq!(prob_far, 0.0);

    // Moving apart
    let prob_separate = calc.coalescence_probability(0.5e-6, -0.1);
    assert_eq!(prob_separate, 0.0);
}

#[test]
fn test_interaction_range() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let r1 = 5e-6;
    let r2 = 5e-6;
    let v1 = 1e-15;
    let v2 = 1e-15;
    let phase = 0.0;

    // Beyond interaction range
    let distance = 200e-6; // Beyond 100 μm range
    let force = calc
        .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
        .unwrap();

    assert_eq!(force.secondary, 0.0); // No force at large distance
    assert_eq!(force.interaction_type, InteractionType::Neutral);
}

#[test]
fn test_invalid_radius() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let result = calc.primary_bjerknes_force(0.0, 100e3, 1e6);
    assert!(result.is_err());
}

#[test]
fn test_zero_distance_error() {
    let config = BjerknesConfig::default();
    let calc = BjerknesCalculator::new(config);

    let result = calc.secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 0.0);
    assert!(result.is_err());
}
