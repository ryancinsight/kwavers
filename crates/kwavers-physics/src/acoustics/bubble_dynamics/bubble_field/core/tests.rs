use ndarray::Array3;

use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};

use super::BubbleField;

#[test]
fn test_single_bubble_no_coupling() {
    let params = BubbleParameters::default();
    let mut field = BubbleField::new((10, 10, 10), params.clone());
    field.add_center_bubble(&params);

    let pressure = Array3::<f64>::zeros((10, 10, 10));
    let dp_dt = Array3::<f64>::zeros((10, 10, 10));

    field.update(&pressure, &dp_dt, 1e-8, 0.0);
    assert_eq!(field.bubbles.len(), 1);
}

#[test]
fn test_two_bubble_coupling_at_equilibrium() {
    let params = BubbleParameters::default();
    let mut field = BubbleField::with_spacing((20, 10, 10), params.clone(), (1e-6, 1e-6, 1e-6));

    field.add_bubble(5, 5, 5, BubbleState::new(&params));
    field.add_bubble(15, 5, 5, BubbleState::new(&params));

    let corrections = field.compute_secondary_pressures();
    for &val in corrections.values() {
        assert!(val.is_finite(), "Correction must be finite, got {val}");
    }
}

#[test]
fn test_coupling_threshold_skips_distant_bubbles() {
    let mut params = BubbleParameters::default();
    params.r0 = 1e-6;

    let mut field = BubbleField::with_spacing((20, 10, 10), params.clone(), (1e-3, 1e-3, 1e-3));
    field.coupling_threshold = 0.01;

    let mut s1 = BubbleState::new(&params);
    s1.wall_velocity = 1.0;
    s1.wall_acceleration = 1e6;

    field.add_bubble(0, 5, 5, s1);
    field.add_bubble(10, 5, 5, BubbleState::new(&params));

    let corrections = field.compute_secondary_pressures();
    for &val in corrections.values() {
        assert_eq!(val, 0.0, "Distant bubble coupling should be skipped");
    }
}

#[test]
fn test_nonzero_coupling_within_threshold() {
    let mut params = BubbleParameters::default();
    params.r0 = 1e-6;

    let mut field = BubbleField::with_spacing((10, 10, 10), params.clone(), (1e-6, 1e-6, 1e-6));
    field.coupling_threshold = 0.01;

    let mut s1 = BubbleState::new(&params);
    s1.radius = 1e-6;
    s1.wall_velocity = 5.0;
    s1.wall_acceleration = 1e8;

    field.add_bubble(0, 5, 5, s1);
    field.add_bubble(5, 5, 5, BubbleState::new(&params));

    let corrections = field.compute_secondary_pressures();
    let corr_at_j = corrections[&(5, 5, 5)];
    assert!(
        corr_at_j.abs() > 0.0,
        "Correction must be non-zero for coupled bubbles within threshold"
    );
    assert!(corr_at_j.is_finite());
}
