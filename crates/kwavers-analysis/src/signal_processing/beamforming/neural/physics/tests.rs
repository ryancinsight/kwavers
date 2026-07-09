use super::*;
use Array;

#[test]
fn test_physics_constraints_creation() {
    let pc = PhysicsConstraints::new(0.8, 0.5, 0.2);
    assert_eq!(pc.reciprocity_weight(), 0.8);
    assert_eq!(pc.coherence_weight(), 0.5);
    assert_eq!(pc.sparsity_weight(), 0.2);
}

#[test]
fn test_physics_constraints_default() {
    let pc = PhysicsConstraints::default();
    assert_eq!(pc.reciprocity_weight(), 1.0);
    assert_eq!(pc.coherence_weight(), 0.5);
    assert_eq!(pc.sparsity_weight(), 0.1);
}

#[test]
fn test_reciprocity_smoothing() {
    let pc = PhysicsConstraints::new(1.0, 0.0, 0.0);
    let mut image = Array3::zeros((5, 5, 1));
    image[[2, 2, 0]] = 100.0; // Central spike

    let result = pc.apply_reciprocity(&image);

    // Center should be slightly reduced due to averaging
    assert!(result[[2, 2, 0]] < 100.0);
    // Neighbors should receive some signal
    assert!(result[[1, 2, 0]] > 0.0);
    assert!(result[[3, 2, 0]] > 0.0);
}

#[test]
fn test_coherence_diffusion() {
    let pc = PhysicsConstraints::new(0.0, 1.0, 0.0);
    let mut image = Array3::zeros((5, 5, 1));
    image[[2, 2, 0]] = 100.0; // Central spike

    let result = pc.apply_coherence(&image);

    // Diffusion should smooth the spike
    assert!(result[[2, 2, 0]] < 100.0);
    // Energy should spread to neighbors
    assert!(result[[1, 2, 0]] > 0.0 || result[[2, 1, 0]] > 0.0);
}

#[test]
fn test_sparsity_thresholding() {
    let pc = PhysicsConstraints::new(0.0, 0.0, 1.0);
    let mut image = Array3::from_elem((5, 5, 1), 1.0);
    image[[2, 2, 0]] = 100.0; // Strong peak

    let result = pc.apply_sparsity(&image);

    // Low values should be zeroed
    assert_eq!(result[[0, 0, 0]], 0.0);
    // Peak should be preserved (though reduced)
    assert!(result[[2, 2, 0]] > 50.0);
}

#[test]
fn test_combined_constraints() {
    let pc = PhysicsConstraints::default();
    let image = Array::from_shape_fn((10, 10, 5), |(i, j, _k)| {
        if i == 5 && j == 5 {
            100.0
        } else {
            (i as f32 + j as f32) * 0.1
        }
    });

    let result = pc.apply(&image).unwrap();

    // Should be well-formed
    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(result.dim() == image.dim());
}

#[test]
fn test_adaptive_weight_update() {
    let mut pc = PhysicsConstraints::default();
    let initial_reciprocity = pc.reciprocity_weight();

    let feedback = BeamformingFeedback {
        improvement: -0.1,
        error_gradient: 0.5,
        signal_quality: 0.7,
    };

    pc.update(&feedback).unwrap();

    // Weights should be reduced after negative feedback
    assert!(pc.reciprocity_weight() < initial_reciprocity);
    assert!(pc.coherence_weight() < 0.5);
}

#[test]
fn test_boundary_preservation() {
    let pc = PhysicsConstraints::default();
    let image = Array3::from_elem((5, 5, 1), 10.0);

    let result = pc.apply(&image).unwrap();

    // Boundary values should be approximately preserved
    assert!((result[[0, 0, 0]] - 10.0).abs() < 5.0);
    assert!((result[[4, 4, 0]] - 10.0).abs() < 5.0);
}
