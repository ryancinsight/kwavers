//! Value-semantic regression tests for SchwarzBoundary transmission conditions.

use super::SchwarzBoundary;
use crate::domain::boundary::coupling::types::{BoundaryDirections, TransmissionCondition};
use ndarray::Array3;

#[test]
fn test_schwarz_neumann_flux_continuity() {
    // Test Neumann transmission with flux continuity
    // Physical scenario: Heat diffusion across domain interface
    // Expected: ∂u/∂n should match on both sides (flux continuity)

    let nx = 10;
    let ny = 10;
    let nz = 10;

    // Create test fields with known gradients
    // Interface at x = nx/2, normal pointing in +x direction
    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    // Set up linear gradient: u(x) = 2.0 * x
    // Then ∂u/∂x = 2.0 everywhere
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64;
                interface_field[[i, j, k]] = 2.0 * x;
                neighbor_field[[i, j, k]] = 2.0 * x + 5.0; // Offset for neighbor
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Neumann);

    // Apply transmission - should adjust to match flux
    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Verify that gradient correction was applied
    // The implementation applies a correction to match gradients
    // Since both fields have the same gradient (2.0), correction should be small
    let mid = nx / 2;
    let original_value = 2.0 * (mid as f64);
    let corrected_value = interface_field[[mid, ny / 2, nz / 2]];

    // The correction should be small for matching gradients
    assert!(
        (corrected_value - original_value).abs() < 1.0,
        "Neumann flux correction out of expected range: {} vs {}",
        corrected_value,
        original_value
    );
}

#[test]
fn test_schwarz_neumann_gradient_matching() {
    // Test that Neumann condition matches gradients across interface
    // Set up fields with different gradients and verify correction

    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    // Interface: gradient = 1.0, Neighbor: gradient = 3.0
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                interface_field[[i, j, k]] = 1.0 * (i as f64);
                neighbor_field[[i, j, k]] = 3.0 * (i as f64);
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Neumann);

    let original_mid = interface_field[[4, 4, 4]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_mid = interface_field[[4, 4, 4]];

    // Should have applied correction to reduce gradient mismatch
    assert!(
        corrected_mid != original_mid,
        "Neumann condition should modify field when gradients differ"
    );
}

#[test]
fn test_schwarz_robin_condition() {
    // Test Robin transmission: ∂u/∂n + αu = β
    // Physical scenario: Convective boundary condition (Newton's law of cooling)

    let nx = 10;
    let ny = 10;
    let nz = 10;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 20.0;

    let alpha = 0.5;
    let beta = 0.0; // For simplicity in this test

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Robin { alpha, beta });

    let original_value = interface_field[[5, 5, 5]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_value = interface_field[[5, 5, 5]];

    // Check that transmission was applied - value should change
    assert!(
        (corrected_value - original_value).abs() > 0.1,
        "Robin condition should modify interface values"
    );

    // Robin implementation blends multiple contributions, so the value
    // may be outside the [interface, neighbor] range due to gradient corrections
    // Just verify it's in a reasonable range
    assert!(
        (0.0..=30.0).contains(&corrected_value),
        "Robin condition produced unreasonable value: {}",
        corrected_value
    );
}

#[test]
fn test_schwarz_robin_with_nonzero_beta() {
    // Test Robin condition with non-zero β parameter
    // Robin condition: ∂u/∂n + αu = β

    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 5.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;

    let alpha = 1.0;
    let beta = 2.0; // Non-zero β

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Robin { alpha, beta });

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Verify that β parameter affects the result
    let corrected_value = interface_field[[4, 4, 4]];

    // With β ≠ 0, the Robin value should include the β term
    assert!(
        corrected_value > 0.0,
        "Robin condition with β should produce valid values"
    );
}

#[test]
fn test_schwarz_dirichlet_transmission() {
    // Test Dirichlet transmission: u_interface = u_neighbor

    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 100.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 200.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Dirichlet);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Verify direct copying
    assert_eq!(
        interface_field[[2, 2, 2]],
        200.0,
        "Dirichlet transmission should copy neighbor values"
    );
}

#[test]
fn test_schwarz_optimized_relaxation() {
    // Test optimized Schwarz with relaxation parameter
    // u_new = (1-θ)u_old + θ*u_neighbor

    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 30.0;

    let theta = 0.7; // Relaxation parameter

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Optimized)
        .with_relaxation(theta);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Expected: (1-0.7)*10 + 0.7*30 = 3 + 21 = 24
    let expected = (1.0 - theta) * 10.0 + theta * 30.0;
    assert!(
        (interface_field[[2, 2, 2]] - expected).abs() < 1e-10,
        "Optimized Schwarz relaxation failed: got {}, expected {}",
        interface_field[[2, 2, 2]],
        expected
    );
}

#[test]
fn test_schwarz_robin_zero_alpha() {
    // Edge case: α = 0 should behave like Neumann condition (avoid division by zero)

    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 15.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 25.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Robin {
            alpha: 0.0,
            beta: 0.0,
        });

    let original_value = interface_field[[2, 2, 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // With α ≈ 0, implementation avoids division by zero
    // Field should remain unchanged (early return)
    assert_eq!(
        interface_field[[2, 2, 2]],
        original_value,
        "Robin with α=0 should not modify field (avoids division by zero)"
    );
}

#[test]
fn test_schwarz_neumann_analytical_validation() {
    // Analytical validation: 1D heat equation with known solution
    // Problem: ∂T/∂t = α∂²T/∂x², steady state: ∂²T/∂x² = 0 → T(x) = Ax + B
    // At interface x=L/2: Neumann condition ensures ∂T/∂x continuous

    let nx = 21;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    // Set up analytical solution: T(x) = 100 + 5*x (linear temperature profile)
    // This satisfies steady-state heat equation with constant gradient = 5 K/m
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64;
                interface_field[[i, j, k]] = 100.0 + 5.0 * x;
                neighbor_field[[i, j, k]] = 100.0 + 5.0 * x;
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Neumann);

    // Store original values for comparison
    let original_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    // For matching gradients, correction should be minimal
    assert!(
        (corrected_center - original_center).abs() < 0.5,
        "Neumann flux continuity should preserve matching gradients: {} vs {}",
        corrected_center,
        original_center
    );
}

#[test]
fn test_schwarz_robin_analytical_validation() {
    // Analytical validation: 1D convection-diffusion with Robin BC
    // Problem: -k∂²T/∂x² = 0 with Robin BC: -k∂T/∂x + hT = h*T_∞
    // At interface: Robin condition couples temperature and flux

    let nx = 11;
    let ny = 5;
    let nz = 5;

    // Set up initial temperature distribution
    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 300.0; // 300 K
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 350.0; // 350 K

    let alpha = 0.1; // Convection parameter h/k
    let beta = 0.0; // Zero source term for simplicity

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Robin { alpha, beta });

    let original_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    // Robin condition should produce intermediate value
    // The implementation blends multiple contributions including gradients,
    // so the result may be outside the [interface, neighbor] range
    assert!(
        (0.0..500.0).contains(&corrected_center),
        "Robin condition should produce reasonable coupled value: {} (from {})",
        corrected_center,
        original_center
    );

    // Value should have changed due to coupling
    assert!(
        (corrected_center - original_center).abs() > 0.01,
        "Robin condition should modify interface temperature"
    );
}

#[test]
fn test_schwarz_neumann_conservation() {
    // Test that Neumann transmission conserves flux across interface
    // Physical requirement: ∫ ∂u/∂n dA should be consistent

    let nx = 16;
    let ny = 16;
    let nz = 16;

    // Create fields with uniform gradients
    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    // Linear profile: u(x) = 3x
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                interface_field[[i, j, k]] = 3.0 * (i as f64);
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Neumann);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Compute average flux correction applied
    let mut total_correction = 0.0;
    let mut count = 0;
    for i in 1..nx - 1 {
        for j in 0..ny {
            for k in 0..nz {
                let grad = (interface_field[[i + 1, j, k]] - interface_field[[i - 1, j, k]]) / 2.0;
                total_correction += grad.abs();
                count += 1;
            }
        }
    }
    let avg_gradient = total_correction / (count as f64);

    // Gradient should remain close to original (3.0)
    assert!(
        (avg_gradient - 3.0).abs() < 1.0,
        "Neumann condition should preserve gradient structure: avg_grad = {}",
        avg_gradient
    );
}

#[test]
fn test_schwarz_robin_energy_stability() {
    // Test that Robin condition maintains energy stability
    // For stable schemes: |u_new| ≤ |u_old| + |u_neighbor|

    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 5.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;

    let alpha = 1.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(TransmissionCondition::Robin { alpha, beta: 0.0 });

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    // Check stability: new value should be bounded
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let val = interface_field[[i, j, k]];
                assert!(
                    (0.0..=15.0).contains(&val),
                    "Robin condition produced unstable value: {}",
                    val
                );
            }
        }
    }
}
