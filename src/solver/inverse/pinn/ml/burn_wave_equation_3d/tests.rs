//! Integration tests for 3D wave equation PINN
//!
//! This module contains end-to-end integration tests that validate the complete
//! workflow from solver creation through training to prediction. Tests cover:
//!
//! - Rectangular, spherical, and cylindrical geometries
//! - Homogeneous and heterogeneous media
//! - Collocation point generation
//! - Boundary condition enforcement (future)
//! - Multi-region domains with interfaces (future)

use super::*;
use burn::backend::{Autodiff, NdArray};

use crate::core::error::{KwaversError, KwaversResult, SystemError};

type TestBackend = Autodiff<NdArray>;

#[test]
fn test_end_to_end_rectangular_domain() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16, 16],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Synthetic training data: simple pattern
    let x_data = vec![0.5, 0.6, 0.7];
    let y_data = vec![0.5, 0.5, 0.5];
    let z_data = vec![0.5, 0.5, 0.5];
    let t_data = vec![0.1, 0.2, 0.3];
    let u_data = vec![0.0, 0.1, 0.0];

    // Train
    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 10)?;
    assert_eq!(metrics.epochs_completed, 10);
    assert!(metrics.training_time_secs > 0.0);

    // Predict
    let x_test = vec![0.5, 0.6];
    let y_test = vec![0.5, 0.5];
    let z_test = vec![0.5, 0.5];
    let t_test = vec![0.15, 0.25];

    let u_pred = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
    assert_eq!(u_pred.len(), 2);
    assert!(u_pred.iter().all(|&p| p.is_finite()));
    Ok(())
}

#[test]
fn test_end_to_end_spherical_domain() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 50, // More points needed due to filtering
        learning_rate: 1e-2,
        ..Default::default()
    };

    // Spherical domain centered at (0.5, 0.5, 0.5) with radius 0.3
    let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Training data inside sphere
    let x_data = vec![0.5, 0.6];
    let y_data = vec![0.5, 0.5];
    let z_data = vec![0.5, 0.5];
    let t_data = vec![0.1, 0.2];
    let u_data = vec![0.0, 0.0];

    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 5)?;
    assert_eq!(metrics.epochs_completed, 5);
    Ok(())
}

#[test]
fn test_end_to_end_cylindrical_domain() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 30,
        learning_rate: 1e-2,
        ..Default::default()
    };

    // Cylindrical domain: center (0.5, 0.5), z in [0, 1], radius 0.3
    let geometry = Geometry3D::cylindrical(0.5, 0.5, 0.0, 1.0, 0.3);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Training data inside cylinder
    let x_data = vec![0.5, 0.6];
    let y_data = vec![0.5, 0.5];
    let z_data = vec![0.5, 0.5];
    let t_data = vec![0.1, 0.2];
    let u_data = vec![0.0, 0.0];

    solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 5)?;
    Ok(())
}

#[test]
fn test_heterogeneous_layered_medium() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Layered medium: two materials separated at z=0.5
    let wave_speed = |_x: f32, _y: f32, z: f32| if z < 0.5 { 1500.0 } else { 3000.0 };

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Verify wave speeds at different depths
    assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.3)?, 1500.0);
    assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);

    // Training data across both layers
    let x_data = vec![0.5, 0.5, 0.5, 0.5];
    let y_data = vec![0.5, 0.5, 0.5, 0.5];
    let z_data = vec![0.3, 0.4, 0.6, 0.7]; // Two in each layer
    let t_data = vec![0.1, 0.2, 0.1, 0.2];
    let u_data = vec![0.0, 0.0, 0.0, 0.0];

    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 10)?;
    assert_eq!(metrics.epochs_completed, 10);
    Ok(())
}

#[test]
fn test_radially_varying_medium() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Radially varying speed: fast in center, slow outside
    let wave_speed = |x: f32, y: f32, z: f32| {
        let dx = x - 0.5;
        let dy = y - 0.5;
        let dz = z - 0.5;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 0.3 {
            2500.0 // Fast center
        } else {
            1500.0 // Slow periphery
        }
    };

    let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Verify radial variation
    assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.5)?, 2500.0); // Center
    assert_eq!(solver.get_wave_speed(0.9, 0.5, 0.5)?, 1500.0); // Periphery
    Ok(())
}

#[test]
fn test_collocation_points_rectangular() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 2.0, 0.0, 3.0, 0.0, 4.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

    let (x_colloc, y_colloc, z_colloc, t_colloc) =
        solver.generate_collocation_points(&config, &device);

    // All points should be generated for rectangular domain
    let n_generated = x_colloc.dims()[0];
    assert_eq!(n_generated, config.num_collocation_points);

    // Verify shapes match
    assert_eq!(y_colloc.dims()[0], n_generated);
    assert_eq!(z_colloc.dims()[0], n_generated);
    assert_eq!(t_colloc.dims()[0], n_generated);

    // Verify bounds (sample check via data extraction)
    let x_colloc_data = x_colloc.into_data();
    let x_data = x_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let y_colloc_data = y_colloc.into_data();
    let y_data = y_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let z_colloc_data = z_colloc.into_data();
    let z_data = z_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let t_colloc_data = t_colloc.into_data();
    let t_data = t_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;

    assert!(x_data.iter().all(|&x| (0.0..=2.0).contains(&x)));
    assert!(y_data.iter().all(|&y| (0.0..=3.0).contains(&y)));
    assert!(z_data.iter().all(|&z| (0.0..=4.0).contains(&z)));
    assert!(t_data.iter().all(|&t| (0.0..=1.0).contains(&t)));
    Ok(())
}

#[test]
fn test_collocation_points_spherical_filtering() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.2);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

    let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
        solver.generate_collocation_points(&config, &device);

    // Spherical geometry should filter points (sphere volume << bounding box)
    let n_generated = x_colloc.dims()[0];
    assert!(n_generated > 0);
    assert!(n_generated < config.num_collocation_points);
    Ok(())
}

#[test]
fn test_training_metrics_completeness() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 10,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    let x_data = vec![0.5];
    let y_data = vec![0.5];
    let z_data = vec![0.5];
    let t_data = vec![0.1];
    let u_data = vec![0.0];

    let epochs = 5;
    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, epochs)?;

    // Verify metric completeness
    assert_eq!(metrics.epochs_completed, epochs);
    assert_eq!(metrics.total_loss.len(), epochs);
    assert_eq!(metrics.data_loss.len(), epochs);
    assert_eq!(metrics.pde_loss.len(), epochs);
    assert_eq!(metrics.bc_loss.len(), epochs);
    assert_eq!(metrics.ic_loss.len(), epochs);
    assert!(metrics.training_time_secs > 0.0);

    // All losses should be finite
    assert!(metrics.total_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.data_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.pde_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.bc_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.ic_loss.iter().all(|&l| l.is_finite()));
    Ok(())
}

#[test]
fn test_prediction_shape_consistency() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    // Test with different batch sizes
    for n in [1, 5, 10, 50] {
        let x_test = vec![0.5; n];
        let y_test = vec![0.5; n];
        let z_test = vec![0.5; n];
        let t_test = vec![0.1; n];

        let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
        assert_eq!(predictions.len(), n);
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }
    Ok(())
}

#[test]
fn test_boundary_condition_types() {
    // Verify boundary condition enum variants
    let dirichlet = BoundaryCondition3D::Dirichlet;
    let neumann = BoundaryCondition3D::Neumann;
    let absorbing = BoundaryCondition3D::Absorbing;
    let periodic = BoundaryCondition3D::Periodic;

    // Ensure all variants are distinct
    assert!(matches!(dirichlet, BoundaryCondition3D::Dirichlet));
    assert!(matches!(neumann, BoundaryCondition3D::Neumann));
    assert!(matches!(absorbing, BoundaryCondition3D::Absorbing));
    assert!(matches!(periodic, BoundaryCondition3D::Periodic));
}

#[test]
fn test_geometry_bounding_box_variants() {
    // Rectangular
    let rect = Geometry3D::rectangular(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = rect.bounding_box();
    assert_eq!(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    );

    // Spherical
    let sphere = Geometry3D::spherical(1.0, 2.0, 3.0, 0.5);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = sphere.bounding_box();
    assert!((x_min - 0.5).abs() < 1e-6);
    assert!((x_max - 1.5).abs() < 1e-6);
    assert!((y_min - 1.5).abs() < 1e-6);
    assert!((y_max - 2.5).abs() < 1e-6);
    assert!((z_min - 2.5).abs() < 1e-6);
    assert!((z_max - 3.5).abs() < 1e-6);

    // Cylindrical
    let cylinder = Geometry3D::cylindrical(1.0, 2.0, 0.0, 5.0, 0.5);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = cylinder.bounding_box();
    assert!((x_min - 0.5).abs() < 1e-6);
    assert!((x_max - 1.5).abs() < 1e-6);
    assert!((y_min - 1.5).abs() < 1e-6);
    assert!((y_max - 2.5).abs() < 1e-6);
    assert!((z_min - 0.0).abs() < 1e-6);
    assert!((z_max - 5.0).abs() < 1e-6);
}

#[test]
fn test_geometry_contains_variants() {
    // Rectangular
    let rect = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    assert!(rect.contains(0.5, 0.5, 0.5));
    assert!(!rect.contains(1.5, 0.5, 0.5));

    // Spherical
    let sphere = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    assert!(sphere.contains(0.5, 0.5, 0.5)); // Center
    assert!(!sphere.contains(1.0, 0.5, 0.5)); // Outside

    // Cylindrical
    let cylinder = Geometry3D::cylindrical(0.5, 0.5, 0.0, 1.0, 0.3);
    assert!(cylinder.contains(0.5, 0.5, 0.5)); // Center axis
    assert!(!cylinder.contains(1.0, 0.5, 0.5)); // Outside radius
    assert!(!cylinder.contains(0.5, 0.5, 1.5)); // Outside z-bounds
}

#[test]
fn test_config_defaults() {
    let config = BurnPINN3DConfig::default();

    assert_eq!(config.hidden_layers, vec![100, 100, 100]);
    assert_eq!(config.num_collocation_points, 10000);
    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.batch_size, 1000);
    assert_eq!(config.max_grad_norm, 1.0);

    let weights = config.loss_weights;
    assert_eq!(weights.data_weight, 1.0);
    assert_eq!(weights.pde_weight, 1.0);
    assert_eq!(weights.bc_weight, 1.0);
    assert_eq!(weights.ic_weight, 1.0);
}
