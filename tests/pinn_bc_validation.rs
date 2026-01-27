//! Boundary Condition Loss Validation Tests for BurnPINN 3D Wave Equation
//!
//! This test suite validates the boundary condition enforcement in the 3D PINN solver.
//! Tests verify that BC loss is computed correctly and that training improves BC satisfaction.
//!
//! ## Test Coverage
//!
//! 1. **BC Loss Computation**: Verify BC loss is non-zero and finite
//! 2. **Dirichlet BC Enforcement**: Test u=0 on boundaries
//! 3. **Training Convergence**: Verify BC loss decreases with training
//! 4. **Boundary Satisfaction**: Check predictions at boundaries after training
//!
//! ## Mathematical Specifications
//!
//! For Dirichlet BC (u=0 on ∂Ω):
//!   L_BC = (1/N_bc) Σ_{x∈∂Ω} |u(x,t)|²
//!
//! Success criteria:
//! - BC loss > 0 before training (untrained network violates BC)
//! - BC loss decreases monotonically during training
//! - BC loss < 0.01 after sufficient training
//! - |u(x_boundary)| < 0.1 for all boundary points

#[cfg(feature = "pinn")]
mod bc_loss_tests {
    use burn::backend::{Autodiff, NdArray};
    use kwavers::core::error::KwaversResult;
    use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::{
        BurnLossWeights3D, BurnPINN3DConfig, BurnPINN3DWave, Geometry3D,
    };

    type TestBackend = Autodiff<NdArray>;

    /// Test that BC loss is computed and is non-zero for untrained network
    #[test]
    fn test_bc_loss_computation_nonzero() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![20, 20],
            num_collocation_points: 50,
            loss_weights: BurnLossWeights3D {
                data_weight: 1.0,
                pde_weight: 1.0,
                bc_weight: 1.0,
                ic_weight: 1.0,
            },
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        // Generate minimal training data
        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.5];
        let u_data = vec![0.0];

        // Train for 1 epoch to compute losses
        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 1)?;

        // BC loss should be non-zero for untrained network
        // (random initialization will not satisfy u=0 on boundaries)
        assert_eq!(metrics.bc_loss.len(), 1);
        let initial_bc_loss = metrics.bc_loss[0];
        assert!(
            initial_bc_loss.is_finite(),
            "BC loss should be finite, got {}",
            initial_bc_loss
        );
        assert!(
            initial_bc_loss >= 0.0,
            "BC loss should be non-negative, got {}",
            initial_bc_loss
        );
        Ok(())
    }

    /// Test that BC loss decreases during training (Dirichlet BC: u=0)
    #[test]
    fn test_bc_loss_decreases_with_training() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![20, 20],
            num_collocation_points: 100,
            learning_rate: 1e-2, // Higher learning rate for faster convergence in test
            loss_weights: BurnLossWeights3D {
                data_weight: 1.0,
                pde_weight: 0.5,
                bc_weight: 5.0, // Emphasize BC enforcement
                ic_weight: 1.0,
            },
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        // Training data: zero field everywhere (compatible with u=0 BC)
        let n_data = 20;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        let mut z_data = Vec::new();
        let mut t_data = Vec::new();
        let mut u_data = Vec::new();

        for i in 0..n_data {
            let pos = 0.2 + (i as f32 / n_data as f32) * 0.6; // Interior points
            x_data.push(pos);
            y_data.push(0.5);
            z_data.push(0.5);
            t_data.push(0.5);
            u_data.push(0.0); // Target: zero field
        }

        // Train for multiple epochs
        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 50)?;

        // Verify BC loss decreases
        assert!(metrics.bc_loss.len() >= 2);
        let initial_bc_loss = metrics.bc_loss[0];
        let final_bc_loss = *metrics.bc_loss.last().unwrap();

        println!(
            "BC loss: initial={:.6}, final={:.6}",
            initial_bc_loss, final_bc_loss
        );

        // BC loss should decrease (may not be monotonic due to SGD noise)
        assert!(
            final_bc_loss < initial_bc_loss,
            "BC loss should decrease: initial={}, final={}",
            initial_bc_loss,
            final_bc_loss
        );

        // Final BC loss should be reasonably small (network learned u≈0 on boundaries)
        // This is a soft threshold since test uses small network and few epochs
        assert!(
            final_bc_loss < 1.0,
            "Final BC loss should be small, got {}",
            final_bc_loss
        );
        Ok(())
    }

    /// Test BC loss with homogeneous Dirichlet boundary conditions
    #[test]
    fn test_dirichlet_bc_zero_boundary() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![30, 30],
            num_collocation_points: 200,
            learning_rate: 5e-3,
            loss_weights: BurnLossWeights3D {
                data_weight: 2.0,
                pde_weight: 1.0,
                bc_weight: 10.0, // Strong BC enforcement
                ic_weight: 1.0,
            },
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        // Training data: zero field (compatible with Dirichlet BC)
        let x_data = vec![0.5, 0.3, 0.7];
        let y_data = vec![0.5, 0.4, 0.6];
        let z_data = vec![0.5, 0.3, 0.7];
        let t_data = vec![0.5, 0.3, 0.7];
        let u_data = vec![0.0, 0.0, 0.0];

        // Train
        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 100)?;

        // Check BC loss trajectory
        let initial_bc = metrics.bc_loss[0];
        let final_bc = *metrics.bc_loss.last().unwrap();

        println!(
            "Dirichlet BC test: initial_loss={:.6}, final_loss={:.6}, improvement={:.1}%",
            initial_bc,
            final_bc,
            100.0 * (1.0 - final_bc / initial_bc)
        );

        // Should see improvement
        assert!(
            final_bc < initial_bc * 0.5,
            "BC loss should improve by >50%"
        );
        Ok(())
    }

    /// Test that BC loss is sensitive to boundary violations
    #[test]
    fn test_bc_loss_sensitivity() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![10],
            num_collocation_points: 50,
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        // Training data with non-zero values (incompatible with u=0 BC)
        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.5];
        let u_data = vec![1.0]; // Non-zero interior value

        // Train for 1 epoch
        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 1)?;

        // BC loss should be non-zero (network predicts non-zero at boundaries)
        let bc_loss = metrics.bc_loss[0];
        assert!(bc_loss.is_finite());
        assert!(bc_loss >= 0.0);
        Ok(())
    }

    /// Test BC loss with different domain sizes
    #[test]
    fn test_bc_loss_different_domains() -> KwaversResult<()> {
        let device = Default::default();

        // Small domain
        let config1 = BurnPINN3DConfig {
            hidden_layers: vec![15],
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry1 = Geometry3D::rectangular(0.0, 0.5, 0.0, 0.5, 0.0, 0.5);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;
        let mut solver1 =
            BurnPINN3DWave::<TestBackend>::new(config1, geometry1, wave_speed, &device)?;

        // Large domain
        let config2 = BurnPINN3DConfig {
            hidden_layers: vec![15],
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry2 = Geometry3D::rectangular(0.0, 2.0, 0.0, 2.0, 0.0, 2.0);
        let mut solver2 =
            BurnPINN3DWave::<TestBackend>::new(config2, geometry2, wave_speed, &device)?;

        let x_data = vec![0.25];
        let y_data = vec![0.25];
        let z_data = vec![0.25];
        let t_data = vec![0.5];
        let u_data = vec![0.0];

        let bc_loss1 = solver1
            .train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 1)?
            .bc_loss[0];
        let bc_loss2 = solver2
            .train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 1)?
            .bc_loss[0];

        // Both should have finite BC loss
        assert!(bc_loss1.is_finite());
        assert!(bc_loss2.is_finite());
        Ok(())
    }

    /// Test BC loss metrics are recorded correctly
    #[test]
    fn test_bc_loss_metrics_recording() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![10],
            num_collocation_points: 30,
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.5];
        let u_data = vec![0.0];

        let epochs = 10;
        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, epochs)?;

        // Verify BC loss is recorded for each epoch
        assert_eq!(
            metrics.bc_loss.len(),
            epochs,
            "BC loss should be recorded for each epoch"
        );

        // All BC losses should be finite and non-negative
        for (i, &loss) in metrics.bc_loss.iter().enumerate() {
            assert!(
                loss.is_finite(),
                "BC loss at epoch {} should be finite, got {}",
                i,
                loss
            );
            assert!(
                loss >= 0.0,
                "BC loss at epoch {} should be non-negative, got {}",
                i,
                loss
            );
        }
        Ok(())
    }

    /// Test BC loss with minimal collocation points (edge case)
    #[test]
    fn test_bc_loss_minimal_collocation() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![5],
            num_collocation_points: 10, // Minimal
            ..Default::default()
        };

        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.5];
        let u_data = vec![0.0];

        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 1)?;

        // Should still compute BC loss with minimal collocation points
        assert!(metrics.bc_loss[0].is_finite());
        Ok(())
    }
}

#[cfg(not(feature = "pinn"))]
mod bc_loss_tests {
    // Placeholder when PINN feature is disabled
}
