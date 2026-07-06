//! Burn-based 2D Wave Equation Physics-Informed Neural Network with Automatic Differentiation
//!
//! This module implements a PINN for the 2D acoustic wave equation using the Burn deep learning
//! framework with native automatic differentiation. This extends the 1D implementation to handle
//! two spatial dimensions with complex geometries and boundary conditions.
//!
//! ## Modules
//!
//! - `config`: Configuration structs for PINN architecture and training.
//! - `geometry`: 2D geometry definitions and boundary condition types.
//! - `model`: The Neural Network model itself (`BurnPINN2DWave`).
//! - `optimizer`: Simple gradient descent optimizer.
//! - `trainer`: Training loop and physics loss calculation.
//! - `inference`: Real-time inference engine with quantization and SIMD/GPU support.

pub mod config;
pub mod geometry;
pub mod inference;
pub mod model;
pub mod optimizer;
pub mod trainer;

// Re-export main types for convenience
pub use config::{BoundaryCondition2D, BurnLossWeights2D, BurnPINN2DConfig, BurnTrainingMetrics2D};
pub use geometry::{BurnWave2dGeometry, BurnWave2dInterfaceCondition};
pub use inference::{ActivationType, QuantizedNetwork, RealTimePINNInference};
pub use model::{BurnPINN2DWave, WaveSpeedFn};
pub use optimizer::SimpleOptimizer2D;
pub use trainer::BurnPINN2DTrainer;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    type TestBackend = coeus_core::MoiraiBackend;

    #[test]
    fn test_geometry_rectangular() {
        let geom = BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);
        assert!(geom.contains(0.5, 0.5));
        assert!(!geom.contains(1.5, 0.5));
        assert!(!geom.contains(0.5, 1.5));
    }

    #[test]
    fn test_geometry_circular() {
        let geom = BurnWave2dGeometry::circular(0.0, 0.0, 1.0);
        assert!(geom.contains(0.5, 0.5));
        assert!(!geom.contains(1.5, 0.0));
        assert!(geom.contains(0.0, 0.0));
    }

    #[test]
    fn test_geometry_polygonal() {
        let vertices = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let geom = BurnWave2dGeometry::polygonal(vertices, vec![]);

        assert!(geom.contains(0.5, 0.3));
        assert!(!geom.contains(0.0, 0.8));
        assert!(geom.contains(0.75, 0.0));
    }

    #[test]
    fn test_geometry_parametric_curve() {
        let x_func: Box<dyn Fn(f64) -> f64 + Send + Sync> = Box::new(|t: f64| t.cos());
        let y_func: Box<dyn Fn(f64) -> f64 + Send + Sync> = Box::new(|t: f64| t.sin());

        let geom = BurnWave2dGeometry::parametric_curve(
            x_func,
            y_func,
            0.0,
            2.0 * PI,
            (-1.1, 1.1, -1.1, 1.1),
        );

        assert!(geom.contains(1.0, 0.0));
        assert!(!geom.contains(2.0, 2.0));
    }

    #[test]
    fn test_geometry_multi_region() {
        let rect1 = BurnWave2dGeometry::rectangular(0.0, 1.0, 0.0, 1.0);
        let rect2 = BurnWave2dGeometry::rectangular(1.0, 2.0, 0.0, 1.0);

        let regions = vec![(rect1, 0), (rect2, 1)];
        let geom = BurnWave2dGeometry::multi_region(regions, vec![]);

        assert!(geom.contains(0.5, 0.5));
        assert!(geom.contains(1.5, 0.5));
        assert!(!geom.contains(2.5, 0.5));
    }

    #[test]
    fn test_burn_pinn_2d_creation() {
        let config = BurnPINN2DConfig::default();
        let result = BurnPINN2DWave::<TestBackend>::new(config);
        let _wave = result.unwrap();
    }

    #[test]
    fn test_burn_pinn_2d_invalid_config() {
        let config = BurnPINN2DConfig {
            hidden_layers: vec![],
            ..Default::default()
        };
        let result = BurnPINN2DWave::<TestBackend>::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_burn_pinn_2d_forward_pass() {
        let config = BurnPINN2DConfig::default();
        let pinn = BurnPINN2DWave::<TestBackend>::new(config).unwrap();

        // Create dummy input [batch_size=2, input_dim=3]
        // But the forward method takes x, y, t separately as [batch, 1]
        let backend = TestBackend::default();
        let batch_size = 2;
        let x = coeus_autograd::Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &[0.5_f32, 0.6], &backend),
            false,
        );
        let y = coeus_autograd::Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &[0.5_f32, 0.6], &backend),
            false,
        );
        let t = coeus_autograd::Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &[0.0_f32, 0.1], &backend),
            false,
        );

        let output = pinn.forward(&x, &y, &t);
        assert_eq!(output.tensor.shape(), &[2, 1]);
    }
}
