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
pub use geometry::{Geometry2D, InterfaceCondition};
pub use inference::{ActivationType, QuantizedNetwork, RealTimePINNInference};
pub use model::{BurnPINN2DWave, WaveSpeedFn};
pub use optimizer::SimpleOptimizer2D;
pub use trainer::BurnPINN2DTrainer;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use std::f64::consts::PI;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_geometry_rectangular() {
        let geom = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
        assert!(geom.contains(0.5, 0.5));
        assert!(!geom.contains(1.5, 0.5));
        assert!(!geom.contains(0.5, 1.5));
    }

    #[test]
    fn test_geometry_circular() {
        let geom = Geometry2D::circular(0.0, 0.0, 1.0);
        assert!(geom.contains(0.5, 0.5));
        assert!(!geom.contains(1.5, 0.0));
        assert!(geom.contains(0.0, 0.0));
    }

    #[test]
    fn test_geometry_polygonal() {
        let vertices = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let geom = Geometry2D::polygonal(vertices, vec![]);

        assert!(geom.contains(0.5, 0.3));
        assert!(!geom.contains(0.0, 0.8));
        assert!(geom.contains(0.75, 0.0));
    }

    #[test]
    fn test_geometry_parametric_curve() {
        let x_func: Box<dyn Fn(f64) -> f64 + Send + Sync> = Box::new(|t: f64| t.cos());
        let y_func: Box<dyn Fn(f64) -> f64 + Send + Sync> = Box::new(|t: f64| t.sin());

        let geom =
            Geometry2D::parametric_curve(x_func, y_func, 0.0, 2.0 * PI, (-1.1, 1.1, -1.1, 1.1));

        assert!(geom.contains(1.0, 0.0));
        assert!(!geom.contains(2.0, 2.0));
    }

    #[test]
    fn test_geometry_multi_region() {
        let rect1 = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
        let rect2 = Geometry2D::rectangular(1.0, 2.0, 0.0, 1.0);

        let regions = vec![(rect1, 0), (rect2, 1)];
        let geom = Geometry2D::multi_region(regions, vec![]);

        assert!(geom.contains(0.5, 0.5));
        assert!(geom.contains(1.5, 0.5));
        assert!(!geom.contains(2.5, 0.5));
    }

    #[test]
    fn test_burn_pinn_2d_creation() {
        let device = Default::default();
        let config = BurnPINN2DConfig::default();
        let result = BurnPINN2DWave::<TestBackend>::new(config, &device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_burn_pinn_2d_invalid_config() {
        let device = Default::default();
        let config = BurnPINN2DConfig {
            hidden_layers: vec![],
            ..Default::default()
        };
        let result = BurnPINN2DWave::<TestBackend>::new(config, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_burn_pinn_2d_forward_pass() {
        let device = Default::default();
        let config = BurnPINN2DConfig::default();
        let pinn = BurnPINN2DWave::<TestBackend>::new(config, &device).unwrap();

        // Create dummy input [batch_size=2, input_dim=3]
        // But the forward method takes x, y, t separately as [batch, 1]
        let batch_size = 2;
        let x = burn::tensor::Tensor::<TestBackend, 1>::from_floats([0.5, 0.6], &device)
            .reshape([batch_size, 1]);
        let y = burn::tensor::Tensor::<TestBackend, 1>::from_floats([0.5, 0.6], &device)
            .reshape([batch_size, 1]);
        let t = burn::tensor::Tensor::<TestBackend, 1>::from_floats([0.0, 0.1], &device)
            .reshape([batch_size, 1]);

        let output = pinn.forward(x, y, t);
        assert_eq!(output.shape().dims, [2, 1]);
    }
}
