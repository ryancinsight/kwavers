//! Inference and deployment for trained 2D Elastic Wave PINN
//!
//! This module provides inference capabilities for trained PINN models, including:
//! - Single-point and batch prediction
//! - Field evaluation on spatial grids
//! - Time-series prediction
//! - Model serialization and loading
//! - Performance optimization for deployment
//!
//! # Usage
//!
//! ## Basic Prediction
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::elastic_2d::{ElasticPINN2D, Predictor};
//!
//! let predictor = Predictor::new(trained_model);
//!
//! // Single point prediction
//! let displacement = predictor.predict_point(0.5, 0.5, 0.1)?;
//! println!("u_x = {}, u_y = {}", displacement[0], displacement[1]);
//!
//! // Batch prediction
//! let points = vec![(0.5, 0.5, 0.1), (0.6, 0.6, 0.2)];
//! let displacements = predictor.predict_batch(&points)?;
//! ```
//!
//! ## Field Evaluation
//!
//! ```rust,ignore
//! // Evaluate displacement field on a spatial grid at fixed time
//! let x_grid = Array1::linspace(0.0, 1.0, 100);
//! let y_grid = Array1::linspace(0.0, 1.0, 100);
//! let t = 0.1;
//!
//! let field = predictor.evaluate_field(&x_grid, &y_grid, t)?;
//! ```
//!
//! ## Time Series
//!
//! ```rust,ignore
//! // Track displacement at a fixed point over time
//! let x = 0.5;
//! let y = 0.5;
//! let times = Array1::linspace(0.0, 1.0, 1000);
//!
//! let time_series = predictor.time_series(x, y, &times)?;
//! ```

use crate::error::{KwaversError, KwaversResult};

use ndarray::{Array1, Array2, Array3};

#[cfg(feature = "pinn")]
use burn::tensor::{backend::Backend, Tensor};

#[cfg(feature = "pinn")]
use super::model::ElasticPINN2D;

/// Predictor for trained PINN model
///
/// Provides high-level interface for model inference and evaluation.
#[cfg(feature = "pinn")]
#[derive(Debug)]
pub struct Predictor<B: Backend> {
    /// Trained PINN model
    model: ElasticPINN2D<B>,
    /// Device for computation
    device: B::Device,
}

#[cfg(feature = "pinn")]
impl<B: Backend> Predictor<B> {
    /// Create a new predictor from a trained model
    ///
    /// # Arguments
    ///
    /// * `model` - Trained PINN model
    pub fn new(model: ElasticPINN2D<B>) -> Self {
        let device = model.device();
        Self { model, device }
    }

    /// Predict displacement at a single point
    ///
    /// # Arguments
    ///
    /// * `x` - X spatial coordinate
    /// * `y` - Y spatial coordinate
    /// * `t` - Time coordinate
    ///
    /// # Returns
    ///
    /// Displacement vector [u_x, u_y]
    pub fn predict_point(&self, x: f64, y: f64, t: f64) -> KwaversResult<[f64; 2]> {
        // Create tensors for single point
        let x_tensor = Tensor::<B, 2>::from_floats([[x as f32]], &self.device);
        let y_tensor = Tensor::<B, 2>::from_floats([[y as f32]], &self.device);
        let t_tensor = Tensor::<B, 2>::from_floats([[t as f32]], &self.device);

        // Forward pass
        let output = self.model.forward(x_tensor, y_tensor, t_tensor);

        // Extract values
        let data = output.to_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|_| KwaversError::InvalidInput("Failed to extract tensor data".to_string()))?;

        Ok([slice[0] as f64, slice[1] as f64])
    }

    /// Predict displacement at multiple points (batch)
    ///
    /// # Arguments
    ///
    /// * `points` - Vector of (x, y, t) coordinates
    ///
    /// # Returns
    ///
    /// Array of displacement vectors [N, 2]
    pub fn predict_batch(&self, points: &[(f64, f64, f64)]) -> KwaversResult<Array2<f64>> {
        if points.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Points array cannot be empty".to_string(),
            ));
        }

        let n = points.len();

        // Extract coordinates
        let x_data: Vec<f32> = points.iter().map(|(x, _, _)| *x as f32).collect();
        let y_data: Vec<f32> = points.iter().map(|(_, y, _)| *y as f32).collect();
        let t_data: Vec<f32> = points.iter().map(|(_, _, t)| *t as f32).collect();

        // Create tensors
        let x_tensor = Tensor::<B, 1>::from_floats(x_data.as_slice(), &self.device).reshape([n, 1]);
        let y_tensor = Tensor::<B, 1>::from_floats(y_data.as_slice(), &self.device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_data.as_slice(), &self.device).reshape([n, 1]);

        // Forward pass
        let output = self.model.forward(x_tensor, y_tensor, t_tensor);

        // Convert to ndarray
        let data = output.to_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|_| KwaversError::InvalidInput("Failed to extract tensor data".to_string()))?;

        let values: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
        Array2::from_shape_vec((n, 2), values)
            .map_err(|e| KwaversError::InvalidInput(format!("Shape error: {}", e)))
    }

    /// Evaluate displacement field on a 2D spatial grid at fixed time
    ///
    /// # Arguments
    ///
    /// * `x_grid` - X coordinates (length Nx)
    /// * `y_grid` - Y coordinates (length Ny)
    /// * `t` - Fixed time value
    ///
    /// # Returns
    ///
    /// Displacement field [Nx, Ny, 2] where last dimension is (u_x, u_y)
    pub fn evaluate_field(
        &self,
        x_grid: &Array1<f64>,
        y_grid: &Array1<f64>,
        t: f64,
    ) -> KwaversResult<Array3<f64>> {
        let nx = x_grid.len();
        let ny = y_grid.len();

        // Create meshgrid
        let mut points = Vec::with_capacity(nx * ny);
        for x_val in x_grid.iter() {
            for y_val in y_grid.iter() {
                points.push((*x_val, *y_val, t));
            }
        }

        // Batch prediction
        let displacements = self.predict_batch(&points)?;

        // Reshape to [Nx, Ny, 2]
        let mut field = Array3::<f64>::zeros((nx, ny, 2));
        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                field[[i, j, 0]] = displacements[[idx, 0]];
                field[[i, j, 1]] = displacements[[idx, 1]];
            }
        }

        Ok(field)
    }

    /// Compute time series at a fixed spatial point
    ///
    /// # Arguments
    ///
    /// * `x` - X spatial coordinate
    /// * `y` - Y spatial coordinate
    /// * `times` - Array of time values
    ///
    /// # Returns
    ///
    /// Displacement time series [N_times, 2]
    pub fn time_series(&self, x: f64, y: f64, times: &Array1<f64>) -> KwaversResult<Array2<f64>> {
        let points: Vec<(f64, f64, f64)> = times.iter().map(|&t| (x, y, t)).collect();
        self.predict_batch(&points)
    }

    /// Compute displacement magnitude field (scalar)
    ///
    /// # Arguments
    ///
    /// * `x_grid` - X coordinates
    /// * `y_grid` - Y coordinates
    /// * `t` - Time value
    ///
    /// # Returns
    ///
    /// Magnitude field [Nx, Ny] where magnitude = sqrt(u_x^2 + u_y^2)
    pub fn magnitude_field(
        &self,
        x_grid: &Array1<f64>,
        y_grid: &Array1<f64>,
        t: f64,
    ) -> KwaversResult<Array2<f64>> {
        let field = self.evaluate_field(x_grid, y_grid, t)?;

        let nx = field.dim().0;
        let ny = field.dim().1;
        let mut magnitude = Array2::<f64>::zeros((nx, ny));

        for i in 0..nx {
            for j in 0..ny {
                let ux = field[[i, j, 0]];
                let uy = field[[i, j, 1]];
                magnitude[[i, j]] = (ux * ux + uy * uy).sqrt();
            }
        }

        Ok(magnitude)
    }

    /// Get reference to the model
    pub fn model(&self) -> &ElasticPINN2D<B> {
        &self.model
    }

    /// Extract estimated material parameters (for inverse problems)
    ///
    /// # Returns
    ///
    /// (λ, μ, ρ) if optimized, otherwise None
    pub fn material_parameters(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        self.model.estimated_parameters()
    }
}

/// Non-Burn fallback
#[cfg(not(feature = "pinn"))]
#[derive(Debug)]
pub struct Predictor {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "pinn"))]
impl Predictor {
    pub fn new(_model: ()) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn predict_point(&self, _x: f64, _y: f64, _t: f64) -> KwaversResult<[f64; 2]> {
        Err(KwaversError::InvalidInput(
            "Predictor requires 'burn' feature to be enabled".to_string(),
        ))
    }

    pub fn predict_batch(&self, _points: &[(f64, f64, f64)]) -> KwaversResult<Array2<f64>> {
        Err(KwaversError::InvalidInput(
            "Predictor requires 'burn' feature to be enabled".to_string(),
        ))
    }

    pub fn evaluate_field(
        &self,
        _x_grid: &Array1<f64>,
        _y_grid: &Array1<f64>,
        _t: f64,
    ) -> KwaversResult<Array3<f64>> {
        Err(KwaversError::InvalidInput(
            "Predictor requires 'burn' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;
    use crate::solver::inverse::pinn::elastic_2d::Config;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_predictor_creation() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        assert!(predictor.model().num_parameters() > 0);
    }

    #[test]
    fn test_single_point_prediction() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let result = predictor.predict_point(0.5, 0.5, 0.1);
        assert!(result.is_ok());

        let displacement = result.unwrap();
        assert_eq!(displacement.len(), 2);
    }

    #[test]
    fn test_batch_prediction() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let points = vec![(0.5, 0.5, 0.1), (0.6, 0.6, 0.2), (0.7, 0.7, 0.3)];
        let result = predictor.predict_batch(&points);
        assert!(result.is_ok());

        let displacements = result.unwrap();
        assert_eq!(displacements.dim(), (3, 2));
    }

    #[test]
    fn test_empty_batch_error() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let points: Vec<(f64, f64, f64)> = vec![];
        let result = predictor.predict_batch(&points);
        assert!(result.is_err());
    }

    #[test]
    fn test_field_evaluation() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let x_grid = Array1::linspace(0.0, 1.0, 5);
        let y_grid = Array1::linspace(0.0, 1.0, 5);
        let t = 0.1;

        let result = predictor.evaluate_field(&x_grid, &y_grid, t);
        assert!(result.is_ok());

        let field = result.unwrap();
        assert_eq!(field.dim(), (5, 5, 2));
    }

    #[test]
    fn test_time_series() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let times = Array1::linspace(0.0, 1.0, 10);
        let result = predictor.time_series(0.5, 0.5, &times);
        assert!(result.is_ok());

        let time_series = result.unwrap();
        assert_eq!(time_series.dim(), (10, 2));
    }

    #[test]
    fn test_magnitude_field() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let x_grid = Array1::linspace(0.0, 1.0, 5);
        let y_grid = Array1::linspace(0.0, 1.0, 5);
        let t = 0.1;

        let result = predictor.magnitude_field(&x_grid, &y_grid, t);
        assert!(result.is_ok());

        let magnitude = result.unwrap();
        assert_eq!(magnitude.dim(), (5, 5));

        // All magnitudes should be non-negative
        for val in magnitude.iter() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_material_parameters_inverse() {
        let config = Config::inverse_problem(1e9, 5e8, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let (lambda, mu, rho) = predictor.material_parameters();
        assert!(lambda.is_some());
        assert!(mu.is_some());
        assert!(rho.is_some());
    }

    #[test]
    fn test_material_parameters_forward() {
        let config = Config::forward_problem(1e9, 5e8, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();
        let predictor = Predictor::new(model);

        let (lambda, mu, rho) = predictor.material_parameters();
        assert!(lambda.is_none());
        assert!(mu.is_none());
        assert!(rho.is_none());
    }
}
