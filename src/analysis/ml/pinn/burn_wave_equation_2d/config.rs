use burn::config::Config;

/// Boundary condition types for 2D domains
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BoundaryCondition2D {
    /// Dirichlet: u = 0 on boundary (sound-hard)
    Dirichlet,
    /// Neumann: ∂u/∂n = 0 on boundary (sound-soft)
    Neumann,
    /// Periodic boundary conditions
    Periodic,
    /// Absorbing boundary conditions
    Absorbing,
}

/// Configuration for Burn-based 2D Wave Equation PINN
#[derive(Debug, Config)]
pub struct BurnPINN2DConfig {
    /// Hidden layer sizes (e.g., [100, 100, 100, 100])
    #[config(default = "vec![100, 100, 100, 100]")]
    pub hidden_layers: Vec<usize>,
    /// Learning rate for optimizer
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Loss function weights
    #[config(default = "BurnLossWeights2D::default()")]
    pub loss_weights: BurnLossWeights2D,
    /// Number of collocation points for PDE residual
    #[config(default = 1000)]
    pub num_collocation_points: usize,
    /// Boundary condition type
    #[config(default = "BoundaryCondition2D::Dirichlet")]
    pub boundary_condition: BoundaryCondition2D,
}

impl Default for BurnPINN2DConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![100, 100, 100, 100],
            learning_rate: 1e-3,
            loss_weights: BurnLossWeights2D::default(),
            num_collocation_points: 1000,
            boundary_condition: BoundaryCondition2D::Dirichlet,
        }
    }
}

/// Loss function weight configuration for 2D PINN
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BurnLossWeights2D {
    /// Weight for data fitting loss (λ_data)
    pub data: f64,
    /// Weight for PDE residual loss (λ_pde)
    pub pde: f64,
    /// Weight for boundary condition loss (λ_bc)
    pub boundary: f64,
    /// Weight for initial condition loss (λ_ic)
    pub initial: f64,
}

impl Default for BurnLossWeights2D {
    fn default() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 10.0, // Higher weight for boundary enforcement
            initial: 10.0,  // Higher weight for initial condition enforcement
        }
    }
}

/// Training metrics for monitorining convergence in 2D
#[derive(Debug, Clone)]
pub struct BurnTrainingMetrics2D {
    /// Total loss history
    pub total_loss: Vec<f64>,
    /// Data loss history
    pub data_loss: Vec<f64>,
    /// PDE residual loss history
    pub pde_loss: Vec<f64>,
    /// Boundary condition loss history
    pub bc_loss: Vec<f64>,
    /// Initial condition loss history
    pub ic_loss: Vec<f64>,
    /// Training time (seconds)
    pub training_time_secs: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
}
