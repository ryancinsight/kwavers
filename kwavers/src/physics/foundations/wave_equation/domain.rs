//! Spatial domain and boundary condition definitions for wave equations

/// Spatial dimension specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDimension {
    /// 1D problems (x-axis only)
    One,
    /// 2D problems (x-y plane)
    Two,
    /// 3D problems (x-y-z volume)
    Three,
}

/// Time integration scheme specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeIntegration {
    /// Explicit time stepping (CFL-limited)
    Explicit,
    /// Implicit time stepping (unconditionally stable)
    Implicit,
    /// Semi-implicit (mixed explicit/implicit)
    SemiImplicit,
}

/// Boundary condition type
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    /// Dirichlet: prescribed field value u = g
    Dirichlet { value: f64 },
    /// Neumann: prescribed normal derivative ∂u/∂n = g
    Neumann { flux: f64 },
    /// Absorbing: perfectly matched layer or damping
    Absorbing { damping: f64 },
    /// Periodic: u(x₀) = u(x₁)
    Periodic,
    /// Free surface: stress-free condition
    FreeSurface,
    /// Rigid: zero displacement
    Rigid,
}

/// Spatial domain specification
#[derive(Debug, Clone)]
pub struct Domain {
    /// Spatial dimension
    pub dimension: SpatialDimension,
    /// Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    pub bounds: Vec<f64>,
    /// Grid resolution [nx, ny, nz]
    pub resolution: Vec<usize>,
    /// Boundary conditions for each face [x_min, x_max, y_min, y_max, z_min, z_max]
    pub boundaries: Vec<BoundaryCondition>,
}

impl Domain {
    /// Create 1D domain
    pub fn new_1d(xmin: f64, xmax: f64, nx: usize, bc: BoundaryCondition) -> Self {
        Self {
            dimension: SpatialDimension::One,
            bounds: vec![xmin, xmax],
            resolution: vec![nx],
            boundaries: vec![bc.clone(), bc],
        }
    }

    /// Create 2D domain
    pub fn new_2d(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
        bc: BoundaryCondition,
    ) -> Self {
        Self {
            dimension: SpatialDimension::Two,
            bounds: vec![xmin, xmax, ymin, ymax],
            resolution: vec![nx, ny],
            boundaries: vec![bc.clone(), bc.clone(), bc.clone(), bc],
        }
    }

    /// Create 3D domain
    pub fn new_3d(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
        nx: usize,
        ny: usize,
        nz: usize,
        bc: BoundaryCondition,
    ) -> Self {
        Self {
            dimension: SpatialDimension::Three,
            bounds: vec![xmin, xmax, ymin, ymax, zmin, zmax],
            resolution: vec![nx, ny, nz],
            boundaries: vec![
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc,
            ],
        }
    }

    /// Get spatial step sizes [dx, dy, dz]
    pub fn spacing(&self) -> Vec<f64> {
        self.resolution
            .iter()
            .enumerate()
            .map(|(i, &n)| (self.bounds[2 * i + 1] - self.bounds[2 * i]) / (n - 1) as f64)
            .collect()
    }
}
