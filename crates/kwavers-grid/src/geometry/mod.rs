//! 🌍 **Spatial Domain Bounded Context**
//!
//! ## Ubiquitous Language
//! - **Computational Domain**: Spatial region Ω ⊂ ℝⁿ where physics is defined
//! - **Geometric Primitives**: Rectangles, spheres, custom shapes for domain definition
//! - **Boundary Classification**: Interior points, boundary points, exterior regions
//! - **Collocation Points**: Spatial locations for PDE evaluation (PINN, quadrature)
//! - **Normal Vectors**: Outward unit normals at boundaries for boundary conditions
//! - **Domain Measures**: Volume, area, surface area for normalization and scaling
//!
//! ## 🎯 Business Value
//! The geometry bounded context enables **solver-agnostic spatial reasoning**:
//! - **Grid Generation**: Forward solvers create computational meshes
//! - **Collocation Sampling**: PINNs place training points strategically
//! - **Boundary Enforcement**: All solvers apply BCs at same geometric locations
//! - **Domain Decomposition**: Multi-region problems with interface conditions
//!
//! ## 📐 Mathematical Foundation
//!
//! A computational domain Ω ⊂ ℝⁿ requires these capabilities:
//!
//! ### Geometric Operations
//! - **Membership Testing**: `x ∈ Ω` (point-in-domain classification)
//! - **Boundary Detection**: `x ∈ ∂Ω` (boundary point identification)
//! - **Normal Computation**: `n̂(x)` for `x ∈ ∂Ω` (outward unit normal vectors)
//! - **Measure Calculation**: `|Ω|` (volume/area), `|∂Ω|` (surface/perimeter)
//!
//! ### Sampling Operations
//! - **Interior Sampling**: Generate points `xᵢ ∈ Ω` for PDE collocation
//! - **Boundary Sampling**: Generate points `xᵢ ∈ ∂Ω` for boundary conditions
//! - **Stratified Sampling**: Adaptive point placement for better convergence
//!
//! ## 🏗️ Architecture
//!
//! ```text
//! GeometricDomain (trait)
//! ├── dimension()           ← ℝⁿ dimension
//! ├── contains()            ← x ∈ Ω
//! ├── classify_point()      ← Interior/Boundary/Exterior
//! ├── bounding_box()        ← AABB for optimization
//! ├── normal()              ← n̂(x) at boundaries
//! ├── measure()             ← |Ω| domain measure
//! ├── sample_interior()     ← Collocation points in Ω
//! └── sample_boundary()     ← Boundary condition points
//! │
//! ├── RectangularDomain     ← [x₀,x₁] × [y₀,y₁] × [z₀,z₁]
//! ├── SphericalDomain       ← Balls, shells, spherical sectors
//! └── CompositeDomain       ← Union/intersection of primitives
//! ```
//!
//! ## 🔄 Solver Integration
//!
//! ### Forward Solvers (FD/FEM/Spectral)
//! ```rust,ignore
//! // Use geometry to build computational grid
//! let geometry = RectangularDomain::new_3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
//! let grid = Grid::from_geometry(&geometry, dx, dy, dz);
//!
//! // Apply boundary conditions at geometric boundaries
//! for boundary_point in geometry.sample_boundary(n_boundary_points) {
//!     apply_bc(&boundary_point, &geometry.normal(&boundary_point));
//! }
//! ```
//!
//! ### Inverse Solvers (PINN/Optimization)
//! ```rust,ignore
//! // Sample collocation points from geometry
//! let interior_points = geometry.sample_interior(n_collocation);
//! let boundary_points = geometry.sample_boundary(n_boundary);
//!
//! // Train PINN on geometric domain
//! pinn.train_on_domain(&geometry, interior_points, boundary_points);
//! ```
//!
//! ## 🎯 Design Patterns
//!
//! ### Strategy Pattern for Domain Types
//! ```rust,ignore
//! // Same algorithms work on any geometric domain
//! fn solve_pde_on_domain<G: GeometricDomain>(domain: &G) {
//!     let collocation = domain.sample_interior(1000);
//!     let boundaries = domain.sample_boundary(100);
//!     // ... PDE solution using domain abstraction
//! }
//!
//! solve_pde_on_domain(&rectangular_domain);
//! solve_pde_on_domain(&spherical_domain);
//! ```
//!
//! ### Composite Pattern for Complex Domains
//! ```rust,ignore
//! // Build complex domains from primitives
//! let liver = SphericalDomain::new(center, radius);
//! let tumor = SphericalDomain::new(tumor_center, tumor_radius);
//! let complex_domain = CompositeDomain::union(&liver, &tumor);
//! ```

use ndarray::{Array1, Array2};

/// Spatial dimension specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeometryDimension {
    One,
    Two,
    Three,
}

impl GeometryDimension {
    #[must_use]
    pub fn as_usize(&self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

/// Identifies which face of the grid domain a boundary element belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFace {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

/// Point location relative to domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLocation {
    Interior,
    Boundary,
    Exterior,
}

/// Abstract geometric domain trait
pub trait GeometricDomain: Send + Sync {
    fn dimension(&self) -> GeometryDimension;
    fn contains(&self, point: &[f64]) -> bool;
    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation;
    fn bounding_box(&self) -> Vec<f64>;
    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>>;
    fn measure(&self) -> f64;
    fn sample_interior(&self, n_points: usize, seed: Option<u64>) -> Array2<f64>;
    fn sample_boundary(&self, n_points: usize, seed: Option<u64>) -> Array2<f64>;
}

mod rectangular;
mod spherical;
#[cfg(test)]
mod tests;

pub use rectangular::RectangularDomain;
pub use spherical::SphericalDomain;
