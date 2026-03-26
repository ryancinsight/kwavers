//! Monte Carlo Photon Transport Solver
//!
//! High-fidelity photon transport simulation using stochastic Monte Carlo methods.
//! Provides an alternative to diffusion approximation for scenarios where the
//! diffusion assumption breaks down (e.g., void regions, ballistic regime, strong anisotropy).
//!
//! # Mathematical Foundation
//!
//! ## Radiative Transfer Equation (RTE)
//!
//! The Monte Carlo method provides a stochastic solution to the RTE:
//!
//! ```text
//! ŝ·∇L(r,ŝ) + μ_t L(r,ŝ) = μ_s ∫_{4π} p(ŝ·ŝ') L(r,ŝ') dΩ' + S(r,ŝ)
//! ```
//!
//! where:
//! - `L(r,ŝ)`: Radiance at position r in direction ŝ
//! - `μ_t = μ_a + μ_s`: Total attenuation coefficient
//! - `p(ŝ·ŝ')`: Phase function (scattering probability)
//! - `S(r,ŝ)`: Source term
//!
//! ## Monte Carlo Algorithm
//!
//! 1. **Photon Launch**: Initialize photon position, direction, weight
//! 2. **Propagation**: Sample free path length: `s = -ln(ξ)/μ_t` (ξ ~ U(0,1))
//! 3. **Interaction**:
//!    - Absorption: `W ← W · μ_s/μ_t` (Russian roulette if W < threshold)
//!    - Scattering: Sample new direction from phase function
//! 4. **Boundary Handling**: Fresnel reflection/refraction at interfaces
//! 5. **Termination**: Continue until photon exits domain or weight < threshold
//!
//! ## Henyey-Greenstein Phase Function
//!
//! ```text
//! p(cos θ) = (1 - g²) / [4π(1 + g² - 2g·cos θ)^(3/2)]
//! ```
//!
//! Sampling: `cos θ = (1 + g² - [(1-g²)/(1 - g + 2g·ξ)]²) / (2g)`
//!
//! # Architecture
//!
//! - Domain layer: `OpticalPropertyData` canonical representation
//! - Physics layer (this module): Monte Carlo transport solver
//! - Execution: CPU (parallel via Rayon) with optional GPU acceleration placeholder
//!
//! # Performance
//!
//! - CPU: ~1M photons/sec/thread (typical)
//! - GPU: ~100M photons/sec (CUDA, future work)
//! - Memory: O(N_photons) for trajectory storage (optional)
//!
//! # Example
//!
//! ```rust,no_run
//! use kwavers::physics::optics::monte_carlo::{MonteCarloSolver, PhotonSource, SimulationConfig};
//! use kwavers::domain::grid::Grid3D;
//! use kwavers::domain::grid::GridDimensions;
//! use kwavers::domain::medium::properties::OpticalPropertyData;
//! use kwavers::physics::optics::map_builder::OpticalPropertyMapBuilder;
//!
//! // Create solver
//! let grid = Grid3D::new(50, 50, 50, 0.001, 0.001, 0.001)?;
//! let optical_map = OpticalPropertyMapBuilder::new(GridDimensions::from_grid(&grid))
//!     .set_background(OpticalPropertyData::soft_tissue())
//!     .build();
//! let solver = MonteCarloSolver::new(grid, optical_map);
//!
//! // Configure simulation
//! let config = SimulationConfig::default()
//!     .num_photons(1_000_000)
//!     .russian_roulette_threshold(0.001);
//!
//! // Define source
//! let source = PhotonSource::pencil_beam([0.025, 0.025, 0.0], [0.0, 0.0, 1.0]);
//!
//! // Run simulation
//! let result = solver.simulate(&source, &config)?;
//! println!("Absorbed energy: {:.3e} J", result.total_absorbed_energy());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod photon;
pub mod result;
pub mod solver;
pub mod source;
pub mod utils;

#[cfg(test)]
mod tests;

pub use config::SimulationConfig;
pub use result::MCResult;
pub use solver::MonteCarloSolver;
pub use source::PhotonSource;
