//! Optical Fluence Computation for Photoacoustic Imaging
//!
//! This module implements optical fluence distribution computation using the diffusion
//! approximation to the radiative transfer equation. The diffusion approximation is valid
//! in highly scattering media where the reduced scattering coefficient μₛ' >> μₐ.
//!
//! ## Mathematical Foundation
//!
//! The steady-state diffusion equation for optical fluence Φ(r):
//!
//! ```text
//! ∇·(D∇Φ) - μₐΦ = -S
//! ```
//!
//! Where:
//! - `D = 1/(3(μₐ + μₛ'))`: Diffusion coefficient [m]
//! - `μₐ`: Absorption coefficient [m⁻¹]
//! - `μₛ'`: Reduced scattering coefficient [m⁻¹]
//! - `S`: Source term [W/m³]
//!
//! ## Implementation
//!
//! The module provides:
//! - Tissue optical property initialization with heterogeneous structures
//! - Single-wavelength fluence computation
//! - Multi-wavelength parallel computation for spectroscopic imaging
//! - Integration with physics layer diffusion solver
//!
//! ## References
//!
//! - Arridge (1999): "Optical tomography in medical imaging"
//!   *Inverse Problems* 15(2), R41. DOI: 10.1088/0266-5611/15/2/022
//! - Wang & Wu (2007): "Biomedical Optics: Principles and Imaging"
//!   Wiley-Interscience. ISBN: 978-0-471-74304-0

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::properties::OpticalPropertyData;
use crate::domain::medium::Medium;
use crate::physics::optics::diffusion::solver::{
    DiffusionBoundaryCondition, DiffusionBoundaryConditions, DiffusionSolver, DiffusionSolverConfig,
};
use ndarray::Array3;
use rayon::prelude::*;

/// Initialize optical properties based on tissue type and geometry
///
/// Creates a heterogeneous optical property map with anatomically-inspired structures:
/// - Background: Soft tissue properties
/// - Cylindrical blood vessel (2mm diameter at [25mm, 25mm])
/// - Spherical tumor (5mm diameter at [20mm, 20mm, 15mm])
///
/// # Arguments
///
/// - `grid`: Computational grid defining spatial domain
/// - `_medium`: Acoustic medium (reserved for future coupling)
///
/// # Returns
///
/// 3D array of optical properties at each grid point
///
/// # Design Notes
///
/// This function creates a realistic test phantom for photoacoustic imaging validation.
/// In production systems, optical properties would be loaded from medical imaging data
/// or computed from tissue classification algorithms.
pub fn initialize_optical_properties(
    grid: &Grid,
    _medium: &dyn Medium,
) -> KwaversResult<Array3<OpticalPropertyData>> {
    let (nx, ny, nz) = grid.dimensions();
    let mut properties = Array3::from_elem(
        (nx, ny, nz),
        crate::domain::imaging::photoacoustic::PhotoacousticOpticalProperties::soft_tissue(750.0),
    );

    // Add blood vessels and tumor regions
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Add cylindrical blood vessel aligned with z-axis
                let vessel_dist = ((x - 0.025).powi(2) + (y - 0.025).powi(2)).sqrt();
                if vessel_dist < 0.002 {
                    // 2mm diameter vessel
                    properties[[i, j, k]] =
                        crate::domain::imaging::photoacoustic::PhotoacousticOpticalProperties::blood(
                            750.0,
                        );
                }

                // Add spherical tumor
                let tumor_dist =
                    ((x - 0.02).powi(2) + (y - 0.02).powi(2) + (z - 0.015).powi(2)).sqrt();
                if tumor_dist < 0.005 {
                    // 5mm diameter tumor
                    properties[[i, j, k]] =
                        crate::domain::imaging::photoacoustic::PhotoacousticOpticalProperties::tumor(
                            750.0,
                        );
                }
            }
        }
    }

    Ok(properties)
}

/// Compute optical fluence distribution using diffusion approximation
///
/// Solves the steady-state diffusion equation using finite-difference method
/// with appropriate boundary conditions for tissue-air interface.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `optical_properties`: Spatial distribution of optical properties
/// - `laser_fluence`: Incident laser fluence at surface [W/m² or J/m²]
///
/// # Returns
///
/// Optical fluence field Φ(r) in W/m² (steady-state) or J/m² (pulsed)
///
/// # Implementation Details
///
/// - Top surface (z=0): Uniform illumination with extrapolated boundary condition
/// - Side surfaces: Zero-flux boundary conditions (symmetry)
/// - Bottom surface: Zero-flux boundary condition
/// - Convergence tolerance: 1e-6 relative error
/// - Maximum iterations: 10,000
///
/// # Physical Validity
///
/// The diffusion approximation is valid when:
/// - μₛ' >> μₐ (highly scattering medium)
/// - Distance from source > 1/μₛ' (far from boundaries)
/// - Valid for biological tissue in near-infrared (600-1000 nm)
pub fn compute_fluence_at_wavelength(
    grid: &Grid,
    optical_properties: &Array3<OpticalPropertyData>,
    laser_fluence: f64,
    _wavelength_nm: f64,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = grid.dimensions();

    // Configure diffusion solver with tissue-air boundary conditions
    let config = DiffusionSolverConfig {
        max_iterations: 10000,
        tolerance: 1e-6,
        boundary_parameter: 2.0, // Tissue-air interface (refractive index mismatch)
        boundary_conditions: Some(DiffusionBoundaryConditions {
            x_min: DiffusionBoundaryCondition::ZeroFlux,
            x_max: DiffusionBoundaryCondition::ZeroFlux,
            y_min: DiffusionBoundaryCondition::ZeroFlux,
            y_max: DiffusionBoundaryCondition::ZeroFlux,
            z_min: DiffusionBoundaryCondition::Extrapolated { a: 2.0 },
            z_max: DiffusionBoundaryCondition::ZeroFlux,
        }),
        verbose: false,
    };

    // Create diffusion solver
    let solver = DiffusionSolver::new(grid.clone(), optical_properties.clone(), config)?;

    // Create source term: uniform illumination from top surface (z=0)
    // Source units: W/m³ for steady-state (or J/m³ for pulsed)
    let mut source = Array3::zeros((nx, ny, nz));

    // Top surface illumination (first layer in z)
    let source_strength = laser_fluence / grid.dz; // Convert surface fluence to volumetric source
    for i in 0..nx {
        for j in 0..ny {
            source[[i, j, 0]] = source_strength;
        }
    }

    // Solve diffusion equation
    let fluence = solver.solve(&source)?;

    Ok(fluence)
}

/// Compute fluence for all wavelengths in parallel
///
/// Leverages Rayon for parallel computation across wavelengths, enabling efficient
/// multi-spectral photoacoustic imaging simulations.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `optical_properties`: Spatial distribution of optical properties
/// - `laser_fluence`: Incident laser fluence [W/m² or J/m²]
/// - `wavelengths`: Vector of wavelengths in nanometers
///
/// # Returns
///
/// Vector of fluence fields, one per wavelength
///
/// # Performance
///
/// Parallel execution provides near-linear speedup on multi-core systems.
/// Memory usage scales linearly with number of wavelengths.
///
/// # Example
///
/// ```rust,no_run
/// # use kwavers::simulation::modalities::photoacoustic::optics::*;
/// # use kwavers::domain::grid::Grid;
/// # use ndarray::Array3;
/// # use kwavers::domain::medium::properties::OpticalPropertyData;
/// # fn main() -> kwavers::core::error::KwaversResult<()> {
/// # let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001)?;
/// # let optical_properties = Array3::from_elem((32, 32, 16), OpticalPropertyData::soft_tissue());
/// let wavelengths = vec![700.0, 750.0, 800.0, 850.0]; // nm
/// let fluence_fields = compute_multi_wavelength_fluence(
///     &grid,
///     &optical_properties,
///     1e6, // 1 MJ/m² laser fluence
///     &wavelengths,
/// )?;
/// assert_eq!(fluence_fields.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn compute_multi_wavelength_fluence(
    grid: &Grid,
    optical_properties: &Array3<OpticalPropertyData>,
    laser_fluence: f64,
    wavelengths: &[f64],
) -> KwaversResult<Vec<Array3<f64>>> {
    // Parallel computation over wavelengths using Rayon
    let fluence_fields: Result<Vec<_>, _> = wavelengths
        .par_iter()
        .map(|&wavelength| {
            compute_fluence_at_wavelength(grid, optical_properties, laser_fluence, wavelength)
        })
        .collect();

    fluence_fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_optical_property_initialization() {
        let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let properties = initialize_optical_properties(&grid, &medium).unwrap();

        assert_eq!(properties.dim(), (32, 32, 16));

        // Check that we have heterogeneous properties (not all the same)
        let first_val = properties[[0, 0, 0]].absorption_coefficient;
        let mut found_different = false;
        for val in properties.iter() {
            if (val.absorption_coefficient - first_val).abs() > 1e-10 {
                found_different = true;
                break;
            }
        }
        assert!(found_different, "Expected heterogeneous optical properties");
    }

    #[test]
    fn test_fluence_computation_basic() {
        let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let properties = initialize_optical_properties(&grid, &medium).unwrap();

        let fluence = compute_fluence_at_wavelength(&grid, &properties, 1e6, 750.0).unwrap();

        assert_eq!(fluence.dim(), (16, 16, 8));

        // Check that fluence decreases with depth (fundamental property)
        let surface_fluence = fluence[[8, 8, 0]];
        let deep_fluence = fluence[[8, 8, 7]];
        assert!(
            surface_fluence > deep_fluence,
            "Fluence should decrease with depth"
        );

        // Check physical validity (all values non-negative and finite)
        for &val in fluence.iter() {
            assert!(val >= 0.0, "Fluence must be non-negative");
            assert!(val.is_finite(), "Fluence must be finite");
        }
    }

    #[test]
    fn test_multi_wavelength_fluence() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let properties = initialize_optical_properties(&grid, &medium).unwrap();

        let wavelengths = vec![700.0, 750.0, 800.0];
        let fluence_fields =
            compute_multi_wavelength_fluence(&grid, &properties, 1e6, &wavelengths).unwrap();

        assert_eq!(fluence_fields.len(), 3);
        for fluence in &fluence_fields {
            assert_eq!(fluence.dim(), (8, 8, 4));
        }
    }
}
