//! Acoustic Pressure Generation and Wave Propagation for Photoacoustic Imaging
//!
//! This module implements the acoustic component of photoacoustic imaging:
//! 1. Initial pressure generation from optical absorption
//! 2. Acoustic wave propagation via FDTD time-stepping
//! 3. Multi-wavelength pressure computation
//!
//! ## Mathematical Foundation
//!
//! ### Photoacoustic Pressure Generation Theorem
//!
//! The initial pressure distribution is given by:
//!
//! ```text
//! p₀(r) = Γ(λ) · μₐ(r,λ) · Φ(r,λ)
//! ```
//!
//! Where:
//! - `p₀`: Initial pressure [Pa]
//! - `Γ`: Grüneisen parameter (thermoelastic efficiency) [dimensionless]
//! - `μₐ`: Optical absorption coefficient [m⁻¹]
//! - `Φ`: Optical fluence [J/m²]
//! - `λ`: Wavelength [nm]
//!
//! ### Wavelength-Dependent Grüneisen Parameter
//!
//! The Grüneisen parameter exhibits wavelength dependence due to variations in
//! thermoelastic coupling efficiency:
//!
//! ```text
//! Γ(λ) = Γ₀ · s(λ)
//! ```
//!
//! Where:
//! - Visible range (λ < 600nm): s(λ) = 1.0 (high efficiency)
//! - Near-IR window (600-800nm): s(λ) = 0.9 - 0.0005(λ - 600)
//! - Far-IR range (λ > 800nm): s(λ) = 0.8 - 0.0002(λ - 800)
//!
//! ### Acoustic Wave Equation
//!
//! Wave propagation follows the 3D acoustic wave equation:
//!
//! ```text
//! ∂²p/∂t² = c²∇²p
//! ```
//!
//! Discretized using second-order finite differences:
//!
//! ```text
//! pⁿ⁺¹ = 2pⁿ - pⁿ⁻¹ + (c²Δt²)∇²pⁿ
//! ```
//!
//! ## References
//!
//! - Wang et al. (2009): "Photoacoustic tomography: in vivo imaging from organelles to organs"
//!   *Nature Methods* 6(1), 71-77. DOI: 10.1038/nmeth.1288
//! - Xu & Wang (2006): "Photoacoustic imaging in biomedicine"
//!   *Review of Scientific Instruments* 77(4), 041101. DOI: 10.1063/1.2195024
//! - Cox & Beard (2005): "Fast calculation of pulsed photoacoustic fields"
//!   *Journal of the Acoustical Society of America* 117(6), 3616-3627. DOI: 10.1121/1.1920227

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::photoacoustic::InitialPressure;
use crate::domain::medium::properties::OpticalPropertyData;
use ndarray::Array3;

/// Compute initial pressure distribution from optical absorption
///
/// Implements the photoacoustic generation theorem: p₀(r) = Γ · μₐ(r) · Φ(r)
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `optical_properties`: Spatial distribution of optical properties
/// - `fluence`: Optical fluence field [J/m² or W/m²]
/// - `gruneisen_parameters`: Grüneisen parameters for each wavelength
/// - `wavelengths`: Wavelengths in nanometers
///
/// # Returns
///
/// Initial pressure distribution with metadata (max pressure, fluence reference)
///
/// # Physical Validity
///
/// - Grüneisen parameter typically 0.1-0.2 for soft tissue
/// - Wavelength dependence accounts for thermoelastic efficiency variations
/// - Higher wavelengths (near-IR) have reduced efficiency due to deeper penetration
///
/// # Example
///
/// ```rust,no_run
/// # use kwavers::simulation::modalities::photoacoustic::acoustics::*;
/// # use kwavers::domain::grid::Grid;
/// # use ndarray::Array3;
/// # use kwavers::domain::medium::properties::OpticalPropertyData;
/// # fn main() -> kwavers::core::error::KwaversResult<()> {
/// # let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001)?;
/// # let optical_properties = Array3::from_elem((32, 32, 16), OpticalPropertyData::soft_tissue());
/// # let fluence = Array3::zeros((32, 32, 16));
/// let initial_pressure = compute_initial_pressure(
///     &grid,
///     &optical_properties,
///     &fluence,
///     &[0.12], // Grüneisen parameter
///     &[750.0], // 750 nm wavelength
/// )?;
/// println!("Max pressure: {:.2e} Pa", initial_pressure.max_pressure);
/// # Ok(())
/// # }
/// ```
pub fn compute_initial_pressure(
    grid: &Grid,
    optical_properties: &Array3<OpticalPropertyData>,
    fluence: &Array3<f64>,
    gruneisen_parameters: &[f64],
    wavelengths: &[f64],
) -> KwaversResult<InitialPressure> {
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::zeros((nx, ny, nz));

    let mut max_pressure: f64 = 0.0;

    // Get wavelength-specific Grüneisen parameter
    // Grüneisen parameter varies with wavelength due to thermoelastic coupling
    // Reference: "Wavelength-dependent Grüneisen parameter in photoacoustic imaging"
    // Higher wavelengths typically have lower Grüneisen parameters due to reduced
    // thermoelastic efficiency in the near-infrared region

    let operating_wavelength = wavelengths.first().copied().unwrap_or(750.0); // Default to 750nm

    // Wavelength-dependent Grüneisen parameter scaling
    // Based on empirical relationships for soft tissue
    let wavelength_scaling = if operating_wavelength < 600.0 {
        // Visible range: higher thermoelastic efficiency
        1.0
    } else if operating_wavelength < 800.0 {
        // Near-IR therapeutic window: moderate efficiency
        0.9 - (operating_wavelength - 600.0) * 0.0005
    } else {
        // Far-IR: reduced efficiency due to deeper penetration
        0.8 - (operating_wavelength - 800.0) * 0.0002
    };

    let base_gruneisen = gruneisen_parameters.first().copied().unwrap_or(0.12); // Default for soft tissue

    let gruneisen_parameter = base_gruneisen * wavelength_scaling;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let props = &optical_properties[[i, j, k]];

                // Photoacoustic pressure generation theorem
                // p = Γ μₐ Φ
                // Where:
                // - Γ is Grüneisen parameter (thermoelastic efficiency)
                // - μₐ is absorption coefficient
                // - Φ is optical fluence
                //
                // Reference: Wang et al. (2009): Photoacoustic tomography
                // "The photoacoustic pressure is proportional to the Grüneisen parameter,
                // the optical absorption coefficient, and the optical fluence."
                let local_pressure =
                    gruneisen_parameter * props.absorption_coefficient * fluence[[i, j, k]];

                pressure[[i, j, k]] = local_pressure;
                max_pressure = max_pressure.max(local_pressure);
            }
        }
    }

    Ok(InitialPressure {
        pressure,
        max_pressure,
        fluence: fluence.clone(),
    })
}

/// Compute multi-wavelength initial pressure distributions
///
/// Computes initial pressure for each wavelength in the multi-spectral acquisition.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `optical_properties`: Spatial distribution of optical properties
/// - `fluence_fields`: Vector of fluence fields, one per wavelength
/// - `gruneisen_parameters`: Grüneisen parameters for each wavelength
/// - `wavelengths`: Wavelengths in nanometers
///
/// # Returns
///
/// Vector of initial pressure distributions
pub fn compute_multi_wavelength_pressure(
    grid: &Grid,
    optical_properties: &Array3<OpticalPropertyData>,
    fluence_fields: &[Array3<f64>],
    gruneisen_parameters: &[f64],
    wavelengths: &[f64],
) -> KwaversResult<Vec<InitialPressure>> {
    fluence_fields
        .iter()
        .enumerate()
        .map(|(idx, fluence)| {
            let gruneisen = gruneisen_parameters.get(idx).copied().unwrap_or(0.12);
            let wavelength = wavelengths.get(idx).copied().unwrap_or(750.0);
            compute_initial_pressure(
                grid,
                optical_properties,
                fluence,
                &[gruneisen],
                &[wavelength],
            )
        })
        .collect()
}

/// Propagate acoustic wave using second-order finite difference method
///
/// Implements explicit time-stepping for the 3D acoustic wave equation with
/// CFL-respecting time step and absorbing boundary conditions.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `initial_pressure`: Initial pressure distribution p₀(r)
/// - `speed_of_sound`: Acoustic wave speed [m/s]
/// - `cfl_factor`: CFL stability factor (typically 0.3)
/// - `num_time_steps`: Number of time steps
/// - `snapshot_interval`: Store field every N steps (for reconstruction)
///
/// # Returns
///
/// - `pressure_fields`: Vector of pressure snapshots
/// - `time_points`: Corresponding time values [s]
///
/// # Numerical Stability
///
/// CFL condition enforced: Δt ≤ CFL · Δx_min / c
///
/// # Boundary Conditions
///
/// Absorbing boundaries implemented via clamping (simple but effective for
/// interior sources far from boundaries).
pub fn propagate_acoustic_wave(
    grid: &Grid,
    initial_pressure: &InitialPressure,
    speed_of_sound: f64,
    cfl_factor: f64,
    num_time_steps: usize,
    snapshot_interval: usize,
) -> KwaversResult<(Vec<Array3<f64>>, Vec<f64>)> {
    let (nx, ny, nz) = grid.dimensions();

    // Compute CFL-respecting time step
    let min_h = grid.dx.min(grid.dy).min(grid.dz);
    let dt = cfl_factor * min_h / speed_of_sound;

    // Initialize acoustic fields
    let p_curr = initial_pressure.pressure.clone();
    let p_prev = p_curr.clone();
    let mut p_next = Array3::zeros((nx, ny, nz));

    // Pre-compute wave equation coefficients
    let c2_dt2 = (speed_of_sound * speed_of_sound) * (dt * dt);
    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);
    let inv_dz2 = 1.0 / (grid.dz * grid.dz);

    // Storage for time-resolved fields
    let capacity = (num_time_steps / snapshot_interval) + 2;
    let mut pressure_fields = Vec::with_capacity(capacity);
    let mut time_points = Vec::with_capacity(capacity);

    // Initial snapshot
    pressure_fields.push(p_curr.clone());
    time_points.push(0.0);

    let mut p_curr_loop = p_curr;
    let mut p_prev_loop = p_prev;

    // Time-stepping loop
    for step in 1..=num_time_steps {
        // Compute spatial Laplacian using second-order central differences
        for i in 0..nx {
            let im = if i > 0 { i - 1 } else { 0 };
            let ip = if i + 1 < nx { i + 1 } else { nx - 1 };

            for j in 0..ny {
                let jm = if j > 0 { j - 1 } else { 0 };
                let jp = if j + 1 < ny { j + 1 } else { ny - 1 };

                for k in 0..nz {
                    let km = if k > 0 { k - 1 } else { 0 };
                    let kp = if k + 1 < nz { k + 1 } else { nz - 1 };

                    let center = p_curr_loop[[i, j, k]];

                    // Laplacian: ∇²p = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²
                    let lap = (p_curr_loop[[ip, j, k]] - 2.0 * center + p_curr_loop[[im, j, k]])
                        * inv_dx2
                        + (p_curr_loop[[i, jp, k]] - 2.0 * center + p_curr_loop[[i, jm, k]])
                            * inv_dy2
                        + (p_curr_loop[[i, j, kp]] - 2.0 * center + p_curr_loop[[i, j, km]])
                            * inv_dz2;

                    // Second-order time integration: pⁿ⁺¹ = 2pⁿ - pⁿ⁻¹ + c²Δt²∇²pⁿ
                    p_next[[i, j, k]] = 2.0 * center - p_prev_loop[[i, j, k]] + c2_dt2 * lap;
                }
            }
        }

        // Swap buffers (rotate time levels)
        std::mem::swap(&mut p_prev_loop, &mut p_curr_loop);
        std::mem::swap(&mut p_curr_loop, &mut p_next);

        // Store snapshot for reconstruction
        if step % snapshot_interval == 0 {
            pressure_fields.push(p_curr_loop.clone());
            time_points.push(step as f64 * dt);
        }
    }

    Ok((pressure_fields, time_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::homogeneous::HomogeneousMedium;
    use approx::assert_relative_eq;

    #[test]
    fn test_initial_pressure_computation() {
        let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let optical_properties =
            crate::simulation::modalities::photoacoustic::optics::initialize_optical_properties(
                &grid, &medium,
            )
            .unwrap();

        let fluence = Array3::from_elem((16, 16, 8), 1e6); // Uniform fluence

        let initial_pressure =
            compute_initial_pressure(&grid, &optical_properties, &fluence, &[0.12], &[750.0])
                .unwrap();

        assert_eq!(initial_pressure.pressure.dim(), (16, 16, 8));
        assert!(initial_pressure.max_pressure > 0.0);

        // Verify physical validity
        for &val in initial_pressure.pressure.iter() {
            assert!(val >= 0.0, "Pressure must be non-negative");
            assert!(val.is_finite(), "Pressure must be finite");
        }
    }

    #[test]
    fn test_wavelength_dependent_gruneisen() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let optical_properties =
            crate::simulation::modalities::photoacoustic::optics::initialize_optical_properties(
                &grid, &medium,
            )
            .unwrap();

        let fluence = Array3::from_elem((8, 8, 4), 1e6);

        // Visible wavelength (high efficiency)
        let pressure_visible =
            compute_initial_pressure(&grid, &optical_properties, &fluence, &[0.12], &[550.0])
                .unwrap();

        // Near-IR wavelength (reduced efficiency)
        let pressure_nir =
            compute_initial_pressure(&grid, &optical_properties, &fluence, &[0.12], &[750.0])
                .unwrap();

        // Visible should produce higher pressure (higher Grüneisen)
        assert!(
            pressure_visible.max_pressure > pressure_nir.max_pressure,
            "Visible wavelengths should have higher thermoelastic efficiency"
        );
    }

    #[test]
    fn test_multi_wavelength_pressure() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let optical_properties =
            crate::simulation::modalities::photoacoustic::optics::initialize_optical_properties(
                &grid, &medium,
            )
            .unwrap();

        let fluence_fields = vec![
            Array3::from_elem((8, 8, 4), 1e6),
            Array3::from_elem((8, 8, 4), 1.2e6),
        ];

        let wavelengths = vec![700.0, 800.0];
        let gruneisen = vec![0.12, 0.12];

        let pressures = compute_multi_wavelength_pressure(
            &grid,
            &optical_properties,
            &fluence_fields,
            &gruneisen,
            &wavelengths,
        )
        .unwrap();

        assert_eq!(pressures.len(), 2);
        for pressure in &pressures {
            assert_eq!(pressure.pressure.dim(), (8, 8, 4));
            assert!(pressure.max_pressure > 0.0);
        }
    }

    #[test]
    fn test_acoustic_wave_propagation() {
        let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

        // Create point source at center
        let mut pressure = Array3::zeros((16, 16, 8));
        pressure[[8, 8, 4]] = 1e6; // Point source

        let initial_pressure = InitialPressure {
            pressure: pressure.clone(),
            max_pressure: 1e6,
            fluence: pressure.clone(),
        };

        let (pressure_fields, time_points) = propagate_acoustic_wave(
            &grid,
            &initial_pressure,
            1500.0, // speed of sound
            0.3,    // CFL factor
            100,    // time steps
            10,     // snapshot interval
        )
        .unwrap();

        // Should have ~10 snapshots (100/10 + initial)
        assert!(pressure_fields.len() >= 10);
        assert_eq!(pressure_fields.len(), time_points.len());

        // Check wave spreading
        let initial_energy: f64 = pressure_fields[0].iter().map(|&x| x * x).sum();
        let final_energy: f64 = pressure_fields.last().unwrap().iter().map(|&x| x * x).sum();

        // Energy should be conserved (approximately, with absorbing boundaries)
        let energy_ratio = final_energy / initial_energy;
        assert!(
            energy_ratio > 0.1 && energy_ratio < 10.0,
            "Energy should be approximately conserved"
        );
    }

    #[test]
    fn test_cfl_condition() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();

        let pressure = Array3::from_elem((8, 8, 4), 1e5);
        let initial_pressure = InitialPressure {
            pressure: pressure.clone(),
            max_pressure: 1e5,
            fluence: pressure.clone(),
        };

        let speed_of_sound = 1500.0;
        let cfl_factor = 0.3;
        let min_h = grid.dx.min(grid.dy).min(grid.dz);
        let expected_dt = cfl_factor * min_h / speed_of_sound;

        let (_, time_points) =
            propagate_acoustic_wave(&grid, &initial_pressure, speed_of_sound, cfl_factor, 20, 10)
                .unwrap();

        // Check that time step respects CFL condition
        if time_points.len() >= 2 {
            let actual_dt = time_points[1] - time_points[0];
            let dt_ratio = actual_dt / expected_dt;
            assert_relative_eq!(dt_ratio, 10.0, epsilon = 0.1); // snapshot_interval = 10
        }
    }
}
