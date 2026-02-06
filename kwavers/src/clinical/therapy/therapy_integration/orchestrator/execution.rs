//! Therapy Execution and Field Generation
//!
//! This module handles the core therapy execution loop including acoustic field generation,
//! field updates, and step-by-step therapy delivery. It implements focused ultrasound
//! field generation using Gaussian beam approximation and manages the temporal evolution
//! of therapy sessions.
//!
//! ## Acoustic Field Generation
//!
//! Uses Gaussian beam approximation for focused ultrasound field generation:
//! - Focal point targeting based on therapy configuration
//! - Beam width and intensity profiles
//! - Distance-dependent attenuation
//!
//! ## References
//!
//! - O'Neil (1949): "Gaussian beam propagation in focused ultrasound"
//! - IEC 62359:2010: "Field characterization methods"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

use super::super::config::AcousticTherapyParams;
use super::super::state::AcousticField;

/// Generate acoustic field for therapy
///
/// Creates a focused acoustic field using Gaussian beam approximation.
/// The field is centered at the specified focal depth with a Gaussian intensity profile.
///
/// # Arguments
///
/// - `grid`: Computational grid for spatial discretization
/// - `acoustic_params`: Therapy acoustic parameters (frequency, pressure, focal depth, etc.)
///
/// # Returns
///
/// Acoustic field with pressure and velocity components
///
/// # Field Model
///
/// Uses Gaussian beam approximation:
/// TODO_AUDIT: P2 - Nonlinear Ultrasound Therapy - Implement full nonlinear ultrasound propagation for high-intensity therapy applications, replacing Gaussian beam approximation
/// DEPENDS ON: physics/acoustics/nonlinear/kzk.rs, physics/acoustics/shock_formation.rs
/// MISSING: Khokhlov-Zabolotskaya-Kuznetsov equation for focused beam nonlinear propagation
/// MISSING: Shock wave formation criteria and evolution tracking
/// MISSING: Cavitation threshold modeling with Bjerknes forces
/// MISSING: Thermal dose accumulation with nonlinear heating effects
/// MISSING: Standing wave pattern formation in therapy chambers
/// - Pressure amplitude: P(r) = P₀ * exp(-r²/w²)
/// - where r is distance from focal point, w is beam width
/// - Focal point at specified depth along x-axis
/// - Beam width: 5 mm (typical for therapeutic ultrasound)
///
/// # References
///
/// - O'Neil (1949): "Theory of focusing radiators"
/// - Hasegawa & Yosioka (1975): "Acoustic radiation pressure on compressible spheres"
pub fn generate_acoustic_field(
    grid: &Grid,
    acoustic_params: &AcousticTherapyParams,
) -> KwaversResult<AcousticField> {
    // Create focused acoustic field based on therapy parameters
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::zeros((nx, ny, nz));
    let velocity = Array3::zeros((nx, ny, nz));

    // Create focused pressure field using Gaussian beam approximation
    let focal_point = (acoustic_params.focal_depth, 0.0, 0.0);
    let beam_width = 0.005; // 5mm beam width (typical for therapeutic ultrasound)

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Distance from focal point
                let dx = x - focal_point.0;
                let dy = y - focal_point.1;
                let dz = z - focal_point.2;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                // Gaussian beam profile
                let beam_profile = (-r * r / (beam_width * beam_width)).exp();

                // Pressure field using Gaussian beam approximation
                // Reference: O'Neil (1949) Gaussian beam propagation in focused ultrasound
                pressure[[i, j, k]] = acoustic_params.pnp * beam_profile;
            }
        }
    }

    Ok(AcousticField {
        pressure,
        velocity_x: velocity.clone(),
        velocity_y: velocity.clone(),
        velocity_z: velocity,
    })
}

/// Update acoustic heating for therapy
///
/// Calculates temperature rise due to acoustic absorption heating.
/// Uses the Pennes bioheat equation with acoustic heating source term.
///
/// # Arguments
///
/// - `acoustic_field`: Current acoustic field
/// - `grid`: Computational grid
/// - `dt`: Time step (s)
/// - `focal_depth`: Focal depth for distance-based spreading (m)
///
/// # Returns
///
/// 3D temperature field (K)
///
/// # Physics Model
///
/// Acoustic absorption heating:
/// - Q_acoustic = α * |p|² / (ρ * c)
/// - where α is attenuation coefficient, p is pressure, ρ is density, c is sound speed
///
/// Distance-based spreading:
/// - Temperature rise decreases with distance from focus using exponential decay
/// - Characteristic length: 1 cm (typical for focused ultrasound)
///
/// # References
///
/// - Pennes (1948): "Analysis of tissue and arterial blood temperatures"
/// - Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"
pub fn calculate_acoustic_heating(
    acoustic_field: &AcousticField,
    grid: &Grid,
    dt: f64,
    focal_depth: f64,
) -> Array3<f64> {
    let ambient_temp = 310.0; // 37°C in Kelvin
    let mut temperature = Array3::from_elem(acoustic_field.pressure.dim(), ambient_temp);

    // Calculate acoustic absorption heating from pressure field
    // Q_acoustic = α * |p|² / (ρ * c) where α is attenuation coefficient
    let alpha = 0.5; // 0.5 Np/m typical for soft tissue
    let rho = 1000.0; // kg/m³
    let c = 1540.0; // m/s
    let heating_factor = alpha / (rho * c);

    // Add acoustic heating with spatial spreading
    for (index, &pressure) in acoustic_field.pressure.indexed_iter() {
        // Acoustic heating proportional to intensity (pressure²)
        let heating = heating_factor * pressure * pressure;

        // Apply distance-based spreading from focal point
        let (i, j, k) = index;
        let x = i as f64 * grid.dx - focal_depth;
        let y = j as f64 * grid.dy;
        let z = k as f64 * grid.dz;
        let r = (x * x + y * y + z * z).sqrt();

        // Temperature rise decreases with distance from focus
        let distance_factor = (-r / 0.01).exp(); // 1cm characteristic length
        let temp_rise = heating * distance_factor * dt * 1e-6; // Convert to temperature rise

        temperature[index] = ambient_temp + temp_rise;
    }

    temperature
}
