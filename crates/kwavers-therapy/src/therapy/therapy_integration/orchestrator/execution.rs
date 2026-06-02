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
//! ## Temperature Output
//!
//! All temperature fields returned by this module are in **degrees Celsius (°C)**,
//! consistent with the `IntensityTracker::update_thermal_dose` contract which
//! applies CEM43 thresholds at 37 °C and 43 °C.
//!
//! ## References
//!
//! - O'Neil (1949): "Gaussian beam propagation in focused ultrasound"
//! - IEC 62359:2010: "Field characterization methods"
//! - Pennes (1948): "Analysis of tissue and arterial blood temperatures"
//! - Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::constants::tissue_thermal::SPECIFIC_HEAT_TISSUE;
use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use ndarray::{Array3, Zip};

use super::super::config::AcousticTherapyParams;
use super::super::state::AcousticField;

/// Generate acoustic field for therapy.
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
/// Uses Gaussian beam approximation: P(r) = P₀·exp(−r²/w²), where r is the
/// distance from the focal point and w = 5 mm is the beam width.
///
/// Acoustic radiation force creates particle velocity v = P/(ρ₀·c₀) along the
/// propagation direction. For a plane-wave approximation the x-component is
/// `vx = p/(ρ₀·c₀)` and the transverse components are zero.
///
/// ## Limitation
///
/// The Gaussian approximation is valid for low-intensity diagnostic levels.
/// For HIFU (>1 kW/cm²), it does not capture:
/// - Shock formation and nonlinear harmonic generation (KZK equation,
///   Zabolotskaya & Khokhlov 1969; Lee & Hamilton 1995)
/// - Cavitation inception and Bjerknes force on bubbles
/// - Thermal dose accumulation with nonlinear heating (Sapareto & Dewey 1984)
///
/// Full nonlinear HIFU propagation via the KZK solver is not yet integrated
/// into this therapy orchestration path.
///
/// # References
///
/// - O'Neil (1949): "Theory of focusing radiators"
/// - Hasegawa & Yosioka (1975): "Acoustic radiation pressure on compressible spheres"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn generate_acoustic_field(
    grid: &Grid,
    acoustic_params: &AcousticTherapyParams,
) -> KwaversResult<AcousticField> {
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
    let velocity_zero = Array3::<f64>::zeros((nx, ny, nz));

    // Gaussian beam approximation: P(r) = P₀ · exp(−r²/w²)
    // Focal point is along the x-axis at the configured focal depth.
    let focal_x = acoustic_params.focal_depth;
    let beam_width_sq = 0.005_f64 * 0.005; // (5 mm)²

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;
    let pnp = acoustic_params.pnp;

    Zip::indexed(pressure.view_mut()).par_for_each(|(i, j, k), p| {
        let x = i as f64 * dx - focal_x;
        let y = j as f64 * dy;
        let z = k as f64 * dz;
        let r_sq = x * x + y * y + z * z;
        let beam_profile = (-r_sq / beam_width_sq).exp();
        *p = pnp * beam_profile;
    });

    // Plane-wave approximation for the axial velocity component:
    // v_x = p / (ρ₀·c₀).  Transverse components are zero.
    // Reference impedance uses the nominal soft-tissue value
    // ρ_water · c_tissue ≈ 1.54 MRayl (Hill & ter Haar 2004, §2.3) sourced from SSOT.
    let z_soft_tissue = DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE;
    let velocity_x = pressure.mapv(|p| p / z_soft_tissue);

    Ok(AcousticField {
        pressure,
        velocity_x,
        velocity_y: velocity_zero.clone(),
        velocity_z: velocity_zero,
    })
}

/// Calculate acoustic absorption heating.
///
/// Returns a 3-D array of **Celsius (°C)** temperatures representing the
/// instantaneous temperature distribution after one therapy step of duration `dt`.
///
/// # Physics Model
///
/// Acoustic absorption heating: Q = α · p² / (ρ₀ · c₀)
/// where α = 0.5 Np/m (soft tissue, Nyborg 1981), ρ₀ = 1000 kg/m³, c₀ = 1540 m/s.
///
/// Temperature rise: ΔT = Q · exp(−r/L) · dt / (ρ₀ · c_p) with characteristic
/// focal length L = 10 mm and specific heat capacity c_p = 3600 J/(kg·K).
///
/// The ambient temperature T₀ = 37 °C is the baseline for CEM43 evaluation.
///
/// # Returns
///
/// 3-D array of temperatures in **degrees Celsius**, with
/// `T_min = 37.0 °C` (no heating outside focal zone) and
/// `T_max = 37.0 + ΔT_peak`.
///
/// # References
///
/// - Pennes (1948): "Analysis of tissue and arterial blood temperatures"
/// - Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"
/// - Sapareto & Dewey (1984): "Thermal dose determination in cancer therapy"
pub fn calculate_acoustic_heating(
    acoustic_field: &AcousticField,
    grid: &Grid,
    dt: f64,
    focal_depth: f64,
) -> Array3<f64> {
    // Tissue constants for soft tissue (Nyborg 1981).
    const ALPHA_NP_M: f64 = 0.5; // absorption coefficient (Np/m)
    const RHO: f64 = DENSITY_WATER_NOMINAL;
    const C0: f64 = SOUND_SPEED_TISSUE;
    let c_p = SPECIFIC_HEAT_TISSUE; // J/(kg·K) — Duck (1990) / ICRP 2002 soft tissue
    const L_FOCAL: f64 = 0.01; // focal characteristic length (10 mm)

    // Q = α p² / (ρ₀ c₀); ΔT = Q dt / (ρ₀ c_p)
    // Combined: ΔT = α p² dt / (ρ₀² c₀ c_p)
    let heating_scale = ALPHA_NP_M * dt / (RHO * RHO * C0 * c_p);

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    let mut temperature =
        Array3::<f64>::from_elem(acoustic_field.pressure.dim(), BODY_TEMPERATURE_C);

    Zip::indexed(temperature.view_mut())
        .and(acoustic_field.pressure.view())
        .par_for_each(|(i, j, k), t, &p| {
            // Radial distance from focal point (on the x-axis).
            let x = i as f64 * dx - focal_depth;
            let y = j as f64 * dy;
            let z = k as f64 * dz;
            let r = (x * x + y * y + z * z).sqrt();
            let distance_factor = (-r / L_FOCAL).exp();
            *t = BODY_TEMPERATURE_C + heating_scale * p * p * distance_factor;
        });

    temperature
}
