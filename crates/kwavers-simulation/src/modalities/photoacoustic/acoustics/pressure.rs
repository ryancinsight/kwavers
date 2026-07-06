//! Initial pressure computation for photoacoustic imaging.
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
//! - `p₀`: Initial pressure (Pa)
//! - `Γ`: Grüneisen parameter (thermoelastic efficiency) (dimensionless)
//! - `μₐ`: Optical absorption coefficient (m^-1)
//! - `Φ`: Optical fluence (J/m^2)
//! - `λ`: Wavelength (nm)
//!
//! ### Wavelength-Dependent Grüneisen Parameter
//!
//! ```text
//! Γ(λ) = Γ₀ · s(λ)
//! ```
//!
//! Where:
//! - Visible range (λ < 600nm): s(λ) = 1.0
//! - Near-IR window (600-800nm): s(λ) = 0.9 - 0.0005(λ - 600)
//! - Far-IR range (λ > 800nm): s(λ) = 0.8 - 0.0002(λ - 800)
//!
//! ## References
//!
//! - Wang et al. (2009): "Photoacoustic tomography: in vivo imaging from organelles to organs"
//! - Xu & Wang (2006): "Photoacoustic imaging in biomedicine"

use kwavers_core::constants::thermodynamic::GRUNEISEN_WATER_20C;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::photoacoustic::InitialPressure;
use kwavers_medium::properties::OpticalPropertyData;
use leto::Array3;
use ndarray::Array3 as NdArray3;

/// Compute initial pressure distribution from optical absorption.
///
/// Implements the photoacoustic generation theorem: p₀(r) = Γ · μₐ(r) · Φ(r)
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn compute_initial_pressure(
    grid: &Grid,
    optical_properties: &NdArray3<OpticalPropertyData>,
    fluence: &Array3<f64>,
    gruneisen_parameters: &[f64],
    wavelengths: &[f64],
) -> KwaversResult<InitialPressure> {
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::zeros([nx, ny, nz]);
    let mut max_pressure: f64 = 0.0;

    let operating_wavelength = wavelengths.first().copied().unwrap_or(750.0);

    let wavelength_scaling = if operating_wavelength < 600.0 {
        1.0
    } else if operating_wavelength < 800.0 {
        (operating_wavelength - 600.0).mul_add(-0.0005, 0.9)
    } else {
        (operating_wavelength - 800.0).mul_add(-0.0002, 0.8)
    };

    let base_gruneisen = gruneisen_parameters
        .first()
        .copied()
        .unwrap_or(GRUNEISEN_WATER_20C);
    let gruneisen_parameter = base_gruneisen * wavelength_scaling;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let props = &optical_properties[[i, j, k]];
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

/// Compute multi-wavelength initial pressure distributions.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn compute_multi_wavelength_pressure(
    grid: &Grid,
    optical_properties: &NdArray3<OpticalPropertyData>,
    fluence_fields: &[Array3<f64>],
    gruneisen_parameters: &[f64],
    wavelengths: &[f64],
) -> KwaversResult<Vec<InitialPressure>> {
    fluence_fields
        .iter()
        .enumerate()
        .map(|(idx, fluence)| {
            let gruneisen = gruneisen_parameters
                .get(idx)
                .copied()
                .unwrap_or(GRUNEISEN_WATER_20C);
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
