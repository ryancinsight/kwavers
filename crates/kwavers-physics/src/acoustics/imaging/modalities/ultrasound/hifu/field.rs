//! Focused HIFU acoustic field calculation.
//!
//! The field model uses a monochromatic Rayleigh-Sommerfeld aperture integral
//! with phase delays chosen so the aperture contributions are in phase at the
//! geometric focus. For a surface particle-velocity amplitude `v0`,
//!
//! ```text
//! p(r) = rho c k v0 / (2 pi) * | integral_A exp(-i k (|r-s| - |f-s|)) / |r-s| dA |
//! v0 = sqrt(2 P / (rho c A))
//! ```
//!
//! where `P` is temporal-average acoustic power and `A = pi a^2`.
//!
//! References: O'Neil (1949), J. Acoust. Soc. Am. 21(5), 516-526;
//! Rayleigh (1896), *The Theory of Sound*, Vol. II.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::hifu::DomainHIFUTransducer;
use kwavers_medium::Medium;
use leto::Array3;
use std::f64::consts::PI;

const RADIAL_APERTURE_SAMPLES: usize = 4;
const ANGULAR_APERTURE_SAMPLES: usize = 16;

#[derive(Debug, Clone, Copy)]
struct ApertureSample {
    x: f64,
    y: f64,
    distance_to_focus: f64,
    area_weight: f64,
}

#[derive(Debug, Clone, Copy)]
struct FieldScale {
    source_pressure_factor: f64,
    wavenumber: f64,
}

/// Compute peak pressure amplitude [Pa] for a focused HIFU transducer.
///
/// The transducer aperture is centered laterally in the computational grid and
/// lies on the `z = 0` plane; the geometric focus is at
/// `(x_center, y_center, focal_length)`.
///
/// # Errors
/// Returns an error when transducer or reference-medium parameters are
/// non-finite or non-positive.
pub fn compute_pressure_field(
    transducer: &DomainHIFUTransducer,
    grid: &Grid,
    medium: &dyn Medium,
) -> KwaversResult<Array3<f64>> {
    let scale = validate_field_inputs(transducer, grid, medium)?;
    let aperture = aperture_samples(transducer.aperture_radius, transducer.focal_length);
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::zeros((nx, ny, nz));
    let min_distance = 0.5 * grid.dx.min(grid.dy).min(grid.dz);

    for k in 0..nz {
        let z = k as f64 * grid.dz;
        for j in 0..ny {
            let y = lateral_coordinate(j, ny, grid.dy);
            for i in 0..nx {
                let x = lateral_coordinate(i, nx, grid.dx);
                pressure[[i, j, k]] = pressure_at_point(x, y, z, min_distance, &aperture, scale);
            }
        }
    }

    Ok(pressure)
}

/// Compute temporal-average intensity [W/m^2] from the HIFU peak pressure field.
///
/// For a harmonic acoustic pressure amplitude, `I = p_peak^2 / (2 rho c)`.
///
/// # Errors
/// Propagates pressure-field validation errors.
pub fn compute_intensity_field(
    transducer: &DomainHIFUTransducer,
    grid: &Grid,
    medium: &dyn Medium,
) -> KwaversResult<Array3<f64>> {
    let pressure = compute_pressure_field(transducer, grid, medium)?;
    let rho = medium.density(grid.nx / 2, grid.ny / 2, 0);
    let c = medium.sound_speed(grid.nx / 2, grid.ny / 2, 0);
    let impedance = rho * c;

    Ok(pressure.mapv(|p| p * p / (2.0 * impedance)))
}

fn validate_field_inputs(
    transducer: &DomainHIFUTransducer,
    grid: &Grid,
    medium: &dyn Medium,
) -> KwaversResult<FieldScale> {
    validate_positive("frequency", transducer.frequency)?;
    validate_positive("acoustic_power", transducer.acoustic_power)?;
    validate_positive("focal_length", transducer.focal_length)?;
    validate_positive("aperture_radius", transducer.aperture_radius)?;

    let density = medium.density(grid.nx / 2, grid.ny / 2, 0);
    let sound_speed = medium.sound_speed(grid.nx / 2, grid.ny / 2, 0);
    validate_positive("reference_density", density)?;
    validate_positive("reference_sound_speed", sound_speed)?;

    let aperture_area = PI * transducer.aperture_radius.powi(2);
    let velocity_amplitude =
        (2.0 * transducer.acoustic_power / (density * sound_speed * aperture_area)).sqrt();
    let wavenumber = TWO_PI * transducer.frequency / sound_speed;
    let source_pressure_factor = density * sound_speed * wavenumber * velocity_amplitude / (TWO_PI);

    Ok(FieldScale {
        source_pressure_factor,
        wavenumber,
    })
}

fn validate_positive(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "HIFU {name} must be finite and positive, got {value}"
        )))
    }
}

fn aperture_samples(radius: f64, focal_length: f64) -> Vec<ApertureSample> {
    let n_samples = RADIAL_APERTURE_SAMPLES * ANGULAR_APERTURE_SAMPLES;
    let area_weight = PI * radius.powi(2) / n_samples as f64;
    let mut samples = Vec::with_capacity(n_samples);

    for radial in 0..RADIAL_APERTURE_SAMPLES {
        let r = radius * ((radial as f64 + 0.5) / RADIAL_APERTURE_SAMPLES as f64).sqrt();
        for angular in 0..ANGULAR_APERTURE_SAMPLES {
            let theta = TWO_PI * (angular as f64 + 0.5) / ANGULAR_APERTURE_SAMPLES as f64;
            let x = r * theta.cos();
            let y = r * theta.sin();
            let distance_to_focus = (r * r + focal_length * focal_length).sqrt();
            samples.push(ApertureSample {
                x,
                y,
                distance_to_focus,
                area_weight,
            });
        }
    }

    samples
}

fn pressure_at_point(
    x: f64,
    y: f64,
    z: f64,
    min_distance: f64,
    aperture: &[ApertureSample],
    scale: FieldScale,
) -> f64 {
    let mut real = 0.0;
    let mut imag = 0.0;

    for sample in aperture {
        let dx = x - sample.x;
        let dy = y - sample.y;
        let distance = (dx * dx + dy * dy + z * z).sqrt().max(min_distance);
        let phase = -scale.wavenumber * (distance - sample.distance_to_focus);
        let contribution = sample.area_weight / distance;
        real += contribution * phase.cos();
        imag += contribution * phase.sin();
    }

    scale.source_pressure_factor * real.hypot(imag)
}

#[inline]
fn lateral_coordinate(index: usize, count: usize, spacing: f64) -> f64 {
    (index as f64 - 0.5 * (count.saturating_sub(1)) as f64) * spacing
}
