use ndarray::Array3;

/// Calculate acoustic energy density at a point
///
/// e = (ρ₀/2)|u|² + p²/(2ρ₀c₀²)
#[inline]
#[must_use]
pub fn acoustic_energy_density(
    pressure: f64,
    velocity: (f64, f64, f64),
    density: f64,
    sound_speed: f64,
) -> f64 {
    let kinetic = 0.5
        * density
        * velocity.2.mul_add(
            velocity.2,
            velocity.1.mul_add(velocity.1, velocity.0.powi(2)),
        );
    let potential = pressure.powi(2) / (2.0 * density * sound_speed.powi(2));
    kinetic + potential
}

/// Calculate acoustic intensity (magnitude of energy flux)
///
/// I = p·u (W/m²)
#[inline]
#[must_use]
pub fn acoustic_intensity(pressure: f64, velocity: (f64, f64, f64)) -> f64 {
    let flux_x = pressure * velocity.0;
    let flux_y = pressure * velocity.1;
    let flux_z = pressure * velocity.2;
    flux_z
        .mul_add(flux_z, flux_y.mul_add(flux_y, flux_x.powi(2)))
        .sqrt()
}

/// Integrate field over volume (trapezoidal rule)
#[must_use]
pub fn integrate_field(field: &Array3<f64>, dx: f64, dy: f64, dz: f64) -> f64 {
    let dv = dx * dy * dz;
    field.sum() * dv
}

/// Calculate RMS (root mean square) of field
#[must_use]
pub fn field_rms(field: &Array3<f64>) -> f64 {
    let n = field.len() as f64;
    let sum_squares = field.iter().map(|&x| x * x).sum::<f64>();
    (sum_squares / n).sqrt()
}
