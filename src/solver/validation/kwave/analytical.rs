//! Analytical solutions for validation
//!
//! Provides exact analytical solutions for simple cases to validate numerical methods

use std::f64::consts::PI;

/// Collection of analytical solutions
#[derive(Debug)]
pub struct AnalyticalSolutions;

impl AnalyticalSolutions {
    /// Plane wave solution in homogeneous medium
    ///
    /// Reference: Kinsler et al., "Fundamentals of Acoustics", 4th ed., 2000
    pub fn plane_wave(x: f64, t: f64, amplitude: f64, frequency: f64, sound_speed: f64) -> f64 {
        let k = 2.0 * PI * frequency / sound_speed; // Wave number
        let omega = 2.0 * PI * frequency; // Angular frequency
        amplitude * (omega * t - k * x).sin()
    }

    /// Point source Green's function in 3D
    ///
    /// Reference: Morse & Ingard, "Theoretical Acoustics", 1968
    pub fn greens_function_3d(
        r: f64,
        t: f64,
        source_time_function: impl Fn(f64) -> f64,
        sound_speed: f64,
    ) -> f64 {
        if r == 0.0 {
            return 0.0; // Avoid singularity
        }

        let retarded_time = t - r / sound_speed;
        if retarded_time < 0.0 {
            0.0 // Causality
        } else {
            source_time_function(retarded_time) / (4.0 * PI * r)
        }
    }

    /// Standing wave in rectangular cavity
    ///
    /// Reference: Blackstock, "Fundamentals of Physical Acoustics", 2000
    pub fn cavity_mode(
        x: f64,
        y: f64,
        z: f64,
        t: f64,
        lx: f64,
        ly: f64,
        lz: f64,
        nx: usize,
        ny: usize,
        nz: usize,
        sound_speed: f64,
    ) -> f64 {
        let kx = nx as f64 * PI / lx;
        let ky = ny as f64 * PI / ly;
        let kz = nz as f64 * PI / lz;

        let omega = sound_speed * (kx * kx + ky * ky + kz * kz).sqrt();

        (kx * x).sin() * (ky * y).sin() * (kz * z).sin() * (omega * t).cos()
    }

    /// Gaussian beam solution (paraxial approximation)
    ///
    /// Reference: Ding & Zhang, "Acoustic beam propagation", JASA 2004
    pub fn gaussian_beam(x: f64, y: f64, z: f64, beam_width: f64, wavelength: f64) -> f64 {
        let k = 2.0 * PI / wavelength;
        let z_r = PI * beam_width * beam_width / wavelength; // Rayleigh range

        let r_squared = x * x + y * y;
        let w_z = beam_width * (1.0 + (z / z_r).powi(2)).sqrt();
        let r_c = z * (1.0 + (z_r / z).powi(2));
        let phase = k * z + k * r_squared / (2.0 * r_c) - (z / z_r).atan();

        (beam_width / w_z) * (-r_squared / (w_z * w_z)).exp() * phase.cos()
    }
}
