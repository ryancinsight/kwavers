use num_complex::Complex64;
use std::f64::consts::PI;

/// 2-D focused Gaussian beam field in the presence of a skull layer and
/// a back-reflecting surface (e.g. contralateral skull).
///
/// ```text
/// P(x, z) = T_skull · (w₀/w(z)) · exp(−x²/w(z)²)
///           · exp(i·k_br·(z−z_f))
///           · SW(z)
/// w(z) = w₀·√(1 + ((z−z_f)/z_R)²)
/// z_R = π·w₀²·f/c_brain
/// SW(z) = 1 + R_back·exp(2i·k_br·(z_back − z))
/// ```
///
/// Output: two flattened row-major Vecs (real, imag) of size NX × NZ.
///
/// # Arguments
/// * `x_arr`, `z_arr` – grid coordinates [m]
/// * `x_f`, `z_f` – focal point [m]
/// * `freq_hz` – frequency [Hz]
/// * `c_brain` – sound speed in brain [m/s]
/// * `w0_m` – beam waist at focus [m]
/// * `skull_transmission` – complex transmission coefficient T (from transfer matrix)
/// * `r_back` – back-wall pressure reflection coefficient (real scalar)
/// * `z_back` – axial position of the back wall [m]
///
/// # Reference
/// Pinton et al. (2012), *IEEE Trans. Ultrason.* 59, 1302;
/// Salahura et al. (2020), *Phys. Med. Biol.* 65, 115006.
pub fn focused_gaussian_beam_2d(
    x_arr: &[f64],
    z_arr: &[f64],
    x_f: f64,
    z_f: f64,
    freq_hz: f64,
    c_brain: f64,
    w0_m: f64,
    skull_transmission: Complex64,
    r_back: f64,
    z_back: f64,
) -> (Vec<f64>, Vec<f64>) {
    let k_br = 2.0 * PI * freq_hz / c_brain;
    let z_r = PI * w0_m * w0_m * freq_hz / c_brain;
    let nx = x_arr.len();
    let nz = z_arr.len();
    let mut real_out = vec![0.0_f64; nx * nz];
    let mut imag_out = vec![0.0_f64; nx * nz];

    for (ix, &x) in x_arr.iter().enumerate() {
        let dx = x - x_f;
        for (iz, &z) in z_arr.iter().enumerate() {
            let dz = z - z_f;
            let w = w0_m * (1.0 + (dz / z_r).powi(2)).sqrt();
            let gauss = (w0_m / w) * (-(dx * dx) / (w * w)).exp();
            let phase_fwd = k_br * dz;
            let p_fwd = Complex64::new(gauss * phase_fwd.cos(), gauss * phase_fwd.sin());
            let sw_phase = 2.0 * k_br * (z_back - z);
            let sw = Complex64::new(1.0 + r_back * sw_phase.cos(), r_back * sw_phase.sin());
            let field = skull_transmission * p_fwd * sw;
            let idx = ix * nz + iz;
            real_out[idx] = field.re;
            imag_out[idx] = field.im;
        }
    }
    (real_out, imag_out)
}
