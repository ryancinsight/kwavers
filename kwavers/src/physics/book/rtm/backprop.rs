use std::f64::consts::PI;

/// 2-D Green's function backpropagation from a focal point.
///
/// ```text
/// P_bwd(x, z) = exp(−i·k·r_f) / √(r_f)
/// r_f = √((x−x_f)² + (z−z_f)²)
/// ```
/// Represents the time-reversed Green's function for a point source at (x_f, z_f).
/// Singularity at r_f = 0 is regularised by a small offset.
///
/// Output: `(real_flat, imag_flat)` for the NX × NZ grid.
///
/// # Arguments
/// * `x_arr`, `z_arr` – grid coordinates [m]
/// * `x_f`, `z_f` – focal point [m]
/// * `k_br` – wavenumber in brain [rad/m]
///
/// # Reference
/// Baysal et al. (1983), *Geophysics* 48, 1514 (RTM formulation).
pub fn backprop_green_function_2d(
    x_arr: &[f64],
    z_arr: &[f64],
    x_f: f64,
    z_f: f64,
    k_br: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nx = x_arr.len();
    let nz = z_arr.len();
    let mut real_out = vec![0.0_f64; nx * nz];
    let mut imag_out = vec![0.0_f64; nx * nz];

    for (ix, &x) in x_arr.iter().enumerate() {
        for (iz, &z) in z_arr.iter().enumerate() {
            let r_f = ((x - x_f).powi(2) + (z - z_f).powi(2)).sqrt().max(1e-12);
            let phase = -k_br * r_f;
            let amp = 1.0 / r_f.sqrt();
            let idx = ix * nz + iz;
            real_out[idx] = amp * phase.cos();
            imag_out[idx] = amp * phase.sin();
        }
    }
    (real_out, imag_out)
}

/// 3-D free-space Green's function backpropagation from a focal point.
///
/// ```text
/// G(x,y,z) = exp(−i·k·r) / (4π·r)
/// r = √((x−x_f)² + (y−y_f)² + (z−z_f)²)
/// ```
/// The `1/r` amplitude law is the 3-D spherical-wave decay (cf. `1/√r` in 2-D).
/// Singularity at `r = 0` is regularised by a 10 μm floor.
///
/// Output: `(real_flat, imag_flat)` for an NX × NY × NZ grid, row-major `[x][y][z]`.
///
/// # Reference
/// Aki & Richards (2002), *Quantitative Seismology* §4.1.
pub fn backprop_green_function_3d(
    x_arr: &[f64],
    y_arr: &[f64],
    z_arr: &[f64],
    x_f: f64,
    y_f: f64,
    z_f: f64,
    k: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nx = x_arr.len();
    let ny = y_arr.len();
    let nz = z_arr.len();
    let n = nx * ny * nz;
    let mut real_out = vec![0.0_f64; n];
    let mut imag_out = vec![0.0_f64; n];
    let scale = 1.0 / (4.0 * PI);

    for (ix, &x) in x_arr.iter().enumerate() {
        for (iy, &y) in y_arr.iter().enumerate() {
            for (iz, &z) in z_arr.iter().enumerate() {
                let r = ((x - x_f).powi(2) + (y - y_f).powi(2) + (z - z_f).powi(2))
                    .sqrt()
                    .max(1e-5);
                let amp = scale / r;
                let phase = -k * r;
                let idx = (ix * ny + iy) * nz + iz;
                real_out[idx] = amp * phase.cos();
                imag_out[idx] = amp * phase.sin();
            }
        }
    }
    (real_out, imag_out)
}
