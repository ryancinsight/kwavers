use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use moirai_parallel::{for_each_chunk_pair_mut_enumerated_with, Adaptive};

/// 2-D Green's function backpropagation from a focal point.
///
/// ```text
/// P_bwd(x, z) = exp(−i·k·r_f) / √(r_f)
/// r_f = √((x−x_f)² + (z−z_f)²)
/// k   = 2π·f / c
/// ```
/// Represents the time-reversed Green's function for a point source at (x_f, z_f).
/// Singularity at r_f = 0 is regularised to 1 pm.
///
/// Output: `(real_flat, imag_flat)` for the NX × NZ grid.
///
/// ## Parallelism
///
/// The outer loop over x-rows is embarrassingly parallel (each row writes to an
/// independent slice `[ix*nz .. (ix+1)*nz]`) and is scheduled through Moirai.
/// `f64::sin_cos` computes both sin and cos in a single instruction on x86
/// (FSINCOS / `__sincosf`).
///
/// # Arguments
/// * `x_arr`, `z_arr` – grid coordinates [m]
/// * `x_f`, `z_f` – focal point [m]
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed in the coupling medium [m/s]
///
/// # Reference
/// Baysal et al. (1983), *Geophysics* 48, 1514 (RTM formulation).
#[must_use]
pub fn backprop_green_function_2d(
    x_arr: &[f64],
    z_arr: &[f64],
    x_f: f64,
    z_f: f64,
    freq_hz: f64,
    c: f64,
) -> (Vec<f64>, Vec<f64>) {
    let k_br = TWO_PI * freq_hz / c;
    let nx = x_arr.len();
    let nz = z_arr.len();
    let mut real_out = vec![0.0_f64; nx * nz];
    let mut imag_out = vec![0.0_f64; nx * nz];

    // Each ix row writes to an independent nz-element slice → race-free.
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        &mut real_out,
        &mut imag_out,
        nz,
        |ix, re_row, im_row| {
            let x = x_arr[ix];
            for (iz, (re, im)) in re_row.iter_mut().zip(im_row.iter_mut()).enumerate() {
                let r_f = ((x - x_f).powi(2) + (z_arr[iz] - z_f).powi(2))
                    .sqrt()
                    .max(1e-12);
                let amp = 1.0 / r_f.sqrt();
                // sin_cos computes both values in one instruction on x86.
                let (sin_ph, cos_ph) = (-k_br * r_f).sin_cos();
                *re = amp * cos_ph;
                *im = amp * sin_ph;
            }
        },
    );
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
/// ## Parallelism
///
/// Outer ix loop distributes independent `[ny×nz]`-element slices through
/// Moirai. `f64::sin_cos` eliminates the separate `cos`/`sin` evaluations.
///
/// Output: `(real_flat, imag_flat)` for an NX × NY × NZ grid, row-major `[x][y][z]`.
///
/// # Reference
/// Aki & Richards (2002), *Quantitative Seismology* §4.1.
#[must_use]
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
    let scale = 1.0 / (FOUR_PI);
    let stride = ny * nz;

    // Each ix slice is [stride] elements, written independently → race-free.
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        &mut real_out,
        &mut imag_out,
        stride,
        |ix, re_slice, im_slice| {
            let x = x_arr[ix];
            let dx2 = (x - x_f).powi(2);
            for (iy, &y) in y_arr.iter().enumerate() {
                let dxy2 = (y - y_f).mul_add(y - y_f, dx2);
                for (iz, &z) in z_arr.iter().enumerate() {
                    let r = (z - z_f).mul_add(z - z_f, dxy2).sqrt().max(1e-5);
                    let amp = scale / r;
                    let (sin_ph, cos_ph) = (-k * r).sin_cos();
                    let idx = iy * nz + iz;
                    re_slice[idx] = amp * cos_ph;
                    im_slice[idx] = amp * sin_ph;
                }
            }
        },
    );
    (real_out, imag_out)
}
