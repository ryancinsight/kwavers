//! Reverse-time migration (RTM) and adaptive beamforming physics for
//! book chapter ch25.
//!
//! Covers: focused Gaussian beam 2-D field, Green's function backpropagation,
//! RTM imaging condition, multi-frequency fusion, temporal modulation
//! frequency schedule, and standing-wave suppression gain.

use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Focused Gaussian beam ────────────────────────────────────────────────────

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
    let z_r = PI * w0_m * w0_m * freq_hz / c_brain; // Rayleigh range
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
            // Forward propagation phase
            let phase_fwd = k_br * dz;
            let p_fwd = Complex64::new(gauss * phase_fwd.cos(), gauss * phase_fwd.sin());
            // Standing-wave factor SW(z) = 1 + R_back·exp(2i·k_br·(z_back−z))
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

// ─── Green's function backpropagation ────────────────────────────────────────

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
            let r_f = ((x - x_f).powi(2) + (z - z_f).powi(2))
                .sqrt()
                .max(1e-12); // regularise singularity
            let phase = -k_br * r_f;
            let amp = 1.0 / r_f.sqrt();
            let idx = ix * nz + iz;
            real_out[idx] = amp * phase.cos();
            imag_out[idx] = amp * phase.sin();
        }
    }
    (real_out, imag_out)
}

// ─── RTM imaging condition ────────────────────────────────────────────────────

/// RTM cross-correlation imaging condition.
///
/// ```text
/// I(x, z) = Re[P_fwd(x,z) · conj(P_bwd(x,z))],  clipped to ≥ 0
/// ```
/// Normalised so that the maximum value is 1.0.
///
/// # Arguments
/// * `p_fwd_real`, `p_fwd_imag` – forward field [NX × NZ, row-major]
/// * `p_bwd_real`, `p_bwd_imag` – backward field [NX × NZ, row-major]
/// * `nx`, `nz` – grid dimensions
///
/// # Reference
/// Claerbout (1971), *Geophysics* 36, 467.
pub fn rtm_imaging_condition(
    p_fwd_real: &[f64],
    p_fwd_imag: &[f64],
    p_bwd_real: &[f64],
    p_bwd_imag: &[f64],
    nx: usize,
    nz: usize,
) -> Vec<f64> {
    let n = nx * nz;
    let mut img = vec![0.0_f64; n];
    for i in 0..n {
        // Re[P_fwd · conj(P_bwd)] = P_fwd_re·P_bwd_re + P_fwd_im·P_bwd_im
        let val = p_fwd_real[i] * p_bwd_real[i] + p_fwd_imag[i] * p_bwd_imag[i];
        img[i] = val.max(0.0);
    }
    let max_val = img.iter().cloned().fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        img.iter_mut().for_each(|v| *v /= max_val);
    }
    img
}

// ─── Multi-frequency fusion ────────────────────────────────────────────────────

/// Fuse multiple RTM images by pixel-wise mean.
///
/// Each image in `images` must have the same length. Returns the
/// element-wise arithmetic mean.
///
/// # Reference
/// Marty et al. (2021), *Phys. Rev. Applied* 15, 024061.
pub fn rtm_multi_frequency_fusion(images: &[Vec<f64>]) -> Vec<f64> {
    if images.is_empty() {
        return Vec::new();
    }
    let n = images[0].len();
    let m = images.len() as f64;
    let mut out = vec![0.0_f64; n];
    for img in images {
        assert_eq!(img.len(), n, "all images must have the same length");
        for (o, &v) in out.iter_mut().zip(img.iter()) {
            *o += v;
        }
    }
    out.iter_mut().for_each(|v| *v /= m);
    out
}

// ─── Temporal modulation frequencies ──────────────────────────────────────────

/// Temporal modulation frequency schedule for standing-wave suppression.
///
/// ```text
/// f_m = f₀ + m · c / (2·M·d_back)   for m = 0..M-1
/// ```
/// Each frequency shifts the standing-wave pattern by one lobe-width, so
/// coherent averaging over M frequencies cancels the standing-wave modulation.
///
/// # Arguments
/// * `f0_hz` – base frequency [Hz]
/// * `m_steps` – number of frequencies M
/// * `c` – sound speed [m/s]
/// * `d_back_m` – distance to the back-reflecting wall [m]
///
/// # Reference
/// Dencks & Schmitz (2005), *Ultrasonics* 43, 183.
pub fn temporal_modulation_frequencies(
    f0_hz: f64,
    m_steps: usize,
    c: f64,
    d_back_m: f64,
) -> Vec<f64> {
    let df = c / (2.0 * m_steps as f64 * d_back_m);
    (0..m_steps).map(|m| f0_hz + m as f64 * df).collect()
}

// ─── Standing-wave suppression gain ──────────────────────────────────────────

/// Analytical gain factor for standing-wave suppression by RTM.
///
/// For a back-reflection coefficient R_back, the peak-to-trough ratio of the
/// standing-wave modulation is `(1 + R_back)²/(1 + R_back²)`. Perfect RTM
/// suppression leaves only the smooth background; the relative gain is:
/// ```text
/// G = (1 + R_back)² / (1 + R_back²)
/// ```
///
/// # Reference
/// Thomas et al. (2017), *Phys. Rev. Lett.* 119, 034301.
#[inline]
pub fn standing_wave_suppression_gain(r_back: f64) -> f64 {
    (1.0 + r_back).powi(2) / (1.0 + r_back * r_back)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_beam_size() {
        let x = vec![-1e-3, 0.0, 1e-3];
        let z = vec![0.0, 5e-3, 10e-3];
        let t = Complex64::new(1.0, 0.0);
        let (re, im) = focused_gaussian_beam_2d(&x, &z, 0.0, 5e-3, 1e6, 1500.0, 1e-3, t, 0.0, 0.1);
        assert_eq!(re.len(), 9);
        assert_eq!(im.len(), 9);
    }

    #[test]
    fn gaussian_beam_peak_at_focus() {
        let x = vec![0.0];
        let z: Vec<f64> = vec![-5e-3, 0.0, 5e-3];
        let t = Complex64::new(1.0, 0.0);
        let (re, _) = focused_gaussian_beam_2d(&x, &z, 0.0, 0.0, 1e6, 1500.0, 1e-3, t, 0.0, 0.1);
        // At x=0, z=z_f: w=w0, amp = w0/w0 * exp(0) = 1 × |T|; other z smaller
        assert!(re[1].abs() >= re[0].abs().min(re[2].abs()));
    }

    #[test]
    fn backprop_normalisation() {
        let x = vec![0.0];
        let z = vec![0.01, 0.02];
        let (re, _) = backprop_green_function_2d(&x, &z, 0.0, 0.0, 1000.0);
        // Closer point (z=0.01) should have larger magnitude (1/sqrt(r))
        let mag0 = re[0].abs();
        let mag1 = re[1].abs();
        assert!(mag0 > mag1, "mag0={} mag1={}", mag0, mag1);
    }

    #[test]
    fn rtm_imaging_normalised_max_one() {
        let fwd_r = vec![1.0, 2.0, 3.0];
        let fwd_i = vec![0.0, 0.0, 0.0];
        let bwd_r = vec![1.0, 1.0, 1.0];
        let bwd_i = vec![0.0, 0.0, 0.0];
        let img = rtm_imaging_condition(&fwd_r, &fwd_i, &bwd_r, &bwd_i, 1, 3);
        let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn multi_freq_fusion_mean() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![2.0, 1.0, 0.0];
        let fused = rtm_multi_frequency_fusion(&[a, b]);
        assert_eq!(fused, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn modulation_frequencies_increasing() {
        let f = temporal_modulation_frequencies(1e6, 5, 1500.0, 0.1);
        assert_eq!(f.len(), 5);
        for i in 1..f.len() {
            assert!(f[i] > f[i - 1]);
        }
    }

    #[test]
    fn suppression_gain_no_reflection_is_one() {
        // R_back = 0 → G = 1/(1) = 1
        assert!((standing_wave_suppression_gain(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn suppression_gain_increases_with_reflection() {
        let g1 = standing_wave_suppression_gain(0.2);
        let g2 = standing_wave_suppression_gain(0.5);
        assert!(g1 > 1.0 && g2 > g1);
    }
}
