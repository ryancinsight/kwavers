//! Magnetic-resonance elastography (MRE) front end.
//!
//! MRE measures harmonic tissue motion by motion-encoding gradients: the acquired
//! MR phase `φ(r, t) = κ · u(r, t)` is proportional to displacement through the
//! encoding sensitivity `κ` [rad/m]. A wave is sampled at several equally-spaced
//! phase offsets spanning one actuation period; this module extracts the **complex
//! first-harmonic displacement field** from that stack and converts it to physical
//! displacement, producing the [`DisplacementField`] that the elastography
//! inversions (LFE, direct, phase-gradient) already consume.
//!
//! This is the acquisition-to-displacement front end; the modulus reconstruction
//! itself lives in `kwavers-solver::inverse::elastography`.
//!
//! # Extraction
//! For `N` offsets `k = 0..N` over one period, the first temporal-harmonic
//! coefficient per voxel is the single-bin DFT
//! `C = (2/N) · Σ_k φ[k] · exp(−i·2π·k/N)`, and the complex displacement is
//! `U = C / κ`. The `2/N` scaling makes `|U|` the displacement amplitude; the DC
//! (mean phase, bin 0) and higher harmonics do not contaminate the fundamental.
//!
//! # References
//! - Muthupillai, R., et al. (1995). "Magnetic resonance elastography by direct
//!   visualization of propagating acoustic strain waves." *Science* 269, 1854–1857.
//! - Manduca, A., et al. (2001). "Magnetic resonance elastography: Non-invasive
//!   mapping of tissue elasticity." *Med. Image Anal.* 5(4), 237–254.

use super::displacement::DisplacementField;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use ndarray::{Array3, Array4};

/// Extract the complex first-harmonic displacement field [m] from an MRE
/// motion-encoded phase-offset stack.
///
/// `phase_stack` has shape `(nx, ny, nz, n_offsets)` of MR phase [rad] sampled at
/// `n_offsets ≥ 2` equally-spaced wave-phase offsets over one actuation period.
/// `encoding_sensitivity_rad_per_m` is `κ` in `φ = κ·u`.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] if `n_offsets < 2` or
///   `encoding_sensitivity_rad_per_m` is non-finite/`≤ 0`.
pub fn extract_first_harmonic(
    phase_stack: &Array4<f64>,
    encoding_sensitivity_rad_per_m: f64,
) -> KwaversResult<Array3<Complex64>> {
    let (nx, ny, nz, n_offsets) = phase_stack.dim();
    if n_offsets < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "MRE extract_first_harmonic requires n_offsets >= 2, got {n_offsets}"
        )));
    }
    if !encoding_sensitivity_rad_per_m.is_finite() || encoding_sensitivity_rad_per_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "MRE encoding_sensitivity_rad_per_m must be finite and > 0, got {encoding_sensitivity_rad_per_m}"
        )));
    }

    let n = n_offsets as f64;
    // Precompute the DFT twiddle factors exp(-i·2π·k/N).
    let twiddles: Vec<Complex64> = (0..n_offsets)
        .map(|k| {
            let angle = -TWO_PI * k as f64 / n;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    let inv_kappa = 1.0 / encoding_sensitivity_rad_per_m;
    let scale = 2.0 / n;

    let mut harmonic = Array3::<Complex64>::from_elem((nx, ny, nz), Complex64::default());
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let mut acc = Complex64::new(0.0, 0.0);
                for (l, tw) in twiddles.iter().enumerate() {
                    acc += *tw * phase_stack[[i, j, k, l]];
                }
                harmonic[[i, j, k]] = acc * scale * inv_kappa;
            }
        }
    }
    Ok(harmonic)
}

/// Real displacement snapshot `Re{U·e^{iθ}}` [m] at temporal phase `θ` [rad].
#[must_use]
pub fn harmonic_snapshot(harmonic: &Array3<Complex64>, snapshot_phase_rad: f64) -> Array3<f64> {
    let rot = Complex64::new(snapshot_phase_rad.cos(), snapshot_phase_rad.sin());
    harmonic.mapv(|u| (u * rot).re)
}

/// Build a [`DisplacementField`] from a `z`-encoded MRE phase stack (the common
/// single-axis acquisition): `uz` is the real first-harmonic snapshot at `θ = 0`
/// (the in-phase component); `ux = uy = 0`.
///
/// # Errors
/// - Propagates [`extract_first_harmonic`] errors.
pub fn mre_displacement_field_z(
    uz_phase_stack: &Array4<f64>,
    encoding_sensitivity_rad_per_m: f64,
) -> KwaversResult<DisplacementField> {
    let harmonic = extract_first_harmonic(uz_phase_stack, encoding_sensitivity_rad_per_m)?;
    let (nx, ny, nz) = harmonic.dim();
    let mut field = DisplacementField::zeros(nx, ny, nz);
    field.uz = harmonic_snapshot(&harmonic, 0.0);
    Ok(field)
}

#[cfg(test)]
mod tests;
