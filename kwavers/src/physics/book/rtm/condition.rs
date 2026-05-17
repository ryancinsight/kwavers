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
        let val = p_fwd_real[i] * p_bwd_real[i] + p_fwd_imag[i] * p_bwd_imag[i];
        img[i] = val.max(0.0);
    }
    let max_val = img.iter().cloned().fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        img.iter_mut().for_each(|v| *v /= max_val);
    }
    img
}

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

/// Source-normalised RTM imaging condition.
///
/// Removes the source-amplitude footprint by dividing the cross-correlation by
/// the forward-field energy at each image point:
/// ```text
/// I_SN(x) = Re[P_fwd · conj(P_bwd)] / (|P_fwd|² + ε)
/// ```
/// where `ε = stab_frac · max_x(|P_fwd(x)|²)` prevents division by zero in
/// skull shadow zones. Result is normalised to `[0, 1]`.
///
/// This condition substantially reduces skull-reflection artefacts and
/// low-illumination bias that afflict the plain cross-correlation condition
/// for transcranial 1024-element apertures.
///
/// # Arguments
/// * `p_fwd_real`, `p_fwd_imag` – forward field (element source or encoded stack)
/// * `p_bwd_real`, `p_bwd_imag` – backward (receiver-injected adjoint) field
/// * `stab_frac` – stabilisation fraction (typical: 1e-4)
///
/// # Reference
/// Guitton, Valenciano & Bevc (2007), *Geophysics* 72, S35.
/// Whitmore (1983), *SEG Annual Meeting* 827.
pub fn rtm_source_normalized_condition(
    p_fwd_real: &[f64],
    p_fwd_imag: &[f64],
    p_bwd_real: &[f64],
    p_bwd_imag: &[f64],
    stab_frac: f64,
) -> Vec<f64> {
    let n = p_fwd_real.len();
    let max_fwd_energy = p_fwd_real
        .iter()
        .zip(p_fwd_imag.iter())
        .map(|(&re, &im)| re * re + im * im)
        .fold(0.0_f64, f64::max);
    let eps = stab_frac * max_fwd_energy;
    let mut img = vec![0.0_f64; n];
    for i in 0..n {
        let cross = p_fwd_real[i] * p_bwd_real[i] + p_fwd_imag[i] * p_bwd_imag[i];
        let fwd_e = p_fwd_real[i] * p_fwd_real[i] + p_fwd_imag[i] * p_fwd_imag[i];
        img[i] = (cross / (fwd_e + eps)).max(0.0);
    }
    let max_val = img.iter().cloned().fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        img.iter_mut().for_each(|v| *v /= max_val);
    }
    img
}

/// Fuse RTM images from multiple aperture elements with per-element weights.
///
/// ```text
/// I_W(x) = Σ_s w_s · I_s(x) / Σ_s w_s
/// ```
/// Weights can encode element solid angle, CT-derived skull transmission, or
/// aperture-diversity scores. All-zero weight vector falls back to uniform mean.
///
/// # Reference
/// Margrave (2003), *CREWES Research Report* §3.2 (aperture weighting in migration).
pub fn rtm_aperture_weighted_fusion(images: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    if images.is_empty() {
        return Vec::new();
    }
    assert_eq!(
        images.len(),
        weights.len(),
        "image count must equal weight count"
    );
    let n = images[0].len();
    let w_sum: f64 = weights.iter().sum();
    let uniform = w_sum <= 0.0;
    let w_total = if uniform { weights.len() as f64 } else { w_sum };
    let mut out = vec![0.0_f64; n];
    for (img, &w) in images.iter().zip(weights.iter()) {
        assert_eq!(img.len(), n, "all images must have the same length");
        let w_eff = if uniform { 1.0 } else { w };
        for (o, &v) in out.iter_mut().zip(img.iter()) {
            *o += w_eff * v;
        }
    }
    out.iter_mut().for_each(|v| *v /= w_total);
    out
}
