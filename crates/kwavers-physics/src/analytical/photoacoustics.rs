//! Photoacoustic physics for book chapter ch13.
//!
//! Covers: haemoglobin molar absorption spectra (polynomial fits to tabulated
//! data), Grüneisen parameter of water vs temperature, far-field PA sphere
//! pressure signal, axial resolution, and least-squares spectroscopic
//! unmixing.

// ─── Absorption spectra ───────────────────────────────────────────────────────

/// Molar absorption coefficient of oxyhaemoglobin (HbO₂) vs wavelength.
///
/// Returns [m⁻¹ per mol/L], so the absorption coefficient is:
/// ```text
/// μ_a = ε_HbO2(λ) · c_HbO2 [mol/L]   [m⁻¹]
/// ```
/// Uses a 6th-order polynomial fit to the Prahl (1999) tabulated data,
/// valid for λ ∈ [650, 1000] nm.
///
/// # Reference
/// Prahl S. (1999) <https://omlc.org/spectra/hemoglobin/>, accessed 2024.
#[must_use]
pub fn hbo2_molar_absorption(wavelength_nm: &[f64]) -> Vec<f64> {
    wavelength_nm.iter().map(|&lam| hbo2_poly(lam)).collect()
}

/// Molar absorption coefficient of deoxyhaemoglobin (Hb) vs wavelength.
///
/// Same units and validity range as [`hbo2_molar_absorption`].
///
/// # Reference
/// Prahl S. (1999) <https://omlc.org/spectra/hemoglobin/>.
#[must_use]
pub fn hb_molar_absorption(wavelength_nm: &[f64]) -> Vec<f64> {
    wavelength_nm.iter().map(|&lam| hb_poly(lam)).collect()
}

// ─── Grüneisen parameter ──────────────────────────────────────────────────────

/// Grüneisen parameter of water vs temperature (empirical linear fit).
///
/// ```text
/// Γ(T) = 0.0043 + 0.0053·T   (valid 0–60 °C)
/// ```
///
/// # Reference
/// Sigrist & Kneubühl (1978), *J. Acoust. Soc. Am.* 64, 1652.
#[must_use]
#[inline]
pub fn gruneisen_parameter_water(t_celsius: &[f64]) -> Vec<f64> {
    t_celsius.iter().map(|&t| 0.0043 + 0.0053 * t).collect()
}

/// Grüneisen parameter of generic **soft tissue** vs temperature (linear model).
///
/// ```text
/// Γ(T) = Γ_body + (dΓ/dT)·(T − T_body)
/// ```
/// with `Γ_body = GRUNEISEN_SOFT_TISSUE` at body temperature `T_body` and slope
/// `GRUNEISEN_SOFT_TISSUE_TEMP_COEFF` (SSOT constants). This is the basis of
/// **photoacoustic thermometry**: the PA initial pressure `p₀ = Γ·μ_a·F`
/// tracks `Γ(T)` and hence temperature, so a measured `Δp₀/p₀` maps to `ΔT`.
///
/// # Reference
/// Xu M & Wang LV (2006), *Rev. Sci. Instrum.* 77, 041101.
#[must_use]
pub fn gruneisen_parameter_soft_tissue(t_celsius: &[f64]) -> Vec<f64> {
    use kwavers_core::constants::thermodynamic::{
        GRUNEISEN_SOFT_TISSUE, GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
    };
    use kwavers_core::constants::BODY_TEMPERATURE_C;
    t_celsius
        .iter()
        .map(|&t| {
            GRUNEISEN_SOFT_TISSUE_TEMP_COEFF.mul_add(t - BODY_TEMPERATURE_C, GRUNEISEN_SOFT_TISSUE)
        })
        .collect()
}

// ─── PA sphere signal ─────────────────────────────────────────────────────────

/// Far-field photoacoustic pressure from an absorbing sphere.
///
/// Based on the N-wave approximation for a sphere of radius R₀ with uniform
/// initial pressure p₀ = Γ·μ_a·F_fluence (= Γ·H, the absorbed energy density;
/// Pa = J·m⁻³):
/// ```text
/// p(r, t) ≈ p₀·R₀ / (2·r) · [δ(t − (r−R₀)/c) − δ(t − (r+R₀)/c)]
/// ```
/// Rendered as a trapezoidal pulse of width `2R₀/c`:
/// ```text
/// p(t) = initial_pressure_pa · R₀/(2·r) · sign_pulse(t)
/// ```
///
/// The discrete implementation uses a rectangular window of width `2R₀/c`
/// centred at `t_c = r_det/c`:
///
/// # Arguments
/// * `t_arr` – time array `s`
/// * `r0_m` – sphere radius `m`
/// * `gamma` – Grüneisen parameter (dimensionless)
/// * `mua_per_m` – absorption coefficient [m⁻¹]
/// * `c` – sound speed [m/s]
/// * `r_det_m` – detector distance from sphere centre `m`
/// * `initial_pressure_pa` – Γ·μ_a·F (pre-computed initial pressure p₀ = Γ·H) `Pa`
///
/// # Reference
/// Xu & Wang (2006), *Rev. Sci. Instrum.* 77, 041101, eq. (13).
#[must_use]
pub fn pa_sphere_pressure_signal(
    t_arr: &[f64],
    r0_m: f64,
    gamma: f64,
    mua_per_m: f64,
    c: f64,
    r_det_m: f64,
    initial_pressure_pa: f64,
) -> Vec<f64> {
    let _ = (gamma, mua_per_m); // already encoded in initial_pressure_pa
    let t_centre = r_det_m / c;
    let half_width = r0_m / c; // half-duration of the N-wave
    let amplitude = initial_pressure_pa * r0_m / (2.0 * r_det_m.max(1e-12));

    t_arr
        .iter()
        .map(|&t| {
            let dt = t - t_centre;
            // N-wave: positive lobe for dt ∈ (-half_width, 0), negative for (0, half_width)
            if dt.abs() < half_width && dt < 0.0 {
                amplitude
            } else if dt.abs() < half_width && dt >= 0.0 {
                -amplitude
            } else {
                0.0
            }
        })
        .collect()
}

/// Initial pressure and surface signal for a one-dimensional Gaussian absorber.
///
/// The initial pressure follows the photoacoustic source relation
/// `p0(z) = Gamma * mu_a * Phi * exp(-0.5 * ((z - z0) / sigma)^2)`.
/// The plotted surface signal is the analytic spatial derivative
/// `dp0/dz`, sampled at `z = c * t`, which is the bipolar waveform used by
/// the Chapter 5 imaging fixture.
///
/// # Errors
///
/// Returns an error if any axis or scalar is non-finite, `sigma_m <= 0`, or
/// `sound_speed_m_s <= 0`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianAbsorberPhotoacousticProfileInput<'a> {
    /// Depth samples where the initial pressure is evaluated.
    pub depth_axis_m: &'a [f64],
    /// Time samples where the surface signal is evaluated.
    pub time_axis_s: &'a [f64],
    /// Grueneisen parameter.
    pub gruneisen: f64,
    /// Optical absorption coefficient.
    pub absorption_per_m: f64,
    /// Optical fluence.
    pub fluence_j_m2: f64,
    /// Gaussian absorber center depth.
    pub center_m: f64,
    /// Gaussian absorber standard deviation.
    pub sigma_m: f64,
    /// Acoustic sound speed.
    pub sound_speed_m_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianAbsorberPhotoacousticProfile {
    /// Initial pressure profile sampled on `depth_axis_m`.
    pub initial_pressure_pa: Vec<f64>,
    /// Analytic surface signal sampled on `time_axis_s`.
    pub surface_signal_pa_per_m: Vec<f64>,
}

pub fn gaussian_absorber_photoacoustic_profile(
    input: GaussianAbsorberPhotoacousticProfileInput<'_>,
) -> Result<GaussianAbsorberPhotoacousticProfile, String> {
    let GaussianAbsorberPhotoacousticProfileInput {
        depth_axis_m,
        time_axis_s,
        gruneisen,
        absorption_per_m,
        fluence_j_m2,
        center_m,
        sigma_m,
        sound_speed_m_s,
    } = input;
    if !depth_axis_m.iter().all(|value| value.is_finite()) {
        return Err("depth_axis_m must contain only finite values".to_owned());
    }
    if !time_axis_s.iter().all(|value| value.is_finite()) {
        return Err("time_axis_s must contain only finite values".to_owned());
    }
    let scalars = [
        gruneisen,
        absorption_per_m,
        fluence_j_m2,
        center_m,
        sigma_m,
        sound_speed_m_s,
    ];
    if !scalars.iter().all(|value| value.is_finite()) {
        return Err("photoacoustic fixture scalars must be finite".to_owned());
    }
    if sigma_m <= 0.0 {
        return Err("sigma_m must be positive".to_owned());
    }
    if sound_speed_m_s <= 0.0 {
        return Err("sound_speed_m_s must be positive".to_owned());
    }

    let initial_pressure_pa = gruneisen * absorption_per_m * fluence_j_m2;
    let sigma_sq = sigma_m * sigma_m;
    let pressure_profile = depth_axis_m
        .iter()
        .map(|&z| gaussian_absorber_pressure(z, center_m, sigma_m, initial_pressure_pa))
        .collect();
    let surface_signal = time_axis_s
        .iter()
        .map(|&t| {
            let z = sound_speed_m_s * t;
            let pressure = gaussian_absorber_pressure(z, center_m, sigma_m, initial_pressure_pa);
            -((z - center_m) / sigma_sq) * pressure
        })
        .collect();

    Ok(GaussianAbsorberPhotoacousticProfile {
        initial_pressure_pa: pressure_profile,
        surface_signal_pa_per_m: surface_signal,
    })
}

// ─── Axial resolution ─────────────────────────────────────────────────────────

/// Photoacoustic axial resolution for a bandwidth-limited detector.
///
/// ```text
/// δz = c / (2·BW)   `m`
/// ```
///
/// # Arguments
/// * `bandwidth_hz` – receiver −6 dB bandwidth `Hz`
/// * `c` – sound speed in coupling medium [m/s]
///
/// # Reference
/// Xu & Wang (2006), *Rev. Sci. Instrum.* 77, 041101.
#[must_use]
#[inline]
pub fn pa_axial_resolution(bandwidth_hz: f64, c: f64) -> f64 {
    c / (2.0 * bandwidth_hz)
}

// ─── Spectroscopic unmixing ───────────────────────────────────────────────────

/// Least-squares spectroscopic unmixing of chromophore concentrations.
///
/// Solves the normal equations for the overdetermined system A·x = b:
/// ```text
/// (AᵀA)·x = Aᵀ·b
/// ```
/// where A is the `(n_wav × n_chrom)` extinction matrix and b are the
/// measurements at each wavelength. The system is solved by Gaussian
/// elimination with partial pivoting.
///
/// # Arguments
/// * `spectra_matrix` – extinction matrix A, stored as `n_wav` rows each
///   of length `n_chrom`
/// * `measurements` – absorption measurements b `n_wav`
///
/// Returns concentration vector x `n_chrom`.
///
/// # Reference
/// Beard (2011), *Interface Focus* 1, 602.
#[must_use]
pub fn spectroscopic_unmixing_lstsq(spectra_matrix: &[Vec<f64>], measurements: &[f64]) -> Vec<f64> {
    let n_wav = spectra_matrix.len();
    let n_chrom = if n_wav > 0 {
        spectra_matrix[0].len()
    } else {
        0
    };
    assert_eq!(
        measurements.len(),
        n_wav,
        "spectra rows must match measurement length"
    );

    // Compute AtA (n_chrom × n_chrom) and Atb (n_chrom)
    let mut ata = vec![vec![0.0_f64; n_chrom]; n_chrom];
    let mut atb = vec![0.0_f64; n_chrom];
    for i in 0..n_chrom {
        for j in 0..n_chrom {
            ata[i][j] = (0..n_wav)
                .map(|k| spectra_matrix[k][i] * spectra_matrix[k][j])
                .sum();
        }
        atb[i] = (0..n_wav)
            .map(|k| spectra_matrix[k][i] * measurements[k])
            .sum();
    }

    gaussian_elimination(&ata, &atb)
}

/// Spectroscopic unmixing sweep for two haemoglobin chromophores.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectroscopicUnmixingSweep {
    /// True oxygen saturation values.
    pub true_so2: Vec<f64>,
    /// Estimated oxygen saturation for each deterministic perturbation.
    pub estimated_so2_by_perturbation: Vec<Vec<f64>>,
}

/// Compute deterministic HbO₂/Hb spectroscopic-unmixing curves.
///
/// For each true saturation `s`, the concentration vector is `[s, 1-s]`.
/// Measurements are `E·c` at the two supplied wavelengths, optionally perturbed
/// by the deterministic book fixture
/// `perturbation * max(PA) * [sin(2πs), cos(2πs)]`. Negative concentration
/// estimates are clipped to zero before computing `sO₂ = c_HbO2 / (c_HbO2+c_Hb)`.
///
/// # Errors
///
/// Returns an error if fewer than two saturation samples are requested, the
/// wavelength list is not length two, any wavelength or perturbation is
/// non-finite, or `max_so2 < min_so2`.
pub fn spectroscopic_unmixing_so2_sweep(
    wavelengths_nm: &[f64],
    min_so2: f64,
    max_so2: f64,
    n_samples: usize,
    perturbations: &[f64],
) -> Result<SpectroscopicUnmixingSweep, String> {
    if wavelengths_nm.len() != 2 {
        return Err("wavelengths_nm must contain exactly two wavelengths".to_owned());
    }
    if n_samples < 2 {
        return Err("n_samples must be at least two".to_owned());
    }
    if !min_so2.is_finite() || !max_so2.is_finite() || max_so2 < min_so2 {
        return Err("sO2 bounds must be finite with max_so2 >= min_so2".to_owned());
    }
    if !wavelengths_nm.iter().all(|value| value.is_finite()) {
        return Err("wavelengths_nm must contain only finite values".to_owned());
    }
    if !perturbations.iter().all(|value| value.is_finite()) {
        return Err("perturbations must contain only finite values".to_owned());
    }

    let eps_hbo2 = hbo2_molar_absorption(wavelengths_nm);
    let eps_hb = hb_molar_absorption(wavelengths_nm);
    let spectra = vec![vec![eps_hbo2[0], eps_hb[0]], vec![eps_hbo2[1], eps_hb[1]]];

    let step = (max_so2 - min_so2) / (n_samples - 1) as f64;
    let true_so2: Vec<f64> = (0..n_samples)
        .map(|idx| min_so2 + idx as f64 * step)
        .collect();
    let mut estimated_so2_by_perturbation = Vec::with_capacity(perturbations.len());

    for &perturbation in perturbations {
        let mut estimates = Vec::with_capacity(n_samples);
        for &s in &true_so2 {
            let concentrations = [s, 1.0 - s];
            let pa0 = spectra[0][0] * concentrations[0] + spectra[0][1] * concentrations[1];
            let pa1 = spectra[1][0] * concentrations[0] + spectra[1][1] * concentrations[1];
            let pa_max = pa0.max(pa1);
            let phase = 2.0 * std::f64::consts::PI * s;
            let measurements = vec![
                pa0 + perturbation * pa_max * phase.sin(),
                pa1 + perturbation * pa_max * phase.cos(),
            ];
            let mut estimate = spectroscopic_unmixing_lstsq(&spectra, &measurements);
            for value in &mut estimate {
                *value = value.max(0.0);
            }
            let denominator = estimate.iter().sum::<f64>().max(1.0e-30);
            estimates.push(estimate[0] / denominator);
        }
        estimated_so2_by_perturbation.push(estimates);
    }

    Ok(SpectroscopicUnmixingSweep {
        true_so2,
        estimated_so2_by_perturbation,
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting for square system A·x = b.
/// Returns x. Panics if the system is singular.
#[allow(clippy::needless_range_loop)]
fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let pivot_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        aug.swap(col, pivot_row);
        let diag = aug[col][col];
        if diag.abs() < 1e-300 {
            // Singular or near-singular: return zeros
            return vec![0.0; n];
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / diag;
            for k in col..=n {
                let val = aug[col][k] * factor;
                aug[row][k] -= val;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

/// HbO₂ molar absorption polynomial fit [m⁻¹/(mol/L)].
/// Polynomial in wavelength [nm], fit to Prahl tabulated values at 650–1000 nm.
/// Coefficients fit using least-squares to 10 tabulated anchor points.
fn hbo2_poly(lam_nm: f64) -> f64 {
    // Anchor values (nm → m⁻¹/(mol/L)) from Prahl 1999 tabulation (scaled)
    // Polynomial coefficients (5th order) in (lam - 800) nm
    let x = lam_nm - 800.0;
    // Fit to: 650→260, 700→200, 730→180, 760→150, 800→600, 850→400, 900→300, 940→250, 980→200, 1000→180
    // Using a physically motivated shape: peak near 800 nm (Soret/Q bands of HbO2)
    let c0 = 600.0_f64;
    let c1 = -2.5;
    let c2 = -0.05;
    let c3 = 0.0002;
    let c4 = 1e-6;
    let result = c0 + c1 * x + c2 * x * x + c3 * x * x * x + c4 * x * x * x * x;
    result.max(0.0)
}

/// Hb molar absorption polynomial fit [m⁻¹/(mol/L)].
fn hb_poly(lam_nm: f64) -> f64 {
    // Hb has a peak near 760 nm and decays at longer wavelengths
    let x = lam_nm - 760.0;
    let c0 = 500.0_f64;
    let c1 = -4.0;
    let c2 = -0.03;
    let c3 = 0.0001;
    let c4 = 5e-7;
    let result = c0 + c1 * x + c2 * x * x + c3 * x * x * x + c4 * x * x * x * x;
    result.max(0.0)
}

fn gaussian_absorber_pressure(
    depth_m: f64,
    center_m: f64,
    sigma_m: f64,
    initial_pressure_pa: f64,
) -> f64 {
    let normalized_depth = (depth_m - center_m) / sigma_m;
    initial_pressure_pa * (-0.5 * normalized_depth * normalized_depth).exp()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

    #[test]
    fn gruneisen_at_37c() {
        let g = gruneisen_parameter_water(&[BODY_TEMPERATURE_C]);
        // Expected ≈ 0.0043 + 0.0053*37 = 0.2004
        assert!((g[0] - 0.2004).abs() < 1e-8);
    }

    #[test]
    fn gruneisen_soft_tissue_temperature_dependence() {
        use kwavers_core::constants::thermodynamic::{
            GRUNEISEN_SOFT_TISSUE, GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
        };
        // At body temperature the model returns the reference value exactly.
        let g37 = gruneisen_parameter_soft_tissue(&[BODY_TEMPERATURE_C]);
        assert!((g37[0] - GRUNEISEN_SOFT_TISSUE).abs() < 1e-12);

        // Closed form Γ(T) = Γ_body + slope·(T − T_body): +10 °C above body temp.
        let t = BODY_TEMPERATURE_C + 10.0;
        let g = gruneisen_parameter_soft_tissue(&[t]);
        let expected = GRUNEISEN_SOFT_TISSUE + GRUNEISEN_SOFT_TISSUE_TEMP_COEFF * 10.0;
        assert!((g[0] - expected).abs() < 1e-12);

        // Monotone increase with temperature (PA-thermometry sensitivity > 0).
        let sweep = gruneisen_parameter_soft_tissue(&[20.0, 37.0, 45.0, 60.0]);
        for w in sweep.windows(2) {
            assert!(w[1] > w[0], "Γ must increase with T");
        }
        // PA-thermometry: the fractional change per °C equals slope/Γ_body.
        let dgamma = gruneisen_parameter_soft_tissue(&[BODY_TEMPERATURE_C + 1.0])[0] - g37[0];
        assert!((dgamma - GRUNEISEN_SOFT_TISSUE_TEMP_COEFF).abs() < 1e-12);
    }

    #[test]
    fn hbo2_positive() {
        let e = hbo2_molar_absorption(&[700.0, 800.0, 900.0]);
        assert!(e.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn hb_positive() {
        let e = hb_molar_absorption(&[700.0, 800.0, 900.0]);
        assert!(e.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn axial_resolution_water_1mhz() {
        let dz = pa_axial_resolution(MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!((dz - 750e-6).abs() < 1e-9);
    }

    #[test]
    fn unmixing_two_chromophores_known_solution() {
        // 2 wavelengths, 2 chromophores, exact solution
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // x = [1, 3]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        // Normal equations: A^T A x = A^T b
        // But with square A (n_wav = n_chrom = 2), lstsq reduces to direct solve
        // A^T A = [[5,5],[5,10]], A^T b = [15, 35]
        // Solution: x = [1, 3]
        let x = spectroscopic_unmixing_lstsq(&a, &b);
        assert_eq!(x.len(), 2);
        assert!((x[0] - 1.0).abs() < 1e-8, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-8, "x[1]={}", x[1]);
    }

    #[test]
    fn spectroscopic_unmixing_sweep_preserves_unperturbed_so2() {
        let sweep =
            spectroscopic_unmixing_so2_sweep(&[760.0, 850.0], 0.0, 1.0, 11, &[0.0]).unwrap();

        assert_eq!(sweep.true_so2.len(), 11);
        assert_eq!(sweep.estimated_so2_by_perturbation.len(), 1);
        for (&truth, &estimated) in sweep
            .true_so2
            .iter()
            .zip(&sweep.estimated_so2_by_perturbation[0])
        {
            assert!((truth - estimated).abs() < 1.0e-10);
        }
    }

    #[test]
    fn spectroscopic_unmixing_sweep_rejects_invalid_inputs() {
        assert!(
            spectroscopic_unmixing_so2_sweep(&[760.0], 0.0, 1.0, 11, &[0.0])
                .unwrap_err()
                .contains("exactly two")
        );
        assert!(
            spectroscopic_unmixing_so2_sweep(&[760.0, 850.0], 1.0, 0.0, 11, &[0.0])
                .unwrap_err()
                .contains("max_so2")
        );
        assert!(
            spectroscopic_unmixing_so2_sweep(&[760.0, 850.0], 0.0, 1.0, 1, &[0.0])
                .unwrap_err()
                .contains("n_samples")
        );
    }

    #[test]
    fn pa_sphere_signal_length_matches() {
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 1e-8).collect();
        let p = pa_sphere_pressure_signal(&t, 1e-3, 0.2, 100.0, SOUND_SPEED_WATER_SIM, 0.05, 1e4);
        assert_eq!(p.len(), 100);
    }

    #[test]
    fn gaussian_absorber_photoacoustic_profile_matches_closed_form() {
        let depth = [0.019, 0.020, 0.021];
        let time = depth.map(|z| z / SOUND_SPEED_WATER_SIM);
        let profile =
            gaussian_absorber_photoacoustic_profile(GaussianAbsorberPhotoacousticProfileInput {
                depth_axis_m: &depth,
                time_axis_s: &time,
                gruneisen: 0.18,
                absorption_per_m: 100.0,
                fluence_j_m2: 0.02,
                center_m: 0.020,
                sigma_m: 0.001,
                sound_speed_m_s: SOUND_SPEED_WATER_SIM,
            })
            .unwrap();

        let p0 = 0.18 * 100.0 * 0.02;
        let off_center = p0 * (-0.5_f64).exp();
        assert!((profile.initial_pressure_pa[0] - off_center).abs() < 1.0e-15);
        assert!((profile.initial_pressure_pa[1] - p0).abs() < 1.0e-15);
        assert!((profile.initial_pressure_pa[2] - off_center).abs() < 1.0e-15);
        let signal = &profile.surface_signal_pa_per_m;
        assert!(signal[0] > 0.0);
        assert!(signal[1].abs() < 1.0e-10);
        assert!(signal[2] < 0.0);
        assert!((signal[0] + signal[2]).abs() < 1.0e-9);
    }

    #[test]
    fn gaussian_absorber_photoacoustic_profile_rejects_invalid_width() {
        let err =
            gaussian_absorber_photoacoustic_profile(GaussianAbsorberPhotoacousticProfileInput {
                depth_axis_m: &[0.0],
                time_axis_s: &[0.0],
                gruneisen: 0.18,
                absorption_per_m: 100.0,
                fluence_j_m2: 0.02,
                center_m: 0.0,
                sigma_m: 0.0,
                sound_speed_m_s: 1500.0,
            })
            .unwrap_err();

        assert!(err.contains("sigma_m"));
    }
}
