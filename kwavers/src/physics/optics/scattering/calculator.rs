//! Mie scattering calculator
//!
//! Provides exact solutions to Maxwell's equations for scattering by spheres
//! using multipole expansion (Riccati-Bessel functions).

use super::parameters::MieParameters;
use super::result::MieResult;
use crate::core::error::{KwaversError, KwaversResult};

/// Mie scattering calculator
#[derive(Debug)]
pub struct MieCalculator {
    /// Maximum number of terms in series expansion
    max_terms: usize,
}

impl Default for MieCalculator {
    fn default() -> Self {
        Self { max_terms: 10000 }
    }
}

impl MieCalculator {
    /// Create new Mie calculator with custom maximum terms
    #[must_use]
    pub fn new(max_terms: usize) -> Self {
        Self { max_terms }
    }

    /// Calculate Mie scattering for given parameters
    ///
    /// # Arguments
    /// * `params` - Mie scattering parameters
    ///
    /// # Returns
    /// Mie scattering results
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if size parameter exceeds 100.0
    pub fn calculate(&self, params: &MieParameters) -> KwaversResult<MieResult> {
        let x = params.size_parameter();

        if x < 0.1 {
            return Ok(self.rayleigh_approximation(params));
        }

        let m = params.relative_index();

        if x > 100.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Size parameter x = {x} too large for current implementation (max 100.0)"
            )));
        }

        let n_max = (x as usize + 1).max(10).min(self.max_terms);

        let (an, bn) = self.calculate_coefficients(num_complex::Complex64::from(x), m, n_max);

        let q_sca = self.scattering_efficiency(&an, &bn, x);
        let q_ext = self.extinction_efficiency(&an, &bn, x);
        let q_abs = q_ext - q_sca;
        let q_bsa = self.backscattering_efficiency(&an, &bn);

        let geometric_cs = MieResult::geometric_cross_section(params.radius);
        let sigma_sca = q_sca * geometric_cs;
        let sigma_ext = q_ext * geometric_cs;
        let sigma_abs = q_abs * geometric_cs;

        let g = self.asymmetry_parameter(&an, &bn, x);
        let p_180 = self.phase_function_180(&an, &bn);

        Ok(MieResult {
            size_parameter: x,
            scattering_efficiency: q_sca,
            extinction_efficiency: q_ext,
            absorption_efficiency: q_abs,
            backscattering_efficiency: q_bsa,
            scattering_cross_section: sigma_sca,
            extinction_cross_section: sigma_ext,
            absorption_cross_section: sigma_abs,
            asymmetry_parameter: g,
            phase_function_180: p_180,
        })
    }

    /// Rayleigh approximation for small particles (x << 1)
    fn rayleigh_approximation(&self, params: &MieParameters) -> MieResult {
        let x = params.size_parameter();
        let m = params.relative_index();

        let m2 = m * m;
        let alpha_term = (m2 - 1.0) / (m2 + 2.0);
        let q_sca = (8.0 / 3.0) * x.powi(4) * alpha_term.norm_sqr();
        let q_ext = q_sca;

        let geometric_cs = MieResult::geometric_cross_section(params.radius);

        MieResult {
            size_parameter: x,
            scattering_efficiency: q_sca,
            extinction_efficiency: q_ext,
            absorption_efficiency: 0.0,
            backscattering_efficiency: q_sca * 0.5,
            scattering_cross_section: q_sca * geometric_cs,
            extinction_cross_section: q_ext * geometric_cs,
            absorption_cross_section: 0.0,
            asymmetry_parameter: 0.0,
            phase_function_180: q_sca,
        }
    }

    /// Calculate Mie scattering coefficients a_n and b_n
    fn calculate_coefficients(
        &self,
        x: num_complex::Complex64,
        m: num_complex::Complex64,
        n_max: usize,
    ) -> (Vec<num_complex::Complex64>, Vec<num_complex::Complex64>) {
        let mut an = Vec::with_capacity(n_max);
        let mut bn = Vec::with_capacity(n_max);

        let (psi, xi) = self.riccati_bessel(x, n_max);
        let (psi_m, xi_m) = self.riccati_bessel(x * m, n_max);

        for n in 1..=n_max {
            let a_n = self.mie_a_coefficient(n, m, &psi, &xi, &psi_m, &xi_m);
            let b_n = self.mie_b_coefficient(n, m, &psi, &xi, &psi_m, &xi_m);

            an.push(a_n);
            bn.push(b_n);
        }

        (an, bn)
    }

    /// Calculate Riccati-Bessel functions ψ_n(z) and ξ_n(z)
    fn riccati_bessel(
        &self,
        z: num_complex::Complex64,
        n_max: usize,
    ) -> (Vec<num_complex::Complex64>, Vec<num_complex::Complex64>) {
        let mut psi = Vec::with_capacity(n_max + 1);
        let mut xi = Vec::with_capacity(n_max + 1);

        psi.push(z.sin());
        xi.push(z.sin() - num_complex::Complex64::I * z.cos());

        psi.push(z.sin() / z - z.cos());
        xi.push(psi[1] - num_complex::Complex64::I * (z.sin() / z + z.cos()));

        for n in 2..=n_max {
            let n_f64 = n as f64;
            let psi_n = ((2.0 * n_f64 - 1.0) / n_f64) * psi[n - 1] - psi[n - 2];
            let xi_n = ((2.0 * n_f64 - 1.0) / n_f64) * xi[n - 1] - xi[n - 2];

            psi.push(psi_n);
            xi.push(xi_n);
        }

        (psi, xi)
    }

    /// Calculate Mie a_n coefficient
    fn mie_a_coefficient(
        &self,
        n: usize,
        m: num_complex::Complex64,
        psi: &[num_complex::Complex64],
        xi: &[num_complex::Complex64],
        psi_m: &[num_complex::Complex64],
        xi_m: &[num_complex::Complex64],
    ) -> num_complex::Complex64 {
        let numerator = psi[n] * (psi_m[n] - m * xi_m[n]) - psi_m[n] * (psi[n] - xi[n]);
        let denominator = xi[n] * (psi_m[n] - m * xi_m[n]) - psi_m[n] * (xi[n] - m * xi_m[n]);

        numerator / denominator
    }

    /// Calculate Mie b_n coefficient
    fn mie_b_coefficient(
        &self,
        n: usize,
        m: num_complex::Complex64,
        psi: &[num_complex::Complex64],
        xi: &[num_complex::Complex64],
        psi_m: &[num_complex::Complex64],
        xi_m: &[num_complex::Complex64],
    ) -> num_complex::Complex64 {
        let numerator = psi_m[n] * (psi[n] - m * xi[n]) - psi[n] * (psi_m[n] - m * xi_m[n]);
        let denominator = psi_m[n] * (xi[n] - m * xi[n]) - xi_m[n] * (psi[n] - m * xi[n]);

        numerator / denominator
    }

    /// Calculate scattering efficiency Q_sca
    fn scattering_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        }

        (2.0 / (x * x)) * sum
    }

    /// Calculate extinction efficiency Q_ext
    fn extinction_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (an[n] + bn[n]).re;
        }

        (2.0 / (x * x)) * sum
    }

    /// Calculate backscattering efficiency Q_bsa
    fn backscattering_efficiency(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
    ) -> f64 {
        let mut sum = num_complex::Complex64::new(0.0, 0.0);

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (-1.0_f64).powi(n as i32) * (an[n] - bn[n]);
        }

        sum.norm_sqr()
    }

    /// Calculate asymmetry parameter g
    fn asymmetry_parameter(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
        x: f64,
    ) -> f64 {
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            let weight = (2.0 * n_f64 + 1.0) / ((n_f64 * (n_f64 + 1.0)) * (2.0 * n_f64 + 1.0));

            sum1 += weight * (an[n] * bn[n].conj() + bn[n] * an[n].conj()).re;
            sum2 += (2.0 * n_f64 + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        }

        if sum2 > 0.0 {
            (4.0 / (x * x)) * sum1 / ((2.0 / (x * x)) * sum2)
        } else {
            0.0
        }
    }

    /// Calculate phase function at 180°
    fn phase_function_180(
        &self,
        an: &[num_complex::Complex64],
        bn: &[num_complex::Complex64],
    ) -> f64 {
        let mut sum = num_complex::Complex64::new(0.0, 0.0);

        for n in 0..an.len() {
            let n_f64 = (n + 1) as f64;
            sum += (2.0 * n_f64 + 1.0) * (-1.0_f64).powi(n as i32) * (an[n] - bn[n]);
        }

        sum.norm_sqr()
    }
}
