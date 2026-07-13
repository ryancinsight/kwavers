//! Mie scattering calculator
//!
//! Implements exact Mie series via the Bohren–Huffman (BHMIE) algorithm:
//! Bohren & Huffman 1983, *Absorption and Scattering of Light by Small Particles*,
//! §4.8 and Appendix A.
//!
//! Coefficient formulas (BH Eq. 4.88):
//! ```text
//!   a_n = ((D_n(mx)/m + n/x) ψ_n(x) − ψ_{n-1}(x))
//!       / ((D_n(mx)/m + n/x) ξ_n(x) − ξ_{n-1}(x))
//!   b_n = ((m·D_n(mx) + n/x) ψ_n(x) − ψ_{n-1}(x))
//!       / ((m·D_n(mx) + n/x) ξ_n(x) − ξ_{n-1}(x))
//! ```
//! where ψ_n, χ_n are the Riccati–Bessel functions of the first and second kind,
//! ξ_n = ψ_n − i·χ_n, and D_n(z) = ψ_n'(z)/ψ_n(z) is the logarithmic derivative,
//! computed by downward recurrence (BH Eq. 4.89):
//! ```text
//!   D_{n-1}(z) = n/z − 1/(D_n(z) + n/z)
//! ```
//! Series truncation follows Wiscombe (1980): N_stop = ⌈x + 4·x^(1/3) + 2⌉.

use super::parameters::MieParameters;
use super::result::MieResult;
use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};

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
    #[must_use]
    pub fn new(max_terms: usize) -> Self {
        Self { max_terms }
    }

    /// Calculate Mie scattering for given parameters.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if the size parameter exceeds 100.0.
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

        // Wiscombe (1980) truncation criterion
        let n_stop = (x + 4.0 * x.cbrt() + 2.0).ceil() as usize;
        let n_max = n_stop.max(3).min(self.max_terms);

        let (an, bn) = self.calculate_coefficients(x, m, n_max);

        let q_sca = self.scattering_efficiency(&an, &bn, x);
        let q_ext = self.extinction_efficiency(&an, &bn, x);
        let q_abs = q_ext - q_sca;
        let q_bsa = self.backscattering_efficiency(&an, &bn, x);

        let geometric_cs = MieResult::geometric_cross_section(params.radius);
        let sigma_sca = q_sca * geometric_cs;
        let sigma_ext = q_ext * geometric_cs;
        let sigma_abs = q_abs * geometric_cs;

        let g = self.asymmetry_parameter(&an, &bn, q_sca, x);
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
        // Q_sca = (8/3) x⁴ |(m²−1)/(m²+2)|²  (BH Eq. 5.8)
        let q_sca = (8.0 / 3.0) * x.powi(4) * alpha_term.norm_sqr();
        // Q_abs = 4x · Im[(m²−1)/(m²+2)]  (BH Eq. 5.11, convention m = n + iκ, κ ≥ 0)
        let q_abs = 4.0 * x * alpha_term.im;
        let q_ext = q_sca + q_abs;

        let geometric_cs = MieResult::geometric_cross_section(params.radius);

        MieResult {
            size_parameter: x,
            scattering_efficiency: q_sca,
            extinction_efficiency: q_ext,
            absorption_efficiency: q_abs,
            backscattering_efficiency: 1.5 * q_sca,
            scattering_cross_section: q_sca * geometric_cs,
            extinction_cross_section: q_ext * geometric_cs,
            absorption_cross_section: q_abs * geometric_cs,
            asymmetry_parameter: 0.0,
            phase_function_180: q_sca,
        }
    }

    /// Logarithmic derivative D_n(z) = ψ_n'(z)/ψ_n(z) by downward recurrence
    /// (Bohren–Huffman §4.8 / Eq. 4.89).
    ///
    /// Starts at N_start = max(N_max, ⌈|z|⌉) + 15 with D_{N_start}(z) = 0
    /// and applies D_{n-1}(z) = n/z − 1/(D_n(z) + n/z) downward.
    fn logarithmic_derivative(&self, z: Complex64, n_max: usize) -> Vec<Complex64> {
        let n_start = n_max.max(z.norm().ceil() as usize) + 15;
        let mut d = vec![Complex64::new(0.0, 0.0); n_start + 1];
        for n in (1..=n_start).rev() {
            let n_over_z = Complex64::new(n as f64, 0.0) / z;
            d[n - 1] = n_over_z - Complex64::new(1.0, 0.0) / (d[n] + n_over_z);
        }
        d.truncate(n_max + 1);
        d
    }

    /// Calculate Mie coefficients a_n and b_n for n = 1..=n_max (BH Eq. 4.88).
    #[allow(clippy::needless_range_loop)]
    fn calculate_coefficients(
        &self,
        x: f64,
        m: Complex64,
        n_max: usize,
    ) -> (Vec<Complex64>, Vec<Complex64>) {
        let mx = m * x;
        let d = self.logarithmic_derivative(mx, n_max);

        // Riccati–Bessel functions on real argument x by upward recurrence
        // ψ_{n} = (2n−1)/x · ψ_{n-1} − ψ_{n-2},  ψ_{-1} = cos x, ψ_0 = sin x
        // χ_{n} = (2n−1)/x · χ_{n-1} − χ_{n-2},  χ_{-1} = −sin x, χ_0 = cos x
        // ξ_n = ψ_n − i χ_n   (since ξ_n(x) = x · h_n^{(1)}(x))
        let mut psi_prev = x.cos(); // ψ_{-1}
        let mut psi_curr = x.sin(); // ψ_0
        let mut chi_prev = -x.sin(); // χ_{-1}
        let mut chi_curr = x.cos(); // χ_0
        let mut xi_prev = Complex64::new(psi_curr, -chi_curr); // ξ_0

        let mut an = Vec::with_capacity(n_max);
        let mut bn = Vec::with_capacity(n_max);

        for n in 1..=n_max {
            let n_f = n as f64;
            let factor = (2.0 * n_f - 1.0) / x;

            let psi_n = factor * psi_curr - psi_prev;
            let chi_n = factor * chi_curr - chi_prev;
            let xi_n = Complex64::new(psi_n, -chi_n);

            // BH Eq. 4.88
            let dn_over_m = d[n] / m;
            let m_dn = m * d[n];
            let n_over_x = Complex64::new(n_f / x, 0.0);

            let psi_nm1 = Complex64::new(psi_curr, 0.0);
            let psi_n_c = Complex64::new(psi_n, 0.0);

            let term_a = dn_over_m + n_over_x;
            let term_b = m_dn + n_over_x;

            let a_n = (term_a * psi_n_c - psi_nm1) / (term_a * xi_n - xi_prev);
            let b_n = (term_b * psi_n_c - psi_nm1) / (term_b * xi_n - xi_prev);

            an.push(a_n);
            bn.push(b_n);

            psi_prev = psi_curr;
            psi_curr = psi_n;
            chi_prev = chi_curr;
            chi_curr = chi_n;
            xi_prev = xi_n;
        }

        (an, bn)
    }

    /// Q_sca = (2/x²) Σ (2n+1)(|a_n|² + |b_n|²)   (BH Eq. 4.61)
    fn scattering_efficiency(&self, an: &[Complex64], bn: &[Complex64], x: f64) -> f64 {
        let mut sum = 0.0;
        for n in 0..an.len() {
            let n_f = (n + 1) as f64;
            sum += (2.0 * n_f + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        }
        (2.0 / (x * x)) * sum
    }

    /// Q_ext = (2/x²) Σ (2n+1) Re(a_n + b_n)   (BH Eq. 4.62)
    fn extinction_efficiency(&self, an: &[Complex64], bn: &[Complex64], x: f64) -> f64 {
        let mut sum = 0.0;
        for n in 0..an.len() {
            let n_f = (n + 1) as f64;
            sum += (2.0 * n_f + 1.0) * (an[n] + bn[n]).re;
        }
        (2.0 / (x * x)) * sum
    }

    /// Q_back = (1/x²) |Σ (2n+1)(−1)ⁿ (a_n − b_n)|²   (BH Eq. 4.82)
    fn backscattering_efficiency(&self, an: &[Complex64], bn: &[Complex64], x: f64) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for n in 0..an.len() {
            let n_idx = (n + 1) as i32;
            let n_f = f64::from(n_idx);
            let sign = if n_idx % 2 == 0 { 1.0 } else { -1.0 };
            sum += (2.0 * n_f + 1.0) * sign * (an[n] - bn[n]);
        }
        sum.norm_sqr() / (x * x)
    }

    /// Asymmetry parameter g = ⟨cos θ⟩ (BH Eq. 4.80):
    /// g·Q_sca = (4/x²) Σ [ n(n+2)/(n+1) · Re(a_n a*_{n+1} + b_n b*_{n+1})
    ///                    + (2n+1)/(n(n+1)) · Re(a_n b*_n) ]
    fn asymmetry_parameter(&self, an: &[Complex64], bn: &[Complex64], q_sca: f64, x: f64) -> f64 {
        if q_sca <= 0.0 || an.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for n in 0..an.len() {
            let n_f = (n + 1) as f64;
            // (2n+1)/(n(n+1)) · Re(a_n b*_n)
            sum += (2.0 * n_f + 1.0) / (n_f * (n_f + 1.0)) * (an[n] * bn[n].conj()).re;
            // n(n+2)/(n+1) · Re(a_n a*_{n+1} + b_n b*_{n+1})
            if n + 1 < an.len() {
                sum += n_f * (n_f + 2.0) / (n_f + 1.0)
                    * (an[n] * an[n + 1].conj() + bn[n] * bn[n + 1].conj()).re;
            }
        }
        (4.0 / (x * x)) * sum / q_sca
    }

    /// Differential scattering at θ = π: |S_1(π)|² with S_1(π) = ½ Σ (2n+1)(−1)ⁿ(a_n − b_n)
    fn phase_function_180(&self, an: &[Complex64], bn: &[Complex64]) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for n in 0..an.len() {
            let n_idx = (n + 1) as i32;
            let n_f = f64::from(n_idx);
            let sign = if n_idx % 2 == 0 { 1.0 } else { -1.0 };
            sum += (2.0 * n_f + 1.0) * sign * (an[n] - bn[n]);
        }
        0.25 * sum.norm_sqr()
    }
}
