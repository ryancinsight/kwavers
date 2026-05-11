use super::OpticalPropertyData;

impl OpticalPropertyData {
    /// Calculate reduced scattering coefficient μ_s' = μ_s(1-g)
    #[must_use]
    pub fn reduced_scattering_coefficient(&self) -> f64 {
        self.scattering_coefficient * (1.0 - self.anisotropy)
    }

    /// Get anisotropy factor (alias for backward compatibility)
    #[must_use]
    pub fn anisotropy_factor(&self) -> f64 {
        self.anisotropy
    }

    /// Total attenuation coefficient μ_t = μ_a + μ_s (m⁻¹)
    #[inline]
    #[must_use]
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.scattering_coefficient
    }

    /// Reduced scattering coefficient μ_s' = μ_s (1 - g) (m⁻¹)
    #[inline]
    #[must_use]
    pub fn reduced_scattering(&self) -> f64 {
        self.scattering_coefficient * (1.0 - self.anisotropy)
    }

    /// Optical penetration depth δ = 1/μ_eff (m)
    #[inline]
    #[must_use]
    pub fn penetration_depth(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering();
        let mu_eff =
            (3.0 * self.absorption_coefficient * (self.absorption_coefficient + mu_s_prime)).sqrt();
        if mu_eff > 0.0 {
            1.0 / mu_eff
        } else {
            f64::INFINITY
        }
    }

    /// Mean free path l_mfp = 1/μ_t (m)
    #[inline]
    #[must_use]
    pub fn mean_free_path(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 { 1.0 / mu_t } else { f64::INFINITY }
    }

    /// Transport mean free path l_tr = 1/(μ_a + μ_s') (m)
    #[inline]
    #[must_use]
    pub fn transport_mean_free_path(&self) -> f64 {
        let mu_tr = self.absorption_coefficient + self.reduced_scattering();
        if mu_tr > 0.0 { 1.0 / mu_tr } else { f64::INFINITY }
    }

    /// Albedo α = μ_s / μ_t (dimensionless)
    #[inline]
    #[must_use]
    pub fn albedo(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 { self.scattering_coefficient / mu_t } else { 0.0 }
    }

    /// Fresnel reflectance at normal incidence R₀
    #[inline]
    #[must_use]
    pub fn fresnel_reflectance_normal(&self) -> f64 {
        let n1 = 1.0;
        let n2 = self.refractive_index;
        let r = (n1 - n2) / (n1 + n2);
        r * r
    }
}
