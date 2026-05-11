use super::ElasticPropertyData;

impl ElasticPropertyData {
    /// Young's modulus E = μ(3λ + 2μ)/(λ + μ) (Pa)
    #[inline]
    #[must_use]
    pub fn youngs_modulus(&self) -> f64 {
        self.mu * 3.0f64.mul_add(self.lambda, 2.0 * self.mu) / (self.lambda + self.mu)
    }

    /// Poisson's ratio ν = λ/(2(λ + μ)) (dimensionless)
    #[inline]
    #[must_use]
    pub fn poisson_ratio(&self) -> f64 {
        self.lambda / (2.0 * (self.lambda + self.mu))
    }

    /// Bulk modulus K = λ + 2μ/3 (Pa)
    #[inline]
    #[must_use]
    pub fn bulk_modulus(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }

    /// Shear modulus (alias for μ)
    #[inline]
    #[must_use]
    pub fn shear_modulus(&self) -> f64 {
        self.mu
    }

    /// P-wave (compressional) speed c_p = √((λ + 2μ)/ρ) (m/s)
    #[inline]
    #[must_use]
    pub fn p_wave_speed(&self) -> f64 {
        (2.0f64.mul_add(self.mu, self.lambda) / self.density).sqrt()
    }

    /// S-wave (shear) speed c_s = √(μ/ρ) (m/s)
    #[inline]
    #[must_use]
    pub fn s_wave_speed(&self) -> f64 {
        (self.mu / self.density).sqrt()
    }
}
