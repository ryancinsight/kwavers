use aequitas::systems::si::units::PerMeter;
use hyperion::transport::reduced_scattering;

use super::OpticalPropertyData;

impl OpticalPropertyData {
    /// Return the absorption coefficient in inverse metres.
    #[must_use]
    pub fn absorption_coefficient(&self) -> f64 {
        self.coefficients.absorption().in_unit::<PerMeter>()
    }

    /// Return the unreduced scattering coefficient in inverse metres.
    #[must_use]
    pub fn scattering_coefficient(&self) -> f64 {
        self.coefficients.scattering().in_unit::<PerMeter>()
    }

    /// Return the reduced scattering coefficient in inverse metres.
    #[must_use]
    pub fn reduced_scattering_coefficient(&self) -> f64 {
        reduced_scattering(*self.coefficients.scattering(), self.anisotropy)
            .expect("invariant: construction validates reduced scattering")
            .in_unit::<PerMeter>()
    }

    /// Return the scattering anisotropy factor.
    #[must_use]
    pub fn anisotropy(&self) -> f64 {
        self.anisotropy.quantity().into_base()
    }

    /// Return the refractive index.
    #[must_use]
    pub const fn refractive_index(&self) -> f64 {
        self.refractive_index
    }

    /// Return Fresnel reflectance at normal incidence from vacuum.
    #[must_use]
    pub fn fresnel_reflectance_normal(&self) -> f64 {
        let ratio = (1.0 - self.refractive_index) / (1.0 + self.refractive_index);
        ratio * ratio
    }
}
