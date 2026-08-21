use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::skull::{AcousticSkullProperties, HeterogeneousSkull};
use leto::Array3;

/// CT Hounsfield units and their provider-owned acoustic skull model.
pub(crate) struct SkullModel {
    hu: Array3<f64>,
    acoustic: HeterogeneousSkull,
}

impl SkullModel {
    /// Converts a finite CT volume through the canonical Hill mixing model.
    pub(crate) fn from_hu(hu: Array3<f64>) -> KwaversResult<Self> {
        let acoustic = HeterogeneousSkull::from_ct_hill(&hu, &AcousticSkullProperties::cortical())?;
        Ok(Self { hu, acoustic })
    }

    /// Returns the source CT Hounsfield-unit volume.
    pub(crate) const fn hu(&self) -> &Array3<f64> {
        &self.hu
    }

    /// Returns the derived density, sound-speed, and attenuation fields.
    pub(crate) const fn acoustic(&self) -> &HeterogeneousSkull {
        &self.acoustic
    }
}
