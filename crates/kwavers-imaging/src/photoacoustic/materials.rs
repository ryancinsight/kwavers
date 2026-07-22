use kwavers_medium::properties::OpticalPropertyData;

#[derive(Debug, Clone, Copy)]
pub struct SpectralSample {
    pub wavelength_nm: f64,
    pub properties: OpticalPropertyData,
}

#[derive(Debug)]
pub struct PhotoacousticMaterialLibrary;

impl PhotoacousticMaterialLibrary {
    /// Return a soft-tissue spectral sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `wavelength_nm` is outside the optical model's domain.
    pub fn soft_tissue_sample(wavelength_nm: f64) -> Result<SpectralSample, String> {
        Ok(SpectralSample {
            wavelength_nm,
            properties: crate::photoacoustic::PhotoacousticOpticalProperties::soft_tissue(
                wavelength_nm,
            )?,
        })
    }

    /// Return a blood spectral sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `wavelength_nm` is outside the optical model's domain.
    pub fn blood_sample(wavelength_nm: f64) -> Result<SpectralSample, String> {
        Ok(SpectralSample {
            wavelength_nm,
            properties: crate::photoacoustic::PhotoacousticOpticalProperties::blood(wavelength_nm)?,
        })
    }

    /// Return a tumor spectral sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `wavelength_nm` is outside the optical model's domain.
    pub fn tumor_sample(wavelength_nm: f64) -> Result<SpectralSample, String> {
        Ok(SpectralSample {
            wavelength_nm,
            properties: crate::photoacoustic::PhotoacousticOpticalProperties::tumor(wavelength_nm)?,
        })
    }
}
