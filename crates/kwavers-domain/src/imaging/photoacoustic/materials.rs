use crate::medium::properties::OpticalPropertyData;

#[derive(Debug, Clone, Copy)]
pub struct SpectralSample {
    pub wavelength_nm: f64,
    pub properties: OpticalPropertyData,
}

#[derive(Debug)]
pub struct PhotoacousticMaterialLibrary;

impl PhotoacousticMaterialLibrary {
    #[must_use]
    pub fn soft_tissue_sample(wavelength_nm: f64) -> SpectralSample {
        SpectralSample {
            wavelength_nm,
            properties:
                crate::imaging::photoacoustic::PhotoacousticOpticalProperties::soft_tissue(
                    wavelength_nm,
                ),
        }
    }

    #[must_use]
    pub fn blood_sample(wavelength_nm: f64) -> SpectralSample {
        SpectralSample {
            wavelength_nm,
            properties:
                crate::imaging::photoacoustic::PhotoacousticOpticalProperties::blood(
                    wavelength_nm,
                ),
        }
    }

    #[must_use]
    pub fn tumor_sample(wavelength_nm: f64) -> SpectralSample {
        SpectralSample {
            wavelength_nm,
            properties:
                crate::imaging::photoacoustic::PhotoacousticOpticalProperties::tumor(
                    wavelength_nm,
                ),
        }
    }
}
