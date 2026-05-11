mod blood_oxygenation;
mod layered_tissue;
mod tumor_detection;
mod vascular;

pub use blood_oxygenation::BloodOxygenationPhantomBuilder;
pub use layered_tissue::LayeredTissuePhantomBuilder;
pub use tumor_detection::TumorDetectionPhantomBuilder;
pub use vascular::VascularPhantomBuilder;

use super::types::PhantomType;
use crate::domain::medium::properties::OpticalPropertyData;

/// Clinical phantom builder with domain-specific presets
#[derive(Debug)]
pub struct PhantomBuilder {
    pub builder: crate::domain::medium::optical_map::OpticalPropertyMapBuilder,
    pub phantom_type: PhantomType,
    pub wavelength_nm: f64,
}

impl PhantomBuilder {
    /// Create blood oxygenation phantom builder
    ///
    /// Designed for spectroscopic photoacoustic imaging validation.
    /// Default wavelength: 800 nm (near-infrared, good tissue penetration)
    #[must_use]
    pub fn blood_oxygenation() -> BloodOxygenationPhantomBuilder {
        BloodOxygenationPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            vessels: Vec::new(),
            tumors: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create layered tissue phantom builder
    ///
    /// Models stratified media (e.g., skin/fat/muscle).
    #[must_use]
    pub fn layered_tissue() -> LayeredTissuePhantomBuilder {
        LayeredTissuePhantomBuilder {
            dimensions: None,
            layers: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create tumor detection phantom builder
    ///
    /// Background tissue with embedded lesions for detection algorithm validation.
    #[must_use]
    pub fn tumor_detection() -> TumorDetectionPhantomBuilder {
        TumorDetectionPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            tumors: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create vascular phantom builder
    ///
    /// Models vessel networks for angiogenesis and perfusion studies.
    #[must_use]
    pub fn vascular() -> VascularPhantomBuilder {
        VascularPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            vessels: Vec::new(),
            wavelength_nm: 800.0,
        }
    }
}
