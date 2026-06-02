//! 3D Shear Wave Elastography Clinical Workflows
//!
//! Implements clinical workflows for 3D SWE including volumetric ROI analysis,
//! multi-planar reconstruction, and clinical decision support.
//!
//! ## Clinical Applications
//!
//! - Liver fibrosis staging (METAVIR F0-F4)
//! - Breast lesion characterization (BI-RADS)
//! - Prostate cancer detection and staging
//! - Thyroid nodule assessment
//! - Musculoskeletal tissue evaluation
//!
//! ## References
//!
//! - Nightingale, K. R., et al. (2015). "Shear wave elastography." *Physics in Medicine
//!   and Biology*, 60(2), R1-R41.
//! - Ferraioli, G., et al. (2018). "Guidelines and good clinical practice recommendations
//!   for contrast enhanced ultrasound (CEUS) in the liver." *Ultrasound in Medicine & Biology*.
//! - Barr, R. G., et al. (2019). "Elastography assessment of liver fibrosis." *Abdominal Radiology*.

pub mod decision_support;
pub mod elasticity_map;
pub mod reconstruction;
pub mod roi;
pub mod statistics;

pub use decision_support::{
    BreastLesionClassification, ClassificationConfidence, FibrosisStage, LiverFibrosisStage,
    Swe3dClinicalDecisionSupport, TissueReference,
};
pub use elasticity_map::{ElasticityMap2D, ElasticityMap3D};
pub use reconstruction::{MultiPlanarReconstruction, SliceOrientation, SlicePositions};
pub use roi::VolumetricROI;
pub use statistics::VolumetricStatistics;
