//! Clinical decision support for 3D SWE workflow classification.

mod classifier;
#[cfg(test)]
mod tests;
mod types;

pub use classifier::Swe3dClinicalDecisionSupport;
pub use types::{
    BreastLesionClassification, ClassificationConfidence, FibrosisStage, LiverFibrosisStage,
    TissueReference,
};
