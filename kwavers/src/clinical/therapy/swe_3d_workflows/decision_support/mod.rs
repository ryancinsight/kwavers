//! Clinical decision support for 3D SWE workflow classification.

mod classifier;
mod types;
#[cfg(test)]
mod tests;

pub use classifier::ClinicalDecisionSupport;
pub use types::{
    BreastLesionClassification, ClassificationConfidence, FibrosisStage, LiverFibrosisStage,
    TissueReference,
};
