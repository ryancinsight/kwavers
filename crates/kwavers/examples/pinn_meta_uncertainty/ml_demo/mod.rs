//! Meta-learning, transfer-learning, uncertainty, and impact demonstration sections.

mod adaptation;
mod uncertainty_and_impact;

pub use adaptation::{demonstrate_meta_learning, demonstrate_transfer_learning};
pub use uncertainty_and_impact::{
    demonstrate_integrated_ml, demonstrate_real_world_impact, demonstrate_uncertainty,
};
