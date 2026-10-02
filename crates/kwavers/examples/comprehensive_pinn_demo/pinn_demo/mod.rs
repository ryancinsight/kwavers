//! PINN ecosystem demonstration sections, grouped by topic.

mod advanced_domains;
mod core_capabilities;
mod deployment;

pub use advanced_domains::{
    demonstrate_meta_learning, demonstrate_physics_domains, demonstrate_uncertainty,
};
pub use core_capabilities::{
    demonstrate_basic_pinn, demonstrate_distributed_training, demonstrate_jit_inference,
    demonstrate_quantization,
};
pub use deployment::{
    demonstrate_applications, demonstrate_cloud_deployment, demonstrate_performance,
};
