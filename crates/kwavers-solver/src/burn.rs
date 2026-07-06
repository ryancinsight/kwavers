//! Local `burn` compatibility module.
//!
//! This module shadows the removed `burn` crate dependency with implementations
//! built on coeus Atlas crates.  Every `use burn::…` in the PINN submodules
//! resolves here — zero changes to those files are required.
//!
//! All types are concrete (`f32`, `MoiraiBackend`); the generic `<B: Backend>`
//! parameter on the shim wrapper types accepts any coeus `BackendOps<f32>`.
//!
//! **Migration note**: As each PINN submodule is fully ported to native coeus
//! API the imports from this module are replaced with direct coeus imports and
//! the module declaration in `lib.rs` is removed.

use crate::inverse::pinn::ml::burn_compat as compat;

// ── tensor ────────────────────────────────────────────────────────────────────
pub mod tensor {
    use super::compat;
    pub use compat::{Bool, ElementConversion, Int, Tensor, TensorData};

    pub mod backend {
        use super::compat;
        pub use compat::{Autodiff, AutodiffBackend, Backend, NdArray};
    }

    pub mod activation {
        use super::compat;
        pub use compat::activation::{relu, sigmoid, tanh};
    }
}

// ── module ────────────────────────────────────────────────────────────────────
pub mod module {
    use super::compat;
    pub use compat::{
        AutodiffModule, Devices, Ignored, Module, ModuleMapper, ModuleRecord as Content,
        ModuleRecord, ModuleVisitor, Param,
    };
    /// `Devices` — returns the default-device list.
    pub use compat::default_devices as make_devices;

    /// Stub for `ModuleDisplayDefault`.
    pub trait ModuleDisplayDefault {}

    /// Stub for `ModuleDisplay`.
    pub trait ModuleDisplay {}
}

// ── nn ────────────────────────────────────────────────────────────────────────
pub mod nn {
    use super::compat;
    pub use compat::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
}

// ── optim ─────────────────────────────────────────────────────────────────────
pub mod optim {
    use super::compat;
    pub use compat::{
        AdamConfig, AdamW, AdamWConfig, GradientsParams, MomentumConfig, Optimizer, Sgd, SgdConfig,
        WeightDecayConfig,
    };

    /// Adam optimizer type.
    pub use compat::Adam;

    pub mod adaptor {
        use super::compat;
        pub use compat::OptimizerAdaptor;
    }
    pub mod decay {
        use super::compat;
        pub use compat::WeightDecayConfig;
    }
    pub mod lr_scheduler {
        use super::compat;
        pub use compat::{ConstantLr, LrScheduler};

        pub mod cosine {
            use super::compat;
            pub use compat::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig};
        }
    }
    pub mod momentum {
        use super::compat;
        pub use compat::MomentumConfig;
    }
}

// ── backend ───────────────────────────────────────────────────────────────────
pub mod backend {
    use super::compat;
    pub use compat::{Autodiff, NdArray};
}

// ── config ────────────────────────────────────────────────────────────────────
pub mod config {
    use super::compat;
    pub use compat::Config;
}

// ── prelude ───────────────────────────────────────────────────────────────────
pub mod prelude {
    use super::compat;
    pub use compat::ElementConversion;
    pub use compat::ElementConversion as ToElement;
}

// ── record ────────────────────────────────────────────────────────────────────
pub mod record {
    use super::compat;
    pub use compat::record::{BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings};
}
