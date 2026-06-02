//! Compute Backend Abstraction Layer.
//!
//! Defines the `Backend` trait per project standards' "Compute backend trait" rule
//! (CPU, GPU, accelerator dispatch mediated by a single trait so additional targets
//! can land as new `impl Backend` blocks without touching algorithms).
//!
//! ## Submodule status
//!
//! - `traits` — **stable**, unconditional, the public Backend trait surface.
//! - `gpu`    — **unstable / bit-rotted**: this subtree was orphaned from the
//!   crate module graph (no `pub mod backend;` existed in `solver/mod.rs` until
//!   2026-05-05). When wired in under `--features gpu` it surfaces 32 compile
//!   errors against current `wgpu` v26 (e.g. `wgpu::Device::queue()` no longer
//!   exists), the current `KwaversError`/`ConfigError` variants, and a few
//!   `wgpu::Instance`/`DeviceDescriptor` field shapes that have shifted upstream.
//!   Tracked for repair in a separate sprint; until then it stays gated behind
//!   the explicit-opt-in `solver_backend_gpu_unstable` feature so the rest of
//!   the codebase can consume the trait without dragging in the broken impl.
//!
//! Once the `gpu` submodule is repaired, drop the extra feature gate and gate
//! it on `feature = "gpu"` alone (its real dependency).

#[cfg(all(feature = "gpu", feature = "solver_backend_gpu_unstable"))]
pub mod gpu;
pub mod traits;

// Re-export main trait + capability/device types so callers consume them via the
// stable `solver::backend::*` path rather than reaching into `traits::*`.
pub use traits::{BackendCapabilities, BackendType, ComputeBackend, ComputeDevice};

#[cfg(test)]
mod backend_surface_tests {
    //! Confirms the `Backend` trait surface is reachable through the public path.
    //!
    //! Until 2026-05-05 the entire `solver::backend` subtree was orphaned from
    //! the crate module graph (no `pub mod backend;` in `solver/mod.rs`), making
    //! this trait unreachable from any external consumer despite being declared
    //! `pub`. These compile-time checks ensure the trait surface stays wired in.

    /// `Backend` reachable via `crate::backend::*` (canonical path).
    #[test]
    fn backend_trait_reachable_via_canonical_path() {
        // dyn-trait reference forms only if the trait is in scope; bound check
        // forms only if the trait is object-safe — both invariants verified.
        fn _assert_object_safe(_: &dyn crate::backend::ComputeBackend) {}
    }

    /// `BackendType` enum variants compile-time pattern-checked.
    #[test]
    fn backend_type_variants_match_specification() {
        use crate::backend::BackendType;
        // Exhaustive match — fails to compile if a variant is added/removed
        // without updating callers, per project standards' "type-system
        // enforcement" rule.
        let cpu = BackendType::CPU;
        let gpu = BackendType::GPU;
        for variant in [cpu, gpu] {
            match variant {
                BackendType::CPU => {}
                BackendType::GPU => {}
            }
        }
    }

    /// `ComputeDevice` and `BackendCapabilities` constructible (positive value
    /// inspection beyond mere `is_some()` / `is_ok()` per anti-mock rule).
    /// # Panics
    /// - Panics if assertion fails: `test-cpu-0`.
    ///
    #[test]
    fn compute_device_and_capabilities_value_inspect() {
        use crate::backend::{BackendCapabilities, BackendType, ComputeDevice};

        let dev = ComputeDevice {
            id: 0,
            name: "test-cpu-0".to_string(),
            backend_type: BackendType::CPU,
            total_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 14 * 1024 * 1024 * 1024,
            compute_units: 16,
            peak_performance: 1.5e12,
        };
        assert_eq!(dev.id, 0);
        assert_eq!(dev.name, "test-cpu-0");
        assert_eq!(dev.backend_type, BackendType::CPU);
        assert_eq!(dev.total_memory, 16 * 1024 * 1024 * 1024);
        assert!(dev.available_memory < dev.total_memory);
        assert_eq!(dev.compute_units, 16);
        assert!((dev.peak_performance - 1.5e12).abs() < 1.0);

        let caps = BackendCapabilities {
            supports_fft: true,
            supports_f64: true,
            supports_f32: true,
            supports_async: false,
            max_parallelism: 16,
            supports_unified_memory: false,
        };
        assert!(caps.supports_fft);
        assert!(caps.supports_f64);
        assert!(caps.supports_f32);
        assert!(!caps.supports_async);
        assert_eq!(caps.max_parallelism, 16);
        assert!(!caps.supports_unified_memory);
    }
}
