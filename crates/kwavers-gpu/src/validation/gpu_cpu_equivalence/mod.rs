//! GPU/CPU Equivalence Validation Module
//!
//! Provides rigorous validation that real GPU and CPU implementations produce
//! equivalent results within their declared floating-point error bounds.
//!
//! FDTD validation executes the provider-owned Hephaestus WGPU contract and
//! reports provider acquisition or dispatch failures explicitly; it never
//! compares the CPU solver against itself.
//!
//! # Mathematical Foundation
//!
//! ## GPU/CPU equivalence for deterministic operations
//!
//! For a deterministic numerical algorithm `f` implemented on both GPU and CPU
//! with IEEE 754-2008 compliant arithmetic, bitwise equality follows only when
//! the operation order and precision are identical. Provider paths with a
//! different order or native precision use the validator's absolute-or-
//! relative error contract instead.
//!
//! Under those conditions, equal inputs produce equal intermediate values and
//! therefore equal outputs. The condition is not assumed for the provider
//! FDTD path: its WGPU implementation is f32-native and its operation order
//! is validated through the derived error bound above.
//!
//! **Reference**: IEEE Std 754-2008, §5.1; Goldberg (1991) "What Every Computer
//! Scientist Should Know About Floating-Point Arithmetic"
//!
//! ## Parallel reduction equivalence
//!
//! **Statement**: For parallel reduction operations where operation order differs:
//!
//! ```text
//! |f_GPU(x) - f_CPU(x)| / |f_CPU(x)| ≤ (n-1) · ε_machine · κ
//! ```
//!
//! Where:
//! - n = number of terms
//! - ε_machine = 2⁻⁵² ≈ 2.22×10⁻¹⁶ (f64 machine epsilon)
//! - κ = condition number of the summation
//!
//! **Proof**: Follows from floating-point error analysis of parallel prefix sums.
//! Each parallel tree reduction differs from sequential summation by at most
//! (n-1) rounding errors, each bounded by ε_machine · |partial_sum|.
//!
//! **Reference**: Higham (2002) "Accuracy and Stability of Numerical Algorithms", Ch. 4
//!
//! ## Acceptance threshold
//!
//! The legacy reduction validator defaults to `max_relative_error < 1×10⁻¹²`.
//! FDTD's provider-native f32 path derives its bound from f32 machine epsilon
//! and the number of stencil operations, then accepts either the absolute or
//! relative bound at each value.
//!
//! The legacy f64 bound is conservative for the reduction workloads it covers;
//! it is not an acceptance claim for the provider-native f32 FDTD path.
//!
//! # IEEE 754 Compliance Requirements
//!
//! This module validates:
//! 1. Bitwise equality where precision and operation order are identical
//! 2. Bounded relative error for reduction operations (summations, norms)
//! 3. Special value handling (NaN, ±Inf) propagates identically
//! 4. Subnormal number consistency between platforms
//!
//! # References
//!
//! - IEEE Std 754-2008: IEEE Standard for Floating-Point Arithmetic
//! - Goldberg (1991): "What Every Computer Scientist Should Know About
//!   Floating-Point Arithmetic"
//! - Higham (2002): "Accuracy and Stability of Numerical Algorithms", Ch. 4
//! - Whitehead & Fit-Florea (2011): "Precision & Performance: Floating Point
//!   and IEEE 754 Compliance for NVIDIA GPUs"

pub mod constants;
pub mod ieee754;
pub mod report;
pub mod runner;
pub mod validator;

pub use constants::{
    DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE, F64_MACHINE_EPSILON, F64_UNIT_ROUNDOFF,
    MAX_DIVERGENT_FRACTION, MEASUREMENT_STEPS, WARMUP_STEPS,
};
pub use ieee754::{ulps_diff, verify_ieee754_compliance, within_ulps};
pub use report::EquivalenceReport;
pub use runner::{
    validate_equivalence_config, validate_gpu_cpu_equivalence,
    validate_gpu_cpu_equivalence_with_config,
};
pub use validator::EquivalenceValidator;
