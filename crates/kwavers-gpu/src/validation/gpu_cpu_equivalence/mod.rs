//! GPU/CPU Equivalence Validation Module
//!
//! Provides rigorous validation that real GPU and CPU implementations produce
//! equivalent results within IEEE 754 machine epsilon bounds.
//!
//! FDTD validation currently returns a report with an unavailable-provider
//! failure reason until the FDTD path has a real provider-generic
//! Leto/Hephaestus GPU trait implementation. WGPU and CUDA belong behind that
//! trait seam; this module must not treat a CPU fallback as GPU equivalence
//! evidence.
//!
//! # Mathematical Foundation
//!
//! ## THEOREM: GPU/CPU Equivalence for Deterministic Operations
//!
//! **Statement**: For a deterministic numerical algorithm f implemented on both
//! GPU and CPU with IEEE 754-2008 compliant arithmetic:
//!
//! ```text
//! f_GPU(x) = f_CPU(x) ∀ x ∈ 𝔽ⁿ (bitwise identical)
//! ```
//!
//! **Proof Sketch**:
//! 1. IEEE 754-2008 guarantees deterministic rounding for basic operations (+, -, *, /, √)
//! 2. For algorithms using only these operations with identical operation ordering:
//!    - Same inputs → same intermediate results → same final results
//! 3. GPU and CPU both implement IEEE 754-2008 for f64/f32
//! 4. Therefore f_GPU(x) = f_CPU(x) (bitwise)
//!
//! **Reference**: IEEE Std 754-2008, §5.1; Goldberg (1991) "What Every Computer
//! Scientist Should Know About Floating-Point Arithmetic"
//!
//! ## THEOREM: Parallel Reduction Equivalence
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
//! ## COROLLARY: Acceptance Threshold
//!
//! For practical n ≤ 10⁹ grid points: max_relative_error < 1×10⁻¹²
//!
//! This bound is conservative (actual error typically < 10⁻¹⁵).
//!
//! # IEEE 754 Compliance Requirements
//!
//! This module validates:
//! 1. Bitwise equality for deterministic operations (stencils, pointwise ops)
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
