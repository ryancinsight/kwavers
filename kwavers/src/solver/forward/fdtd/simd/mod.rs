//! SIMD Bounded Context
//!
//! Encapsulates both generic cache-tiled SIMD pathways and specific hardware
//! accelerators like AVX-512 into deeply nested verticals (<400 loc).

pub mod avx512;
pub mod generic;
