//! Tests for the delay-and-sum module.
//!
//! `DelaySumGPU` and `create_element_positions` are only available when the `gpu`
//! feature is enabled.  Tests requiring a live GPU device are integration tests;
//! structural / layout tests run under `cfg(feature = "gpu")` with a mock-free
//! construction path.

// No GPU-independent tests at this scope — see processor/mod.rs for unit tests
// that do not require a device handle, and the integration test suite for
// end-to-end GPU dispatch verification.
