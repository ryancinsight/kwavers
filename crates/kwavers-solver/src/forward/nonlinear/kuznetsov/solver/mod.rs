//! Core Kuznetsov equation solver implementation.
//!
//! Implements the full Kuznetsov equation for nonlinear acoustic wave propagation:
//! ∇²p - (1/c₀²)∂²p/∂t² = -(β/ρ₀c₀⁴)∂²p²/∂t² - (δ/c₀⁴)∂³p/∂t³ + F

mod diagnostics_impl;
mod model_impl;
mod rhs;
mod wave;

pub use wave::KuznetsovWave;
