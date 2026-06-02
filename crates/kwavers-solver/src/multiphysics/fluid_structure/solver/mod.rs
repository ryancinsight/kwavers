//! Fluid-structure interaction solver — SRP submodules.

pub mod struct_impl;
#[cfg(test)]
mod tests;

pub use struct_impl::FluidStructureSolver;
