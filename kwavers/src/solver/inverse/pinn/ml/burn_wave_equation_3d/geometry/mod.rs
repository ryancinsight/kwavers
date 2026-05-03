//! `Geometry3D` — spatial domain primitives for 3D PINN wave problems.

pub mod shape;
#[cfg(test)]
mod tests;

pub use shape::Geometry3D;
