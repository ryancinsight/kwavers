//! Spatial differential operators used by monolithic residual assembly.

mod laplacian;

pub(in crate::multiphysics::monolithic) use laplacian::laplacian_3d_into;

#[cfg(test)]
mod tests;
