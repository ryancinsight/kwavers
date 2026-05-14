//! Spatial differential operators used by monolithic residual assembly.

mod laplacian;

pub(in crate::solver::multiphysics::monolithic) use laplacian::laplacian_3d_into;

#[cfg(test)]
mod tests;
