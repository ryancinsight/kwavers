//! Viscous properties implementation for heterogeneous media

use crate::medium::{
    heterogeneous::core::HeterogeneousMedium,
    viscous::ViscousProperties,
};

impl ViscousProperties for HeterogeneousMedium {
    #[inline]
    fn shear_viscosity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.shear_viscosity_coeff[[i, j, k]]
    }

    #[inline]
    fn bulk_viscosity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.bulk_viscosity_coeff[[i, j, k]]
    }

    #[inline]
    fn kinematic_viscosity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.viscosity[[i, j, k]]
    }
}