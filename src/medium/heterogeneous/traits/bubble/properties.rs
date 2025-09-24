//! Bubble dynamics properties implementation for heterogeneous media

use crate::medium::{
    bubble::{BubbleProperties, BubbleState},
    heterogeneous::core::HeterogeneousMedium,
};

impl BubbleProperties for HeterogeneousMedium {
    #[inline]
    fn surface_tension(&self, i: usize, j: usize, k: usize) -> f64 {
        self.surface_tension[[i, j, k]]
    }

    #[inline]
    fn ambient_pressure(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.ambient_pressure
    }

    #[inline]
    fn vapor_pressure(&self, i: usize, j: usize, k: usize) -> f64 {
        self.vapor_pressure[[i, j, k]]
    }

    #[inline]
    fn polytropic_index(&self, i: usize, j: usize, k: usize) -> f64 {
        self.polytropic_index[[i, j, k]]
    }

    #[inline]
    fn gas_diffusion_coeff(&self, i: usize, j: usize, k: usize) -> f64 {
        self.gas_diffusion_coeff[[i, j, k]]
    }
}

impl BubbleState for HeterogeneousMedium {
    #[inline]
    fn bubble_radius(&self, i: usize, j: usize, k: usize) -> f64 {
        self.bubble_radius[[i, j, k]]
    }

    #[inline]
    fn bubble_velocity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.bubble_velocity[[i, j, k]]
    }
}