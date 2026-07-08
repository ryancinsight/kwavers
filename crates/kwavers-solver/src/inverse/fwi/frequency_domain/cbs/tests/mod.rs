use super::absorbing::absorbing_weights;
use super::green::{apply_shifted_green, apply_shifted_green_adjoint, shifted_outgoing_green};
use super::grid::{bli_weights, BliConfig, GridSpec};
use super::spectral::{
    apply_shifted_green_pstd_spectral_adjoint_with_boundary,
    apply_shifted_green_pstd_spectral_with_boundary, apply_shifted_green_spectral,
    apply_shifted_green_spectral_adjoint, apply_shifted_green_spectral_adjoint_with_boundary,
    apply_shifted_green_spectral_with_boundary,
};
use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers_transducer::transducers::ElementPosition;
use eunomia::Complex64;

mod absorbing;
mod grid_green;
mod potential;
mod receiver;
mod solver;
mod source_temporal;
mod spectral_green;

pub(super) fn inner_product(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&left, &right)| left.conj() * right)
        .sum()
}

pub(super) fn norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

pub(super) fn angular_mode_for_test(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TWO_PI * signed_index / (count as f64 * spacing_m)
}
