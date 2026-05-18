//! Finite-frequency same-aperture operator construction.

use ndarray::Array2;

use super::active_grid::{ActiveGrid, PlanarPoint};
use super::operator::FiniteFrequencyOperator;
use super::row_matrix::RowMatrix;

pub const C_REF_M_S: f64 = crate::core::constants::fundamental::SOUND_SPEED_TISSUE;

#[derive(Clone, Copy, Debug)]
pub struct SameApertureMedium<'a> {
    pub attenuation_np_per_m_mhz: &'a Array2<f64>,
    pub spacing_m: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SameApertureSettings<'a> {
    pub frequencies_hz: &'a [f64],
    pub receiver_offsets: &'a [usize],
    pub phase_speed_m_s: f64,
}

#[must_use]
pub fn fundamental_operator<'a>(
    medium: SameApertureMedium<'a>,
    therapy_elements: &'a [PlanarPoint],
    active: &'a ActiveGrid,
    settings: SameApertureSettings<'a>,
) -> FiniteFrequencyOperator<'a> {
    FiniteFrequencyOperator::fundamental(medium, therapy_elements, active, settings)
}

#[must_use]
pub fn harmonic_operator<'a>(
    medium: SameApertureMedium<'a>,
    therapy_elements: &'a [PlanarPoint],
    active: &'a ActiveGrid,
    settings: SameApertureSettings<'a>,
) -> FiniteFrequencyOperator<'a> {
    FiniteFrequencyOperator::harmonic(medium, therapy_elements, active, settings)
}

#[must_use]
pub fn ultraharmonic_operator<'a>(
    medium: SameApertureMedium<'a>,
    therapy_elements: &'a [PlanarPoint],
    active: &'a ActiveGrid,
    settings: SameApertureSettings<'a>,
) -> FiniteFrequencyOperator<'a> {
    FiniteFrequencyOperator::ultraharmonic(medium, therapy_elements, active, settings)
}

#[must_use]
pub fn passive_operator<'a>(
    medium: SameApertureMedium<'a>,
    therapy_elements: &'a [PlanarPoint],
    imaging_receivers: &'a [PlanarPoint],
    active: &'a ActiveGrid,
    frequencies_hz: &'a [f64],
) -> FiniteFrequencyOperator<'a> {
    FiniteFrequencyOperator::passive(
        medium,
        therapy_elements,
        imaging_receivers,
        active,
        frequencies_hz,
    )
}

#[must_use]
pub fn build_fundamental_matrix(
    medium: SameApertureMedium<'_>,
    therapy_elements: &[PlanarPoint],
    active: &ActiveGrid,
    settings: SameApertureSettings<'_>,
) -> RowMatrix {
    fundamental_operator(medium, therapy_elements, active, settings).materialize()
}

#[must_use]
pub fn build_harmonic_matrix(
    medium: SameApertureMedium<'_>,
    therapy_elements: &[PlanarPoint],
    active: &ActiveGrid,
    settings: SameApertureSettings<'_>,
) -> RowMatrix {
    harmonic_operator(medium, therapy_elements, active, settings).materialize()
}

#[must_use]
pub fn build_ultraharmonic_matrix(
    medium: SameApertureMedium<'_>,
    therapy_elements: &[PlanarPoint],
    active: &ActiveGrid,
    settings: SameApertureSettings<'_>,
) -> RowMatrix {
    ultraharmonic_operator(medium, therapy_elements, active, settings).materialize()
}

#[must_use]
pub fn build_passive_matrix(
    medium: SameApertureMedium<'_>,
    therapy_elements: &[PlanarPoint],
    imaging_receivers: &[PlanarPoint],
    active: &ActiveGrid,
    frequencies_hz: &[f64],
) -> RowMatrix {
    passive_operator(
        medium,
        therapy_elements,
        imaging_receivers,
        active,
        frequencies_hz,
    )
    .materialize()
}
