//! Dense shifted outgoing Green operator for CBS.

use super::absorbing::AbsorbingBoundary;
use super::grid::GridSpec;
use super::temporal::PstdTemporalBinConfig;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_domain::source::transducers::ElementPosition;
use num_complex::Complex64;

pub(super) fn shifted_wavenumber(reference_wavenumber: f64, epsilon: f64) -> Complex64 {
    Complex64::new(reference_wavenumber * reference_wavenumber, epsilon).sqrt()
}

pub(super) fn shifted_outgoing_green(
    source: ElementPosition,
    receiver: ElementPosition,
    shifted_wavenumber: Complex64,
    min_distance_m: f64,
) -> Complex64 {
    let dx = source.x_m - receiver.x_m;
    let dy = source.y_m - receiver.y_m;
    let dz = source.z_m - receiver.z_m;
    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(min_distance_m);
    (Complex64::new(0.0, 1.0) * shifted_wavenumber * distance).exp() / (FOUR_PI * distance)
}

pub(super) fn apply_shifted_green(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    source_density: &[Complex64],
) -> Vec<Complex64> {
    let centers = grid.centers();
    let shifted = shifted_wavenumber(reference_wavenumber, epsilon);
    let min_distance = grid.min_distance_m();
    let cell_volume = grid.cell_volume_m3();
    centers
        .iter()
        .map(|(_, receiver)| {
            centers
                .iter()
                .zip(source_density.iter())
                .map(|((_, source), &density)| {
                    shifted_outgoing_green(*source, *receiver, shifted, min_distance)
                        * density
                        * cell_volume
                })
                .sum()
        })
        .collect()
}

pub(super) fn apply_shifted_green_operator(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    source_density: &[Complex64],
    operator: GreenOperatorKind,
) -> Vec<Complex64> {
    match operator {
        GreenOperatorKind::DenseFreeSpace => {
            apply_shifted_green(grid, reference_wavenumber, epsilon, source_density)
        }
        GreenOperatorKind::SpectralPeriodic { absorbing_boundary } => {
            super::spectral::apply_shifted_green_spectral_with_boundary(
                grid,
                reference_wavenumber,
                epsilon,
                source_density,
                absorbing_boundary,
            )
        }
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s,
            reference_sound_speed_m_s,
            absorbing_boundary,
            ..
        } => super::spectral::apply_shifted_green_pstd_spectral_with_boundary(
            grid,
            reference_wavenumber,
            epsilon,
            source_density,
            time_step_s,
            reference_sound_speed_m_s,
            absorbing_boundary,
        ),
    }
}

pub(crate) fn apply_shifted_green_adjoint(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    field: &[Complex64],
) -> Vec<Complex64> {
    let centers = grid.centers();
    let shifted = shifted_wavenumber(reference_wavenumber, epsilon);
    let min_distance = grid.min_distance_m();
    let cell_volume = grid.cell_volume_m3();
    centers
        .iter()
        .map(|(_, source)| {
            centers
                .iter()
                .zip(field.iter())
                .map(|((_, receiver), &value)| {
                    shifted_outgoing_green(*source, *receiver, shifted, min_distance).conj()
                        * value
                        * cell_volume
                })
                .sum()
        })
        .collect()
}

pub(crate) fn apply_shifted_green_adjoint_operator(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    field: &[Complex64],
    operator: GreenOperatorKind,
) -> Vec<Complex64> {
    match operator {
        GreenOperatorKind::DenseFreeSpace => {
            apply_shifted_green_adjoint(grid, reference_wavenumber, epsilon, field)
        }
        GreenOperatorKind::SpectralPeriodic { absorbing_boundary } => {
            super::spectral::apply_shifted_green_spectral_adjoint_with_boundary(
                grid,
                reference_wavenumber,
                epsilon,
                field,
                absorbing_boundary,
            )
        }
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s,
            reference_sound_speed_m_s,
            absorbing_boundary,
            ..
        } => super::spectral::apply_shifted_green_pstd_spectral_adjoint_with_boundary(
            grid,
            reference_wavenumber,
            epsilon,
            field,
            time_step_s,
            reference_sound_speed_m_s,
            absorbing_boundary,
        ),
    }
}

/// CBS Green operator discretization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GreenOperatorKind {
    /// Dense free-space outgoing Green convolution over the finite volume.
    DenseFreeSpace,
    /// Pseudospectral inverse of `∇² + k0² + i epsilon`.
    SpectralPeriodic {
        /// Absorbing boundary policy applied as `W G W`.
        absorbing_boundary: AbsorbingBoundary,
    },
    /// Pseudospectral inverse using the PSTD leapfrog/k-space modal symbol.
    SpectralPstdPeriodic {
        /// PSTD time step [s] used by the acquisition generator.
        time_step_s: f64,
        /// Reference homogeneous sound speed [m/s].
        reference_sound_speed_m_s: f64,
        /// Optional finite-window source/bin transfer for one drive frequency.
        temporal_transfer: Option<PstdTemporalBinConfig>,
        /// Absorbing boundary policy applied as `W G W`.
        absorbing_boundary: AbsorbingBoundary,
    },
}
