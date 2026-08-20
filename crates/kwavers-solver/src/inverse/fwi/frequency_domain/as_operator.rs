//! Angular-spectrum split-step Helmholtz forward operator (FWI-024-C).
//!
//! Implements the [`HelmholtzForwardOperator`] seam with a one-way
//! downward-continuation model: the transmitted field is propagated plane by
//! plane along the reconstruction grid's z axis using the FFT angular-spectrum
//! method (Goodman 2005), with an optional split-step slowness phase screen per
//! plane (the one-way approximation of the heterogeneous Helmholtz equation).
//!
//! # Why this operator
//!
//! The convergent Born-series operators ([`super::DenseConvergentBornOperator`],
//! [`super::SpectralConvergentBornOperator`]) are two-way: they include
//! backscatter. For survey-scale transmission problems the useful signal is the
//! forward-scattered field, and a one-way angular-spectrum step is far cheaper
//! per propagation step (two 2-D FFTs per plane instead of a convergent series
//! solve). The split-step phase screen is the standard USCT forward model
//! (Ali 2022; the `FullWaveformInversionUSCT` reference).
//!
//! # Divergence from CBS (recorded, not hidden)
//!
//! Angular-spectrum split-step is **one-way**: it propagates the downgoing
//! field and never forms backscattered energy, so it diverges from the
//! two-way CBS operators wherever reflections matter (strong-contrast
//! interfaces, layered media). The differential oracle tests assert agreement
//! on a weak-contrast phantom within a bound that vanishes as the contrast
//! tends to zero — the regime where the one-way approximation is exact.

use aequitas::systems::si::units::Meter;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers_transducer::transducers::ElementPosition;
use leto::{Array2, Array3};

use super::cbs::GridSpec;
use super::{Config, HelmholtzForwardOperator};

/// Angular-spectrum split-step forward operator configuration.
///
/// Stateless value type held on the dyn-dispatched [`Config::forward_operator`].
#[derive(Clone, Debug, PartialEq)]
pub struct AngularSpectrumSplitStepOperator {
    /// Whether the per-plane slowness phase screen is applied. When disabled
    /// the operator reduces to pure angular-spectrum propagation in the
    /// homogeneous reference medium, which isolates the diffraction path in
    /// tests and diagnostics.
    pub phase_screen: bool,
    /// Radial taper width (in grid cells) applied to the source plane to
    /// suppress FFT wraparound at the lateral boundaries. Zero disables the
    /// taper.
    pub source_taper_cells: usize,
}

impl Default for AngularSpectrumSplitStepOperator {
    fn default() -> Self {
        Self {
            phase_screen: true,
            source_taper_cells: 2,
        }
    }
}

impl AngularSpectrumSplitStepOperator {
    /// Propagate one source plane through the slowness volume, returning the
    /// complex field sampled at the receiver element positions.
    fn propagate_transmit(
        &self,
        slowness_s_per_m: &Array3<f64>,
        source_plane: Array2<Complex64>,
        source_z_index: usize,
        frequency_hz: f64,
        config: &Config,
        receiver_positions: &[ElementPosition],
    ) -> KwaversResult<Vec<Complex64>> {
        let shape = slowness_s_per_m.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        let dz = config.spacing_m;
        let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
        let omega = TWO_PI * frequency_hz;

        // The grid is centered at 0.5*N; a physical coordinate maps to a grid
        // index via `index = coordinate/spacing + 0.5*N`.
        let center = [0.5 * nx as f64, 0.5 * ny as f64, 0.5 * nz as f64];

        let kx_axis = wavenumber_axis(nx, config.spacing_m);
        let ky_axis = wavenumber_axis(ny, config.spacing_m);
        let mut spectrum = Array2::zeros([nx, ny]);
        let reference_wavenumber = omega / config.reference_sound_speed_m_s;

        // Propagate from the source plane forward and backward along z,
        // storing every plane. The source plane itself is stored before any
        // step (the phase screen applies *between* planes).
        let mut planes: Vec<Option<Array2<Complex64>>> = vec![None; nz];

        // Forward pass: source plane → deeper z.
        let mut field = source_plane.clone();
        for (iz, slot) in planes.iter_mut().enumerate().skip(source_z_index) {
            if self.phase_screen {
                apply_phase_screen(
                    &mut field,
                    slowness_s_per_m,
                    iz,
                    omega,
                    reference_slowness,
                    dz,
                )?;
            }
            *slot = Some(field.clone());
            if iz + 1 < nz {
                propagate_plane(
                    &mut field,
                    &mut spectrum,
                    &kx_axis,
                    &ky_axis,
                    dz,
                    reference_wavenumber,
                )?;
            }
        }

        // Backward pass: source plane → shallower z.
        let mut field = source_plane;
        for iz in (0..=source_z_index).rev() {
            if iz < source_z_index && self.phase_screen {
                apply_phase_screen(
                    &mut field,
                    slowness_s_per_m,
                    iz,
                    omega,
                    reference_slowness,
                    dz,
                )?;
            }
            if planes[iz].is_none() {
                planes[iz] = Some(field.clone());
            }
            if iz > 0 {
                // Propagate upward by dz (negative z direction).
                propagate_plane(
                    &mut field,
                    &mut spectrum,
                    &kx_axis,
                    &ky_axis,
                    -dz,
                    reference_wavenumber,
                )?;
            }
        }

        // Sample the propagated field at receiver element positions using
        // nearest-grid-point interpolation in the x-y plane at the receiver's z.
        let mut output = Vec::with_capacity(receiver_positions.len());
        for position in receiver_positions {
            let x_m = position.x.in_unit::<Meter>();
            let y_m = position.y.in_unit::<Meter>();
            let z_m = position.z.in_unit::<Meter>();
            let ix = ((x_m / config.spacing_m) + center[0]).round() as isize;
            let iy = ((y_m / config.spacing_m) + center[1]).round() as isize;
            let iz = ((z_m / config.spacing_m) + center[2]).round() as isize;
            if ix < 0
                || iy < 0
                || iz < 0
                || ix >= nx as isize
                || iy >= ny as isize
                || iz >= nz as isize
            {
                // Receiver outside the propagated volume: no signal.
                output.push(Complex64::new(0.0, 0.0));
                continue;
            }
            let plane = planes[iz as usize].as_ref().ok_or_else(|| {
                KwaversError::InvalidInput(format!("no field stored at z index {iz}"))
            })?;
            output.push(plane[[ix as usize, iy as usize]]);
        }
        Ok(output)
    }
}

/// Sample the source field on the grid's first z-plane for one transmit.
///
/// The source is the superposition of cylindrical waves from each element of
/// the transmit row (the array's `cylindrical_source`), evaluated at every
/// grid point of the first z-plane.
fn build_source_plane(
    sources: &[ElementPosition],
    nx: usize,
    ny: usize,
    spacing_m: f64,
    frequency_hz: f64,
    reference_sound_speed_m_s: f64,
    taper_cells: usize,
) -> Array2<Complex64> {
    let center = [0.5 * nx as f64, 0.5 * ny as f64];
    let omega = TWO_PI * frequency_hz;
    let wavenumber = omega / reference_sound_speed_m_s;
    let mut plane = Array2::<Complex64>::zeros([nx, ny]);
    for ix in 0..nx {
        for iy in 0..ny {
            let x_m = (ix as f64 - center[0]) * spacing_m;
            let y_m = (iy as f64 - center[1]) * spacing_m;
            let mut value = Complex64::new(0.0, 0.0);
            for source in sources {
                let sx = source.x.in_unit::<Meter>();
                let sy = source.y.in_unit::<Meter>();
                let sz = source.z.in_unit::<Meter>();
                let distance = ((x_m - sx).powi(2) + (y_m - sy).powi(2) + sz.powi(2)).sqrt();
                // Cylindrical-wave Green function: exp(ikr)/r (outward wave,
                // unit amplitude at unit distance).
                let min_distance = 0.5 * spacing_m;
                let r = distance.max(min_distance);
                value += Complex64::from_polar(1.0 / r, wavenumber * r);
            }
            // Radial taper to suppress FFT wraparound.
            if taper_cells > 0 {
                let rx = (ix as f64 - center[0]).abs();
                let ry = (iy as f64 - center[1]).abs();
                let edge = (nx as f64 / 2.0 - taper_cells as f64).max(0.0);
                let taper_x = ((edge - rx) / taper_cells as f64).clamp(0.0, 1.0);
                let taper_y = ((edge - ry) / taper_cells as f64).clamp(0.0, 1.0);
                value *= Complex64::new(taper_x.min(taper_y), 0.0);
            }
            plane[[ix, iy]] = value;
        }
    }
    plane
}

/// Propagate a complex plane by `dz` along z using the angular-spectrum method.
fn propagate_plane(
    field: &mut Array2<Complex64>,
    spectrum: &mut Array2<Complex64>,
    kx_axis: &[f64],
    ky_axis: &[f64],
    dz: f64,
    wavenumber: f64,
) -> KwaversResult<()> {
    // Forward FFT.
    *spectrum = fft_2d_complex(field);
    let shape = field.shape();
    let (nx, ny) = (shape[0], shape[1]);
    for ix in 0..nx {
        for iy in 0..ny {
            let kx = kx_axis[ix];
            let ky = ky_axis[iy];
            let kz_sq = wavenumber.mul_add(wavenumber, -(kx * kx)) - ky * ky;
            if kz_sq >= 0.0 {
                let kz = kz_sq.sqrt();
                let phase = Complex64::from_polar(1.0, kz * dz);
                spectrum[[ix, iy]] *= phase;
            } else {
                // Evanescent waves decay.
                let kz = (-kz_sq).sqrt();
                spectrum[[ix, iy]] *= Complex64::new((-kz * dz).exp(), 0.0);
            }
        }
    }
    *field = ifft_2d_complex(spectrum);
    Ok(())
}

/// Apply the split-step slowness phase screen to one z-plane.
fn apply_phase_screen(
    field: &mut Array2<Complex64>,
    slowness_s_per_m: &Array3<f64>,
    iz: usize,
    omega: f64,
    reference_slowness: f64,
    dz: f64,
) -> KwaversResult<()> {
    let shape_field = field.shape();
    let (nx, ny) = (shape_field[0], shape_field[1]);
    let shape = slowness_s_per_m.shape();
    if iz >= shape[2] {
        return Err(KwaversError::InvalidInput(format!(
            "phase-screen z index {iz} exceeds volume depth {}",
            shape[2]
        )));
    }
    for ix in 0..nx {
        for iy in 0..ny {
            let slowness = slowness_s_per_m[[ix, iy, iz]];
            if !slowness.is_finite() || slowness <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "slowness must be positive and finite, got {slowness} at ({ix},{iy},{iz})"
                )));
            }
            let contrast = slowness - reference_slowness;
            let phase = Complex64::from_polar(1.0, -omega * contrast * dz);
            field[[ix, iy]] *= phase;
        }
    }
    Ok(())
}

/// The discrete wavenumber axis for the angular-spectrum method (Goodman 2005).
fn wavenumber_axis(n: usize, spacing_m: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let k = if i < n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            TWO_PI * k / (n as f64 * spacing_m)
        })
        .collect()
}

impl HelmholtzForwardOperator for AngularSpectrumSplitStepOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        self.validate()?;
        let shape = slowness_s_per_m.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                "ASM slowness volume must be nonempty".to_owned(),
            ));
        }
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ASM frequency must be positive and finite, got {frequency_hz}"
            )));
        }
        if transmissions > array.circumferential_elements() {
            return Err(KwaversError::InvalidInput(format!(
                "ASM transmissions {transmissions} exceed circumferential elements {}",
                array.circumferential_elements()
            )));
        }

        let mut output = Array2::<Complex64>::zeros([transmissions, array.element_count()]);
        let center_z = 0.5 * nz as f64;
        for transmit in 0..transmissions {
            let sources = array.cylindrical_source(transmit);
            let source_plane = build_source_plane(
                &sources,
                nx,
                ny,
                config.spacing_m,
                frequency_hz,
                config.reference_sound_speed_m_s,
                self.source_taper_cells,
            );
            // The transmit elements define the source z-plane: use the mean of
            // their z positions mapped into the grid.
            let source_z_m = sources
                .iter()
                .map(|source| source.z.in_unit::<Meter>())
                .sum::<f64>()
                / sources.len() as f64;
            let source_z_index = ((source_z_m / config.spacing_m) + center_z).round() as isize;
            let source_z_index = if source_z_index < 0 {
                0
            } else if source_z_index >= nz as isize {
                nz - 1
            } else {
                source_z_index as usize
            };
            let receiver_values = self.propagate_transmit(
                slowness_s_per_m,
                source_plane,
                source_z_index,
                frequency_hz,
                config,
                array.elements(),
            )?;
            for (receiver_index, value) in receiver_values.into_iter().enumerate() {
                output[[transmit, receiver_index]] = value;
            }
        }
        Ok(output)
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        false
    }

    fn model_id(&self) -> &'static str {
        "angular_spectrum_split_step"
    }

    fn validate(&self) -> KwaversResult<()> {
        // Config values are validated by the caller; the operator's own fields
        // are structurally valid by construction (bool + usize).
        Ok(())
    }

    fn validate_for_grid(&self, _grid: GridSpec) -> KwaversResult<()> {
        Ok(())
    }
}
