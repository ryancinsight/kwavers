//! Periodic homogeneous power-law absorption in the spatial Fourier basis.
//! Each mode is an eigenfunction of the Medium step. A real even gain preserves
//! the phase of complex modes and conjugacy of the Fourier bins of real pressure.
use aequitas::systems::si::quantities::{Dimensionless, Length, Wavenumber};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{get_fft_for_grid, wavenumber_vec_positive_nyquist, Complex64, Fft3d};
use kwavers_medium::absorption::FractionalAbsorptionStep;
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use std::sync::Arc;

/// Cached dimensionless amplitude response on a three-dimensional periodic grid.
#[derive(Debug, Clone)]
pub struct FractionalAbsorptionOperator {
    gains: Array3<Dimensionless>,
    fft: Option<Arc<Fft3d>>,
}

impl FractionalAbsorptionOperator {
    /// Evaluate the calibrated step at every FFT angular wavenumber magnitude.
    ///
    /// # Errors
    /// Rejects empty or oversized grids, invalid spacings and unevaluable modes.
    pub fn new(
        shape: [usize; 3],
        spacing: [Length; 3],
        step: FractionalAbsorptionStep,
    ) -> KwaversResult<Self> {
        if shape.contains(&0)
            || spacing
                .iter()
                .any(|d| !d.as_base().is_finite() || d.into_base() <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "Fractional absorption requires nonempty axes and positive finite grid spacings"
                    .into(),
            ));
        }
        let count = shape
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .ok_or_else(|| {
                KwaversError::InvalidInput("FFT grid size exceeds addressable range".into())
            })?;
        let mut modes = Vec::new();
        modes
            .try_reserve_exact(count)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        // Build the FFT phase per sample first. Dividing that by spacing
        // forms angular wavenumber without overflowing a long grid's period.
        let [kx, ky, kz]: [Vec<Wavenumber>; 3] = std::array::from_fn(|axis| {
            wavenumber_vec_positive_nyquist(shape[axis], 1.0)
                .into_iter()
                .map(|phase| Dimensionless::from_base(phase) / spacing[axis])
                .collect()
        });
        for x in kx {
            for &y in &ky {
                let transverse = x.into_base().hypot(y.into_base());
                for &z in &kz {
                    let magnitude = Wavenumber::from_base(transverse.hypot(z.into_base()));
                    modes.push(step.gain(magnitude)?);
                }
            }
        }
        // A lossless or zero-duration step is the exact identity, including on
        // floating-point inputs which would round during an FFT round trip.
        let fft = modes
            .iter()
            .any(|gain| gain.into_base() != 1.0)
            .then(|| get_fft_for_grid(shape[0], shape[1], shape[2]));
        let gains = Array3::from_shape_vec(shape, modes)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        Ok(Self { gains, fft })
    }

    /// Advance pressure by the elapsed time used to construct the cached gain.
    ///
    /// # Errors
    /// Rejects a mismatched pressure grid before transforming it.
    pub fn apply(&self, pressure: &mut Array3<Complex64>) -> KwaversResult<()> {
        if pressure.shape() != self.gains.shape() {
            return Err(KwaversError::InvalidInput(
                "Pressure shape differs from the absorption grid".into(),
            ));
        }
        let Some(fft) = &self.fft else {
            return Ok(());
        };
        fft.forward_complex_inplace(pressure);
        if let (Some(samples), Some(gains)) = (pressure.as_slice_mut(), self.gains.as_slice()) {
            enumerate_mut_with::<Adaptive, _, _>(samples, |i, v| *v *= gains[i].into_base());
        } else {
            kwavers_core::traversal::zip_mut(pressure.view_mut(), self.gains.view(), |p, gain| {
                *p *= gain.into_base()
            });
        }
        fft.inverse_complex_inplace(pressure);
        Ok(())
    }
}
