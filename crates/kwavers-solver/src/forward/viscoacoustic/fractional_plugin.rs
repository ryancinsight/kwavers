//! Plugin adapter for exact k-space fractional viscoacoustic absorption.

use std::any::Any;

use leto::Array4;

use super::fractional::FractionalAbsorptionOperator;
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_medium::Medium;

/// Configuration for the exact power-law viscoacoustic absorption filter.
#[derive(Debug, Clone, Copy)]
pub struct FractionalViscoacousticConfig {
    /// Absorption coefficient α₀ in Np/m/(rad/s)^y.
    pub alpha0: f64,
    /// Power-law exponent y.
    pub exponent: f64,
}

/// In-place viscoacoustic pressure filter for plugin-based solver chains.
#[derive(Debug)]
pub struct FractionalViscoacousticPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    config: FractionalViscoacousticConfig,
    dt: f64,
    operator: Option<FractionalAbsorptionOperator>,
    spectral_pressure: Option<leto::Array3<Complex64>>,
}

impl FractionalViscoacousticPlugin {
    /// Create a new fractional viscoacoustic filter plugin.
    #[must_use]
    pub fn new(config: FractionalViscoacousticConfig, dt: f64) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "fractional_viscoacoustic".to_owned(),
                name: "Fractional Viscoacoustic Filter".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description:
                    "Exact Treeby-Cox power-law viscoacoustic absorption/dispersion filter"
                        .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            config,
            dt,
            operator: None,
            spectral_pressure: None,
        }
    }
}

impl Plugin for FractionalViscoacousticPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        Vec::new()
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let c0 = kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        self.operator = Some(FractionalAbsorptionOperator::new(
            grid.nx,
            grid.ny,
            grid.nz,
            grid.dx,
            grid.dy,
            grid.dz,
            self.config.alpha0,
            self.config.exponent,
            c0,
            self.dt,
        )?);
        self.spectral_pressure = Some(leto::Array3::zeros((grid.nx, grid.ny, grid.nz)));
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let operator = self.operator.as_ref().ok_or_else(|| {
            KwaversError::InternalError(
                "FractionalViscoacousticPlugin updated before initialize()".to_owned(),
            )
        })?;
        let spectrum = self.spectral_pressure.as_mut().ok_or_else(|| {
            KwaversError::InternalError(
                "FractionalViscoacousticPlugin missing spectral pressure workspace".to_owned(),
            )
        })?;
        let mut pressure = fields
            .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
            .map_err(|_| {
                KwaversError::InvalidInput(
                    "FractionalViscoacousticPlugin requires a pressure field".to_owned(),
                )
            })?;

        let pressure_copy = pressure.to_contiguous();
        for (dst, &src) in spectrum.iter_mut().zip(pressure_copy.iter()) {
            *dst = Complex64::new(src, 0.0);
        }
        operator.apply(spectrum);
        let [nx, ny, nz] = spectrum.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    pressure[[i, j, k]] = spectrum[[i, j, k]].re;
                }
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
