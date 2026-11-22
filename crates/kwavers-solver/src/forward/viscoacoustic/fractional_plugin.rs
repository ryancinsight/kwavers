//! Plugin adapter for exact k-space fractional viscoacoustic absorption.

use aequitas::systems::si::quantities::{Length, Time, Velocity};
use kwavers_medium::absorption::FractionalLaplacian;
use leto::{Array3, Array4};

use super::fractional::FractionalAbsorptionOperator;
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_medium::Medium;

/// In-place viscoacoustic pressure filter for plugin-based solver chains.
#[derive(Debug)]
pub struct FractionalViscoacousticPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    model: FractionalLaplacian,
    initial_elapsed: Time,
    prepared: Option<PreparedFilter>,
}

#[derive(Debug, PartialEq)]
struct StepParameters {
    shape: [usize; 3],
    spacing: [Length; 3],
    speed: Velocity,
    elapsed: Time,
}

impl StepParameters {
    fn new(grid: &Grid, medium: &dyn Medium, elapsed: Time) -> KwaversResult<Self> {
        if !medium.is_homogeneous() {
            return Err(KwaversError::InvalidInput(
                "The homogeneous fractional absorption plugin requires a homogeneous medium".into(),
            ));
        }
        Ok(Self {
            shape: [grid.nx, grid.ny, grid.nz],
            spacing: [grid.dx, grid.dy, grid.dz].map(Length::from_base),
            speed: medium.sound_speed(0, 0, 0),
            elapsed,
        })
    }
}

#[derive(Debug)]
struct PreparedFilter {
    parameters: StepParameters,
    operator: FractionalAbsorptionOperator,
    pressure: Array3<Complex64>,
}

impl PreparedFilter {
    fn new(model: FractionalLaplacian, parameters: StepParameters) -> KwaversResult<Self> {
        let operator = FractionalAbsorptionOperator::new(
            parameters.shape,
            parameters.spacing,
            model.step(parameters.speed, parameters.elapsed)?,
        )?;
        Ok(Self {
            pressure: Array3::zeros(parameters.shape),
            operator,
            parameters,
        })
    }
}

impl FractionalViscoacousticPlugin {
    /// Create a new fractional viscoacoustic filter plugin.
    #[must_use]
    pub fn new(model: FractionalLaplacian, dt: Time) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "fractional_viscoacoustic".to_owned(),
                name: "Fractional Viscoacoustic Filter".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Homogeneous calibrated power-law pressure absorption".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            model,
            initial_elapsed: dt,
            prepared: None,
        }
    }
}

impl Plugin for FractionalViscoacousticPlugin {
    crate::plugin_accessors!();

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let parameters = StepParameters::new(grid, medium, self.initial_elapsed)?;
        self.prepared = Some(PreparedFilter::new(self.model, parameters)?);
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let prepared = self.prepared.as_mut().ok_or_else(|| {
            KwaversError::InternalError(
                "Fractional absorption plugin updated before initialization".into(),
            )
        })?;
        let parameters = StepParameters::new(grid, medium, Time::from_base(dt))?;
        let pressure = fields
            .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
            .map_err(|_| {
                KwaversError::InvalidInput(
                    "FractionalViscoacousticPlugin requires a pressure field".to_owned(),
                )
            })?;

        if pressure.shape() != parameters.shape {
            return Err(KwaversError::InvalidInput(
                "Pressure grid differs from the initialized plugin grid".into(),
            ));
        }
        if prepared.parameters != parameters {
            *prepared = PreparedFilter::new(self.model, parameters)?;
        }
        let spectrum = &mut prepared.pressure;
        for (dst, &src) in spectrum.iter_mut().zip(pressure.as_view().iter()) {
            *dst = Complex64::new(src, 0.0);
        }
        prepared.operator.apply(spectrum)?;
        let output = pressure
            .try_iter_mut()
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        for (dst, src) in output.zip(spectrum.iter()) {
            *dst = src.re;
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }
}
