//! Plugin adapter for the Hybrid Angular Spectrum nonlinear solver.
//!
//! Each `Plugin::update` call propagates the pressure field forward by one
//! axial step of size `config.dz` metres, applying the HAS operator-split
//! diffraction + nonlinearity.

use crate::forward::nonlinear::hybrid_angular_spectrum::{HASConfig, HybridAngularSpectrum};
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::{Array3, Array4};

fn copy_field_into_view(mut dst: leto::ArrayViewMut3<'_, f64>, src: &Array3<f64>) {
    leto_ops::zip_mut_with(&mut dst, &src.view(), |dst_value, src_value| {
        *dst_value = *src_value;
    })
    .expect("invariant: HAS field/view copy shapes match");
}

/// Catalog plugin wrapping the Hybrid Angular Spectrum nonlinear solver.
#[derive(Debug)]
pub struct HybridAngularSpectrumPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    config: HASConfig,
    solver: Option<HybridAngularSpectrum>,
}

impl Default for HybridAngularSpectrumPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridAngularSpectrumPlugin {
    /// Create a new (uninitialized) HAS plugin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "hybrid_angular_spectrum_solver".to_owned(),
                name: "Hybrid Angular Spectrum Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description:
                    "Nonlinear propagation via the Hybrid Angular Spectrum operator-splitting method"
                        .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            config: HASConfig::default(),
            solver: None,
        }
    }
}

impl Plugin for HybridAngularSpectrumPlugin {
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
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.solver = Some(HybridAngularSpectrum::new(grid, self.config.clone())?);
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
        let solver = self.solver.as_ref().ok_or_else(|| {
            KwaversError::InvalidInput(
                "HybridAngularSpectrumPlugin::update called before initialize".to_owned(),
            )
        })?;
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let pressure_view = fields.index_axis::<3>(0, pressure_idx).map_err(|_| {
            KwaversError::InvalidInput(
                "HybridAngularSpectrumPlugin requires a pressure field in the unified field stack"
                    .to_owned(),
            )
        })?;
        let [nx, ny, nz] = pressure_view.shape();
        let pressure =
            Array3::from_shape_vec((nx, ny, nz), pressure_view.iter().copied().collect())
                .expect("invariant: HAS pressure field shape must map to Array3");
        let propagated = solver.propagate(&pressure, self.config.dz)?;
        copy_field_into_view(
            fields
                .index_axis_mut(0, pressure_idx)
                .expect("invariant: pressure field axis index within field stack"),
            &propagated,
        );
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.solver = None;
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
