//! Plugin adapter for the FDTD Westervelt nonlinear solver.

use std::sync::Arc;

use crate::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_signal::Signal;
use kwavers_source::types::SourceFocalProperties;
use kwavers_source::{Source, SourceField};
use leto::{Array3, Array4};

#[derive(Debug)]
struct ArcSourceAdapter {
    inner: Arc<dyn Source>,
}

impl Source for ArcSourceAdapter {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        self.inner.create_mask(grid)
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        self.inner.add_mask_into(grid, mask);
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        self.inner.create_mask_into(grid, mask);
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.inner.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner.positions()
    }

    fn for_each_position(&self, visitor: &mut dyn FnMut((f64, f64, f64))) {
        self.inner.for_each_position(visitor);
    }

    fn signal(&self) -> &dyn Signal {
        self.inner.signal()
    }

    fn source_type(&self) -> SourceField {
        self.inner.source_type()
    }

    fn initial_amplitude(&self) -> f64 {
        self.inner.initial_amplitude()
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.inner.get_source_term(t, x, y, z, grid)
    }

    fn focal_point(&self) -> Option<(f64, f64, f64)> {
        self.inner.focal_point()
    }

    fn focal_depth(&self) -> Option<f64> {
        self.inner.focal_depth()
    }

    fn spot_size(&self) -> Option<f64> {
        self.inner.spot_size()
    }

    fn f_number(&self) -> Option<f64> {
        self.inner.f_number()
    }

    fn rayleigh_range(&self) -> Option<f64> {
        self.inner.rayleigh_range()
    }

    fn numerical_aperture(&self) -> Option<f64> {
        self.inner.numerical_aperture()
    }

    fn focal_gain(&self) -> Option<f64> {
        self.inner.focal_gain()
    }

    fn get_focal_properties(&self) -> Option<SourceFocalProperties> {
        self.inner.get_focal_properties()
    }
}

fn copy_field_into_view(mut dst: leto::ArrayViewMut3<'_, f64>, src: &Array3<f64>) {
    leto_ops::zip_mut_with(&mut dst, &src.view(), |dst_value, src_value| {
        *dst_value = *src_value;
    })
    .expect("invariant: Westervelt FDTD field/view copy shapes match");
}

/// Catalog plugin wrapping the FDTD Westervelt solver.
#[derive(Debug)]
pub struct WesterveltFdtdPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<WesterveltFdtd>,
}

impl Default for WesterveltFdtdPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl WesterveltFdtdPlugin {
    /// Create a new (uninitialized) FDTD Westervelt plugin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "westervelt_fdtd_solver".to_owned(),
                name: "Westervelt FDTD Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description:
                    "Nonlinear full-wave propagation via the explicit FDTD Westervelt equation"
                        .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
        }
    }
}

impl Plugin for WesterveltFdtdPlugin {
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

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        self.solver = Some(WesterveltFdtd::new(
            WesterveltFdtdConfig::default(),
            grid,
            medium,
        ));
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            KwaversError::InvalidInput(
                "WesterveltFdtdPlugin::update called before initialize".to_owned(),
            )
        })?;
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let pressure_field = fields.index_axis::<3>(0, pressure_idx).map_err(|_| {
            KwaversError::InvalidInput(
                "WesterveltFdtdPlugin requires a pressure field in the unified field stack"
                    .to_owned(),
            )
        })?;
        solver.set_pressure_field(pressure_field);

        let sources: Vec<Box<dyn Source>> = context
            .sources
            .iter()
            .cloned()
            .map(|inner| Box::new(ArcSourceAdapter { inner }) as Box<dyn Source>)
            .collect();
        solver.update(medium, grid, &sources, t, dt)?;

        copy_field_into_view(
            fields
                .index_axis_mut(0, pressure_idx)
                .expect("invariant: pressure field axis index within field stack"),
            solver.pressure(),
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
