use super::{adjusted_quality, RenderQuality};

#[cfg(feature = "gpu-visualization")]
use crate::visualization::{
    TransferMode, VisualizationConfig, VisualizationEngine, VisualizationTransferProvider,
};
#[cfg(feature = "gpu-visualization")]
use kwavers_field::UnifiedFieldType;
#[cfg(feature = "gpu-visualization")]
use kwavers_grid::Grid;
#[cfg(feature = "gpu-visualization")]
use leto::Array3;

const TARGET_FPS: f64 = 60.0;

#[test]
fn quality_downgrades_below_the_lower_hysteresis_boundary() {
    assert_eq!(
        adjusted_quality(RenderQuality::Publication, 47.0, TARGET_FPS),
        RenderQuality::Production
    );
    assert_eq!(
        adjusted_quality(RenderQuality::Low, 47.0, TARGET_FPS),
        RenderQuality::Low
    );
}

#[test]
fn quality_upgrades_above_the_upper_hysteresis_boundary() {
    assert_eq!(
        adjusted_quality(RenderQuality::Draft, 73.0, TARGET_FPS),
        RenderQuality::Low
    );
    assert_eq!(
        adjusted_quality(RenderQuality::Publication, 73.0, TARGET_FPS),
        RenderQuality::Publication
    );
}

#[test]
fn quality_is_stable_inside_the_hysteresis_band() {
    for current_fps in [48.0, TARGET_FPS, 72.0] {
        assert_eq!(
            adjusted_quality(RenderQuality::High, current_fps, TARGET_FPS),
            RenderQuality::High
        );
    }
}

#[cfg(feature = "gpu-visualization")]
#[derive(Debug, Default)]
struct RetainingProvider {
    transfer: Option<(UnifiedFieldType, Vec<f32>, TransferMode)>,
}

#[cfg(feature = "gpu-visualization")]
impl VisualizationTransferProvider for RetainingProvider {
    fn device_name(&self) -> &'static str {
        "retaining-provider"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        samples: &[f32],
        mode: TransferMode,
    ) -> kwavers_core::error::KwaversResult<()> {
        self.transfer = Some((field_type, samples.to_vec(), mode));
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.transfer.as_ref().map_or(0, |(_, samples, _)| {
            samples.len() * std::mem::size_of::<f32>()
        })
    }
}

#[cfg(feature = "gpu-visualization")]
#[test]
fn adaptive_quality_reconfigures_the_initialized_renderer() {
    let mut config = VisualizationConfig::quality();
    config.quality = RenderQuality::Publication;
    config.enable_profiling = true;
    config.target_fps = TARGET_FPS;
    let mut engine = VisualizationEngine::create(config)
        .expect("visualization configuration is valid")
        .set_transfer_provider(RetainingProvider::default());
    engine
        .initialize_gpu()
        .expect("retaining provider initializes the visualization pipeline");

    let grid = Grid::new(1, 1, 256, 1.0, 1.0, 1.0).expect("test grid is valid");
    let mut field = Array3::zeros((1, 1, 256));
    *field
        .get_mut([0, 0, 1])
        .expect("impulse index is in bounds") = 1.0;
    let publication = engine
        .renderer
        .as_mut()
        .expect("renderer is initialized")
        .render_field(&field, UnifiedFieldType::Pressure, &grid)
        .expect("publication rendering succeeds");

    engine.metrics.update(20.0, 20.0);
    assert!(engine.config.enable_profiling);
    assert_eq!(engine.metrics.current().fps, 25.0);
    assert_eq!(
        adjusted_quality(
            engine.config.quality,
            engine.metrics.current().fps,
            engine.config.target_fps,
        ),
        RenderQuality::Production
    );
    engine.auto_adjust_quality();

    assert_eq!(engine.config.quality, RenderQuality::Production);
    let production = engine
        .renderer
        .as_mut()
        .expect("renderer remains initialized")
        .render_field(&field, UnifiedFieldType::Pressure, &grid)
        .expect("production rendering succeeds");
    assert_eq!(publication.get(3).copied(), Some(u8::MAX));
    assert_eq!(production.get(3).copied(), Some(0));
}
