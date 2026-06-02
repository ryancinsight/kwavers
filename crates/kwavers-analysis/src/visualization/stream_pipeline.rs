use super::stream::FrameMetadata;

/// Stage execution status.
pub type StageResult<T> = Result<T, String>;

/// Per-stage execution metrics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageMetrics {
    pub frames_processed: usize,
    pub failures: usize,
}

/// Pipeline-wide metrics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineMetrics {
    pub extract: StageMetrics,
    pub render: StageMetrics,
    pub encode: StageMetrics,
}

/// Intermediate representation after field extraction.
#[derive(Debug, Clone)]
pub struct ExtractStage {
    pub metadata: FrameMetadata,
    pub scalar_field: Vec<f32>,
    pub dimensions: [usize; 3],
}

/// Intermediate representation after rendering.
#[derive(Debug, Clone)]
pub struct RenderStage {
    pub metadata: FrameMetadata,
    pub width: usize,
    pub height: usize,
    pub rgba8: Vec<u8>,
}

/// Encoded visualization payload.
#[derive(Debug, Clone)]
pub struct EncodedRenderData {
    pub metadata: FrameMetadata,
    pub width: usize,
    pub height: usize,
    pub rgba8: Vec<u8>,
    pub codec: String,
}

/// Final encoding stage output.
pub type EncodeStage = EncodedRenderData;

/// Lightweight pipeline configuration for tests and orchestration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageStagePipelineConfig {
    pub width: usize,
    pub height: usize,
    pub codec: String,
}

impl Default for StagePipelineConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            codec: "raw-rgba8".to_string(),
        }
    }
}

/// Sequential visualization stage pipeline.
#[derive(Debug, Clone)]
pub struct FlatStagePipeline {
    pub config: StagePipelineConfig,
    pub metrics: PipelineMetrics,
}

impl FlatStagePipeline {
    #[must_use]
    pub fn new(config: StagePipelineConfig) -> Self {
        Self {
            config,
            metrics: PipelineMetrics::default(),
        }
    }

    pub fn extract(&mut self, extracted: ExtractStage) -> StageResult<RenderStage> {
        self.metrics.extract.frames_processed += 1;
        let pixel_count = self.config.width * self.config.height;
        let mut rgba8 = vec![0_u8; pixel_count * 4];

        if extracted.scalar_field.is_empty() {
            self.metrics.extract.failures += 1;
            return Err("extract stage received an empty scalar field".to_string());
        }

        let max_value = extracted
            .scalar_field
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1.0);

        for pixel in rgba8.chunks_exact_mut(4) {
            let normalized = (extracted.scalar_field[0] / max_value).clamp(0.0, 1.0);
            let intensity = (normalized * 255.0).round() as u8;
            pixel[0] = intensity;
            pixel[1] = intensity;
            pixel[2] = intensity;
            pixel[3] = 255;
        }

        Ok(RenderStage {
            metadata: extracted.metadata,
            width: self.config.width,
            height: self.config.height,
            rgba8,
        })
    }

    pub fn render(&mut self, rendered: RenderStage) -> StageResult<EncodeStage> {
        self.metrics.render.frames_processed += 1;
        self.metrics.encode.frames_processed += 1;

        if rendered.rgba8.len() != rendered.width * rendered.height * 4 {
            self.metrics.render.failures += 1;
            self.metrics.encode.failures += 1;
            return Err("render stage produced an invalid RGBA payload size".to_string());
        }

        Ok(EncodedRenderData {
            codec: self.config.codec.clone(),
            metadata: rendered.metadata,
            width: rendered.width,
            height: rendered.height,
            rgba8: rendered.rgba8,
        })
    }
}
