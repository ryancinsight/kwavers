//! Clinical workflow orchestrator — acquisition, processing, fusion, analysis, reporting.

mod acquisition;

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;

use super::super::analysis::{calculate_confidence_score, generate_diagnostic_recommendations};
use super::super::config::{
    ClinicalApplication, ClinicalPhotoacousticConfig, ClinicalWorkflowConfig, ElastographyConfig,
};
use super::super::results::{
    AcquisitionResult, AnalysisResult, ClinicalExaminationResult, ProcessingResult,
    WorkflowTimingMetrics,
};
use super::super::simulation::{
    compute_pa_snr, generate_realistic_elastography_data, generate_realistic_pa_data,
    reconstruct_pa_image,
};
use super::super::state::WorkflowState;
use super::monitor::WorkflowPerformanceMonitor;
use crate::clinical::imaging::photoacoustic::PhotoacousticResult;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::imaging::fusion::{FusedImageResult, FusionConfig};
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use crate::physics::acoustics::imaging::fusion::MultiModalFusion;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Real-time clinical workflow orchestrator.
#[derive(Debug)]
pub struct ClinicalWorkflowOrchestrator {
    pub(super) config: ClinicalWorkflowConfig,
    state: WorkflowState,
    fusion_processor: MultiModalFusion,
    performance_monitor: WorkflowPerformanceMonitor,
}

impl ClinicalWorkflowOrchestrator {
    /// Create a new clinical workflow orchestrator.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: ClinicalWorkflowConfig) -> KwaversResult<Self> {
        let fusion_config = match config.application {
            ClinicalApplication::Oncology => FusionConfig {
                modality_weights: [
                    ("ultrasound".to_owned(), 0.3),
                    ("photoacoustic".to_owned(), 0.4),
                    ("elastography".to_owned(), 0.3),
                ]
                .into(),
                fusion_method: crate::domain::imaging::fusion::ImagingFusionMethod::Probabilistic,
                uncertainty_quantification: true,
                ..Default::default()
            },
            ClinicalApplication::Cardiology => FusionConfig {
                modality_weights: [
                    ("ultrasound".to_owned(), 0.5),
                    ("elastography".to_owned(), 0.5),
                ]
                .into(),
                ..Default::default()
            },
            _ => FusionConfig::default(),
        };

        Ok(Self {
            config,
            state: WorkflowState::Initializing,
            fusion_processor: MultiModalFusion::new(fusion_config),
            performance_monitor: WorkflowPerformanceMonitor::new(),
        })
    }

    /// Execute complete clinical examination workflow.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn execute_examination(
        &mut self,
        patient_id: &str,
    ) -> KwaversResult<ClinicalExaminationResult> {
        let start_time = Instant::now();
        self.performance_monitor.start_monitoring();

        self.state = WorkflowState::Acquiring;
        let acquisition_result = self.acquire_multimodal_data()?;
        self.performance_monitor
            .record_stage("acquisition", start_time.elapsed());

        self.state = WorkflowState::Processing;
        let _processing_result = self.process_realtime_data(acquisition_result)?;
        self.performance_monitor
            .record_stage("processing", start_time.elapsed());

        self.state = WorkflowState::Fusing;
        let fused_result = self.fusion_processor.fuse()?;
        self.performance_monitor
            .record_stage("fusion", start_time.elapsed());

        self.state = WorkflowState::Analyzing;
        let analysis_result = self.perform_ai_analysis(&fused_result)?;
        self.performance_monitor
            .record_stage("analysis", start_time.elapsed());

        self.state = WorkflowState::Reporting;
        let clinical_result = self.generate_clinical_report(
            patient_id,
            fused_result,
            analysis_result,
            start_time.elapsed(),
        )?;

        self.state = WorkflowState::Completed;
        Ok(clinical_result)
    }

    fn acquire_multimodal_data(&mut self) -> KwaversResult<AcquisitionResult> {
        let acquisition_start = Instant::now();

        let ultrasound_data = self.acquire_ultrasound_data()?;
        let pa_result = self.acquire_photoacoustic_data()?;
        let elastography_result = self.acquire_elastography_data()?;

        let acquisition_time = acquisition_start.elapsed();
        if self.config.real_time_enabled
            && acquisition_time > Duration::from_millis(self.config.max_latency_ms / 3)
        {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Acquisition time {}ms exceeds real-time constraint",
                        acquisition_time.as_millis()
                    ),
                },
            ));
        }

        Ok(AcquisitionResult {
            ultrasound_data,
            photoacoustic_result: pa_result,
            elastography_result,
            acquisition_time,
        })
    }

    fn process_realtime_data(
        &mut self,
        acquisition: AcquisitionResult,
    ) -> KwaversResult<ProcessingResult> {
        let processing_start = Instant::now();

        self.fusion_processor
            .register_ultrasound(&acquisition.ultrasound_data)?;
        self.fusion_processor
            .register_photoacoustic(&acquisition.photoacoustic_result.reconstructed_image)?;
        self.fusion_processor
            .register_elastography(&acquisition.elastography_result)?;

        let quality_metrics = self.perform_quality_assessment(&acquisition)?;

        let processing_time = processing_start.elapsed();
        if self.config.real_time_enabled
            && processing_time > Duration::from_millis(self.config.max_latency_ms / 3)
        {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Processing time {}ms exceeds real-time constraint",
                        processing_time.as_millis()
                    ),
                },
            ));
        }

        Ok(ProcessingResult {
            quality_metrics,
            processing_time,
        })
    }

    fn perform_ai_analysis(
        &self,
        fused_result: &FusedImageResult,
    ) -> KwaversResult<AnalysisResult> {
        let tissue_properties = self
            .fusion_processor
            .extract_tissue_properties(fused_result);

        let recommendations = generate_diagnostic_recommendations(&tissue_properties)?;

        let confidence_score = calculate_confidence_score(fused_result, &tissue_properties);
        let confidence_score = if confidence_score.is_nan() || confidence_score.is_infinite() {
            75.0
        } else {
            confidence_score
        };

        Ok(AnalysisResult {
            tissue_properties,
            recommendations,
            confidence_score,
        })
    }

    fn generate_clinical_report(
        &self,
        patient_id: &str,
        fused_result: FusedImageResult,
        analysis: AnalysisResult,
        total_time: Duration,
    ) -> KwaversResult<ClinicalExaminationResult> {
        let performance_metrics = WorkflowTimingMetrics {
            total_time,
            stage_times: self.performance_monitor.get_stage_times(),
            gpu_utilization: self.performance_monitor.get_gpu_utilization(),
            memory_usage_mb: self.performance_monitor.get_memory_usage(),
            real_time_satisfied: total_time < Duration::from_millis(self.config.max_latency_ms),
        };

        Ok(ClinicalExaminationResult {
            patient_id: patient_id.to_owned(),
            timestamp: chrono::Utc::now(),
            fused_image: fused_result,
            tissue_classification: analysis.tissue_properties,
            diagnostic_recommendations: analysis.recommendations,
            quality_metrics: HashMap::new(),
            performance_metrics,
            confidence_score: analysis.confidence_score,
        })
    }

    fn acquire_photoacoustic_data(&self) -> KwaversResult<PhotoacousticResult> {
        let pa_config = ClinicalPhotoacousticConfig {
            _wavelength: 800e-9,
            _optical_energy: 10e-3,
            _absorption_coefficient: 100.0,
            _speed_of_sound: SOUND_SPEED_TISSUE,
            _sampling_frequency: 50e6,
            _num_detectors: 256,
            _detector_radius: 0.025,
            _center_frequency: 5e6,
        };

        let (pressure_fields, time_points) = generate_realistic_pa_data(&pa_config);
        let reconstructed_image = reconstruct_pa_image(&pressure_fields, &pa_config)?;
        let snr = compute_pa_snr(&reconstructed_image);

        Ok(PhotoacousticResult {
            pressure_fields,
            time: time_points,
            reconstructed_image,
            snr,
        })
    }

    fn acquire_elastography_data(&self) -> KwaversResult<ElasticityMap> {
        let elast_config = ElastographyConfig {
            _excitation_frequency: 100.0,
            _push_duration: 200e-6,
            _track_duration: 10e-3,
            _push_focal_depth: 0.03,
            _track_focal_depth: 0.04,
            _frame_rate: 10.0,
            _num_tracking_beams: 8,
        };

        let (youngs_modulus, shear_modulus, shear_wave_speed) =
            generate_realistic_elastography_data(&elast_config);

        Ok(ElasticityMap {
            youngs_modulus,
            shear_modulus,
            shear_wave_speed,
        })
    }

    fn perform_quality_assessment(
        &self,
        acquisition: &AcquisitionResult,
    ) -> KwaversResult<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        metrics.insert("ultrasound_snr".to_owned(), 25.0);
        metrics.insert("ultrasound_cnr".to_owned(), 12.0);
        metrics.insert(
            "photoacoustic_snr".to_owned(),
            acquisition.photoacoustic_result.snr,
        );
        metrics.insert("elastography_snr".to_owned(), 18.0);
        Ok(metrics)
    }

    /// Get current workflow state.
    #[must_use]
    pub fn get_state(&self) -> WorkflowState {
        self.state.clone()
    }

    /// Check if workflow meets real-time constraints.
    #[must_use]
    pub fn check_realtime_performance(&self) -> bool {
        if !self.config.real_time_enabled {
            return true;
        }
        let total_time = self.performance_monitor.get_total_time();
        total_time < Duration::from_millis(self.config.max_latency_ms)
    }
}
