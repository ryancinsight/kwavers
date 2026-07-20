//! Comprehensive liver-assessment workflow.
//!
//! This example composes B-mode imaging, shear-wave elastography,
//! contrast-enhanced ultrasound, uncertainty quantification, safety monitoring,
//! clinical validation, and treatment planning for a liver-fibrosis assessment.
//! The default build presents the workflow contract; the `gpu` feature enables
//! the complete simulation.

use kwavers_core::error::KwaversResult;

#[cfg(feature = "gpu")]
use kwavers_analysis::ml::uncertainty::{
    MlUncertaintyConfig, MlUncertaintyMethod, Seed, UncertaintyQuantifier,
};
#[cfg(feature = "gpu")]
use kwavers_analysis::validation::clinical::ClinicalValidator;
#[cfg(feature = "gpu")]
use kwavers_gpu::gpu::memory::UnifiedMemoryManager;
#[cfg(feature = "gpu")]
use kwavers_grid::Grid;
#[cfg(feature = "gpu")]
use kwavers_medium::heterogeneous::HeterogeneousMedium;
#[cfg(feature = "gpu")]
use kwavers_physics::acoustics::transcranial::safety_monitoring::TranscranialSafetyMonitor;
#[cfg(feature = "gpu")]
use kwavers_simulation::imaging::ceus::ContrastEnhancedUltrasound;

#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/clinical.rs"]
mod clinical;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/execution.rs"]
mod execution;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/metrics.rs"]
mod metrics;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/modalities.rs"]
mod modalities;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/presentation.rs"]
mod presentation;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/results.rs"]
mod results;

#[cfg(feature = "gpu")]
use results::*;

#[cfg(feature = "gpu")]
struct LiverAssessmentWorkflow {
    patient_id: String,
    liver_grid: Grid,
    liver_tissue: HeterogeneousMedium,
    ceus_system: ContrastEnhancedUltrasound,
    safety_monitor: TranscranialSafetyMonitor,
    gpu_memory: UnifiedMemoryManager,
    uncertainty_analyzer: UncertaintyQuantifier,
    clinical_validator: ClinicalValidator,
}

#[cfg(feature = "gpu")]
impl LiverAssessmentWorkflow {
    fn new(patient_id: &str, liver_volume_mm3: (f64, f64, f64)) -> KwaversResult<Self> {
        println!("Initializing liver assessment for patient: {patient_id}");
        println!(
            "Liver volume: {:.1} x {:.1} x {:.1} mm³",
            liver_volume_mm3.0, liver_volume_mm3.1, liver_volume_mm3.2
        );

        let grid_scale = 0.5;
        let grid = Grid::new(
            (liver_volume_mm3.0 * grid_scale) as usize,
            (liver_volume_mm3.1 * grid_scale) as usize,
            (liver_volume_mm3.2 * grid_scale) as usize,
            5e-4,
            5e-4,
            5e-4,
        )?;
        println!(
            "Computational grid: {}x{}x{} cells",
            grid.nx, grid.ny, grid.nz
        );

        let liver_tissue = Self::create_liver_tissue_model(&grid);
        let ceus_system = ContrastEnhancedUltrasound::new(&grid, &liver_tissue, 5.0e6, 3.0)?;
        let safety_monitor = TranscranialSafetyMonitor::new((grid.nx, grid.ny, grid.nz), 0.01, 2e6);
        let uncertainty_analyzer = UncertaintyQuantifier::new(MlUncertaintyConfig {
            method: MlUncertaintyMethod::Hybrid,
            num_samples: 50,
            confidence_level: 0.95,
            dropout_rate: 0.1,
            ensemble_size: 5,
            calibration_size: 100,
            sensitivity_seed: Seed::new(0),
        })?;

        Ok(Self {
            patient_id: patient_id.to_owned(),
            liver_grid: grid,
            liver_tissue,
            ceus_system,
            safety_monitor,
            gpu_memory: UnifiedMemoryManager::new(),
            uncertainty_analyzer,
            clinical_validator: ClinicalValidator::new(),
        })
    }
}

#[cfg(not(feature = "gpu"))]
fn main() -> KwaversResult<()> {
    println!("Comprehensive Clinical Workflow Example");
    println!("========================================");
    println!();
    println!("This example demonstrates a complete liver assessment workflow");
    println!("integrating advanced ultrasound simulation capabilities.");
    println!();
    println!("Note: GPU features required for full workflow execution.");
    println!("Run with --features gpu to enable complete functionality.");
    Ok(())
}

#[cfg(feature = "gpu")]
fn main() -> KwaversResult<()> {
    presentation::run()
}
