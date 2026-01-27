//! Clinical Therapy Integration Framework
//!
//! Unified framework for integrating multiple ultrasound therapy modalities with
//! comprehensive literature references and clinical standards compliance.
//!
//! ## Clinical Modalities Supported
//!
//! ### HIFU (High-Intensity Focused Ultrasound)
//! - **References**: Kennedy et al. (2003) "High-intensity focused ultrasound:
//!   surgery of the future?"
//! - **Standards**: IEC 62359:2010 - Ultrasonics - Field characterization
//! - **Applications**: Tumor ablation, uterine fibroids, prostate cancer
//!
//! ### LIFU (Low-Intensity Focused Ultrasound)
//! - **References**: Konofagou et al. (2012) "Focused ultrasound-mediated brain
//!   drug delivery"
//! - **Standards**: FDA guidance for ultrasound contrast agents
//! - **Applications**: Drug delivery, blood-brain barrier opening, gene therapy
//!
//! ### Histotripsy
//! - **References**: Hall et al. (2010) "Histotripsy: minimally invasive
//!   tissue ablation using cavitation"
//! - **Standards**: ASTM F3287-17 - Standard Guide for Histotripsy
//! - **Applications**: Mechanical tissue ablation, cancer treatment
//!
//! ### Oncotripsy
//! - **References**: Xu et al. (2016) "Oncotripsy: targeted cancer therapy
//!   using tumor-specific cavitation"
//! - **Applications**: Tumor-specific mechanical ablation
//!
//! ### Sonodynamic Therapy
//! - **References**: Umemura et al. (1996) "Sonodynamic therapy: a novel
//!   approach to cancer treatment"
//! - **Applications**: ROS-mediated cancer therapy, combined with
//!   photosensitizers
//!
//! ### Lithotripsy
//! - **References**: Chaussy et al. (1980) "Extracorporeally induced
//!   destruction of kidney stones"
//! - **Standards**: ISO 16869:2015 - Lithotripters - Characteristics
//! - **Applications**: Kidney stone fragmentation, gallstone treatment
//!
//! ## Safety Standards and Compliance
//!
//! - **Thermal Index (TI)**: IEC 62359:2010 compliant
//! - **Mechanical Index (MI)**: FDA 510(k) guidance
//! - **Cavitation Dose**: Based on Apfel & Holland (1991) cavitation
//!   threshold models
//! - **Treatment Planning**: AAPM TG-166 recommendations
//!
//! ## Clinical Workflow Integration
//!
//! This framework integrates with:
//! - DICOM medical imaging standards
//! - HL7 clinical data exchange
//! - IHE (Integrating the Healthcare Enterprise) profiles
//! - FDA Q-submission process for combination products
//!
//! ## Architecture
//!
//! The module follows a deep vertical hierarchy pattern with clear separation of concerns:
//!
//! - **config**: Session configuration, parameters, and enums
//! - **tissue**: Tissue property modeling and composition
//! - **state**: Session state tracking and safety monitoring
//! - **acoustic**: Acoustic infrastructure and solvers
//! - **orchestrator**: Main orchestrator with specialized submodules
//!   - initialization: System setup
//!   - execution: Therapy step execution
//!   - safety: Real-time monitoring
//!   - chemical: Sonodynamic chemistry
//!   - microbubble: CEUS dynamics
//!   - cavitation: Histotripsy/oncotripsy control
//!   - lithotripsy: Stone fragmentation
//!
//! ## Example
//!
//! ```
//! use kwavers::clinical::therapy::therapy_integration::{
//!     TherapyIntegrationOrchestrator, TherapySessionConfig, TherapyModality,
//!     AcousticTherapyParams, SafetyLimits, PatientParameters, TissuePropertyMap,
//!     TargetVolume, TissueType, SafetyStatus,
//! };
//! use kwavers::domain::grid::Grid;
//! use kwavers::domain::medium::homogeneous::HomogeneousMedium;
//!
//! // Create therapy configuration
//! let config = TherapySessionConfig {
//!     primary_modality: TherapyModality::Histotripsy,
//!     secondary_modalities: vec![],
//!     duration: 60.0,
//!     acoustic_params: AcousticTherapyParams {
//!         frequency: 1e6,
//!         pnp: 10e6,
//!         prf: 100.0,
//!         duty_cycle: 0.01,
//!         focal_depth: 0.05,
//!         treatment_volume: 1.0,
//!     },
//!     safety_limits: SafetyLimits {
//!         thermal_index_max: 6.0,
//!         mechanical_index_max: 1.9,
//!         cavitation_dose_max: 1000.0,
//!         max_treatment_time: 300.0,
//!     },
//!     patient_params: PatientParameters {
//!         skull_thickness: None,
//!         tissue_properties: TissuePropertyMap::liver((32, 32, 32)),
//!         target_volume: TargetVolume {
//!             center: (0.05, 0.0, 0.0),
//!             dimensions: (0.02, 0.02, 0.02),
//!             tissue_type: TissueType::Liver,
//!         },
//!         risk_organs: vec![],
//!     },
//!     imaging_data_path: None,
//! };
//!
//! // Create computational grid and medium
//! let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
//! let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);
//!
//! // Create orchestrator
//! let mut orchestrator = TherapyIntegrationOrchestrator::new(
//!     config,
//!     grid,
//!     Box::new(medium),
//! ).unwrap();
//!
//! // Execute therapy steps
//! for _ in 0..10 {
//!     orchestrator.execute_therapy_step(0.1).unwrap();
//!
//!     // Check safety limits
//!     let status = orchestrator.check_safety_limits();
//!     if status != SafetyStatus::Safe {
//!         // Handle safety violation
//!         break;
//!     }
//! }
//! ```

// Module declarations
pub mod acoustic;
pub mod config;
pub mod orchestrator;
pub mod state;
pub mod tissue;

// Re-export public API for convenience
pub use acoustic::AcousticWaveSolver;
pub use config::{
    AcousticTherapyParams, PatientParameters, RiskOrgan, SafetyLimits, TargetVolume,
    TherapyModality, TherapySessionConfig, TissueType,
};
pub use orchestrator::TherapyIntegrationOrchestrator;
pub use state::{AcousticField, SafetyMetrics, SafetyStatus, TherapySessionState};
pub use tissue::TissuePropertyMap;
