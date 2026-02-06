//! Therapy Modalities Module
//!
//! This module provides definitions for different therapeutic ultrasound modalities,
//! including their mechanisms of action, safety characteristics, and clinical applications.
//!
//! ## Architecture
//!
//! This module resides in the **clinical/therapy** layer because therapy modalities
//! are clinical concepts that describe treatment protocols and application strategies,
//! not domain primitives.
//!
//! ## Available Modalities
//!
//! ### Thermal Therapy
//! - **HIFU (High-Intensity Focused Ultrasound)**: Thermal ablation via focused energy
//!   - Frequency: 0.5-3 MHz
//!   - Intensity: 100-10,000 W/cm²
//!   - Mechanism: Coagulative necrosis
//!   - Applications: Tumor ablation, uterine fibroids
//!
//! ### Mechanical Therapy
//! - **Histotripsy**: Mechanical tissue fractionation via cavitation
//!   - Frequency: 0.5-3 MHz
//!   - Peak negative pressure: >10 MPa
//!   - Mechanism: Cavitation cloud formation
//!   - Applications: Tissue debulking, thrombolysis
//!
//! - **BBB Opening (Blood-Brain Barrier)**: Microbubble-mediated permeability
//!   - Frequency: 0.2-0.7 MHz
//!   - Pressure: 0.1-0.6 MPa (with microbubbles)
//!   - Mechanism: Sonoporation
//!   - Applications: Drug delivery to CNS
//!
//! ### Low-Intensity Therapy
//! - **LIFU (Low-Intensity Focused Ultrasound)**: Neuromodulation
//!   - Frequency: 0.2-1 MHz
//!   - Intensity: 0.1-10 W/cm²
//!   - Mechanism: Mechanosensitive ion channels
//!   - Applications: Non-invasive brain stimulation
//!
//! - **Sonoporation**: Cell membrane permeabilization
//!   - Frequency: 1-3 MHz
//!   - Intensity: 0.5-5 W/cm²
//!   - Mechanism: Cavitation-induced pores
//!   - Applications: Gene/drug delivery
//!
//! - **Sonodynamic Therapy**: Sonosensitizer activation
//!   - Frequency: 1-3 MHz
//!   - Intensity: 1-3 W/cm²
//!   - Mechanism: ROS generation
//!   - Applications: Cancer treatment
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::clinical::therapy::modalities::{TherapyModality, TherapyMechanism};
//!
//! let modality = TherapyModality::HIFU;
//!
//! println!("Modality: {:?}", modality);
//! println!("Has thermal effects: {}", modality.has_thermal_effects());
//! println!("Has cavitation: {}", modality.has_cavitation());
//! println!("Primary mechanism: {:?}", modality.primary_mechanism());
//! ```
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: This module was moved from `domain::therapy::modalities` to
//! `clinical::therapy::modalities` in Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ### Old Import (No Longer Valid)
//! ```rust,ignore
//! use crate::domain::therapy::modalities::{TherapyModality, TherapyMechanism};
//! ```
//!
//! ### New Import (Correct Location)
//! ```rust,ignore
//! use crate::clinical::therapy::modalities::{TherapyModality, TherapyMechanism};
//! ```
//!
//! ## Clinical Applications
//!
//! ### Oncology
//! - **HIFU**: Prostate, liver, kidney, pancreatic tumors
//! - **Histotripsy**: Liver tumors, BPH (benign prostatic hyperplasia)
//! - **Sonodynamic**: Brain tumors, soft tissue sarcomas
//!
//! ### Neurology
//! - **LIFU**: Essential tremor, depression, Alzheimer's
//! - **BBB Opening**: Chemotherapy delivery, antibody delivery
//!
//! ### Cardiology
//! - **Histotripsy**: Thrombolysis, valve calcification
//! - **LIFU**: Atrial fibrillation ablation
//!
//! ## Safety Considerations
//!
//! ### Thermal Safety (HIFU)
//! - Monitor near-field heating (< 10 mm from transducer)
//! - Protect skin/ribs from thermal burns
//! - CEM43 < 1 min in non-target tissue
//!
//! ### Mechanical Safety (Histotripsy, BBB Opening)
//! - Mechanical Index (MI) monitoring
//! - Avoid bone interfaces (reflections → hot spots)
//! - Microbubble dose control (BBB opening)
//!
//! ### Neurological Safety (LIFU, BBB Opening)
//! - Skull heating monitoring (transcranial)
//! - BBB closure verification (< 24 hours)
//! - Neurological function testing post-treatment
//!
//! ## References
//!
//! - Elias, W. J., et al. (2016). "A randomized trial of focused ultrasound thalamotomy
//!   for essential tremor." *New England Journal of Medicine*, 375(8), 730-739.
//! - Vlaisavljevich, E., et al. (2015). "Image-guided noninvasive ultrasound liver ablation
//!   using histotripsy." *Ultrasound in Medicine & Biology*, 41(5), 1398-1409.
//! - Carpentier, A., et al. (2016). "Clinical trial of blood-brain barrier disruption by
//!   pulsed ultrasound." *Science Translational Medicine*, 8(343), 343re2.
//!
//! ## Regulatory Status
//!
//! - **FDA Approved**: HIFU (uterine fibroids, prostate, essential tremor)
//! - **Clinical Trials**: Histotripsy (liver, BPH), BBB opening (brain tumors)
//! - **Experimental**: LIFU neuromodulation, sonodynamic therapy

pub mod types;

pub use types::{TherapyMechanism, TherapyModality};
