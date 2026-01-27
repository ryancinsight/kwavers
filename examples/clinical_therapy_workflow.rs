//! Clinical Therapy Workflow Example
//!
//! This example demonstrates how to set up and execute clinical ultrasound therapy
//! workflows using the therapy integration framework. It shows:
//!
//! 1. Patient-specific treatment planning
//! 2. Multi-modal therapy configuration
//! 3. Real-time monitoring and safety control
//! 4. Combined therapy approaches

use kwavers::clinical::therapy::therapy_integration::*;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::homogeneous::HomogeneousMedium;
use ndarray::Array3;
use std::collections::HashMap;

/// Example: Liver tumor treatment with combined histotripsy and sonodynamic therapy
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🩺 Clinical Ultrasound Therapy Workflow Example");
    println!("===============================================\n");

    // 1. Patient-specific treatment planning
    println!("1. Patient-Specific Treatment Planning");
    println!("--------------------------------------");

    let patient_params = create_patient_profile();
    let target_volume = define_tumor_target();

    println!("Patient Profile:");
    println!("  - Tissue: Liver with tumor");
    println!(
        "  - Target: {:.1} mm³ tumor at {:.1} cm depth",
        target_volume.dimensions.0 * target_volume.dimensions.1 * target_volume.dimensions.2 * 1e9,
        target_volume.center.0 * 100.0
    );

    // 2. Configure multi-modal therapy
    println!("\n2. Multi-Modal Therapy Configuration");
    println!("------------------------------------");

    let therapy_config = TherapySessionConfig {
        primary_modality: TherapyModality::Histotripsy,
        secondary_modalities: vec![TherapyModality::Sonodynamic, TherapyModality::Microbubble],
        duration: 300.0, // 5 minutes
        acoustic_params: AcousticTherapyParams {
            frequency: 1.0e6,  // 1 MHz for histotripsy
            pnp: 15e6,         // 15 MPa peak negative pressure
            prf: 100.0,        // 100 Hz pulse repetition
            duty_cycle: 0.005, // 0.5% duty cycle
            focal_depth: target_volume.center.0,
            treatment_volume: 2.0, // 2 cm³ treatment volume
        },
        safety_limits: SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 5000.0,
            max_treatment_time: 600.0, // 10 minutes maximum
        },
        patient_params,
        imaging_data_path: None,
    };

    println!("Therapy Configuration:");
    println!("  - Primary: Histotripsy (mechanical ablation)");
    println!("  - Secondary: Sonodynamic therapy + Microbubble enhancement");
    println!("  - Duration: {} seconds", therapy_config.duration);
    println!(
        "  - Frequency: {:.1} MHz",
        therapy_config.acoustic_params.frequency / 1e6
    );
    println!(
        "  - Peak Pressure: {:.0} MPa",
        therapy_config.acoustic_params.pnp / 1e6
    );

    // 3. Initialize therapy system
    println!("\n3. Therapy System Initialization");
    println!("-------------------------------");

    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001)?; // 6.4cm³ domain
    let medium = HomogeneousMedium::new(1050.0, 1570.0, 0.7, 1.0, &grid); // Liver properties

    let mut therapy_orchestrator =
        TherapyIntegrationOrchestrator::new(therapy_config, grid, Box::new(medium.clone()))?;

    println!("✅ Therapy orchestrator initialized successfully");
    println!("   - Grid: {}x{}x{}", 64, 64, 64);
    println!("   - Domain: {:.1} cm³", 6.4);
    println!("   - Modalities: Histotripsy + Sonodynamic + Microbubble");

    // 4. Execute therapy session with monitoring
    println!("\n4. Therapy Execution with Real-time Monitoring");
    println!("---------------------------------------------");

    let dt = 0.1; // 100ms time steps
    let time_points = vec![0.0, 60.0, 120.0, 180.0, 240.0, 300.0]; // Monitor every minute

    for &target_time in &time_points {
        while therapy_orchestrator.session_state().current_time < target_time {
            // Execute therapy step
            therapy_orchestrator.execute_therapy_step(dt)?;

            // Check safety limits
            let safety_status = therapy_orchestrator.check_safety_limits();

            match safety_status {
                SafetyStatus::Safe => {
                    // Continue therapy
                }
                _ => {
                    println!(
                        "⚠️  Safety limit reached: {:?} at t={:.1}s",
                        safety_status,
                        therapy_orchestrator.session_state().current_time
                    );
                    break;
                }
            }
        }

        // Report progress at monitoring points
        report_therapy_progress(&therapy_orchestrator, target_time);
    }

    // 5. Final treatment assessment
    println!("\n5. Final Treatment Assessment");
    println!("-----------------------------");

    let final_state = therapy_orchestrator.session_state();
    let safety_metrics = &final_state.safety_metrics;

    println!("Treatment Summary:");
    println!(
        "  - Total Duration: {:.1} seconds",
        final_state.current_time
    );
    println!("  - Progress: {:.1}%", final_state.progress * 100.0);
    println!(
        "  - Final Thermal Index: {:.2}",
        safety_metrics.thermal_index
    );
    println!(
        "  - Final Mechanical Index: {:.2}",
        safety_metrics.mechanical_index
    );
    println!(
        "  - Total Cavitation Dose: {:.0}",
        safety_metrics.cavitation_dose
    );

    // Check treatment completion
    if final_state.progress >= 0.95 {
        println!("✅ Treatment completed successfully!");
    } else {
        println!("⚠️  Treatment incomplete - safety limits may have been reached");
    }

    println!("\n🏆 Clinical therapy workflow example completed!");
    println!("📊 This demonstrates integrated multi-modal ultrasound therapy");

    Ok(())
}

/// Create patient-specific profile for liver tumor treatment
fn create_patient_profile() -> PatientParameters {
    // Simulate patient liver properties
    let grid_dims = (64, 64, 64);
    let speed_of_sound = Array3::from_elem(grid_dims, 1570.0); // Liver SoS
    let density = Array3::from_elem(grid_dims, 1050.0); // Liver density
    let mut attenuation = Array3::from_elem(grid_dims, 0.7); // Liver attenuation
    let mut nonlinearity = Array3::from_elem(grid_dims, 6.2); // Liver B/A

    // Simulate tumor region (higher attenuation, different properties)
    for i in 25..35 {
        for j in 25..35 {
            for k in 25..35 {
                attenuation[[i, j, k]] = 1.2; // Higher attenuation in tumor
                nonlinearity[[i, j, k]] = 8.0; // Different nonlinearity
            }
        }
    }

    PatientParameters {
        skull_thickness: None, // Not needed for liver treatment
        tissue_properties: TissuePropertyMap {
            speed_of_sound,
            density,
            attenuation,
            nonlinearity,
        },
        target_volume: define_tumor_target(),
        risk_organs: vec![
            // Define risk organs (e.g., major blood vessels, bile ducts)
            RiskOrgan {
                name: "Hepatic Vein".to_string(),
                bounds: ((0.02, 0.04), (-0.01, 0.01), (-0.01, 0.01)),
                max_dose: 1000.0,
            },
        ],
    }
}

/// Define tumor target volume
fn define_tumor_target() -> TargetVolume {
    TargetVolume {
        center: (0.035, 0.0, 0.0),         // 3.5cm depth
        dimensions: (0.015, 0.015, 0.015), // 1.5cm³ tumor
        tissue_type: TissueType::Tumor,
    }
}

/// Report therapy progress at monitoring points
fn report_therapy_progress(orchestrator: &TherapyIntegrationOrchestrator, target_time: f64) {
    let state = orchestrator.session_state();
    let metrics = &state.safety_metrics;

    println!(
        "📊 Progress at {:.0}s ({:.0}% complete):",
        target_time,
        state.progress * 100.0
    );
    println!(
        "   Thermal Index: {:.2} | Mechanical Index: {:.2}",
        metrics.thermal_index, metrics.mechanical_index
    );
    println!(
        "   Cavitation Dose: {:.0} | Safety Status: {:?}",
        metrics.cavitation_dose,
        orchestrator.check_safety_limits()
    );

    // Report modality-specific metrics
    if let Some(ref microbubbles) = state.microbubble_concentration {
        let microbubbles_arr: &Array3<f64> = microbubbles;
        let avg_concentration: f64 = microbubbles_arr.mean().unwrap_or(0.0);
        println!(
            "   Microbubble Concentration: {:.2e} bubbles/mL",
            avg_concentration
        );
    }

    if let Some(ref cavitation) = state.cavitation_activity {
        let cavitation_arr: &Array3<f64> = cavitation;
        let max_activity: f64 = cavitation_arr
            .iter()
            .cloned()
            .fold(0.0f64, |a: f64, b: f64| a.max(b));
        println!("   Peak Cavitation Activity: {:.3}", max_activity);
    }

    if let Some(ref chemicals) = state.chemical_concentrations {
        let chemicals_map: &HashMap<String, Array3<f64>> = chemicals;
        if let Some(ros) = chemicals_map.get("H2O2") {
            let ros_arr: &Array3<f64> = ros;
            let avg_ros: f64 = ros_arr.mean().unwrap_or(0.0);
            println!("   Average ROS Concentration: {:.2e} M", avg_ros);
        }
    }
}
