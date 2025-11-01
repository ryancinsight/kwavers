//! Automotive Crash Simulation Example
//!
//! This example demonstrates PINN-based nonlinear structural dynamics simulation
//! for vehicle crash analysis. The simulation includes material plasticity, contact
//! mechanics, and FMVSS compliance validation for occupant safety assessment.
//!
//! ## Physics Models
//!
//! - Large deformation structural mechanics with plasticity
//! - Johnson-Cook material model for strain rate effects
//! - Contact mechanics with friction and penetration
//! - Energy absorption and crush zone analysis
//!
//! ## Validation Standards
//!
//! Results validated against FMVSS 208 (Occupant Crash Protection) requirements:
//! - Maximum intrusion limits for occupant compartment
//! - Structural integrity and load path continuity
//! - Energy absorption and deformation metrics

use kwavers::applications::automotive::{
    CrashSimulationSolver, VehicleModel, ImpactScenario, BarrierType, ImpactLocation,
    ComponentType, MaterialModel, TimeIntegrationScheme, ContactModel,
};
use kwavers::error::KwaversResult;
use std::collections::HashMap;

#[cfg(feature = "pinn")]
fn main() -> KwaversResult<()> {
    println!("Kwavers PINN - Automotive Crash Simulation");
    println!("===========================================");

    // Create a simplified sedan vehicle model
    let vehicle = create_test_vehicle()?;

    // Define FMVSS 208 frontal impact scenario
    let scenario = ImpactScenario {
        velocity: 15.0, // 54 km/h (35 mph) - FMVSS 208 test speed
        angle: 0.0,     // Head-on collision
        barrier_type: BarrierType::Rigid, // Fixed rigid barrier
        impact_location: ImpactLocation::Front { overlap: 100.0 }, // 100% overlap
        duration: 0.15, // 150ms crash duration
    };

    println!("Vehicle: {}", vehicle.name);
    println!("Impact Scenario: {:.0} km/h frontal crash into rigid barrier",
             scenario.velocity * 3.6);
    println!("Simulation Duration: {:.0} ms", scenario.duration * 1000.0);
    println!();

    // Create crash simulation solver
    println!("Initializing PINN-based crash simulation solver...");
    let contact_model = ContactModel {
        friction_coefficient: 0.2,
        contact_stiffness: 1e9,
        damping_coefficient: 0.1,
        penetration_tolerance: 1e-6,
    };

    let solver = CrashSimulationSolver::new(
        vehicle.materials.clone(),
        contact_model,
        TimeIntegrationScheme::CentralDifference,
    )?;

    // Run crash simulation
    println!("Running crash simulation...");
    let analysis = solver.simulate_crash(&vehicle, &scenario)?;

    // Display results
    println!("Crash Analysis Results:");
    println!("======================");

    println!("Energy Absorption:");
    println!("  Total energy absorbed: {:.0} kJ", analysis.energy_absorption.total_energy / 1000.0);
    println!("  Crush distance: {:.0} mm", analysis.energy_absorption.crush_distance * 1000.0);
    println!("  Specific energy absorption: {:.0} J/kg", analysis.energy_absorption.specific_energy);
    println!("  Energy absorption efficiency: {:.1}%", analysis.energy_absorption.efficiency * 100.0);

    println!();
    println!("Occupant Compartment Integrity:");
    println!("  Compartment integrity: {:.1}%", analysis.occupant_protection.compartment_integrity * 100.0);
    println!("  Survival space maintained: {}", if analysis.occupant_protection.survival_space { "YES" } else { "NO" });

    println!();
    println!("Intrusion Measurements:");
    println!("  Footwell intrusion: {:.0} mm", analysis.occupant_protection.intrusions.footwell);
    println!("  Toe pan intrusion: {:.0} mm", analysis.occupant_protection.intrusions.toe_pan);
    println!("  Side intrusion: {:.0} mm", analysis.occupant_protection.intrusions.side);

    println!();
    println!("FMVSS Compliance Status:");
    println!("=======================");
    let compliance = &analysis.structural_integrity.fmvs_compliance;
    println!("  FMVSS 208 (Occupant Protection): {}", if compliance.fmvs_208 { "PASS ✓" } else { "FAIL ✗" });
    println!("  FMVSS 214 (Side Impact): {}", if compliance.fmvs_214 { "PASS ✓" } else { "FAIL ✗" });
    println!("  FMVSS 301 (Fuel Integrity): {}", if compliance.fmvs_301 { "PASS ✓" } else { "FAIL ✗" });
    println!("  Overall Compliance: {}", if compliance.overall_compliant { "PASS ✓" } else { "FAIL ✗" });

    // Analyze failure modes
    if analysis.failure_modes.is_empty() {
        println!();
        println!("Structural Analysis: No component failures detected ✓");
    } else {
        println!();
        println!("Component Failures Detected:");
        for failure in &analysis.failure_modes {
            println!("  {}: {:?} (Severity: {:.1}%)",
                    failure.component,
                    failure.failure_type,
                    failure.severity * 100.0);
        }
    }

    // Load path analysis
    println!();
    println!("Load Path Analysis:");
    for load_path in &analysis.structural_integrity.load_paths {
        println!("  {}: Continuity {:.1}%, Capacity {:.0} kN, Deformation {:.0} mm",
                load_path.path_id,
                load_path.continuity * 100.0,
                load_path.load_capacity,
                load_path.deformation);
    }

    // Safety assessment
    println!();
    println!("Safety Assessment:");
    println!("==================");

    let safety_score = calculate_safety_score(&analysis);
    println!("  Overall Safety Score: {:.1}/10.0", safety_score);

    if safety_score >= 8.0 {
        println!("  Assessment: EXCELLENT - Vehicle meets or exceeds safety standards");
    } else if safety_score >= 6.0 {
        println!("  Assessment: GOOD - Vehicle meets minimum safety requirements");
    } else if safety_score >= 4.0 {
        println!("  Assessment: ADEQUATE - Vehicle meets basic safety standards");
    } else {
        println!("  Assessment: POOR - Vehicle requires safety improvements");
    }

    println!();
    println!("PINN-based crash simulation completed successfully!");
    println!("This demonstrates the capability to predict complex vehicle");
    println!("deformation and assess occupant safety using physics-informed neural networks.");

    Ok(())
}

/// Create a test vehicle model for crash simulation
fn create_test_vehicle() -> KwaversResult<VehicleModel> {
    // Define materials used in vehicle
    let mut materials = HashMap::new();

    // High-strength steel for frame
    materials.insert("hs-steel".to_string(), kwavers::applications::automotive::MaterialProperties {
        youngs_modulus: 210e9, // Pa
        poissons_ratio: 0.3,
        density: 7850.0, // kg/m³
        yield_strength: 350e6,
        ultimate_strength: 590e6,
        failure_strain: 0.15,
        hardening_modulus: 1.2e9,
        model_type: MaterialModel::ElastoPlastic {
            hardening: kwavers::applications::automotive::HardeningLaw::Linear,
        },
    });

    // Mild steel for body panels
    materials.insert("mild-steel".to_string(), kwavers::applications::automotive::MaterialProperties {
        youngs_modulus: 200e9,
        poissons_ratio: 0.29,
        density: 7850.0,
        yield_strength: 180e6,
        ultimate_strength: 320e6,
        failure_strain: 0.25,
        hardening_modulus: 0.8e9,
        model_type: MaterialModel::ElastoPlastic {
            hardening: kwavers::applications::automotive::HardeningLaw::PowerLaw { exponent: 0.2 },
        },
    });

    // Aluminum for some components
    materials.insert("aluminum".to_string(), kwavers::applications::automotive::MaterialProperties {
        youngs_modulus: 70e9,
        poissons_ratio: 0.33,
        density: 2700.0,
        yield_strength: 100e6,
        ultimate_strength: 200e6,
        failure_strain: 0.12,
        hardening_modulus: 0.5e9,
        model_type: MaterialModel::ElastoPlastic {
            hardening: kwavers::applications::automotive::HardeningLaw::Linear,
        },
    });

    // Simplified vehicle components
    let components = vec![
        kwavers::applications::automotive::StructuralComponent {
            name: "front_rail_left".to_string(),
            component_type: ComponentType::Frame,
            material_id: "hs-steel".to_string(),
            geometry: kwavers::applications::automotive::ComponentGeometry {
                shape: kwavers::applications::automotive::GeometryShape::Box {
                    length: 1.2, width: 0.08, height: 0.15
                },
                dimensions: vec![1.2, 0.08, 0.15],
                thickness: 0.08,
                mass: 25.0,
            },
            connections: vec![], // Would define connections to other components
        },
        kwavers::applications::automotive::StructuralComponent {
            name: "front_rail_right".to_string(),
            component_type: ComponentType::Frame,
            material_id: "hs-steel".to_string(),
            geometry: kwavers::applications::automotive::ComponentGeometry {
                shape: kwavers::applications::automotive::GeometryShape::Box {
                    length: 1.2, width: 0.08, height: 0.15
                },
                dimensions: vec![1.2, 0.08, 0.15],
                thickness: 0.08,
                mass: 25.0,
            },
            connections: vec![],
        },
        kwavers::applications::automotive::StructuralComponent {
            name: "front_bumper".to_string(),
            component_type: ComponentType::Bumper,
            material_id: "mild-steel".to_string(),
            geometry: kwavers::applications::automotive::ComponentGeometry {
                shape: kwavers::applications::automotive::GeometryShape::Box {
                    length: 1.8, width: 0.1, height: 0.3
                },
                dimensions: vec![1.8, 0.1, 0.3],
                thickness: 0.1,
                mass: 15.0,
            },
            connections: vec![],
        },
        kwavers::applications::automotive::StructuralComponent {
            name: "engine_cradle".to_string(),
            component_type: ComponentType::Frame,
            material_id: "hs-steel".to_string(),
            geometry: kwavers::applications::automotive::ComponentGeometry {
                shape: kwavers::applications::automotive::ComponentGeometryShape::Box {
                    length: 0.8, width: 0.6, height: 0.2
                },
                dimensions: vec![0.8, 0.6, 0.2],
                thickness: 0.08,
                mass: 35.0,
            },
            connections: vec![],
        },
    ];

    Ok(VehicleModel {
        name: "Test Sedan (FMVSS Compliant)".to_string(),
        dimensions: (4.5, 1.8, 1.4), // length, width, height (m)
        mass: 1500.0, // kg
        cg_location: (2.0, 0.0, 0.5), // from front axle, ground level
        components,
        materials,
        occupant_compartment: kwavers::applications::automotive::OccupantCompartment {
            volume: 3.0, // m³
            intrusion_limits: kwavers::applications::automotive::IntrusionLimits {
                footwell: 200.0, // mm
                toe_pan: 150.0,
                side: 300.0,
            },
            integrity_requirements: kwavers::applications::automotive::IntegrityRequirements {
                survival_space: true,
                roof_crush: 100.0, // kN
                fuel_integrity: true,
            },
        },
    })
}

/// Calculate overall safety score from crash analysis
fn calculate_safety_score(analysis: &kwavers::applications::automotive::CrashAnalysis) -> f64 {
    let mut score = 0.0;

    // Energy absorption (0-3 points)
    let energy_score = if analysis.energy_absorption.efficiency > 0.8 {
        3.0
    } else if analysis.energy_absorption.efficiency > 0.6 {
        2.0
    } else if analysis.energy_absorption.efficiency > 0.4 {
        1.0
    } else {
        0.0
    };
    score += energy_score;

    // Occupant protection (0-3 points)
    let occupant_score = if analysis.occupant_protection.compartment_integrity > 0.95 {
        3.0
    } else if analysis.occupant_protection.compartment_integrity > 0.85 {
        2.0
    } else if analysis.occupant_protection.compartment_integrity > 0.75 {
        1.0
    } else {
        0.0
    };
    score += occupant_score;

    // Intrusion limits (0-2 points)
    let intrusion_score = if analysis.occupant_protection.intrusions.footwell < 150.0 &&
                          analysis.occupant_protection.intrusions.toe_pan < 100.0 {
        2.0
    } else if analysis.occupant_protection.intrusions.footwell < 200.0 &&
              analysis.occupant_protection.intrusions.toe_pan < 150.0 {
        1.0
    } else {
        0.0
    };
    score += intrusion_score;

    // Structural integrity (0-2 points)
    let structural_score = if analysis.structural_integrity.integrity_score > 0.9 {
        2.0
    } else if analysis.structural_integrity.integrity_score > 0.7 {
        1.0
    } else {
        0.0
    };
    score += structural_score;

    score
}

/// Additional utility functions for crash analysis

/// Analyze load path continuity
fn analyze_load_paths(analysis: &kwavers::applications::automotive::CrashAnalysis) -> LoadPathAssessment {
    let mut total_continuity = 0.0;
    let mut critical_paths = 0;
    let mut compromised_paths = 0;

    for path in &analysis.structural_integrity.load_paths {
        total_continuity += path.continuity;

        if path.continuity > 0.9 {
            critical_paths += 1;
        } else if path.continuity < 0.7 {
            compromised_paths += 1;
        }
    }

    let avg_continuity = if analysis.structural_integrity.load_paths.is_empty() {
        0.0
    } else {
        total_continuity / analysis.structural_integrity.load_paths.len() as f64
    };

    LoadPathAssessment {
        average_continuity: avg_continuity,
        critical_paths_count: critical_paths,
        compromised_paths_count: compromised_paths,
        overall_integrity: if compromised_paths == 0 { "Excellent" } else if compromised_paths <= 2 { "Good" } else { "Poor" }.to_string(),
    }
}

/// Assess material utilization efficiency
fn assess_material_efficiency(vehicle: &VehicleModel, analysis: &kwavers::applications::automotive::CrashAnalysis) -> MaterialEfficiency {
    // Calculate how effectively materials are used for energy absorption
    let total_vehicle_mass: f64 = vehicle.components.iter()
        .map(|comp| comp.geometry.mass)
        .sum();

    let crush_zone_mass: f64 = vehicle.components.iter()
        .filter(|comp| matches!(comp.component_type, ComponentType::CrushZone))
        .map(|comp| comp.geometry.mass)
        .sum();

    let crush_zone_ratio = crush_zone_mass / total_vehicle_mass;
    let specific_energy = analysis.energy_absorption.specific_energy;

    // Estimate material efficiency based on energy absorption per unit mass
    let material_efficiency = if specific_energy > 300.0 {
        "Excellent"
    } else if specific_energy > 200.0 {
        "Good"
    } else if specific_energy > 100.0 {
        "Adequate"
    } else {
        "Poor"
    }.to_string();

    MaterialEfficiency {
        crush_zone_ratio,
        specific_energy_absorption: specific_energy,
        material_utilization: material_efficiency,
    }
}

struct LoadPathAssessment {
    average_continuity: f64,
    critical_paths_count: usize,
    compromised_paths_count: usize,
    overall_integrity: String,
}

struct MaterialEfficiency {
    crush_zone_ratio: f64,
    specific_energy_absorption: f64,
    material_utilization: String,
}

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("This example requires the 'pinn' feature to be enabled.");
    println!("Run with: cargo run --example automotive_crash_simulation --features pinn");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_creation() {
        let vehicle = create_test_vehicle().unwrap();

        assert_eq!(vehicle.name, "Test Sedan (FMVSS Compliant)");
        assert_eq!(vehicle.mass, 1500.0);
        assert_eq!(vehicle.dimensions, (4.5, 1.8, 1.4));
        assert!(!vehicle.components.is_empty());
        assert!(!vehicle.materials.is_empty());
    }

    #[test]
    fn test_safety_score_calculation() {
        // Create mock crash analysis with good safety performance
        let analysis = kwavers::applications::automotive::CrashAnalysis {
            deformation_field: vec![],
            failure_modes: vec![],
            energy_absorption: kwavers::applications::automotive::EnergyAbsorption {
                total_energy: 500000.0,
                crush_distance: 0.5,
                specific_energy: 333.33,
                efficiency: 0.8,
            },
            occupant_protection: kwavers::applications::automotive::OccupantProtection {
                compartment_integrity: 0.95,
                survival_space: true,
                intrusions: kwavers::applications::automotive::IntrusionMeasurements {
                    footwell: 120.0,
                    toe_pan: 80.0,
                    side: 150.0,
                    roof: 30.0,
                },
            },
            structural_integrity: kwavers::applications::automotive::StructuralIntegrity {
                integrity_score: 0.9,
                fmvs_compliance: kwavers::applications::automotive::FMVSSCompliance {
                    fmvs_208: true,
                    fmvs_214: true,
                    fmvs_301: true,
                    overall_compliant: true,
                },
                load_paths: vec![],
            },
        };

        let safety_score = calculate_safety_score(&analysis);
        assert!(safety_score >= 6.0, "Safety score should be good: {}", safety_score);
    }

    #[test]
    fn test_material_properties() {
        let vehicle = create_test_vehicle().unwrap();

        // Check that high-strength steel has proper properties
        let hs_steel = vehicle.materials.get("hs-steel").unwrap();
        assert_eq!(hs_steel.youngs_modulus, 210e9);
        assert_eq!(hs_steel.yield_strength, 350e6);
        assert!(matches!(hs_steel.model_type, MaterialModel::ElastoPlastic { .. }));

        // Check aluminum properties
        let aluminum = vehicle.materials.get("aluminum").unwrap();
        assert_eq!(aluminum.density, 2700.0);
        assert_eq!(aluminum.youngs_modulus, 70e9);
    }
}
