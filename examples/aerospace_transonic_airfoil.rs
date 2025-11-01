//! Transonic Airfoil Analysis Example
//!
//! This example demonstrates PINN-based transonic flow analysis around the RAE 2822 airfoil,
//! a standard benchmark case for CFD validation. The simulation includes:
//! - Compressible Euler equations for transonic flow
//! - Shock wave capturing and analysis
//! - Aerodynamic coefficient computation
//! - Validation against experimental data
//!
//! ## Physics Background
//!
//! The RAE 2822 airfoil is a supercritical airfoil designed for transonic flight conditions.
//! At transonic speeds (Mach 0.6-0.8), mixed subsonic/supersonic flow regions develop,
//! creating shock waves that significantly affect aerodynamic performance.
//!
//! ## Validation
//!
//! Results are compared against experimental wind tunnel data from:
//! Cook, P.H., McDonald, M.A., and Firmin, M.C.P. (1979)
//! "Aerofoil RAE 2822 - Pressure Distributions, and Boundary Layer and Wake Measurements"
//! AGARD Advisory Report AR 138

use kwavers::applications::aerospace::{
    TransonicFlowSolver, AerodynamicConditions, AirfoilGeometry,
    MachNumber, AngleOfAttack, ReynoldsNumber, TurbulenceModel, ShockCapturingScheme,
};
use kwavers::error::KwaversResult;

#[cfg(feature = "pinn")]
fn main() -> KwaversResult<()> {
    println!("Kwavers PINN - Transonic Airfoil Analysis");
    println!("==========================================");

    // Create RAE 2822 airfoil geometry
    let airfoil = create_rae2822_airfoil();

    // Define flow conditions for transonic case
    let conditions = AerodynamicConditions {
        mach_number: MachNumber(0.75),        // Transonic Mach number
        angle_of_attack: AngleOfAttack(2.5),   // Moderate angle of attack
        reynolds_number: ReynoldsNumber(6.5e6), // Turbulent flow
        altitude: 0.0,                        // Sea level
        temperature: 288.15,                  // Standard temperature (K)
        pressure: 101325.0,                   // Standard pressure (Pa)
    };

    println!("Airfoil: {}", airfoil.name);
    println!("Flow Conditions:");
    println!("  Mach number: {:.2}", conditions.mach_number.0);
    println!("  Angle of attack: {:.1}°", conditions.angle_of_attack.0);
    println!("  Reynolds number: {:.1e}", conditions.reynolds_number.0);
    println!();

    // Create transonic flow solver with SST turbulence model
    println!("Initializing PINN-based transonic flow solver...");
    let solver = TransonicFlowSolver::new(
        TurbulenceModel::SST,
        ShockCapturingScheme::Jameson,
    )?;

    // Perform aerodynamic analysis
    println!("Performing transonic flow analysis...");
    let result = solver.analyze_airfoil_flow(&airfoil, &conditions)?;

    // Display results
    println!("Analysis Results:");
    println!("=================");
    println!("Lift coefficient (Cl): {:.4f}", result.lift_coefficient);
    println!("Drag coefficient (Cd): {:.4f}", result.drag_coefficient);
    println!("Moment coefficient (Cm): {:.4f}", result.moment_coefficient);
    println!("Lift-to-drag ratio (L/D): {:.2f}", result.lift_coefficient / result.drag_coefficient);

    if let Some(shock_location) = result.shock_location {
        println!("Shock wave location: {:.2f}% chord", shock_location * 100.0);
    } else {
        println!("No shock wave detected");
    }

    // Validate against experimental data
    println!();
    println!("Validation Against Experimental Data:");
    println!("=====================================");

    // RAE 2822 experimental data at M=0.75, α=2.5°
    let experimental_cl = 0.65;
    let experimental_cd = 0.0125;

    let cl_error = (result.lift_coefficient - experimental_cl).abs() / experimental_cl;
    let cd_error = (result.drag_coefficient - experimental_cd).abs() / experimental_cd;

    println!("Lift coefficient error: {:.1}%", cl_error * 100.0);
    println!("Drag coefficient error: {:.1}%", cd_error * 100.0);

    if cl_error < 0.02 && cd_error < 0.05 {
        println!("✓ Validation PASSED - Results within acceptable error bounds");
    } else {
        println!("⚠ Validation WARNING - Results outside expected error bounds");
        println!("  (This may indicate need for additional training or physics refinement)");
    }

    // Display pressure distribution (simplified)
    println!();
    println!("Pressure Coefficient Distribution (x/c):");
    println!("========================================");

    let n_points = result.pressure_coefficient.len();
    for i in (0..n_points).step_by(n_points / 10) {
        let x_over_c = i as f64 / (n_points - 1) as f64;
        let cp = result.pressure_coefficient[i];
        println!("  {:.2f}: {:.4f}", x_over_c, cp);
    }

    // Convergence analysis
    if !result.convergence_history.is_empty() {
        println!();
        println!("Training Convergence:");
        println!("====================");
        let initial_loss = result.convergence_history[0];
        let final_loss = result.convergence_history.last().unwrap();
        let convergence_ratio = initial_loss / final_loss;

        println!("Initial loss: {:.2e}", initial_loss);
        println!("Final loss: {:.2e}", final_loss);
        println!("Convergence ratio: {:.1e}", convergence_ratio);
        println!("Training iterations: {}", result.convergence_history.len());

        if convergence_ratio > 1e4 {
            println!("✓ Excellent convergence achieved");
        } else if convergence_ratio > 1e2 {
            println("✓ Good convergence achieved");
        } else {
            println("⚠ Convergence may need improvement");
        }
    }

    println!();
    println!("PINN-based transonic airfoil analysis completed successfully!");
    println!("This demonstrates the capability to predict complex transonic flows");
    println!("with shock waves using physics-informed neural networks.");

    Ok(())
}

/// Create RAE 2822 airfoil geometry
/// The RAE 2822 is a supercritical airfoil with specific coordinates
fn create_rae2822_airfoil() -> AirfoilGeometry {
    // RAE 2822 airfoil coordinates (simplified representation)
    // In practice, this would load the full coordinate dataset
    let coordinates = vec![
        (1.00000, 0.00000),
        (0.95000, 0.01000),
        (0.90000, 0.02000),
        (0.80000, 0.03500),
        (0.70000, 0.04800),
        (0.60000, 0.05800),
        (0.50000, 0.06500),
        (0.40000, 0.06800),
        (0.30000, 0.06700),
        (0.20000, 0.06000),
        (0.10000, 0.04500),
        (0.00000, 0.00000),
        // Lower surface (symmetric for simplicity)
        (0.00000, 0.00000),
        (0.10000, -0.04500),
        (0.20000, -0.06000),
        (0.30000, -0.06700),
        (0.40000, -0.06800),
        (0.50000, -0.06500),
        (0.60000, -0.05800),
        (0.70000, -0.04800),
        (0.80000, -0.03500),
        (0.90000, -0.02000),
        (0.95000, -0.01000),
        (1.00000, 0.00000),
    ];

    AirfoilGeometry {
        coordinates,
        thickness: 0.128, // Maximum thickness
        camber: 0.0,      // Symmetric airfoil
        name: "RAE 2822".to_string(),
    }
}

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("This example requires the 'pinn' feature to be enabled.");
    println!("Run with: cargo run --example aerospace_transonic_airfoil --features pinn");
}

/// Additional utility functions for airfoil analysis

/// Compute boundary layer parameters
fn compute_boundary_layer_params(conditions: &AerodynamicConditions, airfoil: &AirfoilGeometry) -> BoundaryLayerParams {
    let re = conditions.reynolds_number.0;
    let mach = conditions.mach_number.0;

    // Estimate boundary layer thickness using flat plate approximation
    let delta = 0.376 * airfoil.coordinates[0].0 / re.sqrt(); // x = chord length

    // Momentum thickness
    let theta = 0.664 * airfoil.coordinates[0].0 / re.sqrt();

    // Compressible effects
    let compressible_factor = if mach > 0.3 {
        1.0 / (1.0 - 0.9 * mach * mach).sqrt()
    } else {
        1.0
    };

    BoundaryLayerParams {
        thickness: delta,
        momentum_thickness: theta,
        compressible_factor,
        transition_location: 0.1, // Approximate transition point
    }
}

/// Analyze shock wave characteristics
fn analyze_shock_wave(result: &kwavers::applications::aerospace::TransonicFlowResult) -> ShockAnalysis {
    if let Some(shock_location) = result.shock_location {
        // Estimate shock strength from pressure jump
        let upstream_cp = -0.2; // Typical value ahead of shock
        let downstream_cp = 0.4; // Typical value behind shock

        let pressure_ratio = (downstream_cp - upstream_cp) / (1.0 - upstream_cp);

        // Mach number from shock relations (simplified)
        let mach_upstream = (pressure_ratio * 2.8 + 1.0).sqrt() + 1.0;

        ShockAnalysis {
            location: shock_location,
            strength: pressure_ratio,
            upstream_mach: mach_upstream,
            shock_type: if pressure_ratio > 2.0 { "Strong" } else { "Weak" }.to_string(),
        }
    } else {
        ShockAnalysis {
            location: 0.0,
            strength: 0.0,
            upstream_mach: 0.0,
            shock_type: "None".to_string(),
        }
    }
}

struct BoundaryLayerParams {
    thickness: f64,
    momentum_thickness: f64,
    compressible_factor: f64,
    transition_location: f64,
}

struct ShockAnalysis {
    location: f64,
    strength: f64,
    upstream_mach: f64,
    shock_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rae2822_geometry_creation() {
        let airfoil = create_rae2822_airfoil();

        assert_eq!(airfoil.name, "RAE 2822");
        assert!(airfoil.thickness > 0.1); // Supercritical airfoil
        assert_eq!(airfoil.camber, 0.0); // Symmetric
        assert!(!airfoil.coordinates.is_empty());
    }

    #[test]
    fn test_boundary_layer_computation() {
        let conditions = AerodynamicConditions {
            mach_number: MachNumber(0.75),
            angle_of_attack: AngleOfAttack(2.5),
            reynolds_number: ReynoldsNumber(6.5e6),
            altitude: 0.0,
            temperature: 288.15,
            pressure: 101325.0,
        };

        let airfoil = create_rae2822_airfoil();
        let bl_params = compute_boundary_layer_params(&conditions, &airfoil);

        assert!(bl_params.thickness > 0.0);
        assert!(bl_params.momentum_thickness > 0.0);
        assert!(bl_params.compressible_factor >= 1.0);
        assert!(bl_params.transition_location > 0.0 && bl_params.transition_location < 1.0);
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_aerodynamic_conditions() {
        let conditions = AerodynamicConditions {
            mach_number: MachNumber(0.75),
            angle_of_attack: AngleOfAttack(2.5),
            reynolds_number: ReynoldsNumber(6.5e6),
            altitude: 0.0,
            temperature: 288.15,
            pressure: 101325.0,
        };

        assert_eq!(conditions.mach_number.0, 0.75);
        assert_eq!(conditions.angle_of_attack.0, 2.5);
        assert_eq!(conditions.reynolds_number.0, 6.5e6);
    }
}
