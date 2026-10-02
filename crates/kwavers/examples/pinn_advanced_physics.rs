//! Advanced Physics Domains PINN Demonstration
//!
//! This example demonstrates the advanced physics domain capabilities of the PINN framework,
//! showcasing Navier-Stokes fluid dynamics, heat transfer, and structural mechanics.
//!
//! ## Physics Domains Demonstrated
//!
//! ### Navier-Stokes Fluid Dynamics
//! - Incompressible flow simulation
//! - Turbulence modeling and boundary conditions
//! - High-Reynolds number flow regimes
//!
//! ### Heat Transfer
//! - Multi-physics conjugate heat transfer
//! - Phase change and material interfaces
//! - Non-linear thermal properties
//!
//! ### Structural Mechanics
//! - Linear elasticity with geometric nonlinearity
//! - Plasticity models and contact mechanics
//! - Dynamic loading and vibration analysis
//!
//! ## Usage
//!
//! ```bash
//! # Run Navier-Stokes demonstration
//! cargo run --example pinn_advanced_physics -- --navier-stokes
//!
//! # Run heat transfer demonstration
//! cargo run --example pinn_advanced_physics -- --heat-transfer
//!
//! # Run structural mechanics demonstration
//! cargo run --example pinn_advanced_physics -- --structural
//!
//! # Run all physics domains
//! cargo run --example pinn_advanced_physics -- --all
//! ```

use std::io::Write;
use std::time::Instant;

#[cfg(feature = "pinn")]
#[path = "pinn_advanced_physics/physics_demo.rs"]
mod physics_demo;

#[cfg(not(feature = "pinn"))]
mod physics_demo {
    pub fn demonstrate_navier_stokes() {
        eprintln!("❌ PINN feature not enabled. Use --features pinn to enable physics domains.");
    }
    pub fn demonstrate_heat_transfer() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_structural_mechanics() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_multi_physics() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_industrial_applications() {
        eprintln!("❌ PINN feature not enabled.");
    }
}

fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = std::env::args().collect();

    let _ = writeln!(
        std::io::stdout().lock(),
        "🌊 Advanced Physics Domains PINN Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "============================================="
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "🔬 Exploring: Navier-Stokes • Heat Transfer • Structural Mechanics"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Applications: Aerospace • Automotive • Civil Engineering • Energy"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Parse command line arguments
    let demo_mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "--all"
    };

    match demo_mode {
        "--navier-stokes" => {
            physics_demo::demonstrate_navier_stokes();
        }
        "--heat-transfer" => {
            physics_demo::demonstrate_heat_transfer();
        }
        "--structural" => {
            physics_demo::demonstrate_structural_mechanics();
        }
        "--multi-physics" => {
            physics_demo::demonstrate_multi_physics();
        }
        "--industrial" => {
            physics_demo::demonstrate_industrial_applications();
        }
        "--all" => {
            let _ = writeln!(
                std::io::stdout().lock(),
                "🎭 Complete Advanced Physics Demonstration"
            );
            let _ = writeln!(
                std::io::stdout().lock(),
                "=========================================="
            );
            let _ = writeln!(std::io::stdout().lock());

            physics_demo::demonstrate_navier_stokes();
            physics_demo::demonstrate_heat_transfer();
            physics_demo::demonstrate_structural_mechanics();
            physics_demo::demonstrate_multi_physics();
            physics_demo::demonstrate_industrial_applications();
        }
        _ => {
            let _ = writeln!(
                std::io::stdout().lock(),
                "🎭 Complete Advanced Physics Demonstration"
            );
            let _ = writeln!(
                std::io::stdout().lock(),
                "=========================================="
            );
            let _ = writeln!(std::io::stdout().lock());

            physics_demo::demonstrate_navier_stokes();
            physics_demo::demonstrate_heat_transfer();
            physics_demo::demonstrate_structural_mechanics();
            physics_demo::demonstrate_multi_physics();
            physics_demo::demonstrate_industrial_applications();
        }
    }

    let elapsed = start_time.elapsed();
    let _ = writeln!(
        std::io::stdout().lock(),
        "🏆 Advanced Physics Demonstration Complete!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "==========================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ⏱️  Total runtime: {:.2}s",
        elapsed.as_secs_f64()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ✅ All physics domains demonstrated"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   🚀 Ready for industrial applications"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "📚 Physics-Specific Examples:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --navier-stokes: Fluid dynamics simulation"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --heat-transfer: Thermal analysis and coupling"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --structural: Mechanical stress and deformation"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --multi-physics: Coupled physics problems"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --industrial: Real-world engineering applications"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "🌟 PINN: Revolutionizing computational physics!"
    );
}
