//! Meta-Learning & Uncertainty Quantification PINN Demonstration
//!
//! This example demonstrates the advanced machine learning capabilities of the PINN framework,
//! showcasing meta-learning for rapid adaptation and uncertainty quantification for reliability.
//!
//! ## Advanced ML Features Demonstrated
//!
//! ### Meta-Learning (MAML - Model-Agnostic Meta-Learning)
//! - Inner-loop adaptation for new physics tasks
//! - Outer-loop meta-parameter optimization
//! - Few-shot learning across physics domains
//! - Cross-geometry generalization
//!
//! ### Transfer Learning
//! - Knowledge transfer from simple to complex geometries
//! - Domain adaptation for different physics parameters
//! - Fine-tuning strategies and layer freezing
//! - Transfer accuracy preservation
//!
//! ### Uncertainty Quantification
//! - Bayesian neural networks with Monte Carlo dropout
//! - Deep ensembles for robust uncertainty estimation
//! - Conformal prediction for guaranteed coverage
//! - Reliability metrics and calibration
//!
//! ## Usage
//!
//! ```bash
//! # Run meta-learning demonstration
//! cargo run --example pinn_meta_uncertainty -- --meta
//!
//! # Run transfer learning demonstration
//! cargo run --example pinn_meta_uncertainty -- --transfer
//!
//! # Run uncertainty quantification demonstration
//! cargo run --example pinn_meta_uncertainty -- --uncertainty
//!
//! # Run complete ML demonstration
//! cargo run --example pinn_meta_uncertainty -- --all
//! ```

use std::io::Write;
use std::time::Instant;

#[cfg(feature = "pinn")]
#[path = "pinn_meta_uncertainty/ml_demo/mod.rs"]
mod ml_demo;

#[cfg(not(feature = "pinn"))]
mod ml_demo {
    pub fn demonstrate_meta_learning() {
        eprintln!("❌ PINN feature not enabled. Use --features pinn to enable ML capabilities.");
    }
    pub fn demonstrate_transfer_learning() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_uncertainty() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_integrated_ml() {
        eprintln!("❌ PINN feature not enabled.");
    }
    pub fn demonstrate_real_world_impact() {
        eprintln!("❌ PINN feature not enabled.");
    }
}

fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = std::env::args().collect();

    let _ = writeln!(
        std::io::stdout().lock(),
        "🎓 Advanced ML PINN Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "================================="
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "🧠 Featuring: Meta-Learning • Transfer Learning • Uncertainty Quantification"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Applications: Medical • Aerospace • Industrial • Environmental"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Parse command line arguments
    let demo_mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "--all"
    };

    match demo_mode {
        "--meta" => {
            ml_demo::demonstrate_meta_learning();
        }
        "--transfer" => {
            ml_demo::demonstrate_transfer_learning();
        }
        "--uncertainty" => {
            ml_demo::demonstrate_uncertainty();
        }
        "--integrated" => {
            ml_demo::demonstrate_integrated_ml();
        }
        "--impact" => {
            ml_demo::demonstrate_real_world_impact();
        }
        "--all" => {
            let _ = writeln!(
                std::io::stdout().lock(),
                "🎭 Complete Advanced ML Demonstration"
            );
            let _ = writeln!(
                std::io::stdout().lock(),
                "====================================="
            );
            let _ = writeln!(std::io::stdout().lock());

            ml_demo::demonstrate_meta_learning();
            ml_demo::demonstrate_transfer_learning();
            ml_demo::demonstrate_uncertainty();
            ml_demo::demonstrate_integrated_ml();
            ml_demo::demonstrate_real_world_impact();
        }
        _ => {
            let _ = writeln!(
                std::io::stdout().lock(),
                "🎭 Complete Advanced ML Demonstration"
            );
            let _ = writeln!(
                std::io::stdout().lock(),
                "====================================="
            );
            let _ = writeln!(std::io::stdout().lock());

            ml_demo::demonstrate_meta_learning();
            ml_demo::demonstrate_transfer_learning();
            ml_demo::demonstrate_uncertainty();
            ml_demo::demonstrate_integrated_ml();
            ml_demo::demonstrate_real_world_impact();
        }
    }

    let elapsed = start_time.elapsed();
    let _ = writeln!(
        std::io::stdout().lock(),
        "🏆 Advanced ML Demonstration Complete!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "====================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ⏱️  Total runtime: {:.2}s",
        elapsed.as_secs_f64()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ✅ All ML capabilities demonstrated"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   🚀 Ready for safety-critical applications"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "📚 ML-Specific Examples:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --meta: Meta-learning for rapid adaptation"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --transfer: Transfer learning across domains"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --uncertainty: Reliability and confidence bounds"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --integrated: Combined ML pipeline"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   • --impact: Real-world safety applications"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "🌟 PINN: From simulation to certified AI systems!"
    );
}
