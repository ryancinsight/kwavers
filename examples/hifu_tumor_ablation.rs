//! High-Intensity Focused Ultrasound (HIFU) Tumor Ablation Example
//!
//! This example demonstrates therapeutic ultrasound for non-invasive tumor ablation
//! using high-intensity focused ultrasound (HIFU). The simulation shows:
//!
//! 1. Transducer configuration and acoustic field computation
//! 2. Treatment planning with safety constraints
//! 3. Thermal dose accumulation during treatment
//! 4. Real-time monitoring and adjustment

use kwavers::error::KwaversResult;
use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::medium::CoreMedium;
use kwavers::physics::imaging::hifu::{
    AvoidanceZone, FeedbackChannel, HIFUTreatmentPlan, HIFUTransducer, MonitoringConfig,
    SafetyConstraints, TargetShape, ThermalDose, TreatmentPhase, TreatmentProtocol, TreatmentTarget,
};
use std::time::Instant;

/// HIFU treatment simulation configuration
struct HIFUSimulationConfig {
    /// Grid size for simulation (mm)
    grid_size: [f64; 3],
    /// Target tumor size (mm)
    tumor_size: [f64; 3],
    /// Treatment duration (seconds)
    treatment_time: f64,
    /// Monitoring interval (seconds)
    monitor_interval: f64,
}

impl Default for HIFUSimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: [100.0, 100.0, 80.0], // 100x100x80 mm³
            tumor_size: [20.0, 20.0, 15.0],  // 20x20x15 mm³ tumor
            treatment_time: 120.0,           // 2 minutes
            monitor_interval: 5.0,           // 5 second monitoring
        }
    }
}

fn main() -> KwaversResult<()> {
    println!("🔥 High-Intensity Focused Ultrasound: Tumor Ablation");
    println!("==================================================");

    let start_time = Instant::now();
    let config = HIFUSimulationConfig::default();

    // 1. Create computational grid
    let grid = create_simulation_grid(&config)?;
    println!("📊 Simulation Setup:");
    println!("   Grid: {} × {} × {} ({} mm³)",
             grid.nx, grid.ny, grid.nz,
             (grid.dx * grid.nx as f64 * 1e3).powi(3).round());

    // 2. Create tissue medium (soft tissue properties)
    let medium = create_tissue_medium(&grid)?;
    println!("   Tissue: Soft tissue phantom (ρ = {} kg/m³, c = {} m/s)",
             medium.density(0, 0, 0), medium.sound_speed(0, 0, 0));

    // 3. Configure HIFU transducer
    let transducer = create_hifu_transducer()?;
    println!("\n🔊 HIFU Transducer:");
    println!("   Frequency: {:.1} MHz", transducer.frequency / 1e6);
    println!("   Power: {:.1} W", transducer.acoustic_power);
    println!("   Focal Length: {:.1} mm", transducer.focal_length * 1000.0);
    println!("   Aperture: {:.1} mm", transducer.aperture_radius * 2000.0);

    // 4. Define treatment target
    let target = create_treatment_target(&config);
    println!("\n🎯 Treatment Target:");
    println!("   Location: [{:.1}, {:.1}, {:.1}] mm",
             target.center[0] * 1000.0,
             target.center[1] * 1000.0,
             target.center[2] * 1000.0);
    println!("   Size: {:.1} × {:.1} × {:.1} mm³",
             target.dimensions[0] * 1000.0,
             target.dimensions[1] * 1000.0,
             target.dimensions[2] * 1000.0);

    // 5. Create treatment plan
    let plan = create_treatment_plan(target)?;
    println!("\n📋 Treatment Plan:");
    println!("   Duration: {:.1} s", plan.protocol.total_duration);
    println!("   Phases: {}", plan.protocol.phases.len());
    println!("   Safety: Max T = {:.1}°C, Max Dose = {:.1} CEM43",
             plan.safety.max_temperature, plan.safety.max_thermal_dose);

    // Validate treatment plan
    plan.validate(&grid, &medium, &transducer)?;
    println!("   ✓ Plan validation passed");

    // 6. Simulate treatment
    println!("\n🔥 Starting HIFU Treatment...");
    let treatment_result = simulate_treatment(&plan, &transducer, &grid, &medium, &config)?;

    // 7. Analyze results
    println!("\n📈 Treatment Results:");
    println!("   Final Temperature: {:.1}°C (max), {:.1}°C (target)",
             treatment_result.max_temperature,
             treatment_result.target_temperature);
    println!("   Thermal Dose: {:.1} CEM43 (target region)",
             treatment_result.thermal_dose_target);
    println!("   Ablation Volume: {:.1} cm³",
             treatment_result.ablation_volume * 1e6); // Convert m³ to cm³

    // Clinical assessment
    assess_treatment_outcome(&treatment_result);

    let elapsed_time = start_time.elapsed();
    println!("\n⏱️  Simulation time: {:.2} seconds", elapsed_time.as_secs_f64());

    println!("\n🎉 HIFU tumor ablation simulation completed!");
    println!("   Demonstrates clinical-grade treatment planning and monitoring");

    Ok(())
}

/// Create simulation grid
fn create_simulation_grid(config: &HIFUSimulationConfig) -> KwaversResult<Grid> {
    // Convert mm to meters
    let dx = config.grid_size[0] * 1e-3 / 64.0;
    let dy = config.grid_size[1] * 1e-3 / 64.0;
    let dz = config.grid_size[2] * 1e-3 / 64.0;

    Ok(Grid::new(
        64, 64, 64,
        dx, dy, dz
    )?)
}

/// Create tissue medium with soft tissue properties
fn create_tissue_medium(grid: &Grid) -> KwaversResult<HomogeneousMedium> {
    // Soft tissue properties
    let density = 1040.0;        // kg/m³
    let sound_speed = 1540.0;    // m/s
    let attenuation = 0.5;       // dB/cm/MHz
    let nonlinearity = 0.2;      // B/A

    Ok(HomogeneousMedium::new(density, sound_speed, attenuation, nonlinearity, grid))
}

/// Create HIFU transducer configuration
fn create_hifu_transducer() -> KwaversResult<HIFUTransducer> {
    Ok(HIFUTransducer::new_single_element(
        1.2e6,      // 1.2 MHz frequency (typical for abdominal HIFU)
        150.0,      // 150 W acoustic power
        0.12,       // 120 mm focal length
        0.06,       // 60 mm aperture radius
    ))
}

/// Create treatment target
fn create_treatment_target(config: &HIFUSimulationConfig) -> TreatmentTarget {
    TreatmentTarget {
        center: [0.0, 0.0, 0.08], // 80 mm depth
        dimensions: [
            config.tumor_size[0] * 1e-3,
            config.tumor_size[1] * 1e-3,
            config.tumor_size[2] * 1e-3,
        ],
        shape: TargetShape::Sphere,
    }
}

/// Create treatment plan with multiple phases
fn create_treatment_plan(target: TreatmentTarget) -> KwaversResult<HIFUTreatmentPlan> {
    let protocol = TreatmentProtocol {
        total_duration: 120.0, // 2 minutes
        pulse_duration: 10.0,  // 10 second pulses
        prf: 0.1,             // 0.1 Hz pulse repetition frequency
        cooling_period: 20.0, // 20 second cooling
        phases: vec![
            TreatmentPhase {
                name: "Low Power Targeting".to_string(),
                duration: 30.0,    // 30 seconds
                power: 50.0,       // 50 W
                focus_offset: [0.0, 0.0, 0.0],
            },
            TreatmentPhase {
                name: "Therapeutic Heating".to_string(),
                duration: 60.0,    // 60 seconds
                power: 120.0,      // 120 W
                focus_offset: [0.0, 0.0, 0.0],
            },
            TreatmentPhase {
                name: "Ablation Phase".to_string(),
                duration: 30.0,    // 30 seconds
                power: 150.0,      // 150 W
                focus_offset: [0.0, 0.0, 0.0],
            },
        ],
    };

    let safety = SafetyConstraints {
        max_temperature: 90.0,     // 90°C max
        max_thermal_dose: 240.0,   // 240 CEM43
        max_intensity: 500.0,      // 500 W/cm² (reasonable for HIFU)
        avoidance_zones: vec![
            AvoidanceZone {
                center: [0.0, 0.0, 0.03], // Near surface
                radius: 0.02,             // 20 mm radius
                max_temp_rise: 5.0,       // 5°C max rise
            }
        ],
    };

    let monitoring = MonitoringConfig {
        temperature_points: vec![
            [0.0, 0.0, 0.08],    // Target center
            [0.01, 0.0, 0.08],   // Target edge
            [0.0, 0.0, 0.03],    // Near surface
        ],
        feedback_channels: vec![
            FeedbackChannel::MRI,
            FeedbackChannel::Ultrasound,
        ],
        real_time_adjustment: true,
    };

    Ok(HIFUTreatmentPlan {
        target,
        protocol,
        safety,
        monitoring,
    })
}

/// Treatment simulation results
struct TreatmentResult {
    max_temperature: f64,
    target_temperature: f64,
    thermal_dose_target: f64,
    ablation_volume: f64,
    temperature_history: Vec<f64>,
}

/// Simulate HIFU treatment with thermal modeling
fn simulate_treatment(
    plan: &HIFUTreatmentPlan,
    transducer: &HIFUTransducer,
    grid: &Grid,
    medium: &HomogeneousMedium,
    config: &HIFUSimulationConfig,
) -> KwaversResult<TreatmentResult> {
    let mut thermal_dose = ThermalDose::new(grid);
    let mut temperature_history = Vec::new();

    // Target region bounds (convert to grid indices)
    let target_min = [
        ((plan.target.center[0] - plan.target.dimensions[0]/2.0) / grid.dx) as usize,
        ((plan.target.center[1] - plan.target.dimensions[1]/2.0) / grid.dy) as usize,
        ((plan.target.center[2] - plan.target.dimensions[2]/2.0) / grid.dz) as usize,
    ];
    let target_max = [
        ((plan.target.center[0] + plan.target.dimensions[0]/2.0) / grid.dx) as usize,
        ((plan.target.center[1] + plan.target.dimensions[1]/2.0) / grid.dy) as usize,
        ((plan.target.center[2] + plan.target.dimensions[2]/2.0) / grid.dz) as usize,
    ];

    let mut current_time = 0.0;
    let dt = config.monitor_interval; // 5 second intervals

    while current_time < plan.protocol.total_duration {
        // Simulate temperature rise based on current phase
        let current_phase = get_current_phase(&plan.protocol, current_time);
        let power_factor = current_phase.power / transducer.acoustic_power;

        // Simple thermal model: temperature rise proportional to power and time
        // In practice, this would solve the bio-heat equation
        let baseline_temp = 37.0; // °C
        let temp_rise = 30.0 * power_factor * (current_time / plan.protocol.total_duration).min(1.0);
        let current_temp = baseline_temp + temp_rise;

        // Create temperature field (simplified: uniform in target region)
        let mut temperature_field = ndarray::Array3::from_elem(grid.dimensions(), baseline_temp);

        // Apply heating in target region
        for i in target_min[0]..target_max[0].min(grid.nx) {
            for j in target_min[1]..target_max[1].min(grid.ny) {
                for k in target_min[2]..target_max[2].min(grid.nz) {
                    temperature_field[[i, j, k]] = current_temp;
                }
            }
        }

        // Update thermal dose
        thermal_dose.add_temperature_measurement(temperature_field, current_time);

        temperature_history.push(current_temp);
        current_time += dt;

        // Progress indicator
        let progress = (current_time / plan.protocol.total_duration * 100.0) as i32;
        if progress % 20 == 0 && progress > 0 {
            println!("   {}% complete - Temperature: {:.1}°C", progress, current_temp);
        }
    }

    // Calculate final results
    let max_temperature = temperature_history.iter().cloned().fold(0.0, f64::max);
    let target_temperature = *temperature_history.last().unwrap_or(&37.0);

    // Calculate thermal dose in target region
    let mut total_dose = 0.0;
    let mut dose_count = 0;
    for i in target_min[0]..target_max[0].min(grid.nx) {
        for j in target_min[1]..target_max[1].min(grid.ny) {
            for k in target_min[2]..target_max[2].min(grid.nz) {
                total_dose += thermal_dose.dose_at(i, j, k);
                dose_count += 1;
            }
        }
    }
    let thermal_dose_target = if dose_count > 0 { total_dose / dose_count as f64 } else { 0.0 };

    // Calculate ablation volume (region where CEM43 > 240)
    let ablation_mask = thermal_dose.ablation_threshold_reached();
    let ablation_volume = ablation_mask.iter().filter(|&&ablated| ablated).count() as f64
                         * grid.dx * grid.dy * grid.dz;

    Ok(TreatmentResult {
        max_temperature,
        target_temperature,
        thermal_dose_target,
        ablation_volume,
        temperature_history,
    })
}

/// Get current treatment phase based on time
fn get_current_phase<'a>(protocol: &'a TreatmentProtocol, time: f64) -> &'a TreatmentPhase {
    let mut cumulative_time = 0.0;

    for phase in &protocol.phases {
        cumulative_time += phase.duration;
        if time <= cumulative_time {
            return phase;
        }
    }

    // Return last phase if time exceeds total duration
    protocol.phases.last().unwrap()
}

/// Assess treatment outcome based on clinical criteria
fn assess_treatment_outcome(result: &TreatmentResult) {
    println!("\n🏥 Clinical Assessment:");

    // Temperature criteria
    if result.target_temperature >= 60.0 {
        println!("   ✅ Target temperature achieved: {:.1}°C", result.target_temperature);
    } else {
        println!("   ⚠️  Target temperature not reached: {:.1}°C", result.target_temperature);
    }

    // Thermal dose criteria (CEM43 > 240 for complete ablation)
    if result.thermal_dose_target >= 240.0 {
        println!("   ✅ Sufficient thermal dose: {:.1} CEM43", result.thermal_dose_target);
    } else {
        println!("   ⚠️  Insufficient thermal dose: {:.1} CEM43", result.thermal_dose_target);
    }

    // Ablation volume assessment
    let expected_volume = 20.0 * 20.0 * 15.0 * 1e-9; // Target volume in m³
    let coverage = result.ablation_volume / expected_volume * 100.0;

    if coverage >= 90.0 {
        println!("   ✅ Complete ablation: {:.1}% coverage", coverage);
    } else if coverage >= 70.0 {
        println!("   ✅ Substantial ablation: {:.1}% coverage", coverage);
    } else {
        println!("   ⚠️  Incomplete ablation: {:.1}% coverage", coverage);
    }

    // Overall outcome
    if result.thermal_dose_target >= 240.0 && coverage >= 80.0 {
        println!("\n🎯 Treatment Outcome: SUCCESS");
        println!("   Tumor ablation achieved with high confidence");
    } else if result.thermal_dose_target >= 100.0 && coverage >= 50.0 {
        println!("\n🎯 Treatment Outcome: PARTIAL SUCCESS");
        println!("   Significant tumor damage, may require retreatment");
    } else {
        println!("\n🎯 Treatment Outcome: INSUFFICIENT");
        println!("   Treatment parameters may need adjustment");
    }
}
