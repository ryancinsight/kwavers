use kwavers::{
    grid::Grid, init_logging, medium::absorption::TissueType,
    medium::heterogeneous::tissue::HeterogeneousTissueMedium,
};
use log::info;
use ndarray::Array3;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging()?;

    // Create high-resolution grid for accurate tracking
    let domain_size = 0.04f64; // 4 cm cubic domain
    let dx = 0.0002f64; // 0.2 mm spacing
    let n = (domain_size / dx).round() as usize;
    let grid = Grid::new(n, n, n, dx, dx, dx);

    info!(
        "Created {}x{}x{} grid with {} mm spacing",
        n,
        n,
        n,
        dx * 1000.0
    );

    // Create tissue medium with inclusions of different stiffness
    let medium = HeterogeneousTissueMedium::new(grid.clone()?, TissueType::Muscle);

    // Create shear modulus distribution with stiff inclusion
    let mut mu = Array3::<f64>::ones((n, n, n)) * 3.0e3; // 3 kPa background (soft tissue)

    // Add stiffer inclusion (simulating tumor)
    let center = n / 2;
    let radius_f64 = 0.005 / dx;
    let radius = if radius_f64 > usize::MAX as f64 {
        usize::MAX
    } else {
        radius_f64 as usize
    };

    for i in (center - radius)..(center + radius) {
        for j in (center - radius)..(center + radius) {
            for k in (center - radius)..(center + radius) {
                let dx_dist = (i as f64 - center as f64) * dx;
                let dy_dist = (j as f64 - center as f64) * dx;
                let dz_dist = (k as f64 - center as f64) * dx;

                if dx_dist * dx_dist + dy_dist * dy_dist + dz_dist * dz_dist
                    <= (radius as f64 * dx).powi(2)
                {
                    mu[[i, j, k]] = 15.0e3; // 15 kPa (stiffer region)
                }
            }
        }
    }

    info!("Starting elastography simulation...");

    // Create a simple displacement field for demonstration
    let mut displacement = Array3::<f64>::zeros((n, n, n));

    // Add initial displacement at the focus region
    for i in (center - radius)..(center + radius) {
        for j in (center - radius)..(center + radius) {
            for k in (center - radius)..(center + radius) {
                let dx_dist = (i as f64 - center as f64) * dx;
                let dy_dist = (j as f64 - center as f64) * dx;
                let dz_dist = (k as f64 - center as f64) * dx;

                if dx_dist * dx_dist + dy_dist * dy_dist + dz_dist * dz_dist
                    <= (radius as f64 * dx).powi(2)
                {
                    displacement[[i, j, k]] = 1e-6; // 1 μm initial displacement
                }
            }
        }
    }

    // Time parameters for demonstration
    let num_tracking_steps = 100;

    // Track displacement evolution (simplified example)
    for step in 0..num_tracking_steps {
        // Simple displacement evolution (for demonstration)
        displacement *= 0.99; // decay factor

        if step % 10 == 0 {
            // Save displacement field every 10 steps (demonstrates error handling)
            save_displacement_snapshot(&displacement, step)?;
            info!("Tracking step {}/{}", step, num_tracking_steps);
        }
    }

    info!("Simulation complete");

    Ok(())
}

fn save_displacement_snapshot(
    displacement: &Array3<f64>,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("displacement_step_{}.csv", step);
    let mut file = std::fs::File::create(&filename)?;

    for ((i, j, k), &value) in displacement.indexed_iter() {
        if let Err(e) = writeln!(file, "{},{},{},{}", i, j, k, value) {
            eprintln!("Error writing to file: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
