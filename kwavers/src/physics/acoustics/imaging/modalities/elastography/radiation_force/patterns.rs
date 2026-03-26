use std::f64::consts::PI;
use super::impulse::PushPulseParameters;

/// Multi-directional push pulse configuration for 3D SWE
#[derive(Debug, Clone)]
pub struct MultiDirectionalPush {
    /// Individual push pulses with locations and properties
    pub pushes: Vec<DirectionalPush>,
    /// Time delays between pushes (s)
    pub time_delays: Vec<f64>,
    /// Total sequence duration (s)
    pub sequence_duration: f64,
}

/// Individual directional push pulse
#[derive(Debug, Clone)]
pub struct DirectionalPush {
    /// Push location [x, y, z] in meters
    pub location: [f64; 3],
    /// Push direction vector (normalized)
    pub direction: [f64; 3],
    /// Amplitude weighting factor
    pub amplitude_weight: f64,
    /// Custom parameters for this push (optional)
    pub parameters: Option<PushPulseParameters>,
}

impl MultiDirectionalPush {
    /// Create orthogonal push pattern for comprehensive 3D coverage
    ///
    /// Generates pushes along x, y, z axes from a central location
    pub fn orthogonal_pattern(center: [f64; 3], spacing: f64) -> Self {
        let pushes = vec![
            // +X direction
            DirectionalPush {
                location: [center[0] + spacing, center[1], center[2]],
                direction: [1.0, 0.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -X direction
            DirectionalPush {
                location: [center[0] - spacing, center[1], center[2]],
                direction: [-1.0, 0.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // +Y direction
            DirectionalPush {
                location: [center[0], center[1] + spacing, center[2]],
                direction: [0.0, 1.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -Y direction
            DirectionalPush {
                location: [center[0], center[1] - spacing, center[2]],
                direction: [0.0, -1.0, 0.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // +Z direction
            DirectionalPush {
                location: [center[0], center[1], center[2] + spacing],
                direction: [0.0, 0.0, 1.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
            // -Z direction
            DirectionalPush {
                location: [center[0], center[1], center[2] - spacing],
                direction: [0.0, 0.0, -1.0],
                amplitude_weight: 1.0,
                parameters: None,
            },
        ];

        // Time delays for sequential excitation
        let time_delays = vec![0.0, 50e-6, 100e-6, 150e-6, 200e-6, 250e-6];
        let sequence_duration = 300e-6; // 300 μs total

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }

    /// Create compound push pattern for enhanced shear wave generation
    ///
    /// Uses multiple pushes at different angles for better wave interference
    pub fn compound_pattern(center: [f64; 3], radius: f64, n_pushes: usize) -> Self {
        let mut pushes = Vec::new();

        for i in 0..n_pushes {
            let angle = 2.0 * PI * (i as f64) / (n_pushes as f64);
            let x = center[0] + radius * angle.cos();
            let y = center[1] + radius * angle.sin();
            let z = center[2];

            // Alternate between different depths for 3D coverage
            let z_offset = if i % 2 == 0 {
                radius * 0.5
            } else {
                -radius * 0.5
            };
            let location = [x, y, z + z_offset];

            // Direction points radially outward from center
            let direction = [
                (x - center[0]) / radius,
                (y - center[1]) / radius,
                z_offset.signum() * 0.5,
            ];

            pushes.push(DirectionalPush {
                location,
                direction,
                amplitude_weight: 1.0,
                parameters: None,
            });
        }

        // Staggered timing for wave interference
        let time_delays: Vec<f64> = (0..n_pushes)
            .map(|i| i as f64 * 25e-6) // 25 μs spacing
            .collect();

        let sequence_duration = time_delays.last().unwrap_or(&0.0) + 100e-6;

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }

    /// Create focused push pattern for targeted 3D SWE
    ///
    /// Concentrates pushes in a specific region of interest
    pub fn focused_pattern(roi_center: [f64; 3], roi_size: [f64; 3], density: usize) -> Self {
        let mut pushes = Vec::new();

        // Create grid of push locations within ROI
        let nx = (roi_size[0] / 0.005).ceil() as usize; // 5mm spacing
        let ny = (roi_size[1] / 0.005).ceil() as usize;
        let nz = (roi_size[2] / 0.005).ceil() as usize;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = roi_center[0] + (i as f64 - nx as f64 / 2.0) * 0.005;
                    let y = roi_center[1] + (j as f64 - ny as f64 / 2.0) * 0.005;
                    let z = roi_center[2] + (k as f64 - nz as f64 / 2.0) * 0.005;

                    // Weight pushes based on distance from ROI center
                    let distance = ((x - roi_center[0]).powi(2)
                        + (y - roi_center[1]).powi(2)
                        + (z - roi_center[2]).powi(2))
                    .sqrt();
                    let max_distance = roi_size.iter().cloned().fold(0.0, f64::max) / 2.0;
                    let weight = (1.0 - distance / max_distance).max(0.1);

                    pushes.push(DirectionalPush {
                        location: [x, y, z],
                        direction: [0.0, 0.0, 1.0], // Axial direction
                        amplitude_weight: weight,
                        parameters: None,
                    });
                }
            }
        }

        // Limit total pushes for computational efficiency
        if pushes.len() > density {
            pushes.sort_by(|a, b| {
                let dist_a = ((a.location[0] - roi_center[0]).powi(2)
                    + (a.location[1] - roi_center[1]).powi(2)
                    + (a.location[2] - roi_center[2]).powi(2))
                .sqrt();
                let dist_b = ((b.location[0] - roi_center[0]).powi(2)
                    + (b.location[1] - roi_center[1]).powi(2)
                    + (b.location[2] - roi_center[2]).powi(2))
                .sqrt();
                dist_a.partial_cmp(&dist_b).unwrap()
            });
            pushes.truncate(density);
        }

        // Sequential timing
        let time_delays: Vec<f64> = (0..pushes.len())
            .map(|i| i as f64 * 10e-6) // 10 μs spacing
            .collect();

        let sequence_duration = time_delays.last().unwrap_or(&0.0) + 50e-6;

        Self {
            pushes,
            time_delays,
            sequence_duration,
        }
    }
}
