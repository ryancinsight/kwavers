use kwavers_core::constants::SOUND_SPEED_TISSUE;

/// Sensor array geometry specification.
///
/// Defines the spatial arrangement of sensor elements for beamforming
/// delay and apodization calculations.
#[derive(Debug, Clone)]
pub struct SensorGeometry {
    /// 3D positions of sensor elements [x, y, z] in meters.
    pub positions: Vec<[f64; 3]>,

    /// Sampling frequency in Hz.
    pub sampling_frequency: f64,

    /// Speed of sound in medium (m/s). Default: 1540 m/s (soft tissue)
    pub sound_speed: f64,
}

impl SensorGeometry {
    /// Create linear array geometry.
    #[must_use]
    pub fn linear_array(
        num_elements: usize,
        pitch: f64,
        sampling_frequency: f64,
        sound_speed: f64,
    ) -> Self {
        let positions: Vec<[f64; 3]> = (0..num_elements)
            .map(|i| {
                let x = (i as f64 - (num_elements - 1) as f64 / 2.0) * pitch;
                [x, 0.0, 0.0]
            })
            .collect();

        Self {
            positions,
            sampling_frequency,
            sound_speed,
        }
    }

    /// Create phased array geometry (2D).
    #[must_use]
    pub fn phased_array(
        nx: usize,
        ny: usize,
        pitch_x: f64,
        pitch_y: f64,
        sampling_frequency: f64,
        sound_speed: f64,
    ) -> Self {
        let mut positions = Vec::with_capacity(nx * ny);

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 - (nx - 1) as f64 / 2.0) * pitch_x;
                let y = (j as f64 - (ny - 1) as f64 / 2.0) * pitch_y;
                positions.push([x, y, 0.0]);
            }
        }

        Self {
            positions,
            sampling_frequency,
            sound_speed,
        }
    }

    /// Get number of sensor elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.positions.len()
    }
}

impl Default for SensorGeometry {
    fn default() -> Self {
        Self::linear_array(64, 0.0003, 40e6, SOUND_SPEED_TISSUE)
    }
}
