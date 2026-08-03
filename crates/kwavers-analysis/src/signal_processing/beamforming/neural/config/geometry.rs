use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::constants::SOUND_SPEED_TISSUE;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Sensor array geometry specification.
///
/// Defines the spatial arrangement of sensor elements for beamforming
/// delay and apodization calculations.
#[derive(Debug, Clone)]
pub struct SensorGeometry {
    /// 3D positions of sensor elements [x, y, z] in meters.
    pub positions: Vec<[Length<f64>; 3]>,

    /// Sampling frequency in Hz.
    pub sampling_frequency: Frequency<f64>,

    /// Speed of sound in medium (m/s). Default: 1540 m/s (soft tissue)
    pub sound_speed: Velocity<f64>,
}

impl SensorGeometry {
    /// Create linear array geometry.
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] for empty arrays or non-finite
    /// and non-positive physical parameters.
    pub fn linear_array(
        num_elements: usize,
        pitch: Length<f64>,
        sampling_frequency: Frequency<f64>,
        sound_speed: Velocity<f64>,
    ) -> KwaversResult<Self> {
        Self::validate_parameters(
            num_elements,
            1,
            pitch,
            pitch,
            sampling_frequency,
            sound_speed,
        )?;
        let pitch = pitch.in_unit::<Meter>();
        let positions: Vec<[Length<f64>; 3]> = (0..num_elements)
            .map(|i| {
                let x = (i as f64 - (num_elements - 1) as f64 / 2.0) * pitch;
                [
                    Length::from_unit::<Meter>(x),
                    Length::from_unit::<Meter>(0.0),
                    Length::from_unit::<Meter>(0.0),
                ]
            })
            .collect();

        Ok(Self {
            positions,
            sampling_frequency,
            sound_speed,
        })
    }

    /// Create phased array geometry (2D).
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] for empty arrays or non-finite
    /// and non-positive physical parameters.
    pub fn phased_array(
        nx: usize,
        ny: usize,
        pitch_x: Length<f64>,
        pitch_y: Length<f64>,
        sampling_frequency: Frequency<f64>,
        sound_speed: Velocity<f64>,
    ) -> KwaversResult<Self> {
        Self::validate_parameters(nx, ny, pitch_x, pitch_y, sampling_frequency, sound_speed)?;
        let pitch_x = pitch_x.in_unit::<Meter>();
        let pitch_y = pitch_y.in_unit::<Meter>();
        let capacity = nx.checked_mul(ny).ok_or_else(|| {
            KwaversError::InvalidInput(
                "Sensor geometry dimensions overflow element count".to_owned(),
            )
        })?;
        let mut positions = Vec::with_capacity(capacity);

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 - (nx - 1) as f64 / 2.0) * pitch_x;
                let y = (j as f64 - (ny - 1) as f64 / 2.0) * pitch_y;
                positions.push([
                    Length::from_unit::<Meter>(x),
                    Length::from_unit::<Meter>(y),
                    Length::from_unit::<Meter>(0.0),
                ]);
            }
        }

        Ok(Self {
            positions,
            sampling_frequency,
            sound_speed,
        })
    }

    fn validate_parameters(
        nx: usize,
        ny: usize,
        pitch_x: Length<f64>,
        pitch_y: Length<f64>,
        sampling_frequency: Frequency<f64>,
        sound_speed: Velocity<f64>,
    ) -> KwaversResult<()> {
        let pitch_x = pitch_x.in_unit::<Meter>();
        let pitch_y = pitch_y.in_unit::<Meter>();
        let sampling_frequency = sampling_frequency.in_unit::<Hertz>();
        let sound_speed = sound_speed.in_unit::<MeterPerSecond>();
        if nx == 0 || ny == 0 {
            return Err(KwaversError::InvalidInput(
                "Sensor geometry dimensions must be positive".to_owned(),
            ));
        }
        if !pitch_x.is_finite()
            || pitch_x <= 0.0
            || !pitch_y.is_finite()
            || pitch_y <= 0.0
            || !sampling_frequency.is_finite()
            || sampling_frequency <= 0.0
            || !sound_speed.is_finite()
            || sound_speed <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "Sensor geometry requires finite positive pitch, sampling frequency, and sound speed"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Get number of sensor elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.positions.len()
    }
}

impl Default for SensorGeometry {
    fn default() -> Self {
        Self::linear_array(
            64,
            Length::from_unit::<Meter>(0.0003),
            Frequency::from_unit::<Hertz>(40e6),
            Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        )
        .expect("invariant: default sensor geometry is physically valid")
    }
}
