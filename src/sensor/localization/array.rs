// localization/array.rs - Sensor array configuration

use super::Position;
use serde::{Deserialize, Serialize};

/// Individual sensor in array
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor {
    /// Sensor ID
    pub id: usize,
    /// Position in 3D space
    pub position: Position,
    /// Sensor sensitivity
    pub sensitivity: f64,
    /// Directivity pattern (0 = omnidirectional)
    pub directivity: f64,
}

impl Sensor {
    /// Create new sensor
    pub fn new(id: usize, position: Position) -> Self {
        Self {
            id,
            position,
            sensitivity: 1.0,
            directivity: 0.0,
        }
    }

    /// Distance to a point
    pub fn distance_to(&self, point: &Position) -> f64 {
        self.position.distance_to(point)
    }

    /// Time of flight from a point - USING sound_speed parameter
    pub fn time_of_flight(&self, point: &Position, sound_speed: f64) -> f64 {
        self.distance_to(point) / sound_speed
    }
}

/// Array geometry types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize))]
pub enum ArrayGeometry {
    Linear,
    Planar,
    Circular,
    Spherical,
    Arbitrary,
}

/// Sensor array for localization
#[derive(Debug, Clone))]
pub struct SensorArray {
    /// Array sensors
    pub sensors: Vec<Sensor>,
    /// Array geometry
    pub geometry: ArrayGeometry,
    /// Reference sound speed
    sound_speed: f64,
}

impl SensorArray {
    /// Create new sensor array - USING all parameters
    pub fn new(sensors: Vec<Sensor>, sound_speed: f64, geometry: ArrayGeometry) -> Self {
        assert!(!sensors.is_empty(), "Array must have at least one sensor");
        assert!(sound_speed > 0.0, "Sound speed must be positive");

        Self {
            sensors,
            geometry,
            sound_speed,
        }
    }

    /// Get sound speed
    pub fn sound_speed(&self) -> f64 {
        self.sound_speed
    }

    /// Get sensor positions
    pub fn get_sensor_positions(&self) -> Vec<Position> {
        self.sensors.iter().map(|s| s.position).collect()
    }

    /// Get number of sensors
    pub fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    /// Get sensor by ID
    pub fn get_sensor(&self, id: usize) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.id == id)
    }

    /// Calculate array centroid
    pub fn centroid(&self) -> Position {
        let n = self.sensors.len() as f64;
        let sum_x: f64 = self.sensors.iter().map(|s| s.position.x).sum();
        let sum_y: f64 = self.sensors.iter().map(|s| s.position.y).sum();
        let sum_z: f64 = self.sensors.iter().map(|s| s.position.z).sum();

        Position::new(sum_x / n, sum_y / n, sum_z / n)
    }

    /// Calculate array aperture (maximum dimension)
    pub fn aperture(&self) -> f64 {
        let mut max_distance = 0.0;

        for i in 0..self.sensors.len() {
            for j in i + 1..self.sensors.len() {
                let dist = self.sensors[i]
                    .position
                    .distance_to(&self.sensors[j].position);
                max_distance = f64::max(max_distance, dist);
            }
        }

        max_distance
    }
}
