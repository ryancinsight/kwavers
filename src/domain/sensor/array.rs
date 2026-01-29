//! Sensor array geometry and configuration
//!
//! This module defines the spatial configuration of sensor arrays.
//! Sensor arrays are hardware concepts (domain layer).
//! Algorithms that use sensor arrays belong in the analysis layer.

use serde::{Deserialize, Serialize};

/// Position in 3D space (m)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate (m)
    pub x: f64,
    /// Y coordinate (m)
    pub y: f64,
    /// Z coordinate (m)
    pub z: f64,
}

impl Position {
    /// Create new position
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Distance to another position
    #[must_use]
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to array
    #[must_use]
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// From array
    #[must_use]
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

/// Individual sensor in array
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor {
    /// Sensor ID
    pub id: usize,
    /// Position in 3D space (m)
    pub position: Position,
    /// Sensor sensitivity (V/Pa)
    pub sensitivity: f64,
    /// Directivity pattern (0 = omnidirectional)
    pub directivity: f64,
}

impl Sensor {
    /// Create new sensor
    #[must_use]
    pub fn new(id: usize, position: Position) -> Self {
        Self {
            id,
            position,
            sensitivity: 1.0,
            directivity: 0.0,
        }
    }

    /// Distance to a point
    #[must_use]
    pub fn distance_to(&self, point: &Position) -> f64 {
        self.position.distance_to(point)
    }

    /// Time of flight from a point
    #[must_use]
    pub fn time_of_flight(&self, point: &Position, sound_speed: f64) -> f64 {
        self.distance_to(point) / sound_speed
    }
}

/// Array geometry types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ArrayGeometry {
    /// Linear 1D array
    Linear,
    /// Planar 2D array
    Planar,
    /// Circular/annular array
    Circular,
    /// Spherical array
    Spherical,
    /// Arbitrary geometry
    Arbitrary,
}

/// Sensor array configuration
#[derive(Debug, Clone)]
pub struct SensorArray {
    /// Array sensors
    pub sensors: Vec<Sensor>,
    /// Array geometry
    pub geometry: ArrayGeometry,
    /// Reference sound speed (m/s)
    sound_speed: f64,
}

impl SensorArray {
    /// Create new sensor array
    ///
    /// # Arguments
    /// * `sensors` - Vector of sensors (non-empty)
    /// * `sound_speed` - Sound speed in medium (m/s, positive)
    /// * `geometry` - Array geometry type
    ///
    /// # Panics
    /// If sensors is empty or sound_speed <= 0
    #[must_use]
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
    #[must_use]
    pub fn sound_speed(&self) -> f64 {
        self.sound_speed
    }

    /// Get sensor positions
    #[must_use]
    pub fn get_sensor_positions(&self) -> Vec<Position> {
        self.sensors.iter().map(|s| s.position).collect()
    }

    /// Get number of sensors
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    /// Get sensor by ID
    #[must_use]
    pub fn get_sensor(&self, id: usize) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.id == id)
    }

    /// Compute array centroid
    #[must_use]
    pub fn centroid(&self) -> Position {
        let n = self.sensors.len() as f64;
        let sum: Position = self
            .sensors
            .iter()
            .fold(Position::new(0.0, 0.0, 0.0), |acc, s| {
                Position::new(
                    acc.x + s.position.x,
                    acc.y + s.position.y,
                    acc.z + s.position.z,
                )
            });
        Position::new(sum.x / n, sum.y / n, sum.z / n)
    }

    /// Set sound speed
    pub fn set_sound_speed(&mut self, sound_speed: f64) {
        assert!(sound_speed > 0.0, "Sound speed must be positive");
        self.sound_speed = sound_speed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let p1 = Position::new(0.0, 0.0, 0.0);
        let p2 = Position::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sensor_array_centroid() {
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(2.0, 0.0, 0.0)),
        ];
        let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Linear);
        let centroid = array.centroid();
        assert!((centroid.x - 1.0).abs() < 1e-10);
        assert!((centroid.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_of_flight() {
        let sensor = Sensor::new(0, Position::new(0.0, 0.0, 0.0));
        let point = Position::new(0.0, 0.0, 1.5);
        let tof = sensor.time_of_flight(&point, 1500.0);
        assert!((tof - 0.001).abs() < 1e-10);
    }
}
