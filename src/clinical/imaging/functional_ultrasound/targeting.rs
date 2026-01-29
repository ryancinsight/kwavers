//! Stereotactic Targeting System
//!
//! Converts between different coordinate systems and provides safe targeting
//! with anatomical validation.

use super::atlas::BrainAtlas;
use crate::core::error::{KwaversError, KwaversResult};

/// Stereotactic coordinates (based on Bregma reference)
#[derive(Debug, Clone, Copy)]
pub struct StereotacticCoordinates {
    /// Anterior-Posterior (relative to Bregma) [mm]
    pub ap: f64,

    /// Medial-Lateral (relative to midline) [mm]
    pub ml: f64,

    /// Dorsal-Ventral (from brain surface) [mm]
    pub dv: f64,

    /// Confidence in coordinates (0.0-1.0)
    pub confidence: f64,
}

impl StereotacticCoordinates {
    /// Create new stereotactic coordinates
    pub fn new(ap: f64, ml: f64, dv: f64) -> Self {
        Self {
            ap,
            ml,
            dv,
            confidence: 1.0,
        }
    }

    /// Convert to [x, y, z] array
    pub fn to_array(&self) -> [f64; 3] {
        [self.ap, self.ml, self.dv]
    }

    /// Validate coordinates are within brain bounds
    pub fn is_valid(&self) -> bool {
        // Standard mouse brain bounds (relative to Bregma)
        self.ap >= -4.0
            && self.ap <= 3.0
            && self.ml >= -4.5
            && self.ml <= 4.5
            && self.dv >= 0.0
            && self.dv <= 8.0
    }

    /// Check if coordinates are in safe region (away from ventricles/major vessels)
    pub fn is_safe(&self) -> bool {
        // Simplified safety check
        self.is_valid() && self.dv > 0.5 && self.dv < 7.5
    }
}

/// Stereotactic targeting system
#[derive(Debug)]
pub struct TargetingSystem {
    /// Bregma reference point in atlas coordinates [mm]
    bregma: [f64; 3],

    /// Brain dimensions [mm]
    brain_dims: [f64; 3],
}

impl TargetingSystem {
    /// Create new targeting system
    pub fn new(atlas: &BrainAtlas) -> KwaversResult<Self> {
        let bregma = atlas.brain_center();

        // Mouse brain typical dimensions
        let brain_dims = [8.0, 9.0, 8.0];

        Ok(Self { bregma, brain_dims })
    }

    /// Convert voxel coordinates to stereotactic
    pub fn voxel_to_stereotactic(
        &self,
        voxel: &[usize; 3],
        atlas: &BrainAtlas,
    ) -> KwaversResult<StereotacticCoordinates> {
        // Convert voxel to mm in atlas space
        let mm = atlas.voxel_to_mm(&[voxel[0], voxel[1], voxel[2]]);

        // Convert to Bregma-relative stereotactic coordinates
        let ap = mm[0] - self.bregma[0];
        let ml = mm[1] - self.bregma[1];
        let dv = self.bregma[2] - mm[2]; // DV is measured downward from surface

        let coords = StereotacticCoordinates::new(ap, ml, dv);

        Ok(coords)
    }

    /// Convert stereotactic to voxel coordinates
    pub fn stereotactic_to_voxel(
        &self,
        coords: &StereotacticCoordinates,
        atlas: &BrainAtlas,
    ) -> KwaversResult<[usize; 3]> {
        if !coords.is_valid() {
            return Err(KwaversError::InvalidInput(
                "Stereotactic coordinates outside brain bounds".to_string(),
            ));
        }

        // Convert to atlas mm coordinates
        let mm = [
            self.bregma[0] + coords.ap,
            self.bregma[1] + coords.ml,
            self.bregma[2] - coords.dv,
        ];

        atlas.mm_to_voxel(&mm)
    }

    /// Plan trajectory to target
    pub fn plan_trajectory(
        &self,
        start: &StereotacticCoordinates,
        target: &StereotacticCoordinates,
        num_waypoints: usize,
    ) -> KwaversResult<Vec<StereotacticCoordinates>> {
        if !start.is_safe() || !target.is_safe() {
            return Err(KwaversError::InvalidInput(
                "Trajectory endpoints not in safe region".to_string(),
            ));
        }

        let mut trajectory = Vec::with_capacity(num_waypoints);

        for i in 0..num_waypoints {
            let t = i as f64 / (num_waypoints - 1).max(1) as f64;

            let ap = start.ap + t * (target.ap - start.ap);
            let ml = start.ml + t * (target.ml - start.ml);
            let dv = start.dv + t * (target.dv - start.dv);

            trajectory.push(StereotacticCoordinates::new(ap, ml, dv));
        }

        Ok(trajectory)
    }

    /// Get distance between two coordinates [mm]
    pub fn distance(&self, from: &StereotacticCoordinates, to: &StereotacticCoordinates) -> f64 {
        let dx = to.ap - from.ap;
        let dy = to.ml - from.ml;
        let dz = to.dv - from.dv;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get Bregma coordinates
    pub fn bregma(&self) -> [f64; 3] {
        self.bregma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereotactic_coordinates_creation() {
        let coords = StereotacticCoordinates::new(0.5, 1.0, 2.5);
        assert_eq!(coords.ap, 0.5);
        assert_eq!(coords.ml, 1.0);
        assert_eq!(coords.dv, 2.5);
    }

    #[test]
    fn test_stereotactic_validation() {
        let valid = StereotacticCoordinates::new(0.0, 0.0, 3.0);
        assert!(valid.is_valid());
        assert!(valid.is_safe());

        let invalid = StereotacticCoordinates::new(10.0, 0.0, 3.0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_targeting_system_creation() {
        let atlas = BrainAtlas::load_default().unwrap();
        let result = TargetingSystem::new(&atlas);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trajectory_planning() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();

        let start = StereotacticCoordinates::new(0.0, 0.0, 1.0);
        let target = StereotacticCoordinates::new(1.0, 1.0, 3.0);

        let result = targeting.plan_trajectory(&start, &target, 5);
        assert!(result.is_ok());

        let trajectory = result.unwrap();
        assert_eq!(trajectory.len(), 5);
    }

    #[test]
    fn test_distance_calculation() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();

        let p1 = StereotacticCoordinates::new(0.0, 0.0, 0.0);
        let p2 = StereotacticCoordinates::new(3.0, 4.0, 0.0);

        let dist = targeting.distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 0.01); // 3-4-5 triangle
    }
}
