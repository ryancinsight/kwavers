//! Stereotactic Targeting System
//!
//! Converts between different coordinate systems and provides safe targeting
//! with anatomical validation.

use super::atlas::BrainAtlas;
use crate::core::error::{KwaversError, KwaversResult};

/// Stereotactic coordinates (based on Bregma reference)
#[derive(Debug, Clone, Copy)]
pub struct StereotacticCoordinates {
    /// Anterior-Posterior (relative to Bregma) (mm)
    pub ap: f64,

    /// Medial-Lateral (relative to midline) (mm)
    pub ml: f64,

    /// Dorsal-Ventral (from brain surface) (mm)
    pub dv: f64,

    /// Confidence in coordinates (0.0-1.0)
    pub confidence: f64,
}

impl StereotacticCoordinates {
    /// Create new stereotactic coordinates
    #[must_use]
    pub fn new(ap: f64, ml: f64, dv: f64) -> Self {
        Self {
            ap,
            ml,
            dv,
            confidence: 1.0,
        }
    }

    /// Convert to [x, y, z] array
    #[must_use]
    pub fn to_array(&self) -> [f64; 3] {
        [self.ap, self.ml, self.dv]
    }

    /// Validate coordinates are within brain bounds
    #[must_use]
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
    #[must_use]
    pub fn is_safe(&self) -> bool {
        // Simplified safety check
        self.is_valid() && self.dv > 0.5 && self.dv < 7.5
    }
}

/// Stereotactic targeting system
#[derive(Debug)]
pub struct TargetingSystem {
    /// Bregma reference point in atlas coordinates (mm)
    bregma: [f64; 3],
}

impl TargetingSystem {
    /// Create new targeting system
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(atlas: &BrainAtlas) -> KwaversResult<Self> {
        let bregma = atlas.brain_center();

        Ok(Self { bregma })
    }

    /// Convert voxel coordinates to stereotactic
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn stereotactic_to_voxel(
        &self,
        coords: &StereotacticCoordinates,
        atlas: &BrainAtlas,
    ) -> KwaversResult<[usize; 3]> {
        if !coords.is_valid() {
            return Err(KwaversError::InvalidInput(
                "Stereotactic coordinates outside brain bounds".to_owned(),
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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn plan_trajectory(
        &self,
        start: &StereotacticCoordinates,
        target: &StereotacticCoordinates,
        num_waypoints: usize,
    ) -> KwaversResult<Vec<StereotacticCoordinates>> {
        if !start.is_safe() || !target.is_safe() {
            return Err(KwaversError::InvalidInput(
                "Trajectory endpoints not in safe region".to_owned(),
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

    /// Get distance between two coordinates (mm)
    #[must_use]
    pub fn distance(&self, from: &StereotacticCoordinates, to: &StereotacticCoordinates) -> f64 {
        let dx = to.ap - from.ap;
        let dy = to.ml - from.ml;
        let dz = to.dv - from.dv;

        dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
    }

    /// Get Bregma coordinates
    #[must_use]
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
        let _targeting = TargetingSystem::new(&atlas).unwrap();
    }

    #[test]
    fn test_trajectory_planning() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();

        let start = StereotacticCoordinates::new(0.0, 0.0, 1.0);
        let target = StereotacticCoordinates::new(1.0, 1.0, 3.0);

        let trajectory = targeting.plan_trajectory(&start, &target, 5).unwrap();
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

    // ─── Exact value-semantic tests ───────────────────────────────────────────

    /// 3-4-5 right triangle distance is exactly 5.0.
    ///
    /// Δap=3, Δml=4, Δdv=0 → d = √(9+16) = √25 = 5.0 (exact in IEEE 754).
    #[test]
    fn targeting_distance_three_four_five_exact() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();
        let p1 = StereotacticCoordinates::new(0.0, 0.0, 0.0);
        let p2 = StereotacticCoordinates::new(3.0, 4.0, 0.0);
        let dist = targeting.distance(&p1, &p2);
        assert!(
            (dist - 5.0).abs() < 1e-12,
            "3-4-5 distance should be exactly 5.0, got {dist}"
        );
    }

    /// Trajectory endpoints match start and target exactly.
    ///
    /// Linear interpolation at t=0 gives start and at t=1 gives target.
    /// With `num_waypoints=5`: waypoint[0] = start, waypoint[4] = target.
    #[test]
    fn targeting_trajectory_endpoints_match_start_and_target() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();
        let start = StereotacticCoordinates::new(-1.0, 0.5, 1.0);
        let target = StereotacticCoordinates::new(1.0, -0.5, 4.0);
        let traj = targeting.plan_trajectory(&start, &target, 5).unwrap();
        assert_eq!(traj.len(), 5, "expected 5 waypoints");
        // t=0 → start
        assert!(
            (traj[0].ap - start.ap).abs() < 1e-12,
            "first waypoint ap mismatch: {} vs {}",
            traj[0].ap,
            start.ap
        );
        assert!(
            (traj[0].ml - start.ml).abs() < 1e-12,
            "first waypoint ml mismatch"
        );
        assert!(
            (traj[0].dv - start.dv).abs() < 1e-12,
            "first waypoint dv mismatch"
        );
        // t=1 → target
        assert!(
            (traj[4].ap - target.ap).abs() < 1e-12,
            "last waypoint ap mismatch: {} vs {}",
            traj[4].ap,
            target.ap
        );
        assert!(
            (traj[4].dv - target.dv).abs() < 1e-12,
            "last waypoint dv mismatch"
        );
    }

    /// Trajectory midpoint (waypoint[2] of 5) is the exact midpoint of start and target.
    ///
    /// t = 2/4 = 0.5 → ap_mid = (ap_start + ap_end) / 2.
    #[test]
    fn targeting_trajectory_midpoint_is_linear_interpolation() {
        let atlas = BrainAtlas::load_default().unwrap();
        let targeting = TargetingSystem::new(&atlas).unwrap();
        let start = StereotacticCoordinates::new(-2.0, 1.0, 1.0);
        let target = StereotacticCoordinates::new(0.0, -1.0, 3.0);
        let traj = targeting.plan_trajectory(&start, &target, 5).unwrap();
        let ap_mid = (start.ap + target.ap) * 0.5;
        let ml_mid = (start.ml + target.ml) * 0.5;
        let dv_mid = (start.dv + target.dv) * 0.5;
        assert!(
            (traj[2].ap - ap_mid).abs() < 1e-12,
            "midpoint ap: expected {ap_mid}, got {}",
            traj[2].ap
        );
        assert!(
            (traj[2].ml - ml_mid).abs() < 1e-12,
            "midpoint ml: expected {ml_mid}, got {}",
            traj[2].ml
        );
        assert!(
            (traj[2].dv - dv_mid).abs() < 1e-12,
            "midpoint dv: expected {dv_mid}, got {}",
            traj[2].dv
        );
    }

    /// Boundary: stereotactic coordinate at exactly the boundary of `is_valid` returns true.
    ///
    /// Bounds: ap ∈ [−4, 3], ml ∈ [−4.5, 4.5], dv ∈ [0, 8].
    /// Corner point (−4, −4.5, 0) is on the boundary → `is_valid() == true`.
    #[test]
    fn stereotactic_boundary_coordinates_are_valid() {
        let corner = StereotacticCoordinates::new(-4.0, -4.5, 0.0);
        assert!(
            corner.is_valid(),
            "boundary corner must be valid: {corner:?}"
        );
        let just_outside = StereotacticCoordinates::new(-4.0 - 1e-9, -4.5, 0.0);
        assert!(!just_outside.is_valid(), "just outside must be invalid");
    }
}
