use super::cloud::BubbleCloud;
use super::core::BubbleField;
use super::distributions::{SizeDistribution, SpatialDistribution};
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;

#[test]
fn test_bubble_field_creation() {
    let params = BubbleParameters::default();
    let mut field = BubbleField::new((10, 10, 10), params.clone());

    field.add_center_bubble(&params);
    assert_eq!(field.bubbles.len(), 1);
    assert!(field.bubbles.contains_key(&(5, 5, 5)));
}

#[test]
fn test_bubble_cloud_generation() {
    let params = BubbleParameters::default();
    let size_dist = SizeDistribution::Uniform {
        min: 1e-6,
        max: 10e-6,
    };
    let spatial_dist = SpatialDistribution::Uniform;

    let mut cloud = BubbleCloud::new((5, 5, 5), params, size_dist, spatial_dist);
    cloud.generate(1e9, (1e-3, 1e-3, 1e-3)); // Moderate density for small grid

    assert!(!cloud.field.bubbles.is_empty());
}
