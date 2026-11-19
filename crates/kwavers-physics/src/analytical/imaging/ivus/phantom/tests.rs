use super::ivus_vessel_phantom;

#[test]
fn ivus_vessel_phantom_preserves_nested_masks_and_properties() {
    let phantom = ivus_vessel_phantom(128, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();

    assert_eq!(phantom.labels.len(), 128 * 128);
    assert!(phantom.lumen_mask.iter().filter(|&&v| v).count() > 500);
    assert!(phantom
        .lumen_mask
        .iter()
        .zip(&phantom.eel_mask)
        .all(|(&lumen, &eel)| !lumen || eel));
    assert!(phantom
        .fibrous_cap_mask
        .iter()
        .zip(&phantom.plaque_mask)
        .all(|(&cap, &plaque)| !cap || plaque));
    let calcium_mean = masked_mean(&phantom.sound_speed_m_s, &phantom.calcium_mask);
    let lipid_mean = masked_mean(&phantom.sound_speed_m_s, &phantom.lipid_mask);
    assert!(calcium_mean > 2500.0, "calcium_mean={calcium_mean}");
    assert!(lipid_mean < 1500.0, "lipid_mean={lipid_mean}");
}

#[test]
fn ivus_vessel_phantom_is_seed_deterministic() {
    let a = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();
    let b = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 30).unwrap();
    let c = ivus_vessel_phantom(64, 12.0e-3, 0.55e-3, -0.72, 31).unwrap();

    assert_eq!(a.labels, b.labels);
    assert_eq!(a.backscatter, b.backscatter);
    assert_ne!(a.backscatter, c.backscatter);
}

fn masked_mean(values: &[f64], mask: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (&value, &selected) in values.iter().zip(mask) {
        if selected {
            sum += value;
            count += 1;
        }
    }
    sum / count as f64
}
