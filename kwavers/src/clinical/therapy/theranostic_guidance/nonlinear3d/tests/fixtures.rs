//! Synthetic CT fixtures shared by the nonlinear 3-D pipeline tests.

use ndarray::Array3;

/// Synthetic brain CT: ellipsoidal skull shell (cortical bone HU values)
/// wrapping an ellipsoidal brain interior (soft tissue HU), surrounded by
/// air. Used by `nonlinear_3d_brain_helmet_pipeline_is_input_sensitive_
/// through_skull` to exercise the cavitation path through skull voxels
/// where the heterogeneous-power-law absorption (`y = 2` for skull,
/// `y ≈ 1.05` for brain) has its largest physical effect.
pub(super) fn brain_fixture() -> Array3<f64> {
    let n = 28;
    let mut ct = Array3::<f64>::from_elem((n, n, n), -1000.0); // air outside head
    let center = [14.0, 14.0, 14.0];
    // Outer head ellipsoid (skull + scalp): everything inside is body.
    let head_radii = [12.5, 11.5, 10.5];
    // Inner brain ellipsoid (soft tissue, lower HU): inside the skull shell.
    let brain_radii = [11.0, 10.0, 9.0];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let head_r = ellipsoid_radius([x, y, z], center, head_radii);
                if head_r <= 1.0 {
                    // Default: skull HU value inside head but outside brain.
                    ct[[x, y, z]] = 600.0; // cortical bone HU (lower-bound, well above 300 threshold)
                }
                let brain_r = ellipsoid_radius([x, y, z], center, brain_radii);
                if brain_r <= 1.0 {
                    ct[[x, y, z]] = 40.0; // brain tissue HU (soft tissue)
                }
            }
        }
    }
    ct
}

/// Synthetic abdominal CT + segmentation labels with a soft-tissue body
/// ellipsoid containing a centered organ ellipsoid and a peripheral target
/// region (label = 2). Used by `nonlinear_3d_westervelt_fwi_and_cavitation_
/// inverse_are_input_sensitive`.
pub(super) fn abdominal_fixture() -> (Array3<f64>, Array3<i16>) {
    let n = 24;
    let mut ct = Array3::<f64>::from_elem((n, n, n), -1000.0);
    let mut labels = Array3::<i16>::zeros((n, n, n));
    let center = [12.0, 12.0, 12.0];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let body_r = ellipsoid_radius([x, y, z], center, [9.0, 8.0, 7.0]);
                if body_r <= 1.0 {
                    ct[[x, y, z]] = 35.0;
                }
                let organ_r = ellipsoid_radius([x, y, z], center, [5.0, 4.0, 4.0]);
                if organ_r <= 1.0 {
                    labels[[x, y, z]] = 1;
                    ct[[x, y, z]] = 55.0;
                }
                let target_r = ellipsoid_radius([x, y, z], [12.0, 12.0, 11.0], [2.0, 2.0, 2.0]);
                if target_r <= 1.0 {
                    labels[[x, y, z]] = 2;
                    ct[[x, y, z]] = 75.0;
                }
            }
        }
    }
    (ct, labels)
}

/// Normalized ellipsoid radius `(Δx/rx)² + (Δy/ry)² + (Δz/rz)²` — inside
/// the ellipsoid when ≤ 1.
pub(super) fn ellipsoid_radius(idx: [usize; 3], center: [f64; 3], radius: [f64; 3]) -> f64 {
    ((idx[0] as f64 - center[0]) / radius[0]).powi(2)
        + ((idx[1] as f64 - center[1]) / radius[1]).powi(2)
        + ((idx[2] as f64 - center[2]) / radius[2]).powi(2)
}
