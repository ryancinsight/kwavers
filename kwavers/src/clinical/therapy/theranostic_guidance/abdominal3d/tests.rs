use super::super::geometry::Point3;
use super::bowl::bowl_elements;
use super::helpers::distance_3d;
use super::placement::plan_abdominal_array_placement;
use ndarray::Array3;

/// Returns a toy CT volume: a sphere of radius R_body centred at the
/// volume centre, with a smaller sphere of label 1 at offset (R_organ, 0, 0).
fn toy_abdominal_volume(
    n: usize,
    r_body: f64,
    r_organ: f64,
    organ_offset: [f64; 3],
) -> (Array3<f64>, Array3<i16>) {
    let centre = 0.5 * (n as f64 - 1.0);
    let ct = Array3::from_shape_fn((n, n, n), |(ix, iy, iz)| {
        let dx = ix as f64 - centre;
        let dy = iy as f64 - centre;
        let dz = iz as f64 - centre;
        let r = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
        if r < r_body {
            20.0
        } else {
            -1000.0
        }
    });
    let label = Array3::from_shape_fn((n, n, n), |(ix, iy, iz)| {
        let dx = ix as f64 - (centre + organ_offset[0]);
        let dy = iy as f64 - (centre + organ_offset[1]);
        let dz = iz as f64 - (centre + organ_offset[2]);
        let r = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
        if r < r_organ {
            1i16
        } else {
            0i16
        }
    });
    (ct, label)
}

#[test]
fn placement_returns_skin_point_outside_body() {
    let n = 48;
    let r_body = 16.0;
    let r_organ = 4.0;
    let offset = [8.0, 0.0, 0.0];
    let (ct, label) = toy_abdominal_volume(n, r_body, r_organ, offset);
    let spacing_mm = [2.0; 3];
    let result =
        plan_abdominal_array_placement(&ct, &label, spacing_mm, 256, 4, -400.0, "test".to_owned())
            .expect("placement should succeed");

    // skin_contact_m is in body-centroid-relative metres (origin = body centroid).
    // The Euclidean distance from that origin equals the skin radius.
    let skin_r = result
        .skin_contact_m
        .x_m
        .hypot(result.skin_contact_m.y_m)
        .hypot(result.skin_contact_m.z_m);
    let body_r_m = r_body * 2.0e-3;
    // Skin radius should be approximately equal to the body radius (within 1 voxel).
    assert!(
        (skin_r - body_r_m).abs() < 3.0e-3,
        "skin contact r={skin_r:.4} m should be near body radius {body_r_m:.4} m"
    );
}

#[test]
fn bowl_vertex_matches_skin_contact() {
    // At t=0 the first element should be close to the skin contact point.
    let skin = Point3 {
        x_m: 0.05,
        y_m: 0.0,
        z_m: 0.0,
    };
    let focus = Point3 {
        x_m: -0.05,
        y_m: 0.0,
        z_m: 0.0,
    };
    let radius = 0.115;
    let elements = bowl_elements(256, skin, focus, radius).unwrap();
    // The bowl vertex is at t→0, which is close to element[0].
    // With theta_cutout ≈ 10°, element[0] is near the vertex.
    let d_first_to_skin = distance_3d(elements[0], skin);
    assert!(
        d_first_to_skin < radius * 0.25,
        "first bowl element (t≈0) should be near the skin contact vertex, got {d_first_to_skin:.4} m"
    );
}

#[test]
fn all_elements_on_sphere_of_correct_radius() {
    let skin = Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: -0.08,
    };
    let focus = Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 0.05,
    };
    let dist = distance_3d(skin, focus);
    let radius = dist * 1.15;
    let elements = bowl_elements(512, skin, focus, radius).unwrap();
    for (i, el) in elements.iter().enumerate() {
        let r_from_focus = distance_3d(*el, focus);
        let err = (r_from_focus - radius).abs();
        assert!(
            err < 1.0e-10,
            "element {i}: distance from focus = {r_from_focus:.8} m, expected {radius:.8} m, err={err:.2e}"
        );
    }
}

#[test]
fn degenerate_bowl_axis_is_rejected() {
    let point = Point3 {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 0.0,
    };
    let result = bowl_elements(32, point, point, 0.1);
    assert!(result.is_err(), "degenerate focus axis must be rejected");
}

#[test]
fn surface_stride_reduces_point_count() {
    let n = 24;
    let (ct, label) = toy_abdominal_volume(n, 8.0, 2.0, [4.0, 0.0, 0.0]);
    let spacing_mm = [2.0; 3];
    let r1 = plan_abdominal_array_placement(&ct, &label, spacing_mm, 64, 1, -400.0, "t".to_owned())
        .expect("stride 1");
    let r2 = plan_abdominal_array_placement(&ct, &label, spacing_mm, 64, 4, -400.0, "t".to_owned())
        .expect("stride 4");
    assert!(
        r2.body_surface_points_m.len() < r1.body_surface_points_m.len(),
        "larger stride should produce fewer surface points"
    );
}

#[test]
fn empty_organ_mask_is_rejected() {
    let n = 16;
    let ct = Array3::from_elem((n, n, n), 20.0_f64);
    let label = Array3::zeros((n, n, n));
    let result =
        plan_abdominal_array_placement(&ct, &label, [2.0; 3], 64, 2, -400.0, "t".to_owned());
    assert!(result.is_err(), "empty organ mask should return an error");
}

#[test]
fn empty_body_mask_is_rejected() {
    let n = 16;
    let ct = Array3::from_elem((n, n, n), -1024.0_f64);
    let label = Array3::from_elem((n, n, n), 1i16);
    let result =
        plan_abdominal_array_placement(&ct, &label, [2.0; 3], 64, 2, -400.0, "t".to_owned());
    assert!(result.is_err(), "empty body mask should return an error");
}

#[test]
fn focus_is_inside_body_mask() {
    let n = 48;
    let (ct, label) = toy_abdominal_volume(n, 16.0, 4.0, [6.0, 0.0, 0.0]);
    let result =
        plan_abdominal_array_placement(&ct, &label, [2.0; 3], 128, 3, -400.0, "t".to_owned())
            .expect("placement");
    // Focus should be approximately at the organ centroid.
    // The organ offset is 6 voxels * 2mm = 12mm from body centre.
    let focus_x = result.focus_m.x_m;
    assert!(
        focus_x.abs() < 0.020,
        "focus x should be near organ centroid, got {focus_x:.4} m"
    );
}
