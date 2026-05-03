use super::{
    angle_between3, barycentric3, cross3, distance3, dot3, fma3, lerp3, norm3, normalize3,
    triangle_area3,
};

#[test]
fn test_dot3() {
    assert_eq!(dot3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0);
    assert_eq!(dot3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0);
    assert_eq!(dot3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0);
}

#[test]
fn test_cross3() {
    let result = cross3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((result[0] - 0.0).abs() < 1e-15);
    assert!((result[1] - 0.0).abs() < 1e-15);
    assert!((result[2] - 1.0).abs() < 1e-15);

    // Anticommutativity
    let cross_ab = cross3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let cross_ba = cross3([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]);
    assert!((cross_ab[0] + cross_ba[0]).abs() < 1e-15);
    assert!((cross_ab[1] + cross_ba[1]).abs() < 1e-15);
    assert!((cross_ab[2] + cross_ba[2]).abs() < 1e-15);
}

#[test]
fn test_cross_orthogonality() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let c = cross3(a, b);
    assert!(dot3(c, a).abs() < 1e-14);
    assert!(dot3(c, b).abs() < 1e-14);
}

#[test]
fn test_normalize3() {
    let v = [3.0, 4.0, 0.0];
    let n = normalize3(v).unwrap();
    assert!((norm3(n) - 1.0).abs() < 1e-15);

    assert!(normalize3([0.0, 0.0, 0.0]).is_none());
}

#[test]
fn test_distance3() {
    let a = [0.0, 0.0, 0.0];
    let b = [3.0, 4.0, 0.0];
    assert!((distance3(a, b) - 5.0).abs() < 1e-15);
}

#[test]
fn test_triangle_area3() {
    // Right triangle with legs 3, 4
    let area = triangle_area3([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
    assert!((area - 6.0).abs() < 1e-15);
}

#[test]
fn test_barycentric3() {
    let p1 = [0.0, 0.0, 0.0];
    let p2 = [1.0, 0.0, 0.0];
    let p3 = [0.0, 1.0, 0.0];

    // Centroid
    let (u, v, w) = barycentric3([1.0 / 3.0, 1.0 / 3.0, 0.0], p1, p2, p3);
    assert!((u - 1.0 / 3.0).abs() < 1e-14);
    assert!((v - 1.0 / 3.0).abs() < 1e-14);
    assert!((w - 1.0 / 3.0).abs() < 1e-14);

    // Vertex p1
    let (u, v, w) = barycentric3(p1, p1, p2, p3);
    assert!((u - 1.0).abs() < 1e-14);
    assert!(v.abs() < 1e-14);
    assert!(w.abs() < 1e-14);
}

#[test]
fn test_lerp3() {
    let a = [0.0, 0.0, 0.0];
    let b = [2.0, 4.0, 6.0];
    let mid = lerp3(a, b, 0.5);
    assert!((mid[0] - 1.0).abs() < 1e-15);
    assert!((mid[1] - 2.0).abs() < 1e-15);
    assert!((mid[2] - 3.0).abs() < 1e-15);
}

#[test]
fn test_angle_between3() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let angle = angle_between3(a, b).unwrap();
    assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
}

#[test]
fn test_fma3() {
    let result = fma3([1.0, 2.0, 3.0], 2.0, [1.0, 1.0, 1.0]);
    assert!((result[0] - 3.0).abs() < 1e-15);
    assert!((result[1] - 5.0).abs() < 1e-15);
    assert!((result[2] - 7.0).abs() < 1e-15);
}
