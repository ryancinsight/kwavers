pub fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let mag_sq = v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1]));
    if !mag_sq.is_finite() || mag_sq <= f64::EPSILON {
        return [0.0, 0.0, 0.0];
    }
    let mag = mag_sq.sqrt();
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

pub fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

pub fn orthogonal_basis_from_normal3(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = normalize3(normal);

    let v = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    let u = normalize3(cross3(v, n));
    let v = cross3(n, u);

    (u, v)
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
    ]
}
