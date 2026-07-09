use leto::Array3;

pub(super) fn leto_real_field(field: &Array3<f64>) -> leto::Array3<f64> {
    let [nx, ny, nz] = field.shape();
    leto::Array3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("ndarray real field length must match Leto field shape")
}

pub(super) fn ndarray_real_field(field: leto::Array3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    Array3::from_shape_vec([nx, ny, nz], field.into_storage().into_inner())
        .expect("Leto real field length must match ndarray field shape")
}
