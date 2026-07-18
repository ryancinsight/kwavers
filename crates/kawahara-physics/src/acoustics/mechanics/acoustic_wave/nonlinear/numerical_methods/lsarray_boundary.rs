use leto::Array3;

pub(super) fn leto_real_field(field: Array3<f64>) -> Array3<f64> {
    let shape = field.shape();
    Array3::from_shape_vec(shape, field.into_storage().into_inner())
        .expect("Leto real field length must match ndarray field shape")
}
