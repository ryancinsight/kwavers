use kwavers_field::UnifiedFieldType;
use leto::{
    /* s -- no leto equivalent */,
    Array3,
};
use std::collections::HashMap;

/// Return field keys in deterministic stacked-state order.
pub(in crate::multiphysics::monolithic) fn sorted_field_keys(
    fields: &HashMap<UnifiedFieldType, Array3<f64>>,
) -> Vec<UnifiedFieldType> {
    let mut keys: Vec<UnifiedFieldType> = fields.keys().copied().collect();
    keys.sort_by_key(|k| k.index());
    keys
}

/// Flatten field map to a single `Array3<f64>` by stacking along axis 0.
///
/// Fields are stacked in `field_order`. Each field of shape `(nx, ny, nz)`
/// becomes rows `[i*nx .. (i+1)*nx]` in the output of shape
/// `(n_fields*nx, ny, nz)`.
pub(in crate::multiphysics::monolithic) fn flatten_fields(
    fields: &HashMap<UnifiedFieldType, Array3<f64>>,
    field_order: &[UnifiedFieldType],
) -> Array3<f64> {
    if field_order.is_empty() {
        return Array3::zeros((1, 1, 1));
    }

    let first = &fields[&field_order[0]];
    let (nx, ny, nz) = first.dim();
    let n_fields = field_order.len();

    let mut stacked = Array3::zeros((n_fields * nx, ny, nz));
    for (i, ft) in field_order.iter().enumerate() {
        let src = &fields[ft];
        stacked
            .slice_mut(s![i * nx..(i + 1) * nx, .., ..])
            .assign(src);
    }
    stacked
}

/// Unflatten solution vector back to field map.
///
/// This is the inverse of [`flatten_fields`]: it splits the stacked array along
/// axis 0 and writes each block back into the corresponding field.
pub(in crate::multiphysics::monolithic) fn unflatten_fields(
    u: &Array3<f64>,
    fields: &mut HashMap<UnifiedFieldType, Array3<f64>>,
    field_order: &[UnifiedFieldType],
) {
    if field_order.is_empty() {
        return;
    }

    let total_rows = u.dim().0;
    let nx = total_rows / field_order.len();

    for (i, ft) in field_order.iter().enumerate() {
        if let Some(field) = fields.get_mut(ft) {
            field.assign(&u.slice(s![i * nx..(i + 1) * nx, .., ..]));
        }
    }
}
