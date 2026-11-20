//! Value and storage-identity contracts across the owned array boundary.

use super::*;
use numpy::ndarray::{array, s, Array3, ShapeBuilder};
use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

#[test]
fn contiguous_owned_storage_moves_in_both_directions() {
    let _trace = trace();
    let input = tracing::info_span!("arrange").in_scope(|| array![[1.0, -2.0], [4.0, 8.0]]);
    let address = input.as_ptr();
    let (same_allocation, output) = tracing::info_span!("act").in_scope(|| {
        let native = into_leto(input);
        let same = native.as_slice().expect("dense native output").as_ptr() == address;
        (same, into_ndarray(native))
    });
    let _assert = tracing::info_span!("assert").entered();
    assert!(same_allocation);
    assert_eq!(output.as_ptr(), address);
    assert_eq!(output, array![[1.0, -2.0], [4.0, 8.0]]);
}

#[test]
fn strided_and_sliced_owners_preserve_logical_element_order() {
    let _trace = trace();
    let (reversed, sliced, fortran, empty) = tracing::info_span!("arrange").in_scope(|| {
        (
            array![2, 4, 8, 16, 32].slice_move(s![..;-2]),
            array![2, 4, 8, 16, 32].slice_move(s![1..4]),
            Array::from_shape_vec((2, 3).f(), vec![1, 2, 3, 4, 5, 6])
                .expect("valid Fortran layout"),
            Array3::<i32>::zeros((0, 2, 3)),
        )
    });
    let (a, b, c, d) = tracing::info_span!("act").in_scope(|| {
        (
            into_ndarray(into_leto(reversed)),
            into_ndarray(into_leto(sliced)),
            into_ndarray(into_leto(fortran)),
            into_ndarray(into_leto(empty)),
        )
    });
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(a, array![32, 8, 2]);
    assert_eq!(b, array![4, 8, 16]);
    assert_eq!(c, array![[1, 3, 5], [2, 4, 6]]);
    assert_eq!(d.shape(), &[0, 2, 3]);
    assert!(d.is_empty());
}
