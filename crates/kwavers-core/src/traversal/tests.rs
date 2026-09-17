use leto::Array3;

use super::{
    zip_mut, zip_mut_indexed, zip_mut_pair, zip_mut_pair_indexed, zip_mut_triple,
    zip_mut_triple_indexed,
};

/// Past moirai's parallel floor, so the dense path splits into tasks. A cube
/// makes a transposed view shape-compatible with an untransposed one.
const SIDE: usize = 24;
const SHAPE: [usize; 3] = [SIDE; 3];

/// A field whose value encodes its logical index, offset so fields differ.
fn field(offset: f64) -> Array3<f64> {
    let mut values = Array3::zeros(SHAPE);
    for i in 0..SIDE {
        for j in 0..SIDE {
            for k in 0..SIDE {
                values[[i, j, k]] = offset + code([i, j, k]);
            }
        }
    }
    values
}

/// The row-major position of `[i, j, k]`; below 24^3, so exact in `f64`
/// under every offset and product the tests apply.
fn code([i, j, k]: [usize; 3]) -> f64 {
    ((i * SIDE + j) * SIDE + k) as f64
}

/// A C-contiguous field holding, at logical `[i, j, k]`, what `field(offset)`
/// holds at `[k, j, i]`; its transposed view reads as `field(offset)`
/// logically but is F-ordered in memory.
fn reversed(offset: f64) -> Array3<f64> {
    let mut values = Array3::zeros(SHAPE);
    for i in 0..SIDE {
        for j in 0..SIDE {
            for k in 0..SIDE {
                values[[i, j, k]] = offset + code([k, j, i]);
            }
        }
    }
    values
}

fn assert_field(actual: &Array3<f64>, expected: impl Fn([usize; 3]) -> f64) {
    for i in 0..SIDE {
        for j in 0..SIDE {
            for k in 0..SIDE {
                assert_eq!(actual[[i, j, k]], expected([i, j, k]), "at [{i}, {j}, {k}]");
            }
        }
    }
}

#[test]
fn a_transposed_input_pairs_by_logical_index() {
    let source = reversed(0.0);
    let transposed = source.transpose([2, 1, 0]).expect("a permutation of three axes");
    assert!(
        transposed.as_slice().is_none() && transposed.as_slice_memory_order().is_some(),
        "the input must be dense in F order for this case to mean anything"
    );
    let mut out = Array3::zeros(SHAPE);
    zip_mut(out.view_mut(), transposed, |value, input| *value = *input);
    assert_field(&out, code);
}

#[test]
fn a_transposed_output_receives_logical_indices() {
    let mut storage = Array3::zeros(SHAPE);
    let out = storage
        .transpose_mut([2, 1, 0])
        .expect("a permutation of three axes");
    zip_mut_indexed(out, (), |index, value, ()| *value = code(index));
    // Logical `[i, j, k]` of the transposed view is `[k, j, i]` of storage.
    assert_field(&storage, |[i, j, k]| code([k, j, i]));
}

#[test]
fn the_dense_indexed_path_reports_each_position() {
    let offset = field(5.0);
    let mut out = Array3::zeros(SHAPE);
    zip_mut_indexed(out.view_mut(), offset.view(), |index, value, input| {
        *value = input - code(index);
    });
    assert_field(&out, |_| 5.0);
}

#[test]
fn every_input_arity_reads_its_own_field() {
    let fields: Vec<Array3<f64>> = (1..=5).map(|n| field(f64::from(n) * 1.0e6)).collect();
    let v: Vec<_> = fields.iter().map(Array3::view).collect();
    let mut out = Array3::zeros(SHAPE);

    zip_mut(out.view_mut(), v[0], |o, a| *o = *a);
    assert_field(&out, |i| 1.0e6 + code(i));
    zip_mut(out.view_mut(), (v[0], v[1]), |o, (a, b)| *o = b - a);
    assert_field(&out, |_| 1.0e6);
    zip_mut(out.view_mut(), (v[0], v[1], v[2]), |o, (_, _, c)| *o = *c);
    assert_field(&out, |i| 3.0e6 + code(i));
    zip_mut(out.view_mut(), (v[0], v[1], v[2], v[3]), |o, (_, _, _, d)| *o = *d);
    assert_field(&out, |i| 4.0e6 + code(i));
    zip_mut(
        out.view_mut(),
        (v[0], v[1], v[2], v[3], v[4]),
        |o, (a, _, _, _, e)| *o = e - a,
    );
    assert_field(&out, |_| 4.0e6);
    zip_mut(out.view_mut(), (), |o, ()| *o = -*o);
    assert_field(&out, |_| -4.0e6);
}

#[test]
fn every_written_field_gets_its_own_value() {
    let input = field(0.0);
    let (mut a, mut b, mut c) = (
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
    );
    zip_mut_pair(a.view_mut(), b.view_mut(), input.view(), |a, b, x| {
        *a = x + 1.0;
        *b = x + 2.0;
    });
    assert_field(&a, |i| code(i) + 1.0);
    assert_field(&b, |i| code(i) + 2.0);

    zip_mut_triple(
        a.view_mut(),
        b.view_mut(),
        c.view_mut(),
        input.view(),
        |a, b, c, x| {
            *a = x * 2.0;
            *b = x * 3.0;
            *c = x * 4.0;
        },
    );
    assert_field(&a, |i| code(i) * 2.0);
    assert_field(&b, |i| code(i) * 3.0);
    assert_field(&c, |i| code(i) * 4.0);
}

#[test]
fn a_transposed_second_output_is_written_logically() {
    let input = field(0.0);
    let mut first = Array3::zeros(SHAPE);
    let mut second_storage = Array3::zeros(SHAPE);
    let second = second_storage
        .transpose_mut([2, 1, 0])
        .expect("a permutation of three axes");
    zip_mut_pair(first.view_mut(), second, input.view(), |a, b, x| {
        *a = *x;
        *b = *x;
    });
    assert_field(&first, code);
    assert_field(&second_storage, |[i, j, k]| code([k, j, i]));
}

#[test]
#[should_panic(expected = "invariant: every traversed view has the output shape")]
fn a_mismatched_input_is_rejected() {
    let input = Array3::<f64>::zeros([SIDE, SIDE, SIDE - 1]);
    let mut out = Array3::<f64>::zeros(SHAPE);
    zip_mut(out.view_mut(), input.view(), |o, x| *o = *x);
}

#[test]
fn an_empty_field_calls_nothing() {
    let input = Array3::<f64>::zeros([0, SIDE, SIDE]);
    let mut out = Array3::<f64>::zeros([0, SIDE, SIDE]);
    zip_mut_indexed(out.view_mut(), input.view(), |_, _, _| {
        unreachable!("an empty field has no element to visit")
    });
}

#[test]
fn multi_output_indexed_forms_report_each_position() {
    let input = field(7.0);
    let (mut a, mut b, mut c) = (
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
    );
    zip_mut_pair_indexed(a.view_mut(), b.view_mut(), input.view(), |index, a, b, x| {
        *a = x - code(index);
        *b = code(index);
    });
    assert_field(&a, |_| 7.0);
    assert_field(&b, code);

    let mut transposed_storage = Array3::zeros(SHAPE);
    let transposed = transposed_storage
        .transpose_mut([2, 1, 0])
        .expect("a permutation of three axes");
    zip_mut_triple_indexed(a.view_mut(), transposed, c.view_mut(), (), |index, a, t, c, ()| {
        *a = code(index);
        *t = code(index);
        *c = code(index) + 1.0;
    });
    assert_field(&a, code);
    assert_field(&transposed_storage, |[i, j, k]| code([k, j, i]));
    assert_field(&c, |i| code(i) + 1.0);
}
