use leto::{Array2, Array3, ArrayView3, SliceArg};

use super::{
    zip_mut, zip_mut_indexed, zip_mut_many, zip_mut_many_indexed, zip_mut_pair,
    zip_mut_pair_indexed, zip_mut_triple, zip_mut_triple_indexed,
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
    let transposed = source
        .transpose([2, 1, 0])
        .expect("a permutation of three axes");
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
    zip_mut(
        out.view_mut(),
        (v[0], v[1], v[2], v[3]),
        |o, (_, _, _, d)| *o = *d,
    );
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
    zip_mut_pair_indexed(
        a.view_mut(),
        b.view_mut(),
        input.view(),
        |index, a, b, x| {
            *a = x - code(index);
            *b = code(index);
        },
    );
    assert_field(&a, |_| 7.0);
    assert_field(&b, code);

    let mut transposed_storage = Array3::zeros(SHAPE);
    let transposed = transposed_storage
        .transpose_mut([2, 1, 0])
        .expect("a permutation of three axes");
    zip_mut_triple_indexed(
        a.view_mut(),
        transposed,
        c.view_mut(),
        (),
        |index, a, t, c, ()| {
            *a = code(index);
            *t = code(index);
            *c = code(index) + 1.0;
        },
    );
    assert_field(&a, code);
    assert_field(&transposed_storage, |[i, j, k]| code([k, j, i]));
    assert_field(&c, |i| code(i) + 1.0);
}

#[test]
fn six_written_fields_each_get_their_own_value() {
    let input = field(0.0);
    let mut fields = [(); 6].map(|()| Array3::zeros(SHAPE));
    {
        let views = fields.each_mut().map(Array3::view_mut);
        zip_mut_many(views, input.view(), |values, x| {
            for (slot, value) in values.into_iter().enumerate() {
                *value = x + slot as f64;
            }
        });
    }
    for (slot, values) in fields.iter().enumerate() {
        assert_field(values, |i| code(i) + slot as f64);
    }
}

#[test]
fn a_transposed_destination_among_many_is_written_logically() {
    let input = field(0.0);
    let mut dense = Array3::zeros(SHAPE);
    let mut other = Array3::zeros(SHAPE);
    let mut transposed_storage = Array3::zeros(SHAPE);
    {
        let transposed = transposed_storage
            .transpose_mut([2, 1, 0])
            .expect("a permutation of three axes");
        zip_mut_many(
            [dense.view_mut(), transposed, other.view_mut()],
            input.view(),
            |[first, second, third], x| {
                *first = *x;
                *second = *x;
                *third = x + 1.0;
            },
        );
    }
    assert_field(&dense, code);
    assert_field(&other, |i| code(i) + 1.0);
    assert_field(&transposed_storage, |[i, j, k]| code([k, j, i]));
}

#[test]
fn the_many_indexed_form_reports_each_position() {
    let input = field(7.0);
    let mut fields = [(); 4].map(|()| Array3::zeros(SHAPE));
    {
        let views = fields.each_mut().map(Array3::view_mut);
        zip_mut_many_indexed(views, input.view(), |index, values, x| {
            let [a, b, c, d] = values;
            *a = x - code(index);
            *b = code(index);
            *c = code(index) + 1.0;
            *d = code(index) + 2.0;
        });
    }
    assert_field(&fields[0], |_| 7.0);
    assert_field(&fields[1], code);
    assert_field(&fields[2], |i| code(i) + 1.0);
    assert_field(&fields[3], |i| code(i) + 2.0);
}

#[test]
#[should_panic(expected = "invariant: every written field has one shape")]
fn a_many_destination_of_a_different_shape_is_rejected() {
    let input = field(0.0);
    let mut first = Array3::<f64>::zeros(SHAPE);
    let mut second = Array3::<f64>::zeros([SIDE, SIDE, SIDE - 1]);
    zip_mut_many(
        [first.view_mut(), second.view_mut()],
        input.view(),
        |[a, b], x| {
            *a = *x;
            *b = *x;
        },
    );
}

/// Offset of the `n`-th weighted input; distinct per position so a swapped,
/// dropped or repeated member changes the weighted sum.
fn offset(n: usize) -> f64 {
    (n as f64 + 1.0) * 1.0e6
}

/// The weighted sum `sum_n 2^n * (code + offset(n))` over the first `arity`
/// inputs. Every term is an integer below 2^40, so the sum is exact.
fn weighted(arity: usize, index: [usize; 3]) -> f64 {
    (0..arity)
        .map(|n| f64::from(1_u32 << n) * (code(index) + offset(n)))
        .sum()
}

/// Every arity, first all dense and then with one member transposed at each
/// position, which sends the traversal down the logical walk.
#[test]
fn every_tuple_member_is_read_at_its_own_position_on_both_paths() {
    let dense: Vec<Array3<f64>> = (0..5).map(|n| field(offset(n))).collect();
    let stored_reversed: Vec<Array3<f64>> = (0..5).map(|n| reversed(offset(n))).collect();
    let transposed: Vec<ArrayView3<'_, f64>> = stored_reversed
        .iter()
        .map(|a| a.transpose([2, 1, 0]).expect("a permutation of three axes"))
        .collect();

    for arity in 2..=5 {
        for odd in std::iter::once(None).chain((0..arity).map(Some)) {
            let v: Vec<ArrayView3<'_, f64>> = (0..5)
                .map(|n| {
                    if odd == Some(n) {
                        transposed[n]
                    } else {
                        dense[n].view()
                    }
                })
                .collect();
            let mut out = Array3::zeros(SHAPE);
            match arity {
                2 => zip_mut(out.view_mut(), (v[0], v[1]), |o, (a, b)| *o = a + 2.0 * b),
                3 => zip_mut(out.view_mut(), (v[0], v[1], v[2]), |o, (a, b, c)| {
                    *o = a + 2.0 * b + 4.0 * c;
                }),
                4 => zip_mut(
                    out.view_mut(),
                    (v[0], v[1], v[2], v[3]),
                    |o, (a, b, c, d)| {
                        *o = a + 2.0 * b + 4.0 * c + 8.0 * d;
                    },
                ),
                _ => zip_mut(
                    out.view_mut(),
                    (v[0], v[1], v[2], v[3], v[4]),
                    |o, (a, b, c, d, e)| *o = a + 2.0 * b + 4.0 * c + 8.0 * d + 16.0 * e,
                ),
            }
            for i in 0..SIDE {
                for j in 0..SIDE {
                    for k in 0..SIDE {
                        assert_eq!(
                            out[[i, j, k]],
                            weighted(arity, [i, j, k]),
                            "arity {arity}, transposed member {odd:?}, at [{i}, {j}, {k}]"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn the_dense_triple_indexed_path_reports_each_position() {
    let input = field(3.0);
    let (mut a, mut b, mut c) = (
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
        Array3::zeros(SHAPE),
    );
    zip_mut_triple_indexed(
        a.view_mut(),
        b.view_mut(),
        c.view_mut(),
        input.view(),
        |index, a, b, c, x| {
            *a = x - code(index);
            *b = code(index);
            *c = f64::from(u32::try_from(index[0]).expect("a small index"));
        },
    );
    assert_field(&a, |_| 3.0);
    assert_field(&b, code);
    assert_field(&c, |[i, _, _]| {
        f64::from(u32::try_from(i).expect("a small index"))
    });
}

/// `s![start..;2, ..]` in leto slice-argument form.
fn every_other_row(start: isize) -> [SliceArg; 2] {
    [
        SliceArg::Range {
            start: Some(start),
            end: None,
            step: 2,
        },
        SliceArg::All,
    ]
}

/// A stepped rank-2 view is neither C- nor F-dense: the logical walk writes
/// exactly the selected rows, each from its own inputs.
#[test]
fn stepped_rank_two_views_write_only_their_rows() {
    let fill = |scale: usize| Array2::from_shape_fn((6, 5), |[i, j]| (scale * (i * 5 + j)) as f64);
    let (first, second) = (fill(1), fill(100));
    let mut out = Array2::<f64>::zeros((6, 5));
    let rows = every_other_row(0);
    zip_mut_indexed(
        out.slice_with_mut::<2>(&rows).expect("rows within bounds"),
        (
            first.slice_with::<2>(&rows).expect("rows within bounds"),
            second.slice_with::<2>(&rows).expect("rows within bounds"),
        ),
        |[i, j], o, (a, b)| *o = a + b + (i * 10 + j) as f64 * 1.0e6,
    );
    for i in 0..6 {
        for j in 0..5 {
            let expected = if i % 2 == 0 {
                (101 * (i * 5 + j)) as f64 + ((i / 2) * 10 + j) as f64 * 1.0e6
            } else {
                0.0
            };
            assert_eq!(out[[i, j]], expected, "at [{i}, {j}]");
        }
    }
    assert!(out
        .slice_with::<2>(&every_other_row(1))
        .expect("rows within bounds")
        .iter()
        .all(|&v| v == 0.0));
}
