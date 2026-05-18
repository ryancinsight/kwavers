use super::{physical_coordinate, ELEMENTS};
use ndarray::{Array1, Array3};

#[derive(Debug, Clone)]
pub struct NamedLine {
    pub name: &'static str,
    pub samples: Vec<(f64, f64)>,
}

pub fn center_line(field: &Array3<f64>) -> Array1<f64> {
    let (_, ny, nz) = field.dim();
    Array1::from_shape_fn(field.dim().0, |i| field[(i, ny / 2, nz / 2)])
}

pub fn dg_line(field: &Array3<f64>, xi_nodes: &Array1<f64>) -> Vec<(f64, f64)> {
    let mut samples = Vec::with_capacity(ELEMENTS * xi_nodes.len());
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            samples.push((
                physical_coordinate(elem, xi_nodes[node]),
                field[(elem, node, 0)],
            ));
        }
    }
    samples
}

pub fn uniform_line(values: &Array1<f64>) -> Vec<(f64, f64)> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect()
}

pub fn absolute_error(actual: &[(f64, f64)], expected: &[(f64, f64)]) -> Vec<(f64, f64)> {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(&(x, actual), &(_, expected))| (x, (actual - expected).abs()))
        .collect()
}
