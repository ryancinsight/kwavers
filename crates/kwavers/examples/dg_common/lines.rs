use super::{physical_coordinate, ELEMENTS};
use leto::{Array1, Array3};

pub trait LineField3 {
    fn shape3(&self) -> [usize; 3];
    fn value_at(&self, i: usize, j: usize, k: usize) -> f64;
}

impl LineField3 for leto::Array3<f64> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn value_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

#[derive(Debug, Clone)]
pub struct NamedLine {
    pub name: &'static str,
    pub samples: Vec<(f64, f64)>,
}

pub fn center_line<F: LineField3>(field: &F) -> Array1<f64> {
    let shape = field.shape3();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    Array1::from_shape_fn(nx, |[i]| field.value_at(i, ny / 2, nz / 2))
}

pub fn dg_line(field: &Array3<f64>, xi_nodes: &Array1<f64>) -> Vec<(f64, f64)> {
    let mut samples = Vec::with_capacity(ELEMENTS * xi_nodes.len());
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            samples.push((
                physical_coordinate(elem, xi_nodes[node]),
                field[[elem, node, 0]],
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
