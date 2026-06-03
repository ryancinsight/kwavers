use super::{exact_gaussian_pressure, physical_coordinate, NamedLine, DT, ELEMENTS, STEPS};
use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers_core::error::KwaversResult;
use ndarray::{Array1, Array3};

const COMMON_SAMPLING_ORDER: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct CommonGaussianMatrix {
    pub dg_exact_l2: f64,
    pub fdtd_exact_l2: f64,
    pub kspace_exact_l2: f64,
    pub pstd_exact_l2: f64,
    pub fdtd_pstd_l2: f64,
    pub kspace_pstd_l2: f64,
    pub dg_fdtd_l2: f64,
    pub dg_pstd_l2: f64,
}

pub struct CommonGaussianSamples {
    pub matrix: CommonGaussianMatrix,
    pub pressure_lines: Vec<NamedLine>,
    pub error_lines: Vec<NamedLine>,
}

pub fn common_gaussian_samples(
    dg_pressure: &Array3<f64>,
    dg_nodes: &Array1<f64>,
    fdtd_line: &Array1<f64>,
    kspace_line: &Array1<f64>,
    pstd_line: &Array1<f64>,
) -> KwaversResult<CommonGaussianSamples> {
    let (sample_nodes, sample_weights) = gauss_lobatto_quadrature(COMMON_SAMPLING_ORDER + 1)?;
    let final_time = STEPS as f64 * DT;
    let mut x = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut weights = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut exact = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut dg = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut fdtd = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut kspace = Vec::with_capacity(ELEMENTS * sample_nodes.len());
    let mut pstd = Vec::with_capacity(ELEMENTS * sample_nodes.len());

    for elem in 0..ELEMENTS {
        for node in 0..sample_nodes.len() {
            let xi = sample_nodes[node];
            let physical_x = physical_coordinate(elem, xi);
            x.push(physical_x);
            weights.push(sample_weights[node]);
            exact.push(exact_gaussian_pressure(physical_x, final_time));
            dg.push(interpolate_dg_element(dg_pressure, elem, dg_nodes, xi));
            fdtd.push(interpolate_periodic_line(fdtd_line, physical_x));
            kspace.push(interpolate_periodic_line(kspace_line, physical_x));
            pstd.push(interpolate_periodic_line(pstd_line, physical_x));
        }
    }

    let matrix = CommonGaussianMatrix {
        dg_exact_l2: weighted_relative_l2(&dg, &exact, &weights),
        fdtd_exact_l2: weighted_relative_l2(&fdtd, &exact, &weights),
        kspace_exact_l2: weighted_relative_l2(&kspace, &exact, &weights),
        pstd_exact_l2: weighted_relative_l2(&pstd, &exact, &weights),
        fdtd_pstd_l2: weighted_relative_l2(&fdtd, &pstd, &weights),
        kspace_pstd_l2: weighted_relative_l2(&kspace, &pstd, &weights),
        dg_fdtd_l2: weighted_relative_l2(&dg, &fdtd, &weights),
        dg_pstd_l2: weighted_relative_l2(&dg, &pstd, &weights),
    };
    let pressure_lines = vec![
        named_line("exact", &x, &exact),
        named_line("DG", &x, &dg),
        named_line("FDTD", &x, &fdtd),
        named_line("FDTD+k-space", &x, &kspace),
        named_line("PSTD", &x, &pstd),
    ];
    let error_lines = vec![
        error_line("DG error", &x, &dg, &exact),
        error_line("FDTD error", &x, &fdtd, &exact),
        error_line("FDTD+k-space error", &x, &kspace, &exact),
        error_line("PSTD error", &x, &pstd, &exact),
    ];
    Ok(CommonGaussianSamples {
        matrix,
        pressure_lines,
        error_lines,
    })
}

pub fn print_common_solver_matrix(matrix: &CommonGaussianMatrix) {
    println!("{:<36} {:>16.6e}", "common DG vs exact", matrix.dg_exact_l2);
    println!(
        "{:<36} {:>16.6e}",
        "common FDTD vs exact", matrix.fdtd_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "common FDTD+k-space vs exact", matrix.kspace_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "common PSTD vs exact", matrix.pstd_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "common FDTD vs PSTD", matrix.fdtd_pstd_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "common FDTD+k-space vs PSTD", matrix.kspace_pstd_l2
    );
    println!("{:<36} {:>16.6e}", "common DG vs FDTD", matrix.dg_fdtd_l2);
    println!("{:<36} {:>16.6e}", "common DG vs PSTD", matrix.dg_pstd_l2);
}

fn interpolate_dg_element(
    pressure: &Array3<f64>,
    elem: usize,
    nodes: &Array1<f64>,
    xi: f64,
) -> f64 {
    for node in 0..nodes.len() {
        if (xi - nodes[node]).abs() <= 1.0e-14 {
            return pressure[(elem, node, 0)];
        }
    }

    let mut value = 0.0;
    for node in 0..nodes.len() {
        let mut basis = 1.0;
        for other in 0..nodes.len() {
            if other != node {
                basis *= (xi - nodes[other]) / (nodes[node] - nodes[other]);
            }
        }
        value += pressure[(elem, node, 0)] * basis;
    }
    value
}

fn interpolate_periodic_line(values: &Array1<f64>, x: f64) -> f64 {
    let n = values.len();
    let position = x.rem_euclid(n as f64);
    let left = position.floor() as usize;
    let right = (left + 1) % n;
    let fraction = position - left as f64;
    (1.0 - fraction) * values[left] + fraction * values[right]
}

fn weighted_relative_l2(actual: &[f64], expected: &[f64], weights: &[f64]) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for ((&actual, &expected), &weight) in actual.iter().zip(expected).zip(weights) {
        let diff = actual - expected;
        diff_sq += weight * diff * diff;
        expected_sq += weight * expected * expected;
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

fn named_line(name: &'static str, x: &[f64], values: &[f64]) -> NamedLine {
    NamedLine {
        name,
        samples: x
            .iter()
            .zip(values)
            .map(|(&x, &value)| (x, value))
            .collect(),
    }
}

fn error_line(name: &'static str, x: &[f64], values: &[f64], exact: &[f64]) -> NamedLine {
    NamedLine {
        name,
        samples: x
            .iter()
            .zip(values)
            .zip(exact)
            .map(|((&x, &value), &exact)| (x, (value - exact).abs()))
            .collect(),
    }
}
