//! Spatial regularization and multiparameter scoring.
//!
//! The regularizer is the discrete `H1` seminorm of the perturbation from the
//! CT-derived background model, restricted to the body mask. It penalizes
//! isolated checkerboard updates while preserving coherent target-scale
//! contrasts.

use leto::Array3;

use super::stencil::index;
use super::types::Nonlinear3dConfig;

pub(super) fn h1_penalty(
    model: &[f64],
    background: &[f64],
    body: &[bool],
    n: usize,
    weight: f64,
    scale: f64,
) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut edges = 0usize;
    for x in 1..n - 1 {
        for y in 1..n - 1 {
            for z in 1..n - 1 {
                let i = index(x, y, z, n);
                if !body[i] {
                    continue;
                }
                for j in [
                    index(x + 1, y, z, n),
                    index(x, y + 1, z, n),
                    index(x, y, z + 1, n),
                ] {
                    if body[j] {
                        let di = (model[i] - background[i]) / scale;
                        let dj = (model[j] - background[j]) / scale;
                        sum += (di - dj).powi(2);
                        edges += 1;
                    }
                }
            }
        }
    }
    0.5 * weight * sum / edges.max(1) as f64
}

pub(super) fn add_h1_gradient(
    gradient: &mut [f64],
    model: &[f64],
    background: &[f64],
    body: &[bool],
    n: usize,
    weight: f64,
    scale: f64,
) {
    if weight == 0.0 {
        return;
    }
    let normalization = active_edge_count(body, n).max(1) as f64;
    let factor = weight / (scale * scale * normalization);
    for x in 1..n - 1 {
        for y in 1..n - 1 {
            for z in 1..n - 1 {
                let i = index(x, y, z, n);
                if !body[i] {
                    continue;
                }
                for j in [
                    index(x + 1, y, z, n),
                    index(x, y + 1, z, n),
                    index(x, y, z + 1, n),
                ] {
                    if body[j] {
                        let diff = (model[i] - background[i]) - (model[j] - background[j]);
                        gradient[i] += factor * diff;
                        gradient[j] -= factor * diff;
                    }
                }
            }
        }
    }
}

pub(super) fn smooth_gradient(gradient: &mut [f64], body: &[bool], n: usize, steps: usize) {
    let mut scratch = gradient.to_vec();
    for _ in 0..steps {
        scratch.copy_from_slice(gradient);
        for x in 1..n - 1 {
            for y in 1..n - 1 {
                for z in 1..n - 1 {
                    let i = index(x, y, z, n);
                    if !body[i] {
                        continue;
                    }
                    let mut sum = scratch[i];
                    let mut count = 1.0;
                    for j in neighbors(i, n) {
                        if body[j] {
                            sum += scratch[j];
                            count += 1.0;
                        }
                    }
                    gradient[i] = sum / count;
                }
            }
        }
    }
}

pub(super) fn multiparameter_score(
    delta_speed: &[f64],
    delta_beta: &[f64],
    body: &[bool],
    config: &Nonlinear3dConfig,
    n: usize,
) -> Array3<f64> {
    let speed_scale = config.lesion_delta_c_m_s.abs().max(1.0);
    let beta_scale = config.lesion_delta_beta.abs().max(1.0e-6);
    Array3::from_shape_fn((n, n, n), |[x, y, z]| {
        let i = index(x, y, z, n);
        if !body[i] {
            return 0.0;
        }
        let speed_score = if config.lesion_delta_c_m_s < 0.0 {
            (-delta_speed[i] / speed_scale).max(0.0)
        } else {
            (delta_speed[i] / speed_scale).max(0.0)
        };
        let beta_score = if config.lesion_delta_beta < 0.0 {
            (-delta_beta[i] / beta_scale).max(0.0)
        } else {
            (delta_beta[i] / beta_scale).max(0.0)
        };
        (0.65 * speed_score + 0.35 * beta_score).clamp(0.0, 1.0)
    })
}

fn active_edge_count(body: &[bool], n: usize) -> usize {
    let mut edges = 0usize;
    for x in 1..n - 1 {
        for y in 1..n - 1 {
            for z in 1..n - 1 {
                let i = index(x, y, z, n);
                if body[i] {
                    edges += neighbors(i, n).into_iter().filter(|j| body[*j]).count();
                }
            }
        }
    }
    edges / 2
}

fn neighbors(i: usize, n: usize) -> [usize; 6] {
    let n2 = n * n;
    [i - n2, i + n2, i - n, i + n, i - 1, i + 1]
}
