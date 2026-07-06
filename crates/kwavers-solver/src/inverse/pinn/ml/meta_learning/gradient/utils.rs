//! Plain-`Vec<f32>` gradient utilities operating on flat per-parameter
//! gradient snapshots (`Vec<Option<Vec<f32>>>`, one entry per model parameter
//! in `parameters()` order).

/// Compute L2 norm of gradients (for gradient clipping)
///
/// Returns the Euclidean norm: ||∇||₂ = sqrt(Σᵢ gᵢ²)
pub fn gradient_norm(grads: &[Option<Vec<f32>>]) -> f64 {
    let mut sum_squares = 0.0;

    for grad in grads.iter().flatten() {
        sum_squares += grad.iter().map(|&g| (g as f64).powi(2)).sum::<f64>();
    }

    sum_squares.sqrt()
}

/// Clip gradients by global norm
///
/// Scales all gradients proportionally if norm exceeds threshold:
/// ```text
/// if ||∇||₂ > max_norm:
///     ∇ = ∇ * (max_norm / ||∇||₂)
/// ```
///
/// Literature: Pascanu et al. (2013) "On the difficulty of training
/// recurrent neural networks"
pub fn clip_gradients_by_norm(
    grads: Vec<Option<Vec<f32>>>,
    max_norm: f64,
) -> Vec<Option<Vec<f32>>> {
    let norm = gradient_norm(&grads);

    if norm > max_norm {
        let scale = (max_norm / norm) as f32;
        grads
            .into_iter()
            .map(|g| g.map(|v| v.into_iter().map(|x| x * scale).collect()))
            .collect()
    } else {
        grads
    }
}

/// Clip gradients by value (element-wise)
///
/// Clamps each gradient element to [-max_value, max_value]:
/// ```text
/// gᵢ = clamp(gᵢ, -max_value, max_value)
/// ```
pub fn clip_gradients_by_value(grads: Vec<Option<Vec<f32>>>, max_value: f64) -> Vec<Option<Vec<f32>>> {
    let max_value = max_value as f32;
    grads
        .into_iter()
        .map(|g| g.map(|v| v.into_iter().map(|x| x.clamp(-max_value, max_value)).collect()))
        .collect()
}

/// Add gradients element-wise (for gradient accumulation)
pub fn add_gradients(
    grads1: Vec<Option<Vec<f32>>>,
    grads2: &[Option<Vec<f32>>],
) -> Vec<Option<Vec<f32>>> {
    grads1
        .into_iter()
        .zip(grads2.iter())
        .map(|(g1, g2)| match (g1, g2) {
            (Some(t1), Some(t2)) => Some(t1.iter().zip(t2.iter()).map(|(a, b)| a + b).collect()),
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2.clone()),
            (None, None) => None,
        })
        .collect()
}

/// Scale gradients by a constant factor
pub fn scale_gradients(grads: Vec<Option<Vec<f32>>>, scale: f64) -> Vec<Option<Vec<f32>>> {
    let scale = scale as f32;
    grads
        .into_iter()
        .map(|g| g.map(|v| v.into_iter().map(|x| x * scale).collect()))
        .collect()
}
