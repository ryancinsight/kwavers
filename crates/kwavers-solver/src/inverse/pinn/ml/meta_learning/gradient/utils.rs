use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Compute L2 norm of gradients (for gradient clipping)
///
/// Returns the Euclidean norm: ||∇||₂ = sqrt(Σᵢ gᵢ²)
pub fn gradient_norm<B: AutodiffBackend>(grads: &[Option<Tensor<B::InnerBackend, 1>>]) -> f64 {
    let mut sum_squares = 0.0;

    for grad in grads.iter().flatten() {
        let grad_data = grad.to_data();
        let values = grad_data.as_slice::<f32>().unwrap_or(&[]);
        sum_squares += values.iter().map(|&g| (g as f64).powi(2)).sum::<f64>();
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
pub fn clip_gradients_by_norm<B: AutodiffBackend>(
    grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    max_norm: f64,
) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
    let norm = gradient_norm::<B>(&grads);

    if norm > max_norm {
        let scale = max_norm / norm;
        grads
            .into_iter()
            .map(|g| g.map(|t| t.mul_scalar(scale)))
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
pub fn clip_gradients_by_value<B: AutodiffBackend>(
    grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    max_value: f64,
) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
    grads
        .into_iter()
        .map(|g| g.map(|t| t.clamp(-max_value, max_value)))
        .collect()
}

/// Add gradients element-wise (for gradient accumulation)
pub fn add_gradients<B: AutodiffBackend>(
    grads1: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    grads2: &[Option<Tensor<B::InnerBackend, 1>>],
) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
    grads1
        .into_iter()
        .zip(grads2.iter())
        .map(|(g1, g2)| match (g1, g2) {
            (Some(t1), Some(t2)) => Some(t1 + t2.clone()),
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2.clone()),
            (None, None) => None,
        })
        .collect()
}

/// Scale gradients by a constant factor
pub fn scale_gradients<B: AutodiffBackend>(
    grads: Vec<Option<Tensor<B::InnerBackend, 1>>>,
    scale: f64,
) -> Vec<Option<Tensor<B::InnerBackend, 1>>> {
    grads
        .into_iter()
        .map(|g| g.map(|t| t.mul_scalar(scale)))
        .collect()
}
