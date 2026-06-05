use super::*;

#[test]
fn test_adaptive_sampler_creation() {
    let sampler = AdaptiveSampler::new(SamplingStrategy::Uniform, 100, 32);
    assert_eq!(sampler.n_points, 100);
    assert_eq!(sampler.batch_size, 32);
}

#[test]
fn test_uniform_sampling() {
    let mut sampler = AdaptiveSampler::new(SamplingStrategy::Uniform, 50, 0);
    let residuals = vec![1.0; 100];
    let indices = sampler.resample(&residuals).unwrap();
    assert_eq!(indices.len(), 50);
    assert!(indices.iter().all(|&i| i < 100));
}

#[test]
fn test_residual_weighted_sampling() {
    let mut sampler = AdaptiveSampler::with_seed(
        SamplingStrategy::ResidualWeighted {
            alpha: 1.0,
            keep_ratio: 0.0,
        },
        50,
        0,
        42,
    );
    let mut residuals = vec![0.01; 100];
    for r in residuals.iter_mut().take(10) {
        *r = 100.0;
    }
    let indices = sampler.resample(&residuals).unwrap();
    assert_eq!(indices.len(), 50);
    assert!(indices.iter().all(|&i| i < 100));
    let high_residual_count = indices.iter().filter(|&&i| i < 10).count();
    assert!(high_residual_count > 0);
}

#[test]
fn test_batch_iterator() {
    let mut sampler = AdaptiveSampler::new(SamplingStrategy::Uniform, 100, 32);
    let batches: Vec<Vec<usize>> = sampler.iter_batches().collect();
    assert_eq!(batches.len(), 4);
    assert_eq!(batches[0].len(), 32);
    assert_eq!(batches[1].len(), 32);
    assert_eq!(batches[2].len(), 32);
    assert_eq!(batches[3].len(), 4);
    let all_indices: std::collections::HashSet<usize> =
        batches.iter().flat_map(|b| b.iter().copied()).collect();
    assert_eq!(all_indices.len(), 100);
}

#[test]
fn test_importance_threshold() {
    let mut sampler = AdaptiveSampler::new(
        SamplingStrategy::ImportanceThreshold {
            threshold: 1.0,
            top_k_ratio: 0.5,
        },
        20,
        0,
    );
    let mut residuals = vec![0.1; 100];
    for r in residuals.iter_mut().take(40) {
        *r = 2.0;
    }
    let indices = sampler.resample(&residuals).unwrap();
    assert_eq!(indices.len(), 20);
    assert!(indices.iter().all(|&i| i < 40));
}

#[test]
fn test_hybrid_sampling() {
    let mut sampler = AdaptiveSampler::new(
        SamplingStrategy::Hybrid {
            uniform_ratio: 0.5,
            alpha: 1.0,
        },
        100,
        0,
    );
    let mut residuals = vec![0.1; 200];
    for r in residuals.iter_mut().take(20) {
        *r = 10.0;
    }
    let indices = sampler.resample(&residuals).unwrap();
    assert_eq!(indices.len(), 100);
    let high_residual_count = indices.iter().filter(|&&i| i < 20).count();
    assert!(high_residual_count > 10);
    assert!(indices.iter().any(|&i| i >= 100));
}

#[test]
fn test_n_batches() {
    let sampler1 = AdaptiveSampler::new(SamplingStrategy::Uniform, 100, 32);
    assert_eq!(sampler1.n_batches(), 4);

    let sampler2 = AdaptiveSampler::new(SamplingStrategy::Uniform, 100, 0);
    assert_eq!(sampler2.n_batches(), 1);

    let sampler3 = AdaptiveSampler::new(SamplingStrategy::Uniform, 100, 25);
    assert_eq!(sampler3.n_batches(), 4);
}
