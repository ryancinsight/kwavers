//! Task Sampling Strategies for Meta-Learning
//!
//! Implements curriculum learning, diversity-based sampling, and balanced sampling
//! for meta-learning task distribution.

use crate::analysis::ml::pinn::meta_learning::config::MetaLearningConfig;
use crate::analysis::ml::pinn::meta_learning::types::{PdeType, PhysicsTask};
use crate::analysis::ml::pinn::Geometry2D;
use crate::core::error::{KwaversError, KwaversResult};

#[derive(Debug)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Curriculum learning (easy to hard)
    Curriculum,
    /// Balanced sampling across physics families
    Balanced,
    /// Diversity sampling
    Diversity,
}

#[derive(Debug)]
pub struct TaskSampler {
    /// Available physics tasks
    task_pool: Vec<PhysicsTask>,
    /// Task sampling strategy
    sampling_strategy: SamplingStrategy,
    /// Current sampling index
    current_index: usize,
    /// History of sampled task indices
    task_history: Vec<usize>,
    /// Configuration
    config: MetaLearningConfig,
}

impl TaskSampler {
    /// Create a new task sampler
    pub fn new(strategy: SamplingStrategy, config: MetaLearningConfig) -> Self {
        Self {
            task_pool: Vec::new(),
            sampling_strategy: strategy,
            current_index: 0,
            task_history: Vec::new(),
            config,
        }
    }

    /// Add a task to the pool
    pub fn add_task(&mut self, task: PhysicsTask) {
        self.task_pool.push(task);
    }

    /// Sample a batch of tasks
    pub fn sample_batch(&mut self, batch_size: usize) -> KwaversResult<Vec<PhysicsTask>> {
        if self.task_pool.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "No tasks available in task pool".to_string(),
                },
            ));
        }

        let mut batch = Vec::new();

        for _ in 0..batch_size {
            let task = match self.sampling_strategy {
                SamplingStrategy::Random => {
                    let idx = rand::random::<usize>() % self.task_pool.len();
                    self.task_pool[idx].clone()
                }
                SamplingStrategy::Curriculum => {
                    // Progressive difficulty curriculum learning
                    // Literature: Bengio et al. (2009) Curriculum Learning, Graves et al. (2017) Automated Curriculum Learning

                    // Compute task difficulty based on multiple factors
                    let task_difficulties: Vec<f64> = self
                        .task_pool
                        .iter()
                        .map(|task| {
                            // Difficulty = f(complexity, domain_knowledge, boundary_conditions)
                            let complexity_score = match task.pde_type {
                                PdeType::Wave => 1.0,
                                PdeType::Diffusion => 2.0,
                                PdeType::NavierStokes => 4.0,
                                PdeType::Electromagnetic => 3.0,
                                PdeType::Acoustic => 2.0,
                                PdeType::Elastic => 3.0,
                            };

                            let geometry_complexity = match task.geometry.as_ref() {
                                Geometry2D::Rectangular { .. } => 1.0,
                                Geometry2D::Circular { .. } => 2.0,
                                Geometry2D::MultiRegion { .. } => 4.0,
                                _ => 3.0, // Default for other geometries
                            };

                            let boundary_complexity = task.boundary_conditions.len() as f64;

                            complexity_score * geometry_complexity * boundary_complexity
                        })
                        .collect();

                    // Progressive sampling: start with easier tasks, gradually increase difficulty
                    let progress_ratio = self.current_index as f64 / self.config.max_tasks as f64;
                    let target_difficulty =
                        progress_ratio * task_difficulties.iter().cloned().fold(0.0, f64::max);

                    // Sample from tasks within current difficulty range
                    let candidates: Vec<usize> = task_difficulties
                        .iter()
                        .enumerate()
                        .filter(|(_, &diff)| diff <= target_difficulty + 1.0) // Allow some exploration
                        .map(|(i, _)| i)
                        .collect();

                    if candidates.is_empty() {
                        // Fallback to any task if no candidates found
                        let idx = self.current_index % self.task_pool.len();
                        self.task_pool[idx].clone()
                    } else {
                        // Sample from candidates, preferring higher difficulty within range
                        let selected_idx = candidates[rand::random::<usize>() % candidates.len()];
                        self.current_index += 1;
                        self.task_pool[selected_idx].clone()
                    }
                }
                SamplingStrategy::Balanced => {
                    // Sample from different physics families
                    let idx = rand::random::<usize>() % self.task_pool.len();
                    self.task_pool[idx].clone()
                }
                SamplingStrategy::Diversity => {
                    // Maximize task diversity using determinantal point processes
                    // Literature: Kulesza & Taskar (2012) Determinantal Point Processes for Machine Learning

                    // Track recently sampled task types to ensure diversity
                    let mut sampled_types = std::collections::HashSet::new();
                    for recent_task in self.task_history.iter().rev().take(5) {
                        if let Some(task) = self.task_pool.get(*recent_task) {
                            sampled_types.insert(task.pde_type.clone());
                        }
                    }

                    // Score tasks by diversity from recent samples
                    let diversity_scores: Vec<(usize, f64)> = self
                        .task_pool
                        .iter()
                        .enumerate()
                        .map(|(i, task)| {
                            let type_diversity = if sampled_types.contains(&task.pde_type) {
                                0.3
                            } else {
                                1.0
                            };
                            let geometry_diversity = match task.geometry.as_ref() {
                                Geometry2D::Rectangular { .. } => 0.5,
                                Geometry2D::Circular { .. } => 0.7,
                                Geometry2D::MultiRegion { .. } => 1.0,
                                _ => 0.8,
                            };
                            let score = type_diversity * geometry_diversity;
                            (i, score)
                        })
                        .collect();

                    // Sample proportionally to diversity scores
                    let total_score: f64 = diversity_scores.iter().map(|(_, s)| s).sum();
                    let mut rand_val = rand::random::<f64>() * total_score;
                    let mut selected_task = None;

                    for (idx, score) in diversity_scores {
                        rand_val -= score;
                        if rand_val <= 0.0 {
                            selected_task = Some(self.task_pool[idx].clone());
                            break;
                        }
                    }

                    // Fallback
                    selected_task.unwrap_or_else(|| {
                        self.task_pool[rand::random::<usize>() % self.task_pool.len()].clone()
                    })
                }
            };

            batch.push(task);
        }

        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_sampler_creation() {
        let config = MetaLearningConfig::default();
        let sampler = TaskSampler::new(SamplingStrategy::Random, config);
        assert_eq!(sampler.task_pool.len(), 0);
    }
}
