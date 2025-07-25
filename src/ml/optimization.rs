//! Reinforcement learning and optimization algorithms

use crate::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use rand::Rng;

/// Parameter optimizer using reinforcement learning
pub struct ParameterOptimizer {
    learning_rate: f64,
    exploration_rate: f64,
}

impl ParameterOptimizer {
    pub fn new(learning_rate: f64, exploration_rate: f64) -> Self {
        Self {
            learning_rate,
            exploration_rate,
        }
    }
    
    /// Optimize parameters based on current state and target
    pub fn optimize(
        &self,
        current: &HashMap<String, f64>,
        target: &HashMap<String, f64>,
    ) -> KwaversResult<HashMap<String, f64>> {
        if current.is_empty() {
            return Err(KwaversError::Data(
                crate::error::DataError::InsufficientData {
                    required: 1,
                    available: 0,
                },
            ));
        }

        let mut rng = rand::thread_rng();
        let mut next_params = HashMap::new();

        for (key, &cur_val) in current {
            let tgt_val = target.get(key).copied().unwrap_or(cur_val);
            // Basic gradient-free update towards the target.
            let delta = tgt_val - cur_val;
            let exploratory = rng.gen_range(-1.0..1.0) * self.exploration_rate;
            let new_val = cur_val + self.learning_rate * delta + exploratory;
            next_params.insert(key.clone(), new_val);
        }

        Ok(next_params)
    }
}