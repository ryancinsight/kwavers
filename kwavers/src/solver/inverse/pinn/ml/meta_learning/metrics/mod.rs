//! Meta-Learning Metrics and Statistics
//!
//! This module defines performance metrics and statistics for meta-learning training,
//! including loss components, generalization scores, and convergence tracking.
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" ICML 2017
//!
//! 2. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML" ICLR 2019. DOI: 10.48550/arXiv.1810.09502

pub mod loss;
pub mod stats;
#[cfg(test)]
mod tests;

pub use loss::MetaLoss;
pub use stats::MetaLearningStats;
