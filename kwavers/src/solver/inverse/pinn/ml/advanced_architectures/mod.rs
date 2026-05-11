//! Advanced Neural Architectures for Physics-Informed Neural Networks (PINNs)
//!
//! This module implements state-of-the-art neural architectures specifically designed
//! for physics-informed neural networks. These architectures address common PINN
//! convergence issues and improve accuracy for solving partial differential equations.
//!
//! ## Architectures Implemented
//!
//! - **ResNet PINNs**: Residual connections to enable deeper networks and better gradient flow
//! - **Fourier Features**: Frequency-domain embeddings for better representation of oscillatory physics
//!
//! ## References
//!
//! - Wang et al. (2021): "When and why PINNs fail to train: A neural tangent kernel perspective"
//! - Wang et al. (2022): "On the eigenvector bias of Fourier feature networks"

mod fourier;
mod residual;
mod resnet;
#[cfg(test)]
mod tests;

pub use fourier::FourierFeatures;
pub use residual::ResidualBlock;
pub use resnet::{ResNetPINN1D, ResNetPINN2D, ResNetPINNConfig};
