#[derive(Debug, Clone, Copy)]
pub struct DGConfig {
    pub polynomial_order: usize,
    pub basis_type: super::basis::BasisType,
    pub flux_type: super::flux::FluxType,
    pub use_limiter: bool,
    pub limiter_type: super::flux::LimiterType,
    pub shock_threshold: f64,
}
