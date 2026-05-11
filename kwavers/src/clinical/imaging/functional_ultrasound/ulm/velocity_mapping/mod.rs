//! Velocity Mapping from ULM Microbubble Tracks
//!
//! Reconstructs 2D velocity vector fields from accumulated bubble trajectories,
//! enabling quantitative hemodynamic measurements including flow speed, direction,
//! and wall shear stress estimation.
//!
//! ## Algorithm
//!
//! ### Instantaneous Velocity Estimation (Heiles et al. 2022)
//!
//! For each bubble track with detections at positions (x_k, z_k) at frame indices f_k:
//! ```text
//! v_x[k] = (x_{k+1} − x_k) / ((f_{k+1} − f_k) · Δt)   (m/s)
//! v_z[k] = (z_{k+1} − z_k) / ((f_{k+1} − f_k) · Δt)
//! ```
//! The estimate is assigned to the segment midpoint (x̄_k, z̄_k) = ((x_k + x_{k+1})/2, ...).
//!
//! ### Grid Accumulation (bin-and-average)
//!
//! Each velocity estimate votes into the nearest grid cell:
//! ```text
//! V_x[m,n] += v_x[k]   (for all k whose midpoint falls in cell (m,n))
//! count[m,n] += 1
//! ⟨v_x⟩[m,n] = V_x[m,n] / count[m,n]
//! ```
//! Cells with `count < min_count` are set to NaN (insufficient statistics).
//!
//! ### Velocity Magnitude and Direction
//!
//! ```text
//! speed[m,n]     = √(⟨v_x⟩² + ⟨v_z⟩²)    (m/s)
//! direction[m,n] = atan2(⟨v_z⟩, ⟨v_x⟩)   ∈ (−π, π]  (rad)
//! ```
//!
//! ### Wall Shear Stress Estimation (Womersley 1955; Reneman et al. 2006)
//!
//! Wall shear stress τ_w = μ · ∂u/∂n (velocity gradient perpendicular to wall).
//! In the discrete approximation without explicit wall segmentation, the
//! central-difference gradient magnitude of the speed field is used:
//! ```text
//! τ_proxy[m,n] = μ · ‖∇speed[m,n]‖
//!             = μ · √(((speed[m+1,n] − speed[m−1,n])/(2d))²
//!                   + ((speed[m,n+1] − speed[m,n−1])/(2d))²)
//! ```
//! where μ is dynamic blood viscosity [Pa·s] and d is the pixel size (m).
//! Only interior cells with all four neighbors populated (non-NaN) are computed.
//!
//! ## References
//!
//! - Heiles, B., et al. (2022). Performance benchmarking of microbubble-localization
//!   algorithms for ultrasound localization microscopy.
//!   *Nat. Biomed. Eng.* 6(5):605–616. DOI: 10.1038/s41551-021-00824-8
//! - Reneman, R. S., Arts, T., & Hoeks, A. P. G. (2006). Wall shear stress — an important
//!   determinant of endothelial cell function and structure in the arterial system in vivo.
//!   *J. Vasc. Res.* 43(3):251–269. DOI: 10.1159/000091648
//! - Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow and
//!   viscous drag in arteries when the pressure gradient is known.
//!   *J. Physiol.* 127(3):553–563. DOI: 10.1113/jphysiol.1955.sp005276

pub mod config;
pub mod mapper;
pub mod output;
#[cfg(test)]
mod tests;

pub use config::VelocityMapConfig;
pub use mapper::VelocityMapper;
pub use output::VelocityMap;
