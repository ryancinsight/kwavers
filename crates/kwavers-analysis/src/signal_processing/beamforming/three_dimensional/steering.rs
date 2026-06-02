//! 3D Steering Vector — reserved for future far-field MVDR variant.
//!
//! The current CPU MVDR beamformer (`cpu/mvdr/mod.rs`) uses the delay-then-MVDR
//! formulation where receive delays are pre-applied and the steering vector is
//! identically **1** (Synnevåg et al. 2007).  A far-field plane-wave steering
//! vector is required for the direct-MVDR variant where covariance is estimated
//! before delay correction; that variant is not yet implemented.
//!
//! When the direct-MVDR path is added, `compute_steering_vector_3d` should be
//! restored here and wired into `cpu/mvdr/mod.rs`.
