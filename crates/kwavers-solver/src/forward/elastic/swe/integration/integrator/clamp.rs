//! Clamped outer walls for the elastic leapfrog.
//!
//! The one-sided wall closures of the central stencils break the discrete
//! skew-symmetry the leapfrog needs for energy conservation, and wall degrees
//! of freedom with nonzero strain otherwise form an energy-injection channel
//! (`KW-SWE-EDGE-GROWTH-2026-09-17`). Clamping the walls after each
//! acceleration evaluation holds wall velocity at zero, so no work enters
//! through the boundary and the interior keeps its conservation. Without a
//! PML, wall displacement stays exactly at its initial value; with one, the
//! PML's own displacement damping (a contractive factor, never a source)
//! relaxes the walls toward zero. Compact data is unaffected to floating
//! point, since its wall values are already ~zero. An initial field with
//! nonzero wall velocity has it projected out on the first step.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::ElasticWaveField;

/// Clamp the outer walls after an acceleration evaluation: wall velocities
/// and stored wall accelerations are zeroed, so the drift and the kicks
/// that follow cannot move wall displacement off its initial value.
pub(super) fn clamp_walls(field: &mut ElasticWaveField, scratch: &mut ElasticStepScratch) {
    zero_wall_planes(&mut field.vx);
    zero_wall_planes(&mut field.vy);
    zero_wall_planes(&mut field.vz);
    zero_wall_planes(&mut scratch.ax);
    zero_wall_planes(&mut scratch.ay);
    zero_wall_planes(&mut scratch.az);
}

/// Zero the six outer wall planes of a volume field.
fn zero_wall_planes(values: &mut leto::Array3<f64>) {
    let shape = values.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    for j in 0..ny {
        for k in 0..nz {
            values[[0, j, k]] = 0.0;
            values[[nx - 1, j, k]] = 0.0;
        }
    }
    for i in 0..nx {
        for k in 0..nz {
            values[[i, 0, k]] = 0.0;
            values[[i, ny - 1, k]] = 0.0;
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            values[[i, j, 0]] = 0.0;
            values[[i, j, nz - 1]] = 0.0;
        }
    }
}
