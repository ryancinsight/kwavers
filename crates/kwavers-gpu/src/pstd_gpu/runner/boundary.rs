//! Host staging of collocated and staggered PML volumes.

use kwavers_boundary::cpml::CPMLProfiles;

use crate::pstd_gpu::PmlArrays;

pub(super) enum StagedBoundary {
    Identity(Vec<f32>),
    Graded {
        collocated: [Vec<f32>; 3],
        staggered: [Vec<f32>; 3],
    },
}

impl StagedBoundary {
    pub(super) fn as_arrays(&self) -> PmlArrays<'_> {
        let (collocated, staggered) = match self {
            Self::Identity(ones) => ([ones.as_slice(); 3], [ones.as_slice(); 3]),
            Self::Graded {
                collocated,
                staggered,
            } => (
                collocated.each_ref().map(Vec::as_slice),
                staggered.each_ref().map(Vec::as_slice),
            ),
        };
        let [x, y, z] = collocated;
        let [sgx, sgy, sgz] = staggered;
        PmlArrays {
            x,
            y,
            z,
            sgx,
            sgy,
            sgz,
        }
    }
}

pub(super) fn stage(
    [nx, ny, nz]: [usize; 3],
    total: usize,
    dt: f64,
    profiles: Option<&CPMLProfiles>,
) -> StagedBoundary {
    let Some(profiles) = profiles else {
        // Six identical logical fields can borrow the same immutable backing.
        return StagedBoundary::Identity(vec![1.0; total]);
    };

    // Evaluate the half-step damping once at each position along an axis.
    // Replicate it in C order for the backend's dense field contract.
    let exp_half = |sigma: &leto::Array1<f64>| -> Vec<f32> {
        sigma
            .iter()
            .map(|&s| (-s * dt * 0.5).exp() as f32)
            .collect()
    };
    let den = [&profiles.sigma_x, &profiles.sigma_y, &profiles.sigma_z].map(exp_half);
    let vel = [
        &profiles.sigma_x_sgx,
        &profiles.sigma_y_sgy,
        &profiles.sigma_z_sgz,
    ]
    .map(exp_half);
    let mut collocated: [Vec<f32>; 3] = std::array::from_fn(|_| vec![1.0; total]);
    let mut staggered: [Vec<f32>; 3] = std::array::from_fn(|_| vec![1.0; total]);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let flat = (x * ny + y) * nz + z;
                collocated[0][flat] = den[0][x];
                collocated[1][flat] = den[1][y];
                collocated[2][flat] = den[2][z];
                staggered[0][flat] = vel[0][x];
                staggered[1][flat] = vel[1][y];
                staggered[2][flat] = vel[2][z];
            }
        }
    }
    StagedBoundary::Graded {
        collocated,
        staggered,
    }
}

#[cfg(test)]
mod tests;
