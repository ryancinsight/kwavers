//! Active-support graph construction for same-aperture inverse problems.

use leto::Array2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarPoint {
    pub x_m: f64,
    pub y_m: f64,
}

#[derive(Clone, Debug)]
pub struct ActiveGrid {
    pub indices: Vec<(usize, usize)>,
    pub points_m: Vec<PlanarPoint>,
    neighbor_indices: Vec<[Option<usize>; 4]>,
}

impl ActiveGrid {
    #[must_use]
    pub fn len(&self) -> usize {
        (self.indices.shape()[0] * self.indices.shape()[1] * self.indices.shape()[2])
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Apply the four-neighbor graph Laplacian on the active tissue support.
    ///
    /// # Theorem
    ///
    /// For the undirected graph `G = (V, E)` induced by four-neighbor active
    /// CT pixels, define `(Lx)_i = deg(i)x_i - sum_{j in N(i)} x_j`. Then
    /// `x^T L x = sum_{(i,j) in E} (x_i - x_j)^2 >= 0`.
    ///
    /// # Proof
    ///
    /// Expanding `x^T L x` gives one `deg(i)x_i^2` term per vertex and one
    /// `-x_i x_j` term per directed neighbor relation. Each undirected edge is
    /// counted twice in the directed sum, so the edge contribution is
    /// `x_i^2 + x_j^2 - 2x_i x_j = (x_i - x_j)^2`.
    pub fn graph_laplacian_into(&self, values: &[f32], out: &mut [f32]) {
        debug_assert_eq!((values.shape()[0] * values.shape()[1] * values.shape()[2]), (self.indices.shape()[0] * self.indices.shape()[1] * self.indices.shape()[2]));
        debug_assert_eq!((out.shape()[0] * out.shape()[1] * out.shape()[2]), (self.indices.shape()[0] * self.indices.shape()[1] * self.indices.shape()[2]));
        for (row, neighbors) in self.neighbor_indices.iter().enumerate() {
            let center = values[row];
            let mut degree = 0.0;
            let mut sum = 0.0;
            for neighbor in neighbors.iter().flatten() {
                degree += 1.0;
                sum += values[*neighbor];
            }
            out[row] = degree * center - sum;
        }
    }
}

#[must_use]
pub fn active_grid(mask: &Array2<bool>, spacing_m: f64) -> ActiveGrid {
    let [nx, ny] = mask.shape();
    let cx = (nx - 1) as f64 * 0.5;
    let cy = (ny - 1) as f64 * 0.5;
    let mut indices = Vec::new();
    let mut points_m = Vec::new();
    let mut active_lookup = vec![None; nx * ny];
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            active_lookup[linear_index(ix, iy, ny)] = Some((indices.shape()[0] * indices.shape()[1] * indices.shape()[2]));
            indices.push((ix, iy));
            points_m.push(PlanarPoint {
                x_m: (ix as f64 - cx) * spacing_m,
                y_m: (iy as f64 - cy) * spacing_m,
            });
        }
    }
    let neighbor_indices = indices
        .iter()
        .map(|(ix, iy)| active_neighbors(*ix, *iy, nx, ny, &active_lookup))
        .collect();
    ActiveGrid {
        indices,
        points_m,
        neighbor_indices,
    }
}

#[must_use]
pub fn vector_from_image(image: &Array2<f64>, active: &ActiveGrid) -> Vec<f32> {
    active
        .indices
        .iter()
        .map(|(ix, iy)| image[[*ix, *iy]] as f32)
        .collect()
}

#[must_use]
pub fn image_from_vector(
    values: &[f32],
    active: &ActiveGrid,
    shape: (usize, usize),
) -> Array2<f64> {
    let mut image = Array2::<f64>::zeros(shape);
    for ((ix, iy), value) in active.indices.iter().zip(values.iter()) {
        image[[*ix, *iy]] = f64::from(*value);
    }
    image
}

fn active_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    active_lookup: &[Option<usize>],
) -> [Option<usize>; 4] {
    let mut out = [None; 4];
    let mut count = 0;
    for (jx, jy) in lattice_neighbors(ix, iy, nx, ny) {
        if let Some(active) = active_lookup[linear_index(jx, jy, ny)] {
            out[count] = Some(active);
            count += 1;
        }
    }
    out
}

fn lattice_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut out = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        out[count] = (ix - 1, iy);
        count += 1;
    }
    if iy > 0 {
        out[count] = (ix, iy - 1);
        count += 1;
    }
    if ix + 1 < nx {
        out[count] = (ix + 1, iy);
        count += 1;
    }
    if iy + 1 < ny {
        out[count] = (ix, iy + 1);
        count += 1;
    }
    out.into_iter().take(count)
}

const fn linear_index(ix: usize, iy: usize, ny: usize) -> usize {
    ix * ny + iy
}
