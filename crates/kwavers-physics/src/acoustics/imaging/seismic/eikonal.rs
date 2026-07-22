//! First-arrival traveltimes by the Fast Sweeping Method.
//!
//! Solves the eikonal equation `|∇T(x)| = s(x)`, where `T` is the first-arrival
//! traveltime from a source and `s = 1/c` is the slowness, using the Fast
//! Sweeping Method (FSM): Gauss–Seidel iterations with a Godunov upwind
//! discretization, alternating the sweep direction so that every characteristic
//! direction is followed. FSM converges in a fixed number of sweeps (`2^d` per
//! iteration in `d` dimensions) independent of grid size for a single source.
//!
//! Traveltime tables are the input to Kirchhoff migration (diffraction stacking)
//! and to ray-based traveltime tomography.
//!
//! # References
//! - Zhao, H. (2005). "A fast sweeping method for eikonal equations."
//!   *Mathematics of Computation*, 74(250), 603–627.
//! - Sethian, J. A. (1996). "A fast marching level set method for monotonically
//!   advancing fronts." *PNAS*, 93(4), 1591–1595.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;

/// A large sentinel traveltime marking not-yet-reached nodes.
const FAR: f64 = 1.0e30;

/// Godunov solution of `Σ_dim max((T − a_dim)/h_dim, 0)² = f²` at one node.
///
/// `neighbors` holds `(a_dim, h_dim)` for each axis, where `a_dim` is the smaller
/// of the two axis neighbours (or [`FAR`] at a boundary) and `h_dim` the spacing;
/// `f` is the local slowness. Returns the upwind traveltime.
fn godunov_update(neighbors: &[(f64, f64); 3], f: f64) -> f64 {
    // Keep only finite contributions, sorted ascending by neighbour value.
    let mut dims: Vec<(f64, f64)> = neighbors
        .iter()
        .copied()
        .filter(|&(a, _)| a < FAR)
        .collect();
    dims.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    if dims.is_empty() {
        return FAR;
    }
    // Incrementally include dimensions while the candidate stays above the next
    // neighbour value (the upwind validity condition).
    let mut t = FAR;
    for p in 1..=dims.len() {
        // Solve Σ_{i<p} (T − a_i)²/h_i² = f² for the larger root.
        let (mut sa, mut sb, mut sc) = (0.0, 0.0, 0.0); // coefficients of T², T, 1
        for &(a, h) in dims.iter().take(p) {
            let w = 1.0 / (h * h);
            sa += w;
            sb += -2.0 * a * w;
            sc += a * a * w;
        }
        sc -= f * f;
        let disc = sb * sb - 4.0 * sa * sc;
        if disc < 0.0 {
            continue;
        }
        let cand = (-sb + disc.sqrt()) / (2.0 * sa);
        // Valid iff it does not exceed the next (excluded) neighbour.
        let next = dims.get(p).map_or(f64::INFINITY, |&(a, _)| a);
        if cand >= dims[p - 1].0 && cand <= next {
            t = cand;
            break;
        }
        t = cand; // keep last; the full-dimension solve is always accepted
    }
    t
}

/// Fast-sweeping eikonal solver over a [`Grid`] with a heterogeneous slowness.
#[derive(Debug, Clone)]
pub struct EikonalSolver {
    /// Slowness field `s = 1/c` [s/m], shape `[nx, ny, nz]`.
    slowness: Array3<f64>,
    dx: f64,
    dy: f64,
    dz: f64,
    /// Number of sweep iterations (each iteration = `2^d` directional sweeps).
    iterations: usize,
}

impl EikonalSolver {
    /// Build a solver from a sound-speed field `c` [m/s] on `grid`.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when the speed field shape does not
    /// match the grid or contains a non-positive speed.
    pub fn from_sound_speed(grid: &Grid, sound_speed: &Array3<f64>) -> KwaversResult<Self> {
        if sound_speed.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(KwaversError::InvalidInput(
                "sound speed shape does not match grid".to_owned(),
            ));
        }
        if sound_speed.iter().any(|&c| c <= 0.0) {
            return Err(KwaversError::InvalidInput(
                "sound speed must be strictly positive".to_owned(),
            ));
        }
        Ok(Self {
            slowness: sound_speed.mapv(|c| 1.0 / c),
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
            iterations: 4,
        })
    }

    /// Override the sweep-iteration count (default 4; more for strong contrast).
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations.max(1);
        self
    }

    /// Compute the first-arrival traveltime field from a source grid point.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when the source index is outside
    /// the grid.
    pub fn solve(&self, source: (usize, usize, usize)) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = self.slowness.shape();
        if source.0 >= nx || source.1 >= ny || source.2 >= nz {
            return Err(KwaversError::InvalidInput(
                "eikonal source index out of bounds".to_owned(),
            ));
        }
        let mut t = Array3::from_elem([nx, ny, nz], FAR);
        t[[source.0, source.1, source.2]] = 0.0;
        // First-order source initialization: seed the immediate neighbourhood
        // with the analytic local-slowness traveltime `s·|Δx|`. This removes the
        // dominant near-source error of the upwind scheme (the point-source kink
        // is not representable by finite differences), markedly improving
        // accuracy without changing the scheme's order.
        let s0 = self.slowness[[source.0, source.1, source.2]];
        let (sx, sy, sz) = (source.0 as isize, source.1 as isize, source.2 as isize);
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    let (i, j, k) = (sx + di, sy + dj, sz + dk);
                    if i < 0
                        || j < 0
                        || k < 0
                        || i >= nx as isize
                        || j >= ny as isize
                        || k >= nz as isize
                    {
                        continue;
                    }
                    let dist = ((di as f64 * self.dx).powi(2)
                        + (dj as f64 * self.dy).powi(2)
                        + (dk as f64 * self.dz).powi(2))
                    .sqrt();
                    let val = s0 * dist;
                    let cell = &mut t[[i as usize, j as usize, k as usize]];
                    if val < *cell {
                        *cell = val;
                    }
                }
            }
        }

        let xs: [bool; 2] = [true, false];
        for _ in 0..self.iterations {
            for &sx in &xs {
                for &sy in &xs {
                    for &sz in &xs {
                        self.sweep(&mut t, sx, sy, sz);
                    }
                }
            }
        }
        Ok(t)
    }

    /// One directional Gauss–Seidel sweep over the grid.
    fn sweep(&self, t: &mut Array3<f64>, forward_x: bool, forward_y: bool, forward_z: bool) {
        let [nx, ny, nz] = self.slowness.shape();
        let xi: Vec<usize> = order(nx, forward_x);
        let yi: Vec<usize> = order(ny, forward_y);
        let zi: Vec<usize> = order(nz, forward_z);
        for &i in &xi {
            for &j in &yi {
                for &k in &zi {
                    let ax = min_neighbor(t, i, nx, |q| [q, j, k], self.dx);
                    let ay = min_neighbor(t, j, ny, |q| [i, q, k], self.dy);
                    let az = min_neighbor(t, k, nz, |q| [i, j, q], self.dz);
                    let cand = godunov_update(&[ax, ay, az], self.slowness[[i, j, k]]);
                    if cand < t[[i, j, k]] {
                        t[[i, j, k]] = cand;
                    }
                }
            }
        }
    }
}

/// Sweep index order along an axis.
fn order(n: usize, forward: bool) -> Vec<usize> {
    if forward {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    }
}

/// Smaller of the two axis neighbours of node `idx` along one axis, paired with
/// the spacing. Returns `(FAR, h)` for a degenerate (length-1) axis.
fn min_neighbor(
    t: &Array3<f64>,
    idx: usize,
    n: usize,
    at: impl Fn(usize) -> [usize; 3],
    h: f64,
) -> (f64, f64) {
    if n < 2 {
        return (FAR, h);
    }
    let mut best = FAR;
    if idx > 0 {
        let p = at(idx - 1);
        best = best.min(t[[p[0], p[1], p[2]]]);
    }
    if idx + 1 < n {
        let p = at(idx + 1);
        best = best.min(t[[p[0], p[1], p[2]]]);
    }
    (best, h)
}
