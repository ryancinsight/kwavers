// STRIDED ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute blocked index for cache-efficient access
///
/// Reorders iteration to process blocks of elements that fit in cache.
///
/// # Mathematical Specification
///
/// For tile dimensions $(T_x, T_y, T_z)$ and global index $(i, j, k)$:
/// $$
/// \text{tile}_x = \lfloor i / T_x \rfloor, \text{tile}_y = \lfloor j / T_y \rfloor,
/// \text{tile}_z = \lfloor k / T_z \rfloor
/// $$
/// $$ \text{local}_x = i \bmod T_x, \text{local}_y = j \bmod T_y, \text{local}_z = k \bmod T_z $$
///
/// Linearized tile-major order for sequential access within tile.
#[inline]
#[must_use]
pub fn tiled_index(
    i: usize,
    j: usize,
    k: usize,
    nx: usize,
    ny: usize,
    tile_x: usize,
    tile_y: usize,
    _tile_z: usize,
) -> usize {
    let tile_idx_x = i / tile_x;
    let tile_idx_y = j / tile_y;
    let tile_idx_z = k / tile_x; // Using tile_x for z consistency

    let local_x = i % tile_x;
    let local_y = j % tile_y;

    // Tile-major order: z varies slowest, x varies fastest
    let tile_column = tile_idx_z * (nx / tile_x + 1) * (ny / tile_y + 1)
        + tile_idx_y * (nx / tile_x + 1)
        + tile_idx_x;

    let local_offset = local_y * tile_x + local_x;

    tile_column * tile_x * tile_y + local_offset
}

/// Cache blocking parameters optimized for L1/L2 cache sizes
///
/// # Mathematical Justification
///
/// L1 cache typically 32KB-64KB per core:
/// - Fits ~4096 f64 elements per core
/// - Divide by accessed fields for working set size
///
/// L2 cache typically 256KB-1MB per core:
/// - Fits ~32768-131072 f64 elements
#[derive(Debug, Clone, Copy)]
pub struct CacheBlockSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl CacheBlockSize {
    /// Optimal blocking for 1 field (8 bytes per element)
    /// Targets ~32KB L1 cache per field access
    #[must_use]
    pub fn for_l1_single_field() -> Self {
        // 32KB = 4096 f64 elements
        // Cube root for 3D: ~16 per dimension
        Self {
            x: 16,
            y: 16,
            z: 16,
        }
    }

    /// Optimal blocking for 4 fields simultaneously
    /// Targets ~64KB L1 to hold all fields
    #[must_use]
    pub fn for_l1_four_fields() -> Self {
        // 64KB total = 16KB per field = 2048 elements
        // Distribute across dimensions
        Self {
            x: 12,
            y: 12,
            z: 14, // ~2016 elements
        }
    }

    /// Optimal blocking for L2 cache
    #[must_use]
    pub fn for_l2_single_field() -> Self {
        // 256KB L2 = 32768 f64 elements = 32^3
        Self {
            x: 32,
            y: 32,
            z: 32,
        }
    }

    /// Get total elements in block
    #[inline]
    #[must_use]
    pub const fn total_elements(&self) -> usize {
        self.x * self.y * self.z
    }
}

/// Iteration order that traverses 3D space in cache-efficient tiles
#[derive(Debug)]
pub struct TiledIterator3D {
    nx: usize,
    ny: usize,
    nz: usize,
    tile_x: usize,
    tile_y: usize,
    tile_z: usize,
    current_tile_x: usize,
    current_tile_y: usize,
    current_tile_z: usize,
    current_local_x: usize,
    current_local_y: usize,
    current_local_z: usize,
}

impl TiledIterator3D {
    /// Create tiled iterator for dimensions nx×ny×nz with specified tile size
    pub fn new(nx: usize, ny: usize, nz: usize, tile: &CacheBlockSize) -> Self {
        Self {
            nx,
            ny,
            nz,
            tile_x: tile.x.min(nx),
            tile_y: tile.y.min(ny),
            tile_z: tile.z.min(nz),
            current_tile_x: 0,
            current_tile_y: 0,
            current_tile_z: 0,
            current_local_x: 0,
            current_local_y: 0,
            current_local_z: 0,
        }
    }

    /// Get current global (x, y, z) indices
    #[inline]
    #[must_use]
    pub fn current_indices(&self) -> (usize, usize, usize) {
        let x = self.current_tile_x * self.tile_x + self.current_local_x;
        let y = self.current_tile_y * self.tile_y + self.current_local_y;
        let z = self.current_tile_z * self.tile_z + self.current_local_z;
        (x.min(self.nx - 1), y.min(self.ny - 1), z.min(self.nz - 1))
    }
}

impl Iterator for TiledIterator3D {
    type Item = (usize, usize, usize); // (x, y, z)

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_tile_z * self.tile_z >= self.nz {
            return None;
        }

        let result = self.current_indices();

        // Advance within current tile (inner loops first)
        self.current_local_x += 1;
        if self.current_local_x >= self.tile_x
            || self.current_tile_x * self.tile_x + self.current_local_x >= self.nx
        {
            self.current_local_x = 0;
            self.current_local_y += 1;

            if self.current_local_y >= self.tile_y
                || self.current_tile_y * self.tile_y + self.current_local_y >= self.ny
            {
                self.current_local_y = 0;
                self.current_local_z += 1;

                if self.current_local_z >= self.tile_z
                    || self.current_tile_z * self.tile_z + self.current_local_z >= self.nz
                {
                    self.current_local_z = 0;

                    // Move to next tile
                    self.current_tile_x += 1;
                    if self.current_tile_x * self.tile_x >= self.nx {
                        self.current_tile_x = 0;
                        self.current_tile_y += 1;

                        if self.current_tile_y * self.tile_y >= self.ny {
                            self.current_tile_y = 0;
                            self.current_tile_z += 1;
                        }
                    }
                }
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_iterator() {
        let tile = CacheBlockSize { x: 4, y: 4, z: 4 };
        let iter = TiledIterator3D::new(8, 8, 8, &tile);

        let points: Vec<_> = iter.collect();
        assert_eq!(points.len(), 8 * 8 * 8);

        // First points should be in first tile
        assert_eq!(points[0], (0, 0, 0));
        assert_eq!(points[1], (1, 0, 0));
        assert_eq!(points[4], (0, 1, 0)); // Wrapped in y

        // Should cover all unique points
        let unique: std::collections::HashSet<_> = points.iter().cloned().collect();
        assert_eq!(unique.len(), 512);
    }
}
