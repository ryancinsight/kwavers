//! Standard-layout grid coordinate traversal.

/// C-order coordinates advanced without per-element division.
pub(super) struct GridPosition {
    i: usize,
    j: usize,
    k: usize,
}

impl GridPosition {
    /// Decode the first coordinate of a contiguous flat range.
    pub(super) fn from_flat(index: usize, ny: usize, nz: usize) -> Self {
        let yz_len = ny
            .checked_mul(nz)
            .expect("invariant: grid plane length fits usize");
        let i = index / yz_len;
        let remainder = index % yz_len;
        Self {
            i,
            j: remainder / nz,
            k: remainder % nz,
        }
    }

    /// Advance one element in C storage order.
    pub(super) fn advance(&mut self, ny: usize, nz: usize) {
        self.k += 1;
        if self.k == nz {
            self.k = 0;
            self.j += 1;
            if self.j == ny {
                self.j = 0;
                self.i += 1;
            }
        }
    }

    /// Return the current logical coordinate.
    pub(super) fn coordinates(&self) -> [usize; 3] {
        [self.i, self.j, self.k]
    }
}

#[cfg(test)]
mod tests {
    use super::GridPosition;

    #[test]
    fn decodes_nonzero_chunk_start() {
        let position = GridPosition::from_flat(256, 8, 7);
        assert_eq!(position.coordinates(), [4, 4, 4]);
    }

    #[test]
    fn advances_across_rows_and_planes() {
        let mut row_boundary = GridPosition::from_flat(2, 2, 3);
        row_boundary.advance(2, 3);
        assert_eq!(row_boundary.coordinates(), [0, 1, 0]);

        let mut plane_boundary = GridPosition::from_flat(5, 2, 3);
        plane_boundary.advance(2, 3);
        assert_eq!(plane_boundary.coordinates(), [1, 0, 0]);
    }
}
