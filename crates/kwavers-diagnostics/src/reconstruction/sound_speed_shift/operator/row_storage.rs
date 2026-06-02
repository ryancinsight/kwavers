//! Compressed row storage for nonzero ray/pixel segment lengths.

#[derive(Clone, Debug)]
pub(super) struct RayRowStorage {
    sample_indices: Vec<usize>,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    lengths_m: Vec<f64>,
}

impl RayRowStorage {
    pub(super) fn new() -> Self {
        Self {
            sample_indices: Vec::new(),
            row_offsets: vec![0],
            columns: Vec::new(),
            lengths_m: Vec::new(),
        }
    }

    pub(super) fn push_nonempty_row<I>(&mut self, sample_index: usize, entries: I) -> bool
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let start = self.columns.len();
        self.sample_indices.push(sample_index);
        for (column, length_m) in entries {
            self.columns.push(column);
            self.lengths_m.push(length_m);
        }
        if self.columns.len() == start {
            self.sample_indices.pop();
            return false;
        }
        self.row_offsets.push(self.columns.len());
        true
    }

    #[must_use]
    pub(super) fn row_count(&self) -> usize {
        self.sample_indices.len()
    }

    #[must_use]
    pub(super) fn sample_index(&self, row: usize) -> usize {
        self.sample_indices[row]
    }

    pub(super) fn row_entries(&self, row: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        self.columns[start..end]
            .iter()
            .copied()
            .zip(self.lengths_m[start..end].iter().copied())
    }

    #[must_use]
    pub(super) fn nonzero_count(&self) -> usize {
        self.columns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RayRowStorage;

    #[test]
    fn stores_rows_in_flat_offsets_and_values() {
        let mut rows = RayRowStorage::new();
        assert!(rows.push_nonempty_row(4, [(2, 0.25), (5, 0.75)]));
        assert!(rows.push_nonempty_row(9, [(3, 1.25)]));

        assert_eq!(rows.row_count(), 2);
        assert_eq!(rows.sample_index(0), 4);
        assert_eq!(rows.sample_index(1), 9);
        assert_eq!(rows.nonzero_count(), 3);
        assert_eq!(
            rows.row_entries(0).collect::<Vec<_>>(),
            vec![(2, 0.25), (5, 0.75)]
        );
        assert_eq!(rows.row_entries(1).collect::<Vec<_>>(), vec![(3, 1.25)]);
    }

    #[test]
    fn rejects_empty_rows_without_offset_mutation() {
        let mut rows = RayRowStorage::new();

        assert!(!rows.push_nonempty_row(3, std::iter::empty::<(usize, f64)>()));

        assert_eq!(rows.row_count(), 0);
        assert_eq!(rows.nonzero_count(), 0);
    }
}
