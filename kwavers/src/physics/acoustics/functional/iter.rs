//! Lazy Iterators and Evaluation
//!
//! This module provides lazy iterators for field operations,
//! enabling efficient deferred computation and chaining.

use ndarray::Array3;

/// Lazy field iterator that applies transformations on demand
#[derive(Debug)]
pub struct LazyFieldIterator<'a, T, F, U> {
    field: &'a Array3<T>,
    transform: F,
    index: usize,
    _phantom: std::marker::PhantomData<U>,
}

impl<'a, T, F, U> LazyFieldIterator<'a, T, F, U>
where
    F: Fn(&T) -> U,
{
    /// Create a new lazy field iterator
    pub fn new(field: &'a Array3<T>, transform: F) -> Self {
        Self {
            field,
            transform,
            index: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, F, U> Iterator for LazyFieldIterator<'a, T, F, U>
where
    F: Fn(&T) -> U,
{
    type Item = ((usize, usize, usize), U);

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.field.len();
        if self.index >= total {
            return None;
        }

        let shape = self.field.dim();
        let k = self.index % shape.2;
        let j = (self.index / shape.2) % shape.1;
        let i = self.index / (shape.1 * shape.2);

        let value = (self.transform)(&self.field[[i, j, k]]);
        self.index += 1;

        Some(((i, j, k), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.field.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T, F, U> ExactSizeIterator for LazyFieldIterator<'a, T, F, U>
where
    F: Fn(&T) -> U,
{
    fn len(&self) -> usize {
        self.field.len().saturating_sub(self.index)
    }
}

/// Chunked lazy iterator for processing fields in chunks
#[derive(Debug)]
pub struct ChunkedFieldIterator<'a, T> {
    field: &'a Array3<T>,
    chunk_size: usize,
    current_chunk: usize,
    total_chunks: usize,
}

impl<'a, T> ChunkedFieldIterator<'a, T> {
    /// Create a new chunked iterator
    #[must_use]
    pub fn new(field: &'a Array3<T>, chunk_size: usize) -> Self {
        let total_elements = field.len();
        let total_chunks = total_elements.div_ceil(chunk_size);

        Self {
            field,
            chunk_size,
            current_chunk: 0,
            total_chunks,
        }
    }
}

impl<'a, T> Iterator for ChunkedFieldIterator<'a, T> {
    type Item = Vec<((usize, usize, usize), &'a T)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk >= self.total_chunks {
            return None;
        }

        let start_idx = self.current_chunk * self.chunk_size;
        let end_idx = ((self.current_chunk + 1) * self.chunk_size).min(self.field.len());

        let shape = self.field.dim();
        let mut chunk = Vec::with_capacity(end_idx - start_idx);

        for linear_idx in start_idx..end_idx {
            let k = linear_idx % shape.2;
            let j = (linear_idx / shape.2) % shape.1;
            let i = linear_idx / (shape.1 * shape.2);

            chunk.push(((i, j, k), &self.field[[i, j, k]]));
        }

        self.current_chunk += 1;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_chunks.saturating_sub(self.current_chunk);
        (remaining, Some(remaining))
    }
}

/// Extension trait for creating lazy iterators from fields
pub trait LazyIterExt<T> {
    /// Create a lazy iterator with transformation
    fn lazy_iter<F, U>(&self, transform: F) -> LazyFieldIterator<'_, T, F, U>
    where
        F: Fn(&T) -> U;

    /// Create a chunked iterator
    fn chunked_iter(&self, chunk_size: usize) -> ChunkedFieldIterator<'_, T>;
}

impl<T> LazyIterExt<T> for Array3<T> {
    fn lazy_iter<F, U>(&self, transform: F) -> LazyFieldIterator<'_, T, F, U>
    where
        F: Fn(&T) -> U,
    {
        LazyFieldIterator::new(self, transform)
    }

    fn chunked_iter(&self, chunk_size: usize) -> ChunkedFieldIterator<'_, T> {
        ChunkedFieldIterator::new(self, chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lazy_iterator() {
        let field = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as f64);
        let mut iter = field.lazy_iter(|&x| x * 2.0);

        let first = iter.next().unwrap();
        assert_eq!(first.0, (0, 0, 0));
        assert_abs_diff_eq!(first.1, 0.0);

        assert_eq!(iter.len(), 7); // 7 elements remaining
    }

    #[test]
    fn test_chunked_iterator() {
        let field: Array3<f64> = Array3::from_elem((2, 2, 2), 1.0);
        let chunks: Vec<_> = field.chunked_iter(3).collect();

        assert_eq!(chunks.len(), 3); // 8 elements / 3 = 3 chunks (last partial)
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 3);
        assert_eq!(chunks[2].len(), 2); // Partial chunk
    }
}
