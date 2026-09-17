//! The read-only side of a traversal: zero or more views walked in lockstep
//! with the fields a kernel writes.

use leto::ArrayView;

/// A fixed set of read-only views that a traversal reads beside its outputs.
///
/// Implemented for `()`, one [`ArrayView`], and tuples of two to five views,
/// so a kernel's closure receives `()`, `&U`, or a tuple of references it can
/// destructure (`|out, (a, b)| ...`).
///
/// Every element is paired with its output by *logical* row-major position.
/// The dense path therefore requires C-contiguous storage ([`ArrayView::as_slice`]);
/// a view in any other layout sends the whole traversal to the logical walk,
/// which pairs by index whatever the strides are.
pub trait ZipInputs<'a, const N: usize> {
    /// One element of each view, in declaration order.
    type Refs: Copy;
    /// The C-contiguous storage of each view.
    type Slices: Copy + Sync;

    /// Bytes one traversal unit reads from these views.
    const UNIT_BYTES: usize;

    /// Panics unless every view has `shape`.
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]);

    /// The C-contiguous storage of every view, or `None` if any view is laid
    /// out otherwise.
    fn slices(&self) -> Option<Self::Slices>;

    /// Elements at flat row-major position `position` of the slices.
    fn at(slices: Self::Slices, position: usize) -> Self::Refs;

    /// Elements of every view in logical row-major order.
    fn logical(self) -> impl Iterator<Item = Self::Refs>;
}

impl<'a, const N: usize> ZipInputs<'a, N> for () {
    type Refs = ();
    type Slices = ();

    const UNIT_BYTES: usize = 0;

    #[inline]
    fn assert_shape(&self, _shape: [usize; N]) {}

    #[inline]
    fn slices(&self) -> Option<()> {
        Some(())
    }

    #[inline]
    fn at((): (), _position: usize) {}

    #[inline]
    fn logical(self) -> impl Iterator<Item = ()> {
        core::iter::repeat(())
    }
}

impl<'a, A: Sync, const N: usize> ZipInputs<'a, N> for ArrayView<'a, A, N> {
    type Refs = &'a A;
    type Slices = &'a [A];

    const UNIT_BYTES: usize = size_of::<A>();

    #[inline]
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]) {
        assert_eq!(
            self.shape(),
            shape,
            "invariant: every traversed view has the output shape"
        );
    }

    #[inline]
    fn slices(&self) -> Option<&'a [A]> {
        self.as_slice()
    }

    #[inline]
    fn at(slices: &'a [A], position: usize) -> &'a A {
        &slices[position]
    }

    #[inline]
    fn logical(self) -> impl Iterator<Item = &'a A> {
        self.iter()
    }
}

impl<'a, A: Sync, B: Sync, const N: usize> ZipInputs<'a, N>
    for (ArrayView<'a, A, N>, ArrayView<'a, B, N>)
{
    type Refs = (&'a A, &'a B);
    type Slices = (&'a [A], &'a [B]);

    const UNIT_BYTES: usize = size_of::<A>() + size_of::<B>();

    #[inline]
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]) {
        ZipInputs::<N>::assert_shape(&self.0, shape);
        ZipInputs::<N>::assert_shape(&self.1, shape);
    }

    #[inline]
    fn slices(&self) -> Option<Self::Slices> {
        Some((self.0.as_slice()?, self.1.as_slice()?))
    }

    #[inline]
    fn at((a, b): Self::Slices, position: usize) -> Self::Refs {
        (&a[position], &b[position])
    }

    #[inline]
    fn logical(self) -> impl Iterator<Item = Self::Refs> {
        self.0.iter().zip(self.1.iter())
    }
}

impl<'a, A: Sync, B: Sync, C: Sync, const N: usize> ZipInputs<'a, N>
    for (ArrayView<'a, A, N>, ArrayView<'a, B, N>, ArrayView<'a, C, N>)
{
    type Refs = (&'a A, &'a B, &'a C);
    type Slices = (&'a [A], &'a [B], &'a [C]);

    const UNIT_BYTES: usize = size_of::<A>() + size_of::<B>() + size_of::<C>();

    #[inline]
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]) {
        ZipInputs::<N>::assert_shape(&self.0, shape);
        ZipInputs::<N>::assert_shape(&self.1, shape);
        ZipInputs::<N>::assert_shape(&self.2, shape);
    }

    #[inline]
    fn slices(&self) -> Option<Self::Slices> {
        Some((self.0.as_slice()?, self.1.as_slice()?, self.2.as_slice()?))
    }

    #[inline]
    fn at((a, b, c): Self::Slices, position: usize) -> Self::Refs {
        (&a[position], &b[position], &c[position])
    }

    #[inline]
    fn logical(self) -> impl Iterator<Item = Self::Refs> {
        self.0
            .iter()
            .zip(self.1.iter())
            .zip(self.2.iter())
            .map(|((a, b), c)| (a, b, c))
    }
}

impl<'a, A: Sync, B: Sync, C: Sync, D: Sync, const N: usize> ZipInputs<'a, N>
    for (
        ArrayView<'a, A, N>,
        ArrayView<'a, B, N>,
        ArrayView<'a, C, N>,
        ArrayView<'a, D, N>,
    )
{
    type Refs = (&'a A, &'a B, &'a C, &'a D);
    type Slices = (&'a [A], &'a [B], &'a [C], &'a [D]);

    const UNIT_BYTES: usize = size_of::<A>() + size_of::<B>() + size_of::<C>() + size_of::<D>();

    #[inline]
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]) {
        ZipInputs::<N>::assert_shape(&self.0, shape);
        ZipInputs::<N>::assert_shape(&self.1, shape);
        ZipInputs::<N>::assert_shape(&self.2, shape);
        ZipInputs::<N>::assert_shape(&self.3, shape);
    }

    #[inline]
    fn slices(&self) -> Option<Self::Slices> {
        Some((
            self.0.as_slice()?,
            self.1.as_slice()?,
            self.2.as_slice()?,
            self.3.as_slice()?,
        ))
    }

    #[inline]
    fn at((a, b, c, d): Self::Slices, position: usize) -> Self::Refs {
        (&a[position], &b[position], &c[position], &d[position])
    }

    #[inline]
    fn logical(self) -> impl Iterator<Item = Self::Refs> {
        self.0
            .iter()
            .zip(self.1.iter())
            .zip(self.2.iter())
            .zip(self.3.iter())
            .map(|(((a, b), c), d)| (a, b, c, d))
    }
}

impl<'a, A: Sync, B: Sync, C: Sync, D: Sync, E: Sync, const N: usize> ZipInputs<'a, N>
    for (
        ArrayView<'a, A, N>,
        ArrayView<'a, B, N>,
        ArrayView<'a, C, N>,
        ArrayView<'a, D, N>,
        ArrayView<'a, E, N>,
    )
{
    type Refs = (&'a A, &'a B, &'a C, &'a D, &'a E);
    type Slices = (&'a [A], &'a [B], &'a [C], &'a [D], &'a [E]);

    const UNIT_BYTES: usize =
        size_of::<A>() + size_of::<B>() + size_of::<C>() + size_of::<D>() + size_of::<E>();

    #[inline]
    #[track_caller]
    fn assert_shape(&self, shape: [usize; N]) {
        ZipInputs::<N>::assert_shape(&self.0, shape);
        ZipInputs::<N>::assert_shape(&self.1, shape);
        ZipInputs::<N>::assert_shape(&self.2, shape);
        ZipInputs::<N>::assert_shape(&self.3, shape);
        ZipInputs::<N>::assert_shape(&self.4, shape);
    }

    #[inline]
    fn slices(&self) -> Option<Self::Slices> {
        Some((
            self.0.as_slice()?,
            self.1.as_slice()?,
            self.2.as_slice()?,
            self.3.as_slice()?,
            self.4.as_slice()?,
        ))
    }

    #[inline]
    fn at((a, b, c, d, e): Self::Slices, position: usize) -> Self::Refs {
        (
            &a[position],
            &b[position],
            &c[position],
            &d[position],
            &e[position],
        )
    }

    #[inline]
    fn logical(self) -> impl Iterator<Item = Self::Refs> {
        self.0
            .iter()
            .zip(self.1.iter())
            .zip(self.2.iter())
            .zip(self.3.iter())
            .zip(self.4.iter())
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
    }
}
