//! Moirai-backed iteration for strided RTM 3-D views.

use moirai_parallel::{for_each_index_with, Adaptive};
use ndarray::ArrayViewMut3;

#[derive(Clone, Copy)]
struct StridedMutView {
    base: *mut f64,
    strides: [isize; 3],
}

// SAFETY: `StridedMutView` is constructed only from an exclusive
// `ArrayViewMut3`. `for_each_view_mut` schedules each logical index exactly
// once, so the raw pointer is split into disjoint mutable element references.
unsafe impl Send for StridedMutView {}

// SAFETY: shared access to the pointer wrapper is sound for the same reason as
// `Send`: every scheduled logical index maps to one unique mutable element.
unsafe impl Sync for StridedMutView {}

impl StridedMutView {
    unsafe fn ptr_at(self, i: usize, j: usize, k: usize) -> *mut f64 {
        let offset = i as isize * self.strides[0]
            + j as isize * self.strides[1]
            + k as isize * self.strides[2];
        // SAFETY: callers provide logical indices inside the source view's
        // shape, and ndarray strides map them into the same allocation.
        unsafe { self.base.offset(offset) }
    }
}

pub(super) fn for_each_view_mut<F>(mut view: ArrayViewMut3<'_, f64>, f: F)
where
    F: Fn((usize, usize, usize), &mut f64) + Send + Sync,
{
    let (nx, ny, nz) = view.dim();
    let len = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .expect("invariant: RTM 3-D view length must fit usize");
    let strides = {
        let strides = view.strides();
        [strides[0], strides[1], strides[2]]
    };
    let view_ptr = StridedMutView {
        base: view.as_mut_ptr(),
        strides,
    };
    let f = &f;

    for_each_index_with::<Adaptive, _>(len, move |linear| {
        let plane = ny * nz;
        let i = linear / plane;
        let rem = linear % plane;
        let j = rem / nz;
        let k = rem % nz;
        // SAFETY: `linear` is unique for this invocation, and row-major logical
        // index decoding covers every element in the mutable view exactly once.
        f((i, j, k), unsafe { &mut *view_ptr.ptr_at(i, j, k) });
    });
}
