//! Layout-generic point kernels for elastic stress divergence.

use super::fd_stencils::{fd1_flat_x, fd1_flat_y, fd1_flat_z, fd1_x, fd1_y, fd1_z};
use leto::ArrayView3;

pub(super) trait StencilField: Copy {
    fn value(self, point: StencilPoint) -> f64;

    fn dx(self, point: StencilPoint) -> f64;

    fn dy(self, point: StencilPoint) -> f64;

    fn dz(self, point: StencilPoint) -> f64;
}

#[derive(Clone, Copy)]
pub(super) struct StencilPoint {
    pub(super) index: usize,
    pub(super) position: [usize; 3],
    pub(super) shape: [usize; 3],
    pub(super) spacing: [f64; 3],
}

#[derive(Clone, Copy)]
pub(super) struct FlatField<'a>(pub(super) &'a [f64]);

impl StencilField for FlatField<'_> {
    #[inline]
    fn value(self, point: StencilPoint) -> f64 {
        self.0[point.index]
    }

    #[inline]
    fn dx(self, point: StencilPoint) -> f64 {
        let [i, _, _] = point.position;
        let [nx, ny, nz] = point.shape;
        fd1_flat_x(self.0, point.index, i, nx, ny * nz, point.spacing[0])
    }

    #[inline]
    fn dy(self, point: StencilPoint) -> f64 {
        let [_, j, _] = point.position;
        let [_, ny, nz] = point.shape;
        fd1_flat_y(self.0, point.index, j, ny, nz, point.spacing[1])
    }

    #[inline]
    fn dz(self, point: StencilPoint) -> f64 {
        let [_, _, k] = point.position;
        let [_, _, nz] = point.shape;
        fd1_flat_z(self.0, point.index, k, nz, point.spacing[2])
    }
}

#[derive(Clone, Copy)]
pub(super) struct StridedField<'a>(pub(super) ArrayView3<'a, f64>);

impl StencilField for StridedField<'_> {
    #[inline]
    fn value(self, point: StencilPoint) -> f64 {
        let [i, j, k] = point.position;
        self.0[[i, j, k]]
    }

    #[inline]
    fn dx(self, point: StencilPoint) -> f64 {
        let [i, j, k] = point.position;
        fd1_x(self.0, i, j, k, point.shape[0], point.spacing[0])
    }

    #[inline]
    fn dy(self, point: StencilPoint) -> f64 {
        let [i, j, k] = point.position;
        fd1_y(self.0, i, j, k, point.shape[1], point.spacing[1])
    }

    #[inline]
    fn dz(self, point: StencilPoint) -> f64 {
        let [i, j, k] = point.position;
        fd1_z(self.0, i, j, k, point.shape[2], point.spacing[2])
    }
}

#[inline]
pub(super) fn stress_components<F: StencilField>(fields: [F; 5], point: StencilPoint) -> [f64; 6] {
    let [ux, uy, uz, lambda, mu] = fields;
    let exx = ux.dx(point);
    let eyy = uy.dy(point);
    let ezz = uz.dz(point);
    let exy_2 = ux.dy(point) + uy.dx(point);
    let exz_2 = ux.dz(point) + uz.dx(point);
    let eyz_2 = uy.dz(point) + uz.dy(point);
    let la = lambda.value(point);
    let mv = mu.value(point);
    let la2mu = 2.0f64.mul_add(mv, la);
    [
        la2mu.mul_add(exx, la * (eyy + ezz)),
        mv * exy_2,
        mv * exz_2,
        la2mu.mul_add(eyy, la * (exx + ezz)),
        mv * eyz_2,
        la2mu.mul_add(ezz, la * (exx + eyy)),
    ]
}

#[inline]
pub(super) fn divergence_components<F: StencilField>(
    fields: [F; 6],
    point: StencilPoint,
) -> [f64; 3] {
    let [sxx, sxy, sxz, syy, syz, szz] = fields;
    [
        sxx.dx(point) + sxy.dy(point) + sxz.dz(point),
        sxy.dx(point) + syy.dy(point) + syz.dz(point),
        sxz.dx(point) + syz.dy(point) + szz.dz(point),
    ]
}
