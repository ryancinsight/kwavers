use std::convert::TryInto;

use kwavers_math::fft::Complex64 as KwComplex;
use leto::{Array1 as LetoArray1, Array2 as LetoArray2, Array3 as LetoArray3};
use leto::{
    Array1,
    Array2,
    Array3,
};

pub fn nc_to_ec1(arr: Array1<eunomia::Complex64>) -> Array1<KwComplex> {
    arr.map(|c| KwComplex::new(c.re, c.im))
}

pub fn ec_to_nc1(arr: Array1<KwComplex>) -> Array1<eunomia::Complex64> {
    arr.map(|c| eunomia::Complex64::new(c.re, c.im))
}

pub fn nc_to_ec2(arr: Array2<eunomia::Complex64>) -> Array2<KwComplex> {
    arr.map(|c| KwComplex::new(c.re, c.im))
}

pub fn ec_to_nc2(arr: Array2<KwComplex>) -> Array2<eunomia::Complex64> {
    arr.map(|c| eunomia::Complex64::new(c.re, c.im))
}

pub fn nc_to_ec3(arr: Array3<eunomia::Complex64>) -> Array3<KwComplex> {
    arr.map(|c| KwComplex::new(c.re, c.im))
}

pub fn ec_to_nc3(arr: Array3<KwComplex>) -> Array3<eunomia::Complex64> {
    arr.map(|c| eunomia::Complex64::new(c.re, c.im))
}

pub fn nd_to_leto1<T: Clone>(arr: Array1<T>) -> LetoArray1<T> {
    arr.into()
}

pub fn nd_to_leto2<T: Clone>(arr: Array2<T>) -> LetoArray2<T> {
    arr.into()
}

pub fn nd_to_leto3<T: Clone>(arr: Array3<T>) -> LetoArray3<T> {
    arr.into()
}

pub fn leto1_to_nd1<T>(arr: LetoArray1<T>) -> Array1<T> {
    arr.try_into().expect("contiguous")
}

pub fn leto2_to_nd2<T>(arr: LetoArray2<T>) -> Array2<T> {
    arr.try_into().expect("contiguous")
}

pub fn leto3_to_nd3<T>(arr: LetoArray3<T>) -> Array3<T> {
    arr.try_into().expect("contiguous")
}

