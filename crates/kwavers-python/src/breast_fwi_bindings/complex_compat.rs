use std::convert::TryInto;

use leto::{Array1 as LetoArray1, Array2 as LetoArray2, Array3 as LetoArray3};
use numpy::ndarray::{Array1 as NdArray1, Array2 as NdArray2, Array3 as NdArray3};

pub fn nd_to_leto1<T: Clone>(arr: NdArray1<T>) -> LetoArray1<T> {
    arr.into()
}

pub fn nd_to_leto2<T: Clone>(arr: NdArray2<T>) -> LetoArray2<T> {
    arr.into()
}

pub fn nd_to_leto3<T: Clone>(arr: NdArray3<T>) -> LetoArray3<T> {
    arr.into()
}

pub fn leto1_to_nd1<T>(arr: LetoArray1<T>) -> NdArray1<T> {
    arr.try_into().expect("contiguous")
}

pub fn leto2_to_nd2<T>(arr: LetoArray2<T>) -> NdArray2<T> {
    arr.try_into().expect("contiguous")
}

pub fn leto3_to_nd3<T>(arr: LetoArray3<T>) -> NdArray3<T> {
    arr.try_into().expect("contiguous")
}
