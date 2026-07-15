use std::convert::TryInto;

use leto::{Array2 as LetoArray2, Array3 as LetoArray3};
use numpy::ndarray::{Array2 as NdArray2, Array3 as NdArray3};

pub fn nd_to_leto2<T: Clone>(arr: NdArray2<T>) -> LetoArray2<T> {
    arr.into()
}

pub fn nd_to_leto3<T: Clone>(arr: NdArray3<T>) -> LetoArray3<T> {
    arr.into()
}

pub fn leto2_to_nd2<T>(arr: LetoArray2<T>) -> NdArray2<T> {
    arr.try_into().expect("contiguous")
}

pub fn leto3_to_nd3<T>(arr: LetoArray3<T>) -> NdArray3<T> {
    arr.try_into().expect("contiguous")
}
