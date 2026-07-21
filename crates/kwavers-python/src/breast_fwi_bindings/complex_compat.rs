use leto::{Array1 as LetoArray1, Array2 as LetoArray2, Array3 as LetoArray3};
use numpy::ndarray::{Array1 as NdArray1, Array2 as NdArray2, Array3 as NdArray3};

pub fn nd_to_leto1<T: Clone>(arr: NdArray1<T>) -> LetoArray1<T> {
    let shape = arr.shape();
    let data: Vec<T> = arr.iter().cloned().collect();
    LetoArray1::from_shape_vec(shape[0], data).expect("valid shape")
}

pub fn nd_to_leto2<T: Clone>(arr: NdArray2<T>) -> LetoArray2<T> {
    let shape = arr.shape();
    let data: Vec<T> = arr.iter().cloned().collect();
    LetoArray2::from_shape_vec([shape[0], shape[1]], data).expect("valid shape")
}

pub fn nd_to_leto3<T: Clone>(arr: NdArray3<T>) -> LetoArray3<T> {
    let shape = arr.shape();
    let data: Vec<T> = arr.iter().cloned().collect();
    LetoArray3::from_shape_vec([shape[0], shape[1], shape[2]], data).expect("valid shape")
}

pub fn leto1_to_nd1<T: Clone>(arr: LetoArray1<T>) -> NdArray1<T> {
    let shape = arr.shape();
    NdArray1::from_shape_vec([shape[0]], arr.into_vec()).expect("valid shape")
}

pub fn leto2_to_nd2<T: Clone>(arr: LetoArray2<T>) -> NdArray2<T> {
    let shape = arr.shape();
    NdArray2::from_shape_vec([shape[0], shape[1]], arr.into_vec()).expect("valid shape")
}

pub fn leto3_to_nd3<T: Clone>(arr: LetoArray3<T>) -> NdArray3<T> {
    let shape = arr.shape();
    NdArray3::from_shape_vec([shape[0], shape[1], shape[2]], arr.into_vec()).expect("valid shape")
}
