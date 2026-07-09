mod initialization;
mod pressure_correction;

use leto::Array3;

pub(super) fn zeros_k_mag(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::zeros((nx, ny, nz))
}

pub(super) fn test_k_mag(nx: usize, ny: usize, nz: usize, dk: f64) -> Array3<f64> {
    let mut k = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for kk in 0..nz {
                let ki = if i <= nx / 2 { i } else { nx - i } as f64 * dk;
                let kj = if j <= ny / 2 { j } else { ny - j } as f64 * dk;
                let kkk = if kk <= nz / 2 { kk } else { nz - kk } as f64 * dk;
                k[[i, j, kk]] = (ki * ki + kj * kj + kkk * kkk).sqrt();
            }
        }
    }
    k
}
