use kwavers::math::fft::ProcessorFft3d;
use kwavers::math::fft::Complex64;
use ndarray::Array3;

#[test]
fn test_fft_peak_scaling() {
    let nx = 96; let ny = 64; let nz = 64;
    let mut processor = ProcessorFft3d::new(nx, ny, nz);
    let mut real_in = Array3::<f64>::zeros((nx, ny, nz));
    real_in[[nx/2, ny/2, nz/2]] = 1.0;
    
    let mut complex_out = Array3::<Complex64>::zeros((nx, ny, nz));
    processor.forward_into(&real_in, &mut complex_out);
    
    let dx = 1e-3; let dt = 2e-7; let c = 1500.0;
    
    for i in 0..nx {
        let kx = if i <= nx/2 { i as f64 * (2.0 * std::f64::consts::PI / (nx as f64 * dx)) } else { (i as f64 - nx as f64) * (2.0 * std::f64::consts::PI / (nx as f64 * dx)) };
        for j in 0..ny {
            let ky = if j <= ny/2 { j as f64 * (2.0 * std::f64::consts::PI / (ny as f64 * dx)) } else { (j as f64 - ny as f64) * (2.0 * std::f64::consts::PI / (ny as f64 * dx)) };
            for k in 0..nz {
                let kz = if k <= nz/2 { k as f64 * (2.0 * std::f64::consts::PI / (nz as f64 * dx)) } else { (k as f64 - nz as f64) * (2.0 * std::f64::consts::PI / (nz as f64 * dx)) };
                let k_mag = (kx*kx + ky*ky + kz*kz).sqrt();
                let x = 0.5 * c * dt * k_mag;
                complex_out[[i, j, k]] *= Complex64::new(x.cos(), 0.0);
            }
        }
    }
    
    let mut real_out = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
    processor.inverse_into(&complex_out, &mut real_out, &mut scratch);
    
    let peak = real_out.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    println!("Peak without filter: 1.0");
    println!("Peak with cos filter: {}", peak);
}
