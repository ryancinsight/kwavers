/// Generate coordinate arrays for a given grid dimension
///
/// Creates uniformly spaced coordinate arrays for each spatial dimension.
///
/// # Arguments
///
/// * `dims` - Grid dimensions (nx, ny, nz)
/// * `resolution` - Spatial resolution in each dimension [dx, dy, dz]
///
/// # Returns
///
/// Array of coordinate vectors [x_coords, y_coords, z_coords]
pub fn generate_coordinate_arrays(
    dims: (usize, usize, usize),
    resolution: [f64; 3],
) -> [Vec<f64>; 3] {
    let x_coords: Vec<f64> = (0..dims.0).map(|i| i as f64 * resolution[0]).collect();
    let y_coords: Vec<f64> = (0..dims.1).map(|j| j as f64 * resolution[1]).collect();
    let z_coords: Vec<f64> = (0..dims.2).map(|k| k as f64 * resolution[2]).collect();

    [x_coords, y_coords, z_coords]
}
