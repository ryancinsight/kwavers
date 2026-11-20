pub(super) fn angle_difference(a: f64, b: f64) -> f64 {
    (a - b).sin().atan2((a - b).cos())
}

pub(super) fn square_grid_len(len: usize) -> Result<usize, String> {
    let n = (len as f64).sqrt() as usize;
    if n < 2 || n * n != len {
        return Err(format!(
            "phantom arrays must form a square grid, got length {len}"
        ));
    }
    Ok(n)
}
