//! Shared raster primitives for seismic example artifacts.

use std::io::{self, BufWriter};
use std::path::Path;

/// Write one RGB pixel into a flat byte buffer.
pub(crate) fn put_pixel(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    color: [u8; 3],
) {
    if x >= width || y >= height {
        return;
    }
    let idx = 3 * (y * width + x);
    rgb[idx..idx + 3].copy_from_slice(&color);
}

/// Map sound speed to a five-stop blue-to-red velocity map.
pub(crate) fn velocity_color(c: f64, c_lo: f64, c_hi: f64) -> [u8; 3] {
    let t = ((c - c_lo) / (c_hi - c_lo)).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    [(255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8]
}

/// Encode a flat RGB buffer as a PNG artifact.
pub(crate) fn write_png(path: &Path, rgb: &[u8], width: usize, height: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| io::Error::other(error.to_string()))?;
    writer
        .write_image_data(rgb)
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{put_pixel, velocity_color};

    #[test]
    fn pixel_writer_preserves_bounds_and_value() {
        let mut rgb = [0_u8; 12];
        put_pixel(&mut rgb, 2, 2, 1, 0, [1, 2, 3]);
        put_pixel(&mut rgb, 2, 2, 2, 0, [9, 9, 9]);

        assert_eq!(&rgb[3..6], &[1, 2, 3]);
        assert_eq!(rgb, [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn velocity_map_has_stable_endpoint_colors() {
        assert_eq!(velocity_color(0.0, 0.0, 1.0), [0, 0, 255]);
        assert_eq!(velocity_color(1.0, 0.0, 1.0), [255, 0, 0]);
    }
}
