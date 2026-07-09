use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::{GridSource, SourceMode};
use leto::{
    Array2,
    Array3,
};
use std::f64::consts::PI;

pub const NX: usize = 64;
pub const NY: usize = 72;
pub const NZ: usize = 4;
pub const DX: f64 = 0.25e-3;
pub const C0: f64 = 1500.0;
pub const RHO0: f64 = 1000.0;
pub const CFL: f64 = 0.25;
pub const DT: f64 = CFL * DX / C0;
pub const WAVELENGTH_CELLS: f64 = 8.0;
pub const FREQUENCY: f64 = C0 / (WAVELENGTH_CELLS * DX);
pub const OMEGA: f64 = 2.0 * PI * FREQUENCY;
pub const WAVENUMBER: f64 = 2.0 * PI / (WAVELENGTH_CELLS * DX);
pub const NT: usize = 256;
pub const PML: usize = 8;
pub const SOURCE_Y: usize = 12;
pub const FOCUS_X: usize = NX / 2;
pub const FOCUS_Y: usize = 48;
pub const FOCUS_Z: usize = NZ / 2;
pub const ELEMENT_COUNT: usize = 25;
pub const SOURCE_AMPLITUDE_PA: f64 = 1.0e5;
pub const GATE_START: usize = 192;
pub const GATE_END: usize = 248;
pub const FOCAL_WINDOW_CELLS: usize = 12;

#[derive(Debug, Clone, Copy)]
pub struct Element {
    pub x: usize,
    pub y: usize,
    pub phase_rad: f64,
    pub apodization: f64,
}

pub fn grid() -> anyhow::Result<Grid> {
    Ok(Grid::new(NX, NY, NZ, DX, DX, DX)?)
}

pub fn medium(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, grid)
}

pub fn focus_mm() -> (f64, f64) {
    (FOCUS_X as f64 * DX * 1.0e3, FOCUS_Y as f64 * DX * 1.0e3)
}

pub fn elements() -> Vec<Element> {
    let first_x = FOCUS_X - (ELEMENT_COUNT - 1) / 2;
    let reference = distance_to_focus(FOCUS_X, SOURCE_Y);
    (0..ELEMENT_COUNT)
        .map(|n| {
            let x = first_x + n;
            let distance = distance_to_focus(x, SOURCE_Y);
            Element {
                x,
                y: SOURCE_Y,
                phase_rad: WAVENUMBER * (distance - reference),
                apodization: hamming(n, ELEMENT_COUNT),
            }
        })
        .collect()
}

/// Builds the phased aperture source used by both numerical solvers.
///
/// The water-tank fixture is an embedded 2-D comparison: the analytical
/// reference models a line aperture that is invariant in the out-of-plane
/// direction. The numerical mask must therefore occupy every `z` slice in the
/// thin slab. A single center-slice source would radiate with 3-D geometric
/// spreading and compare a different physical problem against the 2-D
/// focused-array envelope.
pub fn focused_source() -> GridSource {
    let elems = elements();
    let mut p_mask = Array3::<f64>::zeros((NX, NY, NZ));
    let mut p_signal = leto::Array2::<f64>::zeros((elems.len() * NZ, NT));

    for z in 0..NZ {
        for (element_index, element) in elems.iter().enumerate() {
            p_mask[[element.x, element.y, z]] = 1.0;
            let row = z * elems.len() + element_index;
            for step in 0..NT {
                let t = step as f64 * DT;
                p_signal[[row, step]] = element_pressure_at(element, t);
            }
        }
    }

    GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    }
}

pub fn focused_aperture_pressure_at(t: f64) -> f64 {
    let elems = elements();
    let apodization_sum = elems
        .iter()
        .map(|element| element.apodization)
        .sum::<f64>()
        .max(f64::EPSILON);
    elems
        .iter()
        .map(|element| element_pressure_at(element, t))
        .sum::<f64>()
        / apodization_sum
}

pub fn analytical_peak_map() -> Array2<f64> {
    let elems = elements();
    Array2::from_shape_fn((NX, NY), |(i, j)| {
        if j <= SOURCE_Y {
            return 0.0;
        }
        let x = i as f64 * DX;
        let y = j as f64 * DX;
        let mut real = 0.0;
        let mut imag = 0.0;
        for element in &elems {
            let ex = element.x as f64 * DX;
            let ey = element.y as f64 * DX;
            let dx = x - ex;
            let dy = y - ey;
            let distance = (dx * dx + dy * dy).sqrt().max(0.5 * DX);
            let phase = element.phase_rad - WAVENUMBER * distance;
            let spreading = (distance / DX).sqrt();
            real += element.apodization * phase.cos() / spreading;
            imag += element.apodization * phase.sin() / spreading;
        }
        (real * real + imag * imag).sqrt()
    })
}

#[cfg(test)]
pub fn focus_phase_residual_rad() -> f64 {
    let elems = elements();
    let reference = elems[0].phase_rad - WAVENUMBER * distance_to_focus(elems[0].x, elems[0].y);
    elems
        .iter()
        .map(|element| {
            let phase_at_focus =
                element.phase_rad - WAVENUMBER * distance_to_focus(element.x, element.y);
            (phase_at_focus - reference).abs()
        })
        .fold(0.0, f64::max)
}

fn distance_to_focus(x_index: usize, y_index: usize) -> f64 {
    let dx = (x_index as f64 - FOCUS_X as f64) * DX;
    let dy = (y_index as f64 - FOCUS_Y as f64) * DX;
    (dx * dx + dy * dy).sqrt()
}

fn hamming(n: usize, len: usize) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    0.54 - 0.46 * (2.0 * PI * n as f64 / (len - 1) as f64).cos()
}

pub fn element_pressure_at(element: &Element, t: f64) -> f64 {
    SOURCE_AMPLITUDE_PA
        * element.apodization
        * tone_burst_ramp(t)
        * (OMEGA * t + element.phase_rad).sin()
}

fn tone_burst_ramp(t: f64) -> f64 {
    let ramp_duration = 2.0 / FREQUENCY;
    if t >= ramp_duration {
        1.0
    } else {
        0.5 * (1.0 - (PI * t / ramp_duration).cos())
    }
}
