use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::medium::Medium;

fn main() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0)
        .with_acoustic_absorption(0.5, 0.0);
    
    let alpha = medium.absorption_coefficient(0.0, 0.0, 0.0, &grid, 1e6);
    println!("Absorption coefficient: {}", alpha);
}
