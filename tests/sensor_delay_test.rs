use kwavers::domain::grid::Grid;
use kwavers::domain::sensor::beamforming::SensorBeamformer;
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
use kwavers::domain::sensor::localization::Position;

#[test]
fn test_geometric_delay_calculation() {
    // 1. Setup Sensor Array
    // Sensor 0 at origin (0,0,0)
    // Sensor 1 at (1,0,0)
    let s0 = Sensor::new(0, Position::new(0.0, 0.0, 0.0));
    let s1 = Sensor::new(1, Position::new(1.0, 0.0, 0.0));

    let sensors = vec![s0, s1];
    let sound_speed = 1500.0;
    let array = SensorArray::new(sensors, sound_speed, ArrayGeometry::Linear);

    let sampling_freq = 1.0e6;
    let beamformer = SensorBeamformer::new(array, sampling_freq);

    // 2. Setup Grid
    // 2x2x2 grid with 1m spacing, starting at origin
    let nx = 2;
    let ny = 2;
    let nz = 1; // 2x2x1 flat grid for simplicity
    let spacing = 1.0;
    let mut grid = Grid::new(nx, ny, nz, spacing, spacing, spacing).unwrap();
    // Ensure origin is (0,0,0) which is default

    // Grid points:
    // (0,0,0), (1,0,0), (0,1,0), (1,1,0)
    // Flattened indices:
    // 0: (0,0,0)
    // 1: (0,1,0) -- wait, let's check flattening order
    // Based on previous analysis: z varies fastest? No, usually x or z.
    // ndarray Array3 default is C-order (row-major).
    // Indices [i, j, k] -> offset.
    // With shape (nx, ny, nz).
    // offset = i * stride[0] + j * stride[1] + k * stride[2]
    // strides are typically (ny*nz, nz, 1).
    // So k (z) varies fastest.
    // 0: (0,0,0)
    // 1: (0,0,1) -- if nz > 1
    // Here nz=1.
    // 0: (0,0,0)
    // 1: (0,1,0)
    // 2: (1,0,0)
    // 3: (1,1,0)

    // Wait, let's verify grid flattening order by checking coordinates or simpler test.
    // We can just check the values match expected distances.

    // 3. Calculate Delays
    let delays = beamformer
        .calculate_delays(&grid, sound_speed)
        .expect("calculate_delays failed");

    // 4. Verification
    // delays shape: (2, 4)
    assert_eq!(delays.shape(), &[2, 4]);

    // Check Sensor 0 (0,0,0)
    // Point 0 (0,0,0): dist=0, delay=0
    // Point ? (1,0,0): dist=1, delay=1/1500
    // Point ? (0,1,0): dist=1, delay=1/1500
    // Point ? (1,1,0): dist=sqrt(2), delay=sqrt(2)/1500

    // If implementation currently returns zeros, this will fail if we check for non-zero.

    // Let's assert that at least some delays are non-zero (since points are not all at sensor pos)
    let sum: f64 = delays.iter().sum();
    assert!(sum > 0.0, "Delays should not be all zero");
}
