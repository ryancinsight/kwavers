use super::*;
use kwavers_physics::acoustics::bubble_dynamics::{
    EpsteinPlessetDissolution, GasDiffusionParams, ShellPermeationDissolution,
};
use ndarray::Array3;

fn seeded_field(beta: f64, r0: f64) -> ResidualGasField {
    let mut f = ResidualGasField::new((2, 2, 2), r0);
    let dep = Array3::from_elem((2, 2, 2), beta);
    f.deposit(dep.view());
    f
}

#[test]
fn deposit_accumulates_void_fraction() {
    let f = seeded_field(1e-4, 2e-6);
    assert!((f.peak_void_fraction() - 1e-4).abs() < 1e-12);
    assert!((f.representative_radius() - 2e-6).abs() < 1e-12);
}

#[test]
fn dissolution_decays_void_fraction_to_zero() {
    let mut f = seeded_field(1e-4, 2e-6);
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.0));
    let b0 = f.peak_void_fraction();
    // One short interval: β decreases.
    f.dissolve(1e-3, &model);
    let b1 = f.peak_void_fraction();
    assert!(
        b1 < b0 && b1 >= 0.0,
        "β must decay during dissolution: {b0}->{b1}"
    );
    // A long interval (≫ τ_d) fully clears the cloud.
    f.dissolve(1.0, &model);
    assert!(
        f.peak_void_fraction() < 1e-9,
        "residual gas should fully dissolve"
    );
}

#[test]
fn shelled_cloud_persists_longer_than_free() {
    let params = GasDiffusionParams::air_in_water(0.0);
    let free = EpsteinPlessetDissolution::new(params);
    let shelled = ShellPermeationDissolution::lipid_shell(params);
    let dt = 2e-3;
    let mut f_free = seeded_field(1e-4, 2e-6);
    let mut f_shell = seeded_field(1e-4, 2e-6);
    f_free.dissolve(dt, &free);
    f_shell.dissolve(dt, &shelled);
    assert!(
        f_shell.peak_void_fraction() > f_free.peak_void_fraction(),
        "shelled residual cloud must persist longer: free={}, shell={}",
        f_free.peak_void_fraction(),
        f_shell.peak_void_fraction()
    );
}

#[test]
fn sound_speed_field_collapses_where_gas_present() {
    let f = seeded_field(1e-3, 2e-6);
    let c = f.sound_speed_field(1481.0, 998.0, 343.0, 1.2);
    assert!(
        c.iter().all(|&v| v < 600.0 && v > 0.0),
        "Wood sound speed must collapse where β=1e-3"
    );
    // Empty field → liquid sound speed.
    let empty = ResidualGasField::new((2, 2, 2), 2e-6);
    let c0 = empty.sound_speed_field(1481.0, 998.0, 343.0, 1.2);
    assert!(c0.iter().all(|&v| (v - 1481.0).abs() < 1e-3));
}

#[test]
fn attenuation_field_nonzero_only_with_gas() {
    let f = seeded_field(1e-4, 2e-6);
    let a = f.attenuation_field(1e6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4);
    assert!(a.iter().all(|&v| v >= 0.0) && a.iter().any(|&v| v > 0.0));
    let empty = ResidualGasField::new((2, 2, 2), 2e-6);
    let a0 = empty.attenuation_field(1e6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4);
    assert!(a0.iter().all(|&v| v == 0.0));
}

#[test]
fn total_gas_volume_scales_with_voxel_volume() {
    let f = seeded_field(1e-4, 2e-6);
    let dv = (1e-3_f64).powi(3);
    let expected = 1e-4 * 8.0 * dv; // 2×2×2 voxels
    assert!((f.total_gas_volume(dv) - expected).abs() < 1e-18);
}
