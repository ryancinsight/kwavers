// solver/numerics/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
// Import concrete types only if needed for construction/specific functions not using traits.
// For now, assuming LightDiffusion is still concrete here.
use crate::physics::optics::diffusion::LightDiffusion;
use crate::physics::traits::{CavitationModelBehavior, ThermalModelTrait, ChemicalModelTrait};
use log::{debug, warn};
use ndarray::{Array3, Array4, Axis, Zip};

const TOLERANCE: f64 = 1e-8;
const SAFETY_FACTOR: f64 = 0.85;
const MIN_DT_FACTOR: f64 = 0.05;
const MAX_DT_FACTOR: f64 = 1.5;
const MIN_DT: f64 = 1e-14;
const MAX_ATTEMPTS: usize = 20;

const A: [[f64; 5]; 6] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
    [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
    [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
    [
        -8.0 / 27.0,
        2.0,
        -3544.0 / 2565.0,
        1859.0 / 4104.0,
        -11.0 / 40.0,
    ],
];
const B4: [f64; 5] = [
    25.0 / 216.0,
    0.0,
    1408.0 / 2565.0,
    2197. / 4104.0,
    -1.0 / 5.0,
];
const B5: [f64; 6] = [
    16.0 / 135.0,
    0.0,
    6656.0 / 12825.0,
    28561.0 / 56430.0,
    -9.0 / 50.0,
    2.0 / 55.0,
];

#[allow(clippy::too_many_arguments)]
pub fn hybrid_step(
    p_new: &mut Array3<f64>,
    p: &Array3<f64>,
    _p_old: &Array3<f64>,
    cavitation: &mut dyn CavitationModelBehavior,
    light: &mut LightDiffusion,
    thermal: &mut dyn ThermalModelTrait,
    chemical: &mut dyn ChemicalModelTrait, // Changed to trait object
    fields: &mut Array4<f64>,
    grid: &Grid,
    dt_wave: f64,
    t: f64,
    frequency: f64,
    medium: &mut dyn Medium,
) -> Array3<f64> {
    debug!("Hybrid step at t = {:.6e}, dt_wave = {:.6e}", t, dt_wave);

    let mut dt_adaptive = dt_wave;
    let mut attempts = 0;
    let mut light_source = Array3::zeros(p.dim());

    loop {
        attempts += 1;
        debug!(
            "Adaptive attempt {} with dt = {:.6e}",
            attempts, dt_adaptive
        );

        let initial_radius = cavitation.radius().clone();
        let initial_velocity = cavitation.velocity().clone();
        let initial_temp = thermal.temperature().clone();
        let initial_chem = chemical.radical_concentration().clone();
        let initial_light = fields.index_axis(Axis(0), 1).to_owned();

        let mut k_radius: [Array3<f64>; 6] = [0; 6].map(|_| Array3::zeros(p.dim()));
        let mut k_light: [Array3<f64>; 6] = [0; 6].map(|_| Array3::zeros(p.dim()));
        let mut k_temp: [Array3<f64>; 6] = [0; 6].map(|_| Array3::zeros(p.dim()));
        let mut k_chem: [Array3<f64>; 6] = [0; 6].map(|_| Array3::zeros(p.dim()));

        for stage in 0..6 {
            let _stage_dt = A[stage].iter().sum::<f64>() * dt_adaptive;

            if stage > 0 {
                let mut radius_temp = initial_radius.clone();
                let mut light_temp = initial_light.clone();
                let mut temp_temp = initial_temp.clone();
                let mut chem_temp = initial_chem.clone();

                for i in 0..stage {
                    let factor = A[stage][i] * dt_adaptive;
                    Zip::from(&mut radius_temp)
                        .and(&k_radius[i])
                        .for_each(|r, &k| *r += k * factor);
                    Zip::from(&mut light_temp)
                        .and(&k_light[i])
                        .for_each(|l, &k| *l += k * factor);
                    Zip::from(&mut temp_temp)
                        .and(&k_temp[i])
                        .for_each(|t, &k| *t += k * factor);
                    Zip::from(&mut chem_temp)
                        .and(&k_chem[i])
                        .for_each(|c, &k| *c += k * factor);
                }

                medium.update_bubble_state(&radius_temp, &initial_velocity);
                fields.index_axis_mut(Axis(0), 1).assign(&light_temp);
                thermal.set_temperature(&temp_temp); // Use trait method
            }

            let mut p_temp = p_new.clone();
            k_radius[stage] = cavitation.update_cavitation(
                &mut p_temp,
                p,
                grid,
                dt_adaptive,
                medium,
                frequency,
            );
            let mut light_fields = fields.clone();
            // Assuming light is still concrete here, if it becomes Box<dyn Trait> this call needs to adapt
            light.update_light(
                &mut light_fields,
                &k_radius[stage],
                grid,
                medium,
                dt_adaptive,
            );
            fields.assign(&light_fields);
            k_light[stage] = fields.index_axis(Axis(0), 1).to_owned();
            thermal.update_thermal(fields, grid, medium, dt_adaptive, frequency);
            k_temp[stage] = thermal.temperature().clone();
            chemical.update_chemical(
                &p_temp,
                &k_light[stage],
                &Array3::zeros(k_light[stage].dim()),
                cavitation.radius(),
                thermal.temperature(),
                grid,
                dt_adaptive,
                medium,
                frequency,
            );
            k_chem[stage] = chemical.radical_concentration().clone();

            Zip::from(&mut k_light[stage])
                .and(&initial_light)
                .for_each(|k, &init| *k -= init);
            Zip::from(&mut k_temp[stage])
                .and(&initial_temp)
                .for_each(|k, &init| *k -= init);
            Zip::from(&mut k_chem[stage])
                .and(&initial_chem)
                .for_each(|c, &init| *c -= init);
        }

        let mut radius_4th = initial_radius.clone();
        let mut radius_5th = initial_radius.clone();
        let mut light_4th = initial_light.clone();
        let mut light_5th = initial_light.clone();
        let mut temp_4th = initial_temp.clone();
        let mut temp_5th = initial_temp.clone();
        let mut chem_4th = initial_chem.clone();
        let mut chem_5th = initial_chem.clone();

        for i in 0..5 {
            Zip::from(&mut radius_4th)
                .and(&k_radius[i])
                .for_each(|r, &k| *r += k * B4[i] * dt_adaptive);
            Zip::from(&mut light_4th)
                .and(&k_light[i])
                .for_each(|l, &k| *l += k * B4[i] * dt_adaptive);
            Zip::from(&mut temp_4th)
                .and(&k_temp[i])
                .for_each(|t, &k| *t += k * B4[i] * dt_adaptive);
            Zip::from(&mut chem_4th)
                .and(&k_chem[i])
                .for_each(|c, &k| *c += k * B4[i] * dt_adaptive);
        }
        for i in 0..6 {
            Zip::from(&mut radius_5th)
                .and(&k_radius[i])
                .for_each(|r, &k| *r += k * B5[i] * dt_adaptive);
            Zip::from(&mut light_5th)
                .and(&k_light[i])
                .for_each(|l, &k| *l += k * B5[i] * dt_adaptive);
            Zip::from(&mut temp_5th)
                .and(&k_temp[i])
                .for_each(|t, &k| *t += k * B5[i] * dt_adaptive);
            Zip::from(&mut chem_5th)
                .and(&k_chem[i])
                .for_each(|c, &k| *c += k * B5[i] * dt_adaptive);
        }

        let error_radius =
            (&radius_5th - &radius_4th).mapv(|x| x * x).sum().sqrt() / radius_5th.len() as f64;
        let error_light =
            (&light_5th - &light_4th).mapv(|x| x * x).sum().sqrt() / light_5th.len() as f64;
        let error_temp =
            (&temp_5th - &temp_4th).mapv(|x| x * x).sum().sqrt() / temp_5th.len() as f64;
        let error_chem =
            (&chem_5th - &chem_4th).mapv(|x| x * x).sum().sqrt() / chem_5th.len() as f64;
        let error = error_radius
            .max(error_light)
            .max(error_temp)
            .max(error_chem);

        let scale = SAFETY_FACTOR * (TOLERANCE / error.max(f64::EPSILON)).powf(0.25);
        let new_dt = dt_adaptive * scale.clamp(MIN_DT_FACTOR, MAX_DT_FACTOR);

        if error <= TOLERANCE || dt_adaptive <= MIN_DT || attempts >= MAX_ATTEMPTS {
            if attempts >= MAX_ATTEMPTS {
                warn!(
                    "Max attempts ({}) reached at t = {:.6e}, error = {:.6e}",
                    MAX_ATTEMPTS, t, error
                );
            }
            cavitation.set_radius(&radius_5th);
            fields.index_axis_mut(Axis(0), 1).assign(&light_5th);
            thermal.set_temperature(&temp_5th); // Use trait method
            light_source.assign(&k_radius[0]);
            medium.update_bubble_state(&radius_5th, &initial_velocity);
            medium.update_temperature(&temp_5th);
            break;
        } else {
            debug!("Adaptive step rejected: error = {:.6e}", error);
            dt_adaptive = new_dt.min(dt_wave);
            cavitation.set_radius(&initial_radius);
            fields.index_axis_mut(Axis(0), 1).assign(&initial_light);
            thermal.set_temperature(&initial_temp); // Use trait method
            medium.update_bubble_state(&initial_radius, &initial_velocity);
            medium.update_temperature(&initial_temp);
        }
    }

    light_source
}