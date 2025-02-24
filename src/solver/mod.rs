// solver/mod.rs
use crate::boundary::Boundary;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::{
    mechanics::streaming::StreamingModel,
    mechanics::cavitation::CavitationModel,
    mechanics::acoustic_wave::NonlinearWave,
    chemistry::ChemicalModel,
    optics::diffusion::LightDiffusion,
    thermodynamics::heat_transfer::ThermalModel,
    scattering::acoustic::AcousticScatteringModel,
    heterogeneity::HeterogeneityModel,
};
use crate::recorder::Recorder;
use crate::source::Source;
use crate::time::Time;
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, info};
use ndarray::{Array3, Array4, Axis};
use rustfft::num_complex::Complex;

pub const PRESSURE_IDX: usize = 0;
pub const LIGHT_IDX: usize = 1;
pub const TEMPERATURE_IDX: usize = 2;
pub const BUBBLE_RADIUS_IDX: usize = 3;

#[derive(Debug)]
pub struct SimulationFields {
    pub fields: Array4<f64>,
}

impl SimulationFields {
    pub fn new(num_fields: usize, nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            fields: Array4::zeros((num_fields, nx, ny, nz)),
        }
    }
}

#[derive(Debug)]
pub struct Solver {
    pub grid: Grid,
    pub time: Time,
    pub medium: Box<dyn Medium>,
    pub source: Box<dyn Source>,
    pub boundary: Box<dyn Boundary>,
    pub fields: SimulationFields,
    pub prev_pressure: Array3<f64>,
    pub wave: NonlinearWave,
    pub cavitation: CavitationModel,
    pub light: LightDiffusion,
    pub thermal: ThermalModel,
    pub chemical: ChemicalModel,
    pub streaming: StreamingModel,
    pub scattering: AcousticScatteringModel,
    pub heterogeneity: HeterogeneityModel,
}

impl Solver {
    pub fn new(
        grid: Grid,
        time: Time,
        medium: Box<dyn Medium>,
        source: Box<dyn Source>,
        boundary: Box<dyn Boundary>,
    ) -> Self {
        let num_fields = 4;
        let mut fields = SimulationFields::new(num_fields, grid.nx, grid.ny, grid.nz);
        fields.fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(medium.temperature());
        fields.fields.index_axis_mut(Axis(0), BUBBLE_RADIUS_IDX).assign(medium.bubble_radius());

        info!(
            "Initializing Solver with 4D fields: {} fields, grid {}x{}x{}",
            num_fields, grid.nx, grid.ny, grid.nz
        );
        Self {
            grid: grid.clone(),
            time,
            medium,
            source,
            boundary,
            fields,
            prev_pressure: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            wave: NonlinearWave::new(&grid),
            cavitation: CavitationModel::new(&grid, 10e-6),
            light: LightDiffusion::new(&grid, true, true, true),
            thermal: ThermalModel::new(&grid, 293.15, 1e-6, 1e-6),
            chemical: ChemicalModel::new(&grid, true, true),
            streaming: StreamingModel::new(&grid),
            scattering: AcousticScatteringModel::new(&grid),
            heterogeneity: HeterogeneityModel::new(&grid, 1500.0, 0.05),
        }
    }

    pub fn run(&mut self, recorder: &mut Recorder, frequency: f64) {
        let dt = self.time.dt;
        let n_steps = self.time.n_steps;
        info!(
            "Starting simulation: dt = {:.6e}, steps = {}",
            dt, n_steps
        );

        for step in 0..n_steps {
            let t = step as f64 * dt;
            debug!("Step {}: t = {:.6e}", step, t);

            self.prev_pressure.assign(&self.fields.fields.index_axis(Axis(0), PRESSURE_IDX));
            self.wave.update_wave(
                &mut self.fields.fields,
                &self.prev_pressure,
                self.source.as_ref(),
                &self.grid,
                self.medium.as_ref(),
                dt,
                t,
            );

            let mut p_fft = fft_3d(&self.fields.fields, PRESSURE_IDX, &self.grid);
            self.boundary.apply_acoustic_freq(&mut p_fft, &self.grid, step);
            self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&ifft_3d(&p_fft, &self.grid));

            let pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
            let mut p_update = pressure.clone();
            let light_source = self.cavitation.update_cavitation(
                &mut p_update,
                &pressure,
                &self.grid,
                dt,
                self.medium.as_mut(),
                frequency,
            );
            self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&p_update);
            self.fields.fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&light_source);

            self.light.update_light(
                &mut self.fields.fields,
                &light_source,
                &self.grid,
                self.medium.as_ref(),
                dt,
            );

            let pressure_owned = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
            let light_owned = self.fields.fields.index_axis(Axis(0), LIGHT_IDX).to_owned();
            self.thermal.update_thermal(
                &mut self.fields.fields,
                &self.grid,
                self.medium.as_mut(),
                dt,
                frequency,
            );

            self.chemical.update_chemical(
                &pressure_owned,
                &light_owned,
                self.light.emission_spectrum(),
                &self.fields.fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX).to_owned(),
                &self.thermal.temperature,
                &self.grid,
                dt,
                self.medium.as_mut(),
                frequency,
            );

            self.streaming.update_velocity(&pressure_owned, &self.grid, self.medium.as_ref(), dt);
            self.scattering.compute_scattering(
                &pressure_owned,
                &self.fields.fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX).to_owned(),
                &self.grid,
                self.medium.as_ref(),
                frequency,
            );

            let mut light_field = self.fields.fields.index_axis(Axis(0), LIGHT_IDX).to_owned();
            self.boundary.apply_light(&mut light_field, &self.grid, step);
            self.fields.fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&light_field);

            recorder.record(&self.fields.fields, step, t);

            if step % 10 == 0 {
                let center = [self.grid.nx / 2, self.grid.ny / 2, self.grid.nz / 2];
                info!(
                    "Step {}: Pressure = {:.6e}, Light = {:.6e}, Temp = {:.2}",
                    step,
                    self.fields.fields[[PRESSURE_IDX, center[0], center[1], center[2]]],
                    self.fields.fields[[LIGHT_IDX, center[0], center[1], center[2]]],
                    self.fields.fields[[TEMPERATURE_IDX, center[0], center[1], center[2]]]
                );
            }
        }

        info!("Simulation completed at t = {:.6e}", dt * n_steps as f64);
    }
}

pub mod numerics;