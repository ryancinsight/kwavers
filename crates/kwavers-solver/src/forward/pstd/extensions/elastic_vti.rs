//! VTI elastic PSTD extension.
//!
//! Implements a vertically transversely isotropic stress-velocity update with
//! constant stiffness coefficients and pointwise density, advanced via
//! pseudospectral derivatives.

use std::any::Any;
use std::sync::Arc;

use leto::{Array3, Array4};

use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};
use kwavers_medium::Medium;

#[derive(Debug, Clone)]
pub struct VtiConfig {
    pub c11: f64,
    pub c13: f64,
    pub c33: f64,
    pub c44: f64,
    pub c66: f64,
    pub density: Array3<f64>,
}

#[derive(Debug)]
pub struct VtiElasticSolver {
    config: VtiConfig,
    pub vx: Array3<f64>,
    pub vy: Array3<f64>,
    pub vz: Array3<f64>,
    pub sxx: Array3<f64>,
    pub syy: Array3<f64>,
    pub szz: Array3<f64>,
    pub sxz: Array3<f64>,
    fft: Arc<Fft3d>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    spectral_field: Array3<Complex64>,
    spectral_scratch: Array3<Complex64>,
    dvx_dx: Array3<f64>,
    dvy_dy: Array3<f64>,
    dvz_dz: Array3<f64>,
    dvx_dz: Array3<f64>,
    dvz_dx: Array3<f64>,
    dsxx_dx: Array3<f64>,
    dsyy_dy: Array3<f64>,
    dszz_dz: Array3<f64>,
    dsxz_dx: Array3<f64>,
    dsxz_dz: Array3<f64>,
}

fn k_vector(n: usize, spacing: f64) -> Vec<f64> {
    let dk = kwavers_core::constants::numerical::TWO_PI / (n as f64 * spacing);
    (0..n)
        .map(|i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum SpectralAxis {
    X,
    Y,
    Z,
}

impl SpectralAxis {
    fn multiplier_index(self, linear_index: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => linear_index / (ny * nz),
            Self::Y => (linear_index / nz) % ny,
            Self::Z => linear_index % nz,
        }
    }
}

fn spectral_derivative(
    fft: &Fft3d,
    field: &Array3<f64>,
    multiplier: &[f64],
    axis: SpectralAxis,
    spectral_field: &mut Array3<Complex64>,
    spectral_scratch: &mut Array3<Complex64>,
    out: &mut Array3<f64>,
) {
    fft.forward_into(field, spectral_field);

    let ny = field.shape()[1];
    let nz = field.shape()[2];
    if let Some(values) = spectral_field.as_slice_mut() {
        for (index, value) in values.iter_mut().enumerate() {
            let axis_index = axis.multiplier_index(index, ny, nz);
            *value *= Complex64::new(0.0, multiplier[axis_index]);
        }
    } else {
        let [nx, ny, nz] = field.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let factor = match axis {
                        SpectralAxis::X => multiplier[i],
                        SpectralAxis::Y => multiplier[j],
                        SpectralAxis::Z => multiplier[k],
                    };
                    spectral_field[[i, j, k]] *= Complex64::new(0.0, factor);
                }
            }
        }
    }

    fft.inverse_into(spectral_field, out, spectral_scratch);
}

impl VtiElasticSolver {
    #[must_use]
    pub fn new(grid: &Grid, config: VtiConfig) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            config,
            vx: Array3::zeros(shape),
            vy: Array3::zeros(shape),
            vz: Array3::zeros(shape),
            sxx: Array3::zeros(shape),
            syy: Array3::zeros(shape),
            szz: Array3::zeros(shape),
            sxz: Array3::zeros(shape),
            fft: get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            kx: k_vector(grid.nx, grid.dx),
            ky: k_vector(grid.ny, grid.dy),
            kz: k_vector(grid.nz, grid.dz),
            spectral_field: Array3::zeros(shape),
            spectral_scratch: Array3::zeros(shape),
            dvx_dx: Array3::zeros(shape),
            dvy_dy: Array3::zeros(shape),
            dvz_dz: Array3::zeros(shape),
            dvx_dz: Array3::zeros(shape),
            dvz_dx: Array3::zeros(shape),
            dsxx_dx: Array3::zeros(shape),
            dsyy_dy: Array3::zeros(shape),
            dszz_dz: Array3::zeros(shape),
            dsxz_dx: Array3::zeros(shape),
            dsxz_dz: Array3::zeros(shape),
        }
    }

    pub fn step(&mut self, dt: f64, grid: &Grid) {
        spectral_derivative(
            self.fft.as_ref(),
            &self.vx,
            &self.kx,
            SpectralAxis::X,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dvx_dx,
        );
        if grid.ny > 1 {
            spectral_derivative(
                self.fft.as_ref(),
                &self.vy,
                &self.ky,
                SpectralAxis::Y,
                &mut self.spectral_field,
                &mut self.spectral_scratch,
                &mut self.dvy_dy,
            );
        } else {
            self.dvy_dy.fill(0.0);
        }
        spectral_derivative(
            self.fft.as_ref(),
            &self.vz,
            &self.kz,
            SpectralAxis::Z,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dvz_dz,
        );
        spectral_derivative(
            self.fft.as_ref(),
            &self.vx,
            &self.kz,
            SpectralAxis::Z,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dvx_dz,
        );
        spectral_derivative(
            self.fft.as_ref(),
            &self.vz,
            &self.kx,
            SpectralAxis::X,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dvz_dx,
        );

        let c12 = self.config.c11 - 2.0 * self.config.c66;
        for (((((((sxx, syy), szz), sxz), &dvx_dx), &dvy_dy), &dvz_dz), (&dvx_dz, &dvz_dx)) in self
            .sxx
            .iter_mut()
            .zip(self.syy.iter_mut())
            .zip(self.szz.iter_mut())
            .zip(self.sxz.iter_mut())
            .zip(self.dvx_dx.iter())
            .zip(self.dvy_dy.iter())
            .zip(self.dvz_dz.iter())
            .zip(self.dvx_dz.iter().zip(self.dvz_dx.iter()))
        {
            *sxx += dt
                * (self.config.c11 * dvx_dx + c12 * dvy_dy + self.config.c13 * dvz_dz);
            *syy += dt
                * (c12 * dvx_dx + self.config.c11 * dvy_dy + self.config.c13 * dvz_dz);
            *szz += dt
                * (self.config.c13 * (dvx_dx + dvy_dy) + self.config.c33 * dvz_dz);
            *sxz += dt * self.config.c44 * (dvx_dz + dvz_dx);
        }

        spectral_derivative(
            self.fft.as_ref(),
            &self.sxx,
            &self.kx,
            SpectralAxis::X,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dsxx_dx,
        );
        if grid.ny > 1 {
            spectral_derivative(
                self.fft.as_ref(),
                &self.syy,
                &self.ky,
                SpectralAxis::Y,
                &mut self.spectral_field,
                &mut self.spectral_scratch,
                &mut self.dsyy_dy,
            );
        } else {
            self.dsyy_dy.fill(0.0);
        }
        spectral_derivative(
            self.fft.as_ref(),
            &self.szz,
            &self.kz,
            SpectralAxis::Z,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dszz_dz,
        );
        spectral_derivative(
            self.fft.as_ref(),
            &self.sxz,
            &self.kx,
            SpectralAxis::X,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dsxz_dx,
        );
        spectral_derivative(
            self.fft.as_ref(),
            &self.sxz,
            &self.kz,
            SpectralAxis::Z,
            &mut self.spectral_field,
            &mut self.spectral_scratch,
            &mut self.dsxz_dz,
        );

        for (((((((vx, vy), vz), &rho), &dsxx_dx), &dsyy_dy), &dszz_dz), (&dsxz_dx, &dsxz_dz)) in self
            .vx
            .iter_mut()
            .zip(self.vy.iter_mut())
            .zip(self.vz.iter_mut())
            .zip(self.config.density.iter())
            .zip(self.dsxx_dx.iter())
            .zip(self.dsyy_dy.iter())
            .zip(self.dszz_dz.iter())
            .zip(self.dsxz_dx.iter().zip(self.dsxz_dz.iter()))
        {
            if rho <= 0.0 {
                continue;
            }
            let inv_rho = dt / rho;
            *vx += inv_rho * (dsxx_dx + dsxz_dz);
            *vy += inv_rho * dsyy_dy;
            *vz += inv_rho * (dsxz_dx + dszz_dz);
        }
    }

    #[must_use]
    pub fn pressure_field(&self) -> Array3<f64> {
        let mut pressure = Array3::zeros(self.sxx.shape());
        for (((p, &sxx), &syy), &szz) in pressure
            .iter_mut()
            .zip(self.sxx.iter())
            .zip(self.syy.iter())
            .zip(self.szz.iter())
        {
            *p = -(sxx + syy + szz) / 3.0;
        }
        pressure
    }
}

#[derive(Debug)]
pub struct VtiElasticPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    dt: f64,
    c11: f64,
    c13: f64,
    c33: f64,
    c44: f64,
    c66: f64,
    solver: Option<VtiElasticSolver>,
}

impl VtiElasticPlugin {
    #[must_use]
    pub fn new(dt: f64, c11: f64, c13: f64, c33: f64, c44: f64, c66: f64) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "mechanical_stress_vti".to_owned(),
                name: "Mechanical Stress (VTI PSTD)".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Vertical transverse isotropic elastic PSTD propagator".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            dt,
            c11,
            c13,
            c33,
            c44,
            c66,
            solver: None,
        }
    }
}

impl Plugin for VtiElasticPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        Vec::new()
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
            UnifiedFieldType::StressXX,
            UnifiedFieldType::StressYY,
            UnifiedFieldType::StressZZ,
            UnifiedFieldType::StressXZ,
        ]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let solver = VtiElasticSolver::new(
            grid,
            VtiConfig {
                c11: self.c11,
                c13: self.c13,
                c33: self.c33,
                c44: self.c44,
                c66: self.c66,
                density: medium.density_array().to_contiguous(),
            },
        );
        self.solver = Some(solver);
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            KwaversError::InternalError("VtiElasticPlugin updated before initialize()".to_owned())
        })?;
        solver.step(self.dt, grid);
        let pressure = solver.pressure_field();

        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field exists")
            .assign(&pressure);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::VelocityX.index())
            .expect("invariant: velocity-x field exists")
            .assign(&solver.vx);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::VelocityY.index())
            .expect("invariant: velocity-y field exists")
            .assign(&solver.vy);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::VelocityZ.index())
            .expect("invariant: velocity-z field exists")
            .assign(&solver.vz);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::StressXX.index())
            .expect("invariant: stress-xx field exists")
            .assign(&solver.sxx);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::StressYY.index())
            .expect("invariant: stress-yy field exists")
            .assign(&solver.syy);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::StressZZ.index())
            .expect("invariant: stress-zz field exists")
            .assign(&solver.szz);
        fields
            .index_axis_mut::<3>(0, UnifiedFieldType::StressXZ.index())
            .expect("invariant: stress-xz field exists")
            .assign(&solver.sxz);

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
