use super::error::GasTransferError;
use super::grid::RadialGrid;
use aequitas::systems::si::quantities::{AmountOfSubstance, Length, MolarConcentration, Time};
use eunomia::FloatElement;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_physics::acoustics::bubble_dynamics::gas_transfer::{
    spherical_diffusion_metric, GasDiffusionWall, SphericalGasDiffusion,
};
/// Gas exchanged across the two boundaries during one step.
#[derive(Debug, Clone, Copy)]
pub struct GasTransfer<T = f64> {
    /// Signed amount received by the bubble.
    pub into_bubble: AmountOfSubstance<T>,
    /// Signed amount passed from the liquid domain into the far reservoir.
    pub into_reservoir: AmountOfSubstance<T>,
}

/// A successful solve that has yet to replace the accepted liquid field.
pub(super) struct DiffusionTrial<'a, T> {
    solver: &'a mut TransientGasDiffusion<T>,
    exchange: GasTransfer<T>,
    radius: T,
    boundary_excess: T,
}

impl<T: FloatElement> DiffusionTrial<'_, T> {
    pub(super) fn exchange(&self) -> GasTransfer<T> {
        self.exchange
    }

    pub(super) fn commit(self) -> GasTransfer<T> {
        std::mem::swap(&mut self.solver.liquid, &mut self.solver.trial);
        self.solver.radius = self.radius;
        self.solver.boundary_excess = self.boundary_excess;
        self.solver.has_advanced = true;
        self.exchange
    }
}

/// Finite-volume solution of the full moving spherical diffusion equation.
///
/// The spatial scheme uses material cells and a shared flux per face. A
/// midpoint flux balances the change in cell inventories at second order
/// in time. The first step uses backward Euler to damp any initial boundary
/// discontinuity. A failed step leaves the liquid field and wall state intact.
pub struct TransientGasDiffusion<T = f64> {
    reference: T,
    bulk: T,
    time_scale_inverse: T,
    volume_scale: T,
    grid: RadialGrid<T>,
    radius: T,
    boundary_excess: T,
    has_advanced: bool,
    liquid: Vec<T>,
    trial: Vec<T>,
    upper: Vec<T>,
}

impl<T: FloatElement> TransientGasDiffusion<T> {
    /// Initialize a liquid domain uniformly at the bulk concentration and
    /// take the initial interface radius from the reference radius.
    ///
    /// `reservoir_extent` is the initial wall-to-boundary distance.
    /// `initial_concentration` is the liquid concentration at the inner
    /// boundary, which may differ from the uniform initial liquid interior.
    ///
    /// # Errors
    /// Rejects non-finite inputs, non-positive length scales or diffusivity,
    /// negative concentrations, fewer than two cells, or an unrepresentable grid.
    pub fn new(
        problem: SphericalGasDiffusion<T>,
        reservoir_extent: Length<T>,
        cells: usize,
        initial_concentration: MolarConcentration<T>,
    ) -> Result<Self, GasTransferError> {
        let reference = positive(problem.reference_radius.into_base(), "reference radius")?;
        let diffusivity = positive(problem.diffusivity.into_base(), "diffusivity")?;
        let extent = positive(reservoir_extent.into_base(), "reservoir extent")?;
        let bulk = concentration(problem.bulk_concentration.into_base(), "bulk concentration")?;
        let wall = concentration(initial_concentration.into_base(), "wall concentration")?;
        if cells < 2 {
            return Err(GasTransferError::InvalidInput("at least two radial cells"));
        }
        let time_scale_inverse = diffusivity / (reference * reference);
        let volume_scale = T::from_f64(FOUR_PI) * reference * reference * reference;
        if [time_scale_inverse, volume_scale]
            .into_iter()
            .any(|x| !x.is_finite() || x <= T::ZERO)
        {
            return Err(GasTransferError::Unrepresentable);
        }
        Ok(Self {
            reference,
            bulk,
            time_scale_inverse,
            volume_scale,
            grid: RadialGrid::new(extent / reference, cells)?,
            radius: T::ONE,
            boundary_excess: wall - bulk,
            has_advanced: false,
            liquid: vec![T::ZERO; cells],
            trial: vec![T::ZERO; cells],
            upper: vec![T::ZERO; cells],
        })
    }

    /// Excess dissolved gas inventory relative to the bulk concentration.
    #[must_use]
    pub fn liquid_excess(&self) -> AmountOfSubstance<T> {
        let excess = self
            .liquid
            .iter()
            .zip(&self.grid.widths)
            .fold(T::ZERO, |sum, (&value, &width)| sum + value * width);
        AmountOfSubstance::from_base(self.volume_scale * excess)
    }

    /// Evolve the liquid field to the next wall sample and return interval
    /// exchanges in moles. Continue calling through the acoustic off interval.
    ///
    /// Choose a time step that resolves changes in the supplied motion
    /// and boundary concentration.
    ///
    /// # Errors
    /// Rejects an invalid interval, radius, or concentration, or arithmetic
    /// that cannot represent a valid update. A negative reconstructed
    /// concentration requires a shorter time step.
    pub fn advance(
        &mut self,
        interval: Time<T>,
        wall: GasDiffusionWall<T>,
    ) -> Result<GasTransfer<T>, GasTransferError> {
        Ok(self.trial(interval, wall)?.commit())
    }

    pub(super) fn trial(
        &mut self,
        interval: Time<T>,
        wall: GasDiffusionWall<T>,
    ) -> Result<DiffusionTrial<'_, T>, GasTransferError> {
        let interval = positive(interval.into_base(), "time interval")?;
        let radius = positive(wall.radius.into_base(), "wall radius")? / self.reference;
        if !radius.is_finite() || radius <= T::ZERO {
            return Err(GasTransferError::Unrepresentable);
        }
        let boundary =
            concentration(wall.concentration.into_base(), "wall concentration")? - self.bulk;
        let (radius_mid, boundary_mid, implicit_fraction) = if self.has_advanced {
            let half = T::from_f64(0.5);
            (
                self.radius * half + radius * half,
                self.boundary_excess * half + boundary * half,
                half,
            )
        } else {
            (radius, boundary, T::ONE)
        };
        let time = interval * self.time_scale_inverse;
        let scale_radius = radius_mid * radius_mid * radius_mid;
        if [time, scale_radius]
            .into_iter()
            .any(|x| !x.is_finite() || x <= T::ZERO)
        {
            return Err(GasTransferError::Unrepresentable);
        }
        let exchange = self.solve_flux(time, scale_radius, boundary_mid, implicit_fraction)?;
        Ok(DiffusionTrial {
            solver: self,
            exchange,
            radius,
            boundary_excess: boundary,
        })
    }

    fn solve_flux(
        &mut self,
        time: T,
        radius_cubed: T,
        boundary: T,
        implicit: T,
    ) -> Result<GasTransfer<T>, GasTransferError> {
        let cells = self.liquid.len();
        let explicit = T::ONE - implicit;
        let wall_metric = spherical_diffusion_metric(radius_cubed, T::ZERO);
        let [wall_first, wall_second] = self.grid.inner_weights.map(|w| w * wall_metric);
        let mut left = T::ZERO;
        let mut rhs_previous = T::ZERO;
        // Factor and solve for concentration at the time quadrature point.
        // Each row is h_j q_j − θ∆τ(F_right(q) − F_left(q)) = h_j c_j.
        // The row sum before imposing the boundary is h_j > 0. The wall
        // weight on the second cell is negative, so the coupled off-diagonal
        // entries are nonpositive and the wall adds a positive row sum.
        for j in 0..cells {
            let metric = spherical_diffusion_metric(radius_cubed, self.grid.faces[j + 1]);
            let right = metric * self.grid.inverse_distances[j];
            let width = self.grid.widths[j];
            let size = implicit * time;
            let (diagonal, right_neighbor, boundary_term) = if j == 0 {
                (
                    right + wall_first,
                    right - wall_second,
                    (wall_first + wall_second) * boundary,
                )
            } else {
                (left + right, right, T::ZERO)
            };
            let pivot = width
                + size * diagonal
                + if j == 0 {
                    T::ZERO
                } else {
                    size * left * self.upper[j - 1]
                };
            if pivot <= T::ZERO || !pivot.is_finite() {
                return Err(GasTransferError::Unrepresentable);
            }
            let rhs = width * self.liquid[j] + size * (boundary_term + left * rhs_previous);
            self.upper[j] = (T::ZERO - size * right_neighbor) / pivot;
            self.trial[j] = rhs / pivot;
            rhs_previous = self.trial[j];
            left = right;
        }
        for j in (0..cells - 1).rev() {
            let neighbor = self.trial[j + 1];
            self.trial[j] -= self.upper[j] * neighbor;
        }

        let wall_flux =
            wall_first * (self.trial[0] - boundary) + wall_second * (self.trial[1] - boundary);
        let reservoir_flux = left * self.trial[cells - 1];
        for (trial, &old) in self.trial.iter_mut().zip(&self.liquid) {
            *trial = (*trial - explicit * old) / implicit;
        }
        let to_moles = self.volume_scale * time;
        let wall_amount = to_moles * wall_flux;
        let reservoir_amount = to_moles * reservoir_flux;
        if !wall_amount.is_finite()
            || !reservoir_amount.is_finite()
            || self.trial.iter().any(|value| !value.is_finite())
        {
            return Err(GasTransferError::Unrepresentable);
        }
        if self.trial.iter().any(|&c| c < T::ZERO - self.bulk) {
            return Err(GasTransferError::NegativeConcentration);
        }
        Ok(GasTransfer {
            into_bubble: AmountOfSubstance::from_base(wall_amount),
            into_reservoir: AmountOfSubstance::from_base(reservoir_amount),
        })
    }
}

fn positive<T: FloatElement>(value: T, parameter: &'static str) -> Result<T, GasTransferError> {
    if value.is_finite() && value > T::ZERO {
        Ok(value)
    } else {
        Err(GasTransferError::InvalidInput(parameter))
    }
}

fn concentration<T: FloatElement>(
    value: T,
    parameter: &'static str,
) -> Result<T, GasTransferError> {
    if value.is_finite() && value >= T::ZERO {
        Ok(value)
    } else {
        Err(GasTransferError::InvalidInput(parameter))
    }
}
