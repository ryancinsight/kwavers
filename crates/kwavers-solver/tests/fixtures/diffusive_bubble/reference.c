/*
 * Independent spherical Rayleigh-Plesset and gas diffusion reference.
 * For s = r - R(t), C_t = D C_ss + [2D/(R+s) + R_dot*(1 - (R/(R+s))^2)] C_s.
 * Use nodal finite differences on an exponential mesh and implicit BDF4 with
 * a backward Euler start. Use a one-sided derivative for the gas mass flux.
 * Advance the gas amount with the same BDF rule and extrapolate it into the endpoint wall
 * and Henry pressure solves.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double radius;
    double velocity;
    double fraction_change;
} Sample;

static const double frequency = 26600.0;
static const double diffusivity = 2e-9;
static const double ambient = 101325.0;
static const double sigma = 0.0728;
static const double solubility = 0.26;
static const double temperature = 293.15;
static const double gas_constant = 8.314462618;
static const double vapor = 2338.0;
static const double density = 1000.0;
static const double viscosity = 0.001002;

static double gas_pressure(Sample sample, double r0, double p0, double kappa) {
    double inverse_radius = r0/sample.radius;
    return p0*pow((1+sample.fraction_change)*inverse_radius*inverse_radius*inverse_radius, kappa);
}

static double pressure_offset(double cycle, double amplitude) {
    const double pi = acos(-1);
    const double on_time = cycle < 16 ? cycle : cycle - 32;
    if (on_time < 0 || on_time >= 16) {
        return 0;
    }
    const double distance = fmin(fmin(on_time, 16-on_time), 2);
    const double envelope = (1-cos(pi*distance/2))/2;
    return amplitude*envelope*sin(2*pi*cycle);
}

typedef struct {
    double alpha;
    double weights[4];
    double extrapolation[4];
} Formula;

static const Formula formulas[] = {
    {1, {1,0,0,0}, {1,0,0,0}},
    {1.5, {2,-0.5,0,0}, {2,-1,0,0}},
    {11.0/6, {3,-1.5,1.0/3,0}, {3,-3,1,0}},
    {25.0/12, {4,-3,4.0/3,-0.25}, {4,-6,4,-1}},
};

static Sample mechanics(const Sample history[4], Formula formula, double dt,
                        double r0, double p0, double drive, double kappa) {
    double mass = 0, base_r = 0, base_v = 0;
    for (int j = 0; j < 4; ++j) {
        mass += formula.extrapolation[j]*history[j].fraction_change;
        // Take history relative to equilibrium to avoid cancellation.
        base_r += formula.weights[j]*(history[j].radius-r0);
        base_v += formula.weights[j]*history[j].velocity;
    }
    Sample guess = {history[0].radius,history[0].velocity,mass};
    for (int iteration = 0; iteration < 24; ++iteration) {
        const double r = guess.radius, v = guess.velocity;
        const double dp = gas_pressure(guess,r0,p0,kappa) - (ambient-vapor)
            - drive - 2*sigma/r - 4*viscosity*v/r;
        const double a = dp/(density*r)-1.5*v*v/r;
        const double radius = r0+(base_r+dt*v)/formula.alpha;
        const double velocity = (base_v+dt*a)/formula.alpha;
        const int converged = fabs(radius-r) <= 4e-16*r0 && fabs(velocity-v) <= 2e-14;
        guess = (Sample){radius,velocity,mass};
        if (converged) {
            return guess;
        }
    }
    fprintf(stderr,"Mechanics step did not converge\n");
    exit(4);
}

typedef struct {
    double distance;
    double left_d1, right_d1;
    double left_d2, right_d2;
} Node;

static Node stencil(double previous, double center, double next) {
    double left = center-previous, right = next-center;
    return (Node){center, -right/(left*(left+right)), left/(right*(left+right)),
                  2/(left*(left+right)), 2/(right*(left+right))};
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s radius_um radial_intervals steps_per_cycle extent_um pressure_amplitude_pa kappa\n",
                argv[0]);
        return 2;
    }
    const double r0 = strtod(argv[1], NULL)*1e-6;
    const long intervals = strtol(argv[2], NULL, 10);
    const long steps = strtol(argv[3], NULL, 10);
    const double extent = strtod(argv[4], NULL)*1e-6;
    const double amplitude = strtod(argv[5], NULL);
    const double kappa = strtod(argv[6], NULL);
    if (!(r0 > 0 && intervals >= 4 && steps >= 16 && extent > 0 && amplitude > 0
            && amplitude < ambient && kappa >= 1)) {
        return 2;
    }
    const double pi = acos(-1);
    const double dt = 1/(steps*frequency);
    const double p0 = ambient-vapor+2*sigma/r0;
    const double reference_moles = (4*pi/3)*r0*r0*r0*p0/(gas_constant*temperature);
    const double henry = solubility/(ambient-vapor);
    const double conversion = 60e6*r0/(3-2*sigma/(kappa*r0*p0));

    double *distance = calloc(intervals+1, sizeof(*distance));
    Node *node = calloc(intervals, sizeof(*node));
    double *field[4];
    for (int j = 0; j < 4; ++j) {
        field[j] = calloc(intervals, sizeof(*field[j]));
        if (!field[j]) {
            return 3;
        }
    }
    double *next = calloc(intervals, sizeof(*next));
    double *upper = calloc(intervals, sizeof(*upper));
    if (!distance || !node || !next || !upper) {
        return 3;
    }
    const double grading = 8;
    for (long j = 0; j <= intervals; ++j) {
        distance[j] = extent*expm1(grading*j/intervals)/expm1(grading);
    }
    for (long j = 1; j < intervals; ++j) {
        node[j] = stencil(distance[j-1], distance[j], distance[j+1]);
    }
    double wall_first = distance[2]/(distance[1]*(distance[2]-distance[1]));
    double wall_second = -distance[1]/(distance[2]*(distance[2]-distance[1]));
    Sample history[4] = {{r0,0,0}, {r0,0,0}, {r0,0,0}, {r0,0,0}};
    double amount_at_cycle_start = 0;
    printf("cycle,phase,rate_um_per_min,exchanged_mol,radius_m,velocity_m_per_s\n");
    for (long t = 1; t <= 72*steps; ++t) {
        double cycle = (double)t/steps;
        const Formula formula = formulas[t < 4 ? t-1 : 3];
        Sample driving = mechanics(history,formula,dt,r0,p0,pressure_offset(cycle,amplitude),kappa);
        const double wall = henry*(gas_pressure(driving,r0,p0,kappa)-p0);
        next[0] = wall;
        for (long j = 1; j < intervals; ++j) {
            const Node n = node[j];
            double r = driving.radius + n.distance;
            double fraction = driving.radius/r;
            double drift = 2*diffusivity/r + driving.velocity*(1-fraction*fraction);
            double left = dt*(diffusivity*n.left_d2 + drift*n.left_d1);
            double right = dt*(diffusivity*n.right_d2 + drift*n.right_d1);
            double pivot = formula.alpha+left+right+left*upper[j-1];
            double rhs = formula.weights[0]*field[0][j] + formula.weights[1]*field[1][j]
                + formula.weights[2]*field[2][j] + formula.weights[3]*field[3][j];
            upper[j] = -right/pivot;
            next[j] = (rhs + left*next[j-1])/pivot;
        }
        for (long j = intervals-2; j >= 1; --j) {
            next[j] -= upper[j]*next[j+1];
        }
        double gradient = wall_first*(next[1]-wall) + wall_second*(next[2]-wall);
        double uptake = 4*pi*diffusivity*driving.radius*driving.radius*gradient;
        double amount_rhs = dt*uptake/reference_moles;
        for (int j = 0; j < 4; ++j) {
            amount_rhs += formula.weights[j]*history[j].fraction_change;
        }
        driving.fraction_change = amount_rhs/formula.alpha;
        if (t % steps == 0) {
            long cycle_index = t/steps;
            int on = cycle_index <= 16 || (cycle_index > 32 && cycle_index <= 48);
            double delta = driving.fraction_change-amount_at_cycle_start;
            printf("%ld,%s,%.14g,%.14g,%.14g,%.14g\n",cycle_index,on?"on":"off",
                   conversion*frequency*delta,reference_moles*delta,driving.radius,driving.velocity);
            amount_at_cycle_start = driving.fraction_change;
        }
        double *reusable = field[3];
        for (int j = 3; j > 0; --j) {
            field[j] = field[j-1];
            history[j] = history[j-1];
        }
        field[0] = next;
        history[0] = driving;
        next = reusable;
    }
    free(distance);
    free(node);
    for (int j = 0; j < 4; ++j) {
        free(field[j]);
    }
    free(next);
    free(upper);
    return 0;
}
