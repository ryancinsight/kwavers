/*
 * Independent moving spherical diffusion reference.
 * Coordinate s = r - R(t) stays a physical distance from the wall:
 *
 *   C_t = D C_ss + [2D/(R+s) + R_dot*(1 - (R/(R+s))^2)] C_s.
 *
 * Use nodal finite differences on an exponential mesh and implicit BDF2.
 * Bubble uptake is computed from the one-sided wall derivative. Integrate
 * the uptake using the same BDF2 rule as the concentration.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double radius;
    double velocity;
    double wall;
} Sample;

static const double frequency = 26600.0;
static const double diffusivity = 2e-9;
static const double ambient = 101325.0;
static const double sigma = 0.0728;
static const double solubility = 0.26;
static const double temperature = 293.15;
static const double gas_constant = 8.314462618;

static double bulk(double initial_radius) {
    return solubility * (1 + 2*sigma/(initial_radius*ambient));
}

static Sample sample(double cycle, double r0, double amplitude, double kappa) {
    const double pi = acos(-1);
    double envelope = 0;
    double slope = 0;
    if ((cycle < 16) || (cycle >= 32 && cycle < 48)) {
        double on_time = cycle < 16 ? cycle : cycle - 32;
        double distance = fmin(on_time, 16 - on_time);
        if (distance < 2) {
            envelope = (1 - cos(pi*distance/2))/2;
            slope = pi*sin(pi*distance/2)/4 * (on_time < 8 ? 1 : -1);
        } else {
            envelope = 1;
        }
    }
    double phase = 2*pi*cycle;
    double radius = r0*(1 - amplitude*envelope*sin(phase));
    double velocity = -r0*amplitude*frequency
        * (slope*sin(phase) + 2*pi*envelope*cos(phase));
    double wall = bulk(r0) * pow(r0/radius, 3*kappa);
    return (Sample){radius, velocity, wall};
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
        fprintf(stderr, "usage: %s radius_um radial_intervals steps_per_cycle extent_um epsilon kappa\n",
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
            && amplitude < 1 && kappa >= 1)) {
        return 2;
    }
    const double pi = acos(-1);
    const double dt = 1/(steps*frequency);
    const double conversion = 60e6 * gas_constant*temperature /
        (4*pi*r0*r0*(ambient+4*sigma/(3*r0)));

    double *distance = calloc(intervals+1, sizeof(*distance));
    Node *node = calloc(intervals, sizeof(*node));
    double *previous = calloc(intervals, sizeof(*previous));
    double *current = calloc(intervals, sizeof(*current));
    double *next = calloc(intervals, sizeof(*next));
    double *upper = calloc(intervals, sizeof(*upper));
    if (!distance || !node || !previous || !current || !next || !upper) {
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
    double prior_amount = 0, amount = 0, amount_at_cycle_start = 0;
    printf("cycle,phase,rate_um_per_min,exchanged_mol\n");
    for (long t = 1; t <= 72*steps; ++t) {
        double cycle = (double)t/steps;
        Sample driving = sample(cycle,r0,amplitude,kappa);
        double wall = driving.wall - bulk(r0);
        double alpha = t == 1 ? 1 : 1.5;
        next[0] = wall;
        for (long j = 1; j < intervals; ++j) {
            const Node n = node[j];
            double r = driving.radius + n.distance;
            double fraction = driving.radius/r;
            double drift = 2*diffusivity/r + driving.velocity*(1-fraction*fraction);
            double left = dt*(diffusivity*n.left_d2 + drift*n.left_d1);
            double right = dt*(diffusivity*n.right_d2 + drift*n.right_d1);
            double pivot = alpha+left+right+left*upper[j-1];
            double rhs = t == 1 ? current[j] : 2*current[j]-0.5*previous[j];
            upper[j] = -right/pivot;
            next[j] = (rhs + left*next[j-1])/pivot;
        }
        for (long j = intervals-2; j >= 1; --j) {
            next[j] -= upper[j]*next[j+1];
        }
        double gradient = wall_first*(next[1]-wall) + wall_second*(next[2]-wall);
        double uptake = 4*pi*diffusivity*driving.radius*driving.radius*gradient;
        double exchange = t == 1 ? dt*uptake : (2*amount - 0.5*prior_amount + dt*uptake)/alpha;
        prior_amount = amount;
        amount = exchange;
        if (t % steps == 0) {
            long cycle_index = t/steps;
            int on = cycle_index <= 16 || (cycle_index > 32 && cycle_index <= 48);
            double delta = amount-amount_at_cycle_start;
            printf("%ld,%s,%.12g,%.12g\n",cycle_index,on?"on":"off",
                   conversion*frequency*delta,delta);
            amount_at_cycle_start = amount;
        }
        double *reusable = previous;
        previous = current;
        current = next;
        next = reusable;
    }
    free(distance);
    free(node);
    free(previous);
    free(current);
    free(next);
    free(upper);
    return 0;
}
