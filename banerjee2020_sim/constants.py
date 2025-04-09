import numpy as np


PARTICLE_MASS = 7.754657e+10 # Particle mass [Msun/h]

# Cosmology
COSMO_PARAMS = {
    'flat': True,
    'H0': 70,
    'Om0': 0.3,
    'Ob0': 0.0469,
    'sigma8': 0.8355,
    'ns': 1
}

BOX_SIZE = 1_000 # Simulation box length [Mpc/h]

CRIT_DENSITY = 2.77536627e+11 # Critical density [h^2 Msun/Mpc^3]

MEAN_DENSITY = 1e-26 # Critical density [h^2 Msun/Mpc^3]

PERCENT_PARTICLES = 10 # Percent of particles taken from the simulation 

N_RADIAL_BINS = 20 # Number of radial bins to use 

# Define radial bins that logarithmically go from R_SOFT to R_MAX
R_SOFT = 0.015 # Softening length [Mpc/h]
R_MAX = 5 # Max halo radius [Mpc/h]
RADIUS_BINS = np.logspace(np.log10(R_SOFT), np.log10(R_MAX), num=N_RADIAL_BINS+1, base=10)
RADIUS = 0.5 * (RADIUS_BINS[1:] + RADIUS_BINS[:-1]) # Radial bin middle points [Mpc/h]
VOLUME = (4/3) * np.pi * np.diff(RADIUS_BINS ** 3)

MASS_BIN_EDGES = np.array([13.40, 13.55, 13.70, 13.85, 14.00, 14.15, 14.30, 14.45, 14.65, 15.00]) # Halo mass bins
MASS_BIN_STRS = ['13.40-13.55', '13.55-13.70', '13.70-13.85', '13.85-14.00', '14.00-14.15', '14.15-14.30', '14.30-14.45', '14.45-14.65', '14.65-15.00']
MASS = 0.5 * (MASS_BIN_EDGES[1:] + MASS_BIN_EDGES[:-1])

G_GRAV = 4.3e-09 # Newton's gravitational constant [km^2 Mpc Msun^-1 s^-2]

MEMBSIZE = int(10 * 1000**3) # Particle file size

# Colorblind-friendly color cycle
COLORS = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']


# ORBITING MODEL (Salazar et al. 2024)
# Slopes and pivots for the Morb power laws of rh (halo radius) and alpha_inf (asymptotic slope)
RH_S = 0.226
RH_P = 0.8403 # Mpc/h
ALPHA_S = -0.050
ALPHA_P = 2.018
INNER_SCALING = 0.037 # a
M_P = 1e+14 # Msun/h, pivot mass


# HALO FITTING
FITTING_MASK = np.where(RADIUS > (6 * R_SOFT))

# SHIELDS ET AL. (2025)
LNR_BIN_EDGES = np.array([-1.18129078, -0.53366139, -0.29854228, -0.06342316,  0.17169595,
        0.40681506,  1.09861229]) # Minimum lnR, mean - (2 * std), mean - std, mean, mean + std, mean + (2*std), max lnR

S_ALPHA = 1.0016474457286733 # Slope of the best-fit line to lnR vs. delta alpha infinity (fig. 2)
ALPHA_0 = 0.04933732028434024 # Y-intercept of the best-fit line to lnR vs. delta alpha infinity (fig. 2)
S_ARF = -1.06050296107201 # Slope of the best-fit line to R vs. aRF (fig. 4)
ARF_0 = 1.967430451024661 # Y-intercept of the best-fit line to R vs. aRF (fig. 4)
