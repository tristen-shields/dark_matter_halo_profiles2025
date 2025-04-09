from .loading import *

from scipy.integrate import quad
import matplotlib.pyplot as plt
# from numba import jit


def rh_st(Morb):
    '''
    Stacked profile halo radius given from power law in Salazar et al. (2024).
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
    '''
    return RH_P * ((Morb / M_P) ** RH_S)


def alpha_inf_st(Morb):
    '''
    Stacked profile asymptotic slope given from power law in Salazar et al. (2024).
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
    '''
    return ALPHA_P * ((Morb / M_P) ** ALPHA_S)


def compute_density(part_distances, vol_factor=1):
    '''
    Counts the number of particles in the halo, N, then finds the total mass in bins of volume.
        part_distances: Particle distances from the center of the halo (1darray)
        vol_factor(=1): Fraction of halo's volume to use in the density calculation (int/float)
    '''
    N, __ = np.histogram(part_distances, RADIUS_BINS) # Number of particles in each radial bin
    M = N * PARTICLE_MASS # Total mass of the particles

    return M / (vol_factor * VOLUME)


def stacked_parameters(Morb):
    '''
    Power laws for rh (halo radius) and alpha_inf (asymptotic slope) from the stacked profile in Salazar et al. (2024).
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
    '''
    rh = RH_P * ((Morb / M_P) ** RH_S)
    alpha_inf = ALPHA_P * ((Morb / M_P) ** ALPHA_S)
    return rh, alpha_inf


def orb_model(r, Morb, rh=False, alpha_inf=False):
    '''
    Orbiting density fitting function from Salazar et al. (2024).
        r: Radius [Mpc/h] (float)
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
        rh(=False): Halo radius parameter [Mpc/h] (float)
        alpha_inf(=False): Asymptotic slope of the halo's profile (float)
    '''
    if rh == False:
        rh, __ = stacked_parameters(Morb)
    if alpha_inf == False:
        __, alpha_inf = stacked_parameters(Morb)
    
    A = normalize_orb_model(Morb, rh=rh, alpha_inf=alpha_inf)
    x = r / rh # Dimensionless radius
    alpha = (alpha_inf * x) / (x + INNER_SCALING)
    return A * ((x / INNER_SCALING) ** (-alpha)) * np.exp(- (x ** 2) / 2)


def orb_model_integrand(r, Morb, rh=False, alpha_inf=False):
    '''
    Integrand for normalize_orb_model(), i.e. 4*pi*r^2*orb_model().
        r: Dummy variable (integrated over by scipy)
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
        rh(=False): Halo radius parameter [Mpc/h] (float)
        alpha_inf(=False): Asymptotic slope of the halo's profile (float)
    '''
    if rh == False:
        rh, __ = stacked_parameters(Morb)
    if alpha_inf == False:
        __, alpha_inf = stacked_parameters(Morb)
    
    x = r / rh # Dimensionless radius
    alpha = (alpha_inf * x) / (x + INNER_SCALING)
    return 4 * np.pi * (r ** 2) * ((x / INNER_SCALING) ** (-alpha)) * np.exp(- (x ** 2) / 2)


def normalize_orb_model(Morb, rh=False, alpha_inf=False):
    '''
    Finds normalization constant for orb_model(), s.t. the model gives the halo's Morb when integrated to infinity.
        r: Radius [Mpc/h] (float)
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
        rh(=False): Halo radius parameter [Mpc/h] (float)
        alpha_inf(=False): Asymptotic slope of the halo's profile (float)
    '''
    integrand = quad(orb_model_integrand, 0, np.inf, args=(Morb, rh, alpha_inf))[0]
    return Morb / integrand


def orb_model_tilde_integrand(x, alpha_inf):
    '''
    Integrand to use to solve for orb_model_tilde().
        alpha_inf: Asymptotic slope of the halo's profile (float)
    '''
    alpha = (alpha_inf * x) / (x + INNER_SCALING)
    return (x ** 2) * ((x / INNER_SCALING) ** -alpha) * np.exp(-(x ** 2) / 2)


def orb_model_tilde(orb_dens, Morb, rh, alpha_inf):
    '''
    Takes in data for a halo's orbiting density, as well as its best fit rh and alpha_inf parameters, and rewrites the orb_model() to be of the form exp(-x^2/2) 
    (see Shields et al. 2025).
        orb_dens: Orbiting density of the halo [Msun*Mpc^{-3}] (1darray, length of RADIUS)
        Morb: Mass of halo's orbiting particles [Msun/h] (float)
        rh(=False): Halo radius parameter [Mpc/h] (float)
        alpha_inf(=False): Asymptotic slope of the halo's profile (float)

    Returns:
        orb_model_tilde: Halo's profile in the form exp(-x^2/2) (1darray, length of RADIUS)
        x: r/rh for the halo (float)
    '''
    x = RADIUS / rh
    I = quad(orb_model_tilde_integrand, 0, np.inf, args=alpha_inf)[0]
    A_tilde = Morb / (4 * np.pi * (rh ** 3) * I)
    alpha = (alpha_inf * x) / (x + INNER_SCALING)
    orb_model_tilde = (1 / A_tilde) * ((x / INNER_SCALING) ** alpha) * orb_dens 
    return orb_model_tilde, x


def chi_squared(data, model, delta=0.05):
    '''
    Chi squared cost function with a Poisson error term, a fractional error term, and 1 (prevents division by 0) all in the denominator. Converts the data and model 
    (presumed to be densities) to number counts.
        data: Data array (1darray)
        model: Model array (1darray, same shape as data)
        delta(=0.05): Assumed fractional error in the data (float)
    '''
    data = (VOLUME[FITTING_MASK] / PARTICLE_MASS) * data[FITTING_MASK]
    model = (VOLUME[FITTING_MASK] / PARTICLE_MASS) * model[FITTING_MASK]

    return np.sum(((data - model) ** 2) / (((delta * data) ** 2) + data + 1))


def cost(x, data, Morb, fit_keyword, delta=0.05):
    '''
    Cost function to be minimized with Iminuit in the halo class, feeds into chi_squared().
        x: Free variable to be minimized over 
        data: Data array to feed to chi_squared() (ndarray)
        Morb: Mass of the orbiting particles of the halo [Msun/h] (float)
        fit_keyword: Method of determining the model parameters 'rh' (halo radius) and 'alpha_inf' (asymptotic slope) 
            'simultaneous': Fit for both rh and alpha_inf simultaneously, returning two best fit parameter values
            ==> x is a array (rh, alpha_inf)
            'calibrated': Use the written function to determine alpha_inf from rh, therefore only fitting for rh 
            ==> x becomes just rh
        delta(=0.05): Assumed fractional error in the data (float)   
    '''
    if fit_keyword == 'simultaneous':
        model = orb_model(RADIUS, Morb, rh=x[0], alpha_inf=x[1])

    elif fit_keyword == 'calibrated':
        R = x / rh_st(Morb)
        alph_inf = alpha_inf(R, alpha_inf_st(Morb))
        model = orb_model(RADIUS, Morb, rh=x, alpha_inf=alph_inf)
    
    return chi_squared(data, model, delta=delta)


def bin_massive_halos():
    '''
    Take all massive (see HALO_MASS_MASK in loading.py) halos, and organize them by log10(Morb) into MASS_BIN_EDGES by returning indices assigned to each halo that 
    correspond to their mass bin.
    '''
    Morb = HALO_CATALOG['Morb'][HALO_MASS_MASK] 
    log_Morb = np.log10(Morb)
    return np.digitize(log_Morb, MASS_BIN_EDGES)


def alpha_inf(rh, Morb):
    '''
    Eq. (9) from Shields et al. (2025).
        R: Ratio of the best-fit rh to the rh from the stacked profile at the halo's mass (float)
        alpha_inf_st: Alpha infinity from the stacked profile at the halo's mass (float)
    '''
    R = rh / rh_st(Morb)
    return alpha_inf_st(Morb) + ALPHA_0 + (S_ALPHA * np.log(R))


def alpha_inf_aRF(rh, aRF):
    R = ARF_0 + (S_ARF * aRF)
    return ALPHA_0 + (ALPHA_P * ((rh / (RH_P * R)) ** (ALPHA_S / RH_S))) + (S_ALPHA * np.log(R))


def rh(Morb, aRF):
    return (ARF_0 + (S_ARF * aRF)) * rh_st(Morb) 


def rh_scatter(Np, var_i, var_0):
    '''
    Fitting function used to measure the intrinsic scatter (sig_0) in rh.
        var_i: Variance in rh of each halo mass bin
        var_0: Variance in rh as Np approaches infinity, i.e. scatter not due to number of particles
        Np: Number of particles
    '''
    return np.sqrt(var_0 + (var_i / Np))


# JACKKNIFE ERRORS
def jk_error(values):
    '''
    Calculate jackknife error of a set of values.
    '''
    average = np.average(values)
    N = len(values) # Number of values/measurements

    return np.sqrt(((N - 1) / N) * np.sum((values - average)**2))



# MISC


# @jit
def is_in_set_nb(a, b):
    '''
    Faster alternative to np.isin()
    '''
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)



# PLOTTING FUNCTIONS


def latex():
    '''
    Sets matplotlib parameters to format in Latex, as well as sets a good figure size.
    '''
    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Serif',
    'figure.dpi': 250
    })
    