from .functions import *

from iminuit import minimize 


class halo(object):
    def __init__(self, hid):
        '''
        Takes in an HID (int) to distinguish a halo in the simulation.
        '''
        self.hid = hid 
        self.ind = np.where(HALO_CATALOG['OHID'][:] == hid)[0][0] # Halo index for all files that match halo catalog's 'OHID' dataset
        self.part_mask = HALO_PARTICLE_DICT[f'{self.hid}'][:] # Masks out the particles belonging to the halo in the particle catalog
        self.orb_mask = np.where(PARTICLE_TAGS['CLASS'][self.part_mask] == True) # Mask of halo's orbiting particles
        self.inf_mask = np.where(PARTICLE_TAGS['CLASS'][self.part_mask] == False) # Mask of halo's infalling particles

        # Halo physical quantities
        self.x, self.y, self.z = HALO_CATALOG['x'][self.ind], HALO_CATALOG['y'][self.ind], HALO_CATALOG['z'][self.ind] # Halo position
        self.vx, self.vy, self.vz = HALO_CATALOG['vx'][self.ind], HALO_CATALOG['vy'][self.ind], HALO_CATALOG['vz'][self.ind] # Halo velocity
        self.max_circ_velocity = HALO_CATALOG['Vmax'][self.ind]
        self.Rt, self.R200m, = HALO_CATALOG['Rt'][self.ind], HALO_CATALOG['R200m'][self.ind] # Radius definitions
        self.Mt, self.M200m, self.Morb = HALO_CATALOG['Mt'][self.ind], HALO_CATALOG['M200m'][self.ind], HALO_CATALOG['Morb'][self.ind] 

        # Stacked profile power law parameters from Salazar et al. (2024)
        self.rh_st, self.alpha_inf_st = stacked_parameters(self.Morb)

        # Halo particles
        self.a_acc = PARTICLE_A_ACC['a_inf'][self.part_mask][self.orb_mask]
        

    def density(self, vol_factor=1):
        '''
        Calculate density profile of all of the halo's particles.
            vol_factor(=1): Fraction of halo volume to use (int/float)

            Returns: Density profile in radial bins (1darray, length 20)
        '''
        part_distances = ORBIT_CATALOG['Rp'][self.part_mask, 0]
        return compute_density(part_distances, vol_factor=vol_factor)
        

    def orb_density(self, vol_factor=1):
        '''
        Calculate density profile of the halo's orbiting particles.
            vol_factor(=1): Fraction of halo volume to use (int/float)

            Returns: Density profile in radial bins (1darray, length 20)
        '''
        orb_part_distances = ORBIT_CATALOG['Rp'][self.part_mask, 0][self.orb_mask]
        return compute_density(orb_part_distances, vol_factor=vol_factor)


    def inf_density(self, vol_factor=1):
        '''
        Calculate density profile of the halo's infalling particles.
            vol_factor(=1): Fraction of halo volume to use (int/float)

            Returns: Density profile in radial bins (1darray, length 20)
        '''
        orb_part_distances = ORBIT_CATALOG['Rp'][self.part_mask, 0][self.inf_mask]
        return compute_density(orb_part_distances, vol_factor=vol_factor)
    

    def fit_orb_parameters(self, fit_keyword, rh0, alpha0, delta=0.05):
        '''
        Fit the orbiting density model to the halo's data. 
            fit_keyword: Method of determining the model parameters 'rh' (halo radius) and 'alpha_inf' (asymptotic slope)
                'power_law': Use the power laws in mass from Salazar et al. (2024) 
                'simultaneous': Fit for both rh and alpha_inf simultaneously, returning two best fit parameter values
                'calibrated': Use the written function to determine alpha_inf from rh, therefore only fitting for rh
            rh0: Initial guess for the halo radius parameter (float)
            alpha0: Initial guess for the alpha infinity parameter (float)
            mask: Mask to apply to the fitted data (1darray, same shape as halo's orb_density())

            Returns: Minimized result (float or ndarray), Minimized chi squared value (float)
        '''
        data = self.orb_density()
        if fit_keyword == 'simultaneous':
            result = minimize(cost, [rh0, alpha0], args=(data, self.Morb, fit_keyword, delta), bounds=([0, 3 * rh0], [0, 3 * alpha0]),
                            method='simplex', options={'stra':2, 'maxfun':500}, tol=1e-4)
            return *result.x, result.fun
        
        elif fit_keyword == 'calibrated':
            result = minimize(cost, rh0, args=(data, self.Morb, fit_keyword, delta), bounds=[0, 3 * rh0], method='simplex', options={'stra':2, 'maxfun':500}, tol=1e-4)
            R = result.x / self.rh_st
            return *result.x, alpha_inf(R, self.alpha_inf_st), result.fun
        

    def a_60(self, bins=200, show_plot=False):
        '''
        a_form, or the scale factor a at which the CDF(a_acc) for the halo's orbiting particles is 0.6.
            bins(=200): Number of bins to use when computing CDF(a_acc) (int)
        '''
        P, a_acc_bins = np.histogram(self.a_acc, bins=bins)
        P_unity = P / P.sum() # Normalized to unity
        cdf = P_unity.cumsum()

        if show_plot == True:
            plt.plot(a_acc_bins[1:], cdf)

        return a_acc_bins[1:][cdf > 0.6][0]


