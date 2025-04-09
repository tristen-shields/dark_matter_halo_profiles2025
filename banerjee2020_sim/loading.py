import h5py as h5 
from os.path import join

from .constants import *


BANERJEE_PATH = '/spiff/edgarmsc/simulations/Banerjee'

# Halo catalog (ROCKSTAR haloes mass logM_200m [13.0, 15.4])
HALO_CATALOG = h5.File(join(BANERJEE_PATH, 'rockstar/halo_catalogue.hdf5'), 'r')
    # <KeysViewHDF5 ['ID', 'M200m', 'Morb', 'Mt', 'OHID', 'PID', 'R200m', 'Rt', 'STATUS', 'Vmax', 'vx', 'vy', 'vz', 'x', 'y', 'z']>
        # 'OHID' ==> Halo ID
        # 'M200m', 'Morb', 'Mt' ==> mass definitions, [Msun]
        # 'R200m', 'Rt' ==> radius definitions, [Mpc/h]
        # 'x', 'y', 'z' ==> halo positions in the simulation [Mpc/h]
        # 'vx', 'vy', 'vz' ==> halo velocities in the simulation [km/s]
        # 'vmax' ==> maximum circular velocity [km/s]

LOG_MORB = np.log10(HALO_CATALOG['Morb'][()])
HALO_MASS_MASK = (LOG_MORB > MASS_BIN_EDGES[0]) & (LOG_MORB < MASS_BIN_EDGES[-1]) # Mask to get the halos in mass bins
MASSIVE_HIDS = HALO_CATALOG['OHID'][HALO_MASS_MASK] # All HIDs of haloes in mass bins

# Particle catalog, at z = 0 (1% of particles from the simulation)
PARTICLE_CATALOG = h5.File(join(BANERJEE_PATH, 'snap/snap_099/particle_catalogue.hdf5'), 'r')
    # <KeysViewHDF5 ['PID', 'vx', 'vy', 'vz', 'x', 'y', 'z']>
        # 'PID' ==> Particle ID
        # 'x', 'y', 'z' ==> particle positions in the simulation [Mpc/h]
        # 'vx', 'vy', 'vz' ==> particle velocities in the simulation [km/s]

# Particle tags (orbiting or infalling)
PARTICLE_TAGS = h5.File(join(BANERJEE_PATH, 'data_garcia_23/particle_classification.hdf5'), 'r')
# <KeysViewHDF5 ['CLASS']> ==> 0 or False for infalling particles, 1 or True for orbiting particles

# Scale factors of the simulation
SCALE_FACTOR = h5.File(join(BANERJEE_PATH, 'scale_factor.hdf5'), 'r') 
    # <KeysViewHDF5 ['scale_factor']>

# Particle orbit catalog; particle positions and velocities with respect to their parent haloes within a 5 Mpc/h box of each halo
# NOTE: one particle can be in multiple halo's boxes, so there are repeat PIDs 
# NOTE: if you change MEMBSIZE, the file cannot be accessed
ORBIT_CATALOG = h5.File(join(BANERJEE_PATH, 'orbits/orbit_catalogue_%d.hdf5'), 'r', driver='family', memb_size=MEMBSIZE)
    # <KeysViewHDF5 ['HID', 'PID', 'Rp', 'Vrp', 'Vtp']>
        # 'HID' ==> Box halo's ID, corresponds to 'OHID' in the HALO_CATALOG
        # 'PID' ==> Particle ID, corresponds to the 'PID' in the PARTICLE_CATALOG
        # 'Rp' ==> Radial distance to the halo's center
        # 'Vrp' ==> Radial velocity of the particle with respect to the halo's center
        # 'Vtp' ==> Tangential velocity of the particle with respect to the halo's center
        # All extensions are (N, 100) with N being the number of particles and the 100 representing timesteps 
            # Most of the time, we only need (N, 0), the most recent timestep (z = 0)

# Halo catalog masks for halo mass bins
HALO_CAT_MBIN_MASKS = h5.File(join(BANERJEE_PATH, 'data/mass_bin_haloes.hdf5'), 'r')
    # The keys are the MASS_BIN_STRS from constants.py
    # Mask sizes and ordering matches HALO_CATALOG

# Particle orbit catalog masks for halo mass bins 
PARTICLE_CAT_MBIN_MASKS = h5.File(join(BANERJEE_PATH, 'data/mass_bin_particles.hdf5'), 'r')
    # The keys are the MASS_BIN_STRS from constants.py
    # Mask sizes and ordering matches ORBIT_CATALOG

# Particle orbit to halo ID match
HALO_PARTICLE_DICT = h5.File(join(BANERJEE_PATH, 'halo_particle_dict.hdf5'), 'r')
    # Keys correspond to the 'OHID' dataset in HALO_CATALOG
    # Each dataset contains the indices in the orbit catalog of the particles belonging to a given halo 

PARTICLE_A_ACC = h5.File(join(BANERJEE_PATH, 'data_garcia_23/a_inf.hdf5'), 'r') # Particle accretion times


# SHIELDS ET AL. (2025)
HALO_DENSITIES = h5.File('shields_data_24/halo_densities.hdf5', 'r')
SIMULT_PARAMS = h5.File('shields_data_24/halo_simultaneous_fit_orb_model_params.hdf5', 'r') # Best fit orbiting model parameters from fitting both rh and alpha_inf 
# simultaneously
CALIB_PARAMS = h5.File('shields_data_24/halo_calibrated_fit_orb_model_params.hdf5', 'r')

