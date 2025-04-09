from banerjee2020_sim.halos import *
import multiprocessing as mp
from os import nice


'''
For each massive halo, store its full, orbiting, and infalling density profiles.
'''

nice(10)


Nhids = 192042 # Excludes last two haloes, needs to be divisible by 6 for 6 cores
hids = HALO_CATALOG['OHID'][HALO_MASS_MASK][:Nhids] # Nhids haloes in mass bins
split_hids = np.split(hids, 6) # 6 groups of HIDs for 6 multiprocessing cores

def get_densities(hids):
    dens = np.zeros((int(Nhids/6), 3, 20))

    for i, hid in enumerate(hids):
        hid = int(hid)
        h = halo(hid)
        
        dens[i, 0, :] = h.density()
        dens[i, 1, :] = h.orb_density()
        dens[i, 2, :] = h.inf_density()

        print(f'{i+1}/{Nhids / 6}')

    return dens 

# Multiprocessing
pool = mp.Pool(processes=6)
results = pool.map(get_densities, split_hids)
pool.close()
pool.join()

results = np.concatenate(np.array(results), axis=0)

full_dens = results[:, 0, :]
orb_dens = results[:, 1, :]
inf_dens = results[:, 2, :]

hdf = h5.File('shields_data_24/halo_densities.hdf5', 'w')
hdf.create_dataset('Full', (Nhids, 20), data=full_dens)
hdf.create_dataset('Orb', (Nhids, 20), data=orb_dens)
hdf.create_dataset('Inf', (Nhids, 20), data=inf_dens)
hdf.close()
