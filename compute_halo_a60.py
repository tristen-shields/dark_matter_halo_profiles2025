from banerjee2020_sim.halos import * 
import multiprocessing as mp 
from os import nice 


'''
Create an organizational file that stores the accretion times of each halo's particles, organized by parent halo.
'''

nice(10)

Nhids = 192042 # Excludes last two haloes, needs to be divisible by 6 for 6 cores
hids = HALO_CATALOG['OHID'][HALO_MASS_MASK][:Nhids] # Nhids haloes in mass bins
Nproc = 9
split_hids = np.split(hids, Nproc) # 6 groups of HIDs for 6 multiprocessing cores
bins = 200 # Number of bins to use when using np.histogram() to create a_acc CDF
downsample = round((10 ** MASS[0]) / PARTICLE_MASS) # Whether or not to calculate a_acc from all particles (False) or downsample to smallest mass bin
print(downsample)


def get_aRF(hids):
    data = np.zeros(len(hids))

    for i, hid in enumerate(hids):
        hid = int(hid)
        h = halo(hid)
        
        data[i] = h.a_60(bins=bins, choose_random=downsample) 

        print(f'{i+1}/{len(hids)}')

    return data 

# Multiprocessing
pool = mp.Pool(processes=Nproc)
results = pool.map(get_aRF, split_hids)
pool.close()
pool.join()

results = np.concatenate(np.array(results), axis=0)

np.save('a60_downsampled.npy', results)
