from banerjee2020_sim.halos import * 

import multiprocessing as mp 
from os import nice 


'''
Create an organizational file that stores the accretion times of each halo's particles, organized by parent halo.
'''

nice(10)

Nhids = 192042 # Excludes last two haloes, needs to be divisible by 6 for 6 cores
hids = HALO_CATALOG['OHID'][HALO_MASS_MASK][:Nhids] # Nhids haloes in mass bins
split_hids = np.split(hids, 6) # 6 groups of HIDs for 6 multiprocessing cores
bins = 200 # Number of bins to use when using np.histogram() to create a_acc CDF
Morb = HALO_CATALOG['Morb'][HALO_MASS_MASK][:Nhids]
inds = np.digitize(np.log10(Morb), MASS_BIN_EDGES)

# Save the mean accretion time for each mass bin for the computation of a_RF 
mbin_med_a_acc = np.zeros(len(MASS_BIN_STRS))
for i in range(len(MASS_BIN_STRS)):
    bin_a_acc = PARTICLE_A_ACC['a_inf'][PARTICLE_CAT_MBIN_MASKS[MASS_BIN_STRS[i]][:]]
    mbin_med_a_acc[i] = np.median(bin_a_acc) # Mask is to exclude particles that don't accrete, and are assigned a_acc=0.4 
    print(mbin_med_a_acc[i])


def get_aRF(hids):
    data = np.zeros((int(Nhids/6)))

    for i, hid in enumerate(hids):
        hid = int(hid)
        ind = inds[np.where(hids == hid)]
        h = halo(hid)
        
        data[i] = h.a_60(bins=bins) / mbin_med_a_acc[ind-1]

        print(f'{i+1}/{Nhids / 6}')

    return data 

# Multiprocessing
pool = mp.Pool(processes=6)
results = pool.map(get_aRF, split_hids)
pool.close()
pool.join()

results = np.concatenate(np.array(results), axis=0)

np.save('shields_data_24/aRF.npy', results)
