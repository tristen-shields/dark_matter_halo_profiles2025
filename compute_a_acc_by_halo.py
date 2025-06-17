import multiprocessing as mp
from os import nice
from banerjee2020_sim.halos import *


'''
For each halo not in the lowest mass bin, compute the orbiting densities. 
'''

nice(10)

Nhids = 192042 # Has to be divisible by Nproc
hids = HALO_CATALOG['OHID'][HALO_MASS_MASK][:Nhids].astype(int)

R = CALIB_PARAMS['rh'][()] / CALIB_PARAMS['rh_st'][()]
lnR = np.log(R)
Morb = HALO_CATALOG['Morb'][HALO_MASS_MASK][:192042]

# Concatenate particle accretion times into bins of mass, lnR
hdf = h5.File('shields_data_24/P_a_acc.hdf5', 'w')

# P_a_acc = np.zeros((9, 5, 200)) # (Mass bins, lnR bins)
count = 0
for i in range(len(MASS_BIN_STRS)):
    grp = hdf.create_group(MASS_BIN_STRS[i])
    Morb_mask = (np.log10(Morb) >= MASS_BIN_EDGES[i]) & (np.log10(Morb) < MASS_BIN_EDGES[i + 1])
    mbin_hids = hids[Morb_mask]
    for j in range(6):
        lnR_mask = (lnR[Morb_mask] >= LNR_BIN_EDGES[j]) & (lnR[Morb_mask] < LNR_BIN_EDGES[j + 1])
        Rbin_hids = mbin_hids[lnR_mask]
        halo_a_acc_arrs = []
        for k in range(len(Rbin_hids)):
            h = halo(Rbin_hids[k])
            halo_a_acc = h.a_acc
            halo_a_acc_arrs.append(halo_a_acc)
            print(f'{count+1}/192042')
            count += 1 
        a_acc = np.concatenate(halo_a_acc_arrs) 
        P, a_acc_bins = np.histogram(a_acc, bins=200, density=True)
        # A = 1.0 / (P * np.sum(np.diff(a_acc_bins)))
        grp.create_dataset(f'{LNR_BIN_EDGES[j]:.2f} < lnR < {LNR_BIN_EDGES[j+1]:.2f}', data=P)
        # P_a_acc[i, j, :] = A * P

# np.save('shields_data_24/P_a_acc.npy', P_a_acc)
hdf.close()