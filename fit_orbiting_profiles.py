from banerjee2020_sim.halos import *
import multiprocessing as mp
from os import nice


'''
For each massive halo, fit its data to the orbiting density model. Keyword 'simultaneous' fits for both the halo radius rh and the asymptotic slope alpha_inf at the same 
time. Keyword 'calibrated' writes alpha_inf as a function of rh, and only fits for rh. 
'''

nice(10)

fit_keyword = 'calibrated'
delta = 0.05 # Fractional error to assume for the halo densities

Nhids = 192042 # Excludes last two haloes, needs to be divisible by 6 for 6 cores
hids = HALO_CATALOG['OHID'][HALO_MASS_MASK][:Nhids] # Nhids haloes in mass bins
split_hids = np.split(hids, 6) # 6 groups of HIDs for 6 multiprocessing cores

def get_densities(hids):
    data = np.zeros((int(Nhids/6), 5))

    for i, hid in enumerate(hids):
        hid = int(hid)
        h = halo(hid)

        rh, alpha_inf, chis = h.fit_orb_parameters(fit_keyword, rh0=h.rh_st, alpha0=h.alpha_inf_st, delta=delta)
        
        data[i, 0] = rh # Best fit halo radius
        data[i, 1] = h.rh_st # Stacked profile halo radius
        data[i, 2] = alpha_inf # Best fit asymptotic slope
        data[i, 3] = h.alpha_inf_st # Stacked profile asymptotic slope
        data[i, 4] = chis # Minimized chi squared value

        print(f'{i+1}/{Nhids / 6}')

    return data 

# Multiprocessing
pool = mp.Pool(processes=6)
results = pool.map(get_densities, split_hids)
pool.close()
pool.join()

results = np.concatenate(np.array(results), axis=0)

hdf = h5.File(f'shields_data_24/halo_{fit_keyword}_fit_orb_model_params.hdf5', 'w')
hdf.create_dataset('rh', (Nhids), data=results[:, 0])
hdf.create_dataset('rh_st', (Nhids), data=results[:, 1])
hdf.create_dataset('alpha_inf', (Nhids), data=results[:, 2])
hdf.create_dataset('alpha_inf_st', (Nhids), data=results[:, 3])
hdf.create_dataset('chis', (Nhids), data=results[:, 4])
hdf.close()