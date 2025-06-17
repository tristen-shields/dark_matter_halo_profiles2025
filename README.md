The Density Profile of Dynamical Halos (Shields et al. 2025)

This repository contains the code used to take simulation data (available on __) and perform all the computations as well as to make all of the plots from the paper.

\textbf{Dependencies/Prerequisites}: The code in this repository is written purely in the Python (version 3.12.0) programming language, and relies on the following imported libraries: numpy 1.26.4, astropy 7.0.2, h5py 3.12.1, scipy 1.15.1, Matplotlib 3.10.0, and iminuit 2.30.1

\textbf{Documentation}: For every Python file, there is a string at the top (below imports) explaining the purpose of the file. Each function contains a string that explains the purpose of the function, as well as what it takes as arguments and what it outputs.

This code is based on a system of imports from the folder banerjee2020_sim, representing all of the datafiles, functions, and classes used in the analysis of the paper. 
  constants.py: Contains all of the constants with information from the simulation, cosmological parameters, and numbers derived from the paper 
analysis.
  loading.py: Contains/Loads all of the filepaths of data used in the analysis.
  functions.py: Contains every function used in the analysis that doesn't take in a halo ID.
  halos.py: Contains a halo class, which, upon feeding a halo ID from the catalog, will give useful information and functions having to do with that particular halo.
  Within this folder, each file imports from another: loading imports from constants, loading imports from functions, and halos imports from functions. Therefore, any Python notebook or file outside the folder can simply import from banerjee2020_sim.halos to get all of the constants, data, functions and classes used in the analysis. 

To reproduce the results of the paper, do the following:
  1.) Run compute_halo_densities.py to obtain orbiting, infalling, and full density profiles for each halo
  2.) Run fit_orbiting_profiles.py with fit_keyword = 'simultaneous' to fit our model to all of the halos to obtain best fit values of the halo radius (rh) and asymptotic slope (alpha_infinity)
  3.) After determining the relationship between rh and alpha_infinity (see orb_paper_plots.ipynb), re-run fit_orbiting_profiles.py with fit_keyword = 'calibrated'. This writes alpha_infinity as a function of rh and therefore only fits each halo for rh
  4.) Run compute_a_acc_by_halo.py to get particle CDFs binned by Morb then lnR, as described in Section IV.
  5.) Run compute_halo_a60.py to determine the a60 parameter (scale factor of the Universe when a given halo accretes 60\% of its particles)
  6.) To remake all of the plots of the paper, as well as to derive all numbers and uncertainties reported in the paper, run orb_profile_paper_plots.ipynb from top to bottom.
