#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""

import numpy
import dynamical_functions as dyn
import binned_profile as bp
import matplotlib.pyplot as plt




# HOW TO OBTAIN A DYNAMICAL DENSITY PROFILE FROM A 
# SIMULATION SNAPSHOT: EXAMPLE



# Load a snapshot (here loaded with pynbody as an example)
cutout_size = 120 # Radius where to place the "reflecting" boundary (in kpc)
low_res_snap, low_res_halo = bp.load_snap_halo(file_name = 
                        'Halo1459_DMO_lowres/output_00101', halo_number = 53, 
                        cutout_size=cutout_size)


# Choose the parameters
maximum_radius = cutout_size # minimum radius is zero by default
                                          
number_bins = 2520   # choose the number of bins so that the bin width is 
                     # approx equal to the softening length of the 
                     # simulation
                     
number_of_iterations = 3 # choose how many times to iterate the profile 
                         # (we recommend at least 3 iterations)



# Calculate the (iterated) dynamical density profile.
# Lower_errs and upper_errs refer to the lower bounds and upper bounds of the 
# errors for the dynamical profile, respectively.
dynamical_density, lower_errs, upper_errs = dyn.calculate_dynamical_density_profile(
                            low_res_halo, maximum_radius, 
                            number_bins, number_of_iterations)




# Can calculate the traditional binned profile to compare it to 
# the dynamical one
min_radius_binned = 0.096 #in kpc
max_radius_binned = cutout_size
number_bins_binned = 41

binned_profile, bins_profile, y_errors = bp.calculate_binned_profile(
                        low_res_halo, min_radius_binned, max_radius_binned, 
                        number_bins_binned, bin_type='log')




# And plot the dynamical and binned profiles
plt.style.use('tableau-colorblind10')

radii = numpy.linspace(0, cutout_size, number_bins+1)
radii_centres = 0.5 * (radii[:-1] + radii[1:])


plt.plot(radii_centres, dynamical_density, '-', label=r'Dynamical profile')
plt.fill_between(radii_centres, lower_errs, upper_errs, alpha=.45)

plt.errorbar(bins_profile, binned_profile, yerr=y_errors, fmt='d', 
             markersize=3, label='Binned profile')

plt.yscale('log')   
plt.xscale('log')
plt.grid(alpha = 0.1)
plt.tick_params(axis='both')
plt.ylabel(r'$\rho_{DM}$ [M$_{\odot}$ kpc$^{-3}$]')
plt.xlabel(r'Radius [kpc]')

plt.show()
