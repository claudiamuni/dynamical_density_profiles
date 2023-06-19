#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""

import numpy
import dynamical_functions as dyn
import binned_profile as bp
import matplotlib.pyplot as plt


# EXAMPLE OF HOW TO USE THE CODE TO OBTAIN A DYNAMICAL DENSITY PROFILE

# Load a snapshot
cutout_size = 120 # The radius where to place the "reflecting" boundary
low_res_snap, low_res_halo = bp.load_snap_halo(file_name = 
                        'Halo1459_DMO_lowres/output_00101', halo_number = 53, 
                        cutout_size=cutout_size)


# Choose the parameters
minimum_radius = 0
maximum_radius = cutout_size
num_particles_profile = len(low_res_halo)
number_bins = 1260*2 # choose the number of bins so that the bin width is 
                     # approx equal to half the softening length of the 
                     # simulation
number_of_iterations = 5 # choose how manyy iterations for the profile 
                         # (we recommend at least 3 iterations if possible)



# Calculate the (iterated) dynamical density profile.
# Lower_errs and upper_errs refer to the lower bounds and upper bounds of the 
# errors for the dynamical profile, respectively.
dynamical_density, lower_errs, upper_errs = dyn.calculate_dynamical_density_profile(
                            low_res_halo, minimum_radius, maximum_radius, 
                            number_bins, num_particles_profile, 
                            number_of_iterations)




# Can calculate the traditional binned profile to compare it to 
# the dynamical one
min_radius_binned = 0.096
max_radius_binned = cutout_size
number_bins_binned = 41

binned_profile, bins_profile, y_errors = bp.calculate_binned_profile(
                        low_res_halo, min_radius_binned, max_radius_binned, 
                        number_bins_binned, bin_type='log')




# Plot the dynamical and binned profiles
plt.style.use('tableau-colorblind10')

radii = numpy.linspace(minimum_radius, cutout_size, number_bins+1)
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

