import numpy
import dynamical_functions as dyn
import binned_profile as bp
import matplotlib.pyplot as plt
import pynbody


# HOW TO OBTAIN A DYNAMICAL DENSITY PROFILE FROM A 
# SIMULATION SNAPSHOT: EXAMPLE


# Load a snapshot (here loaded with pynbody as an example)
cutout_size = 1 # Radius where to place the "reflecting" boundary (in kpc)
low_res_snap, low_res_halo = bp.load_snap_halo(file_name = 
                        '/your_path_to_simulation_snapshot', 
                        halo_number = 0, 
                        cutout_size=cutout_size)



# Choose the parameters
maximum_radius = cutout_size # minimum radius is zero by default
                                          
number_bins = 100   # choose the number of bins so that the bin width is 
                     # approx equal to the softening length of the 
                     # simulation
                     
number_of_iterations = 5 # choose how many times to iterate the profile 
                         # (we recommend at least 3 iterations)



# Calculate the (iterated) dynamical density profile.
# Lower_errs and upper_errs refer to the lower bounds and upper bounds of the 
# errors for the dynamical profile, respectively.
# (( If the simulation is DM + stars, 'dynamical_density' will be a 2D array 
# which stores the DM profile as first entry and stellar profile as second entry. ))
dynamical_density, lower_errs, upper_errs = dyn.calculate_dynamical_density_profile(
                            low_res_halo, maximum_radius,
                            number_bins, number_of_iterations)


dynamical_density_DM, lower_errs_DM, upper_errs_DM = dynamical_density[0], lower_errs[0], upper_errs[0]
dynamical_density_stars, lower_errs_stars, upper_errs_stars = dynamical_density[1], lower_errs[1], upper_errs[1]



# And plot the dynamical and binned profiles
plt.style.use('tableau-colorblind10')

radii = numpy.linspace(0, cutout_size, number_bins+1)
radii_centres = 0.5 * (radii[:-1] + radii[1:])


plt.plot(radii_centres, dynamical_density_DM, '-', label=r'Dynamical profile DM')
plt.fill_between(radii_centres, lower_errs_DM, upper_errs_DM, alpha=.45)

plt.plot(radii_centres, dynamical_density_stars, '-', label=r'Dynamical profile stars')
plt.fill_between(radii_centres, lower_errs_stars, upper_errs_stars, alpha=.45)



# Can calculate the traditional binned profile to compare it to 
# the dynamical one
min_radius_binned = 0.05 #in kpc
max_radius_binned = cutout_size
number_bins_binned = 41

binned_profile_dm, bins_profile_dm, y_errors_dm = bp.calculate_binned_profile(
                        low_res_halo.dm, min_radius_binned, max_radius_binned, 
                        number_bins_binned, bin_type='log')

binned_profile_stars, bins_profile_stars, y_errors_stars = bp.calculate_binned_profile(
                        low_res_halo.s, min_radius_binned, max_radius_binned, 
                        number_bins_binned, bin_type='log')

plt.errorbar(bins_profile_dm, binned_profile_dm, yerr=y_errors_dm, fmt='d', 
             markersize=3, label='Binned profile DM')

plt.errorbar(bins_profile_stars, binned_profile_stars, yerr=y_errors_stars, fmt='d', 
             markersize=3, label='Binned profile stars')

plt.yscale('log')   
plt.xscale('log')
plt.grid(alpha = 0.1)
plt.tick_params(axis='both')
plt.ylabel(r'$\rho_{DM}$ [M$_{\odot}$ kpc$^{-3}$]')
plt.xlabel(r'Radius [kpc]')
plt.legend()

plt.show()


