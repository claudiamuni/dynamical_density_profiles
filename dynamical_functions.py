#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""


import numpy
from tqdm import tqdm
import extract_potential as pot
import iteration_functions as it
from functools import partial
import scipy.stats


# Value of G in kpc/M⊙⋅(km/s)^2
G_const= 4.30091e-6 



def initial_conditions(snapshot, particle_numbers):
    '''
    Calculates the initial conditions (x, y, z, vx, vy, vz) 
    of the given particles.
    
    Inputs
    ----------
    snapshot : SimSnap object
    particle_numbers : int or array
    '''
    ICs_pos = numpy.array(snapshot['pos'][particle_numbers])
    ICs_vel = numpy.array(snapshot['vel'][particle_numbers])
    IC_x, IC_y, IC_z = ICs_pos[:,0], ICs_pos[:,1], ICs_pos[:,2]
    IC_vx, IC_vy, IC_vz = ICs_vel[:,0], ICs_vel[:,1], ICs_vel[:,2]    
    return IC_x, IC_y, IC_z, IC_vx, IC_vy, IC_vz




def angular_momentum(IC_x, IC_y, IC_z, IC_vx, IC_vy, IC_vz):
    '''
    Calculates the magnitude of the angular momentum
    from the particles initial conditions.
    '''
    lz = (IC_x * IC_vy)-(IC_y * IC_vx)
    lx = (IC_y * IC_vz)-(IC_z * IC_vy)
    ly = (IC_z * IC_vx)-(IC_x * IC_vz)  
    l_ang_mom = numpy.sqrt(lx**2 + ly**2 + lz**2)
    return l_ang_mom



def total_energy(IC_vx, IC_vy, IC_vz, pot_energy):
    '''
    Calculates the energies (kinetic + potential)
    of the particles from the initial conditions.
    '''
    kin_energy = (IC_vx**2.)/2.+(IC_vy**2.)/2.+(IC_vz**2.)/2.
    return(pot_energy + kin_energy)




def unnormalised_probability(energy_of_orbit, l_ang_mom, radii, 
                             potential_interp):
    '''
    Calculates the *UNNORMALISED* probability density of finding a 
    particle at different radii. This probability traces the radial 
    orbits of the particle.
    
    Inputs
    ----------
    energy_of_orbit : array
    l_ang_mom : array
    radii : array
    potential_interp : scipy.interpolate._interpolate.interp1d
    '''
    
    array_ang_mom = numpy.repeat(l_ang_mom, len(radii)).reshape(-1, 
                                                            len(radii))
    array_energy = numpy.repeat(energy_of_orbit, len(radii)).reshape(-1, 
                                                            len(radii))
    
    unnormalised_prob = 1/numpy.sqrt( array_energy - ( 
                        (array_ang_mom**2)/(2* (radii**2)) ) - 
                        potential_interp(radii)) 
    
    # take the real part
    unnormalised_prob = numpy.where(numpy.isnan(unnormalised_prob), 0, 
                                  unnormalised_prob) 

    
    return unnormalised_prob







def calculate_corrections(snapshot, unnormalised_probs, radii, l_ang_mom, 
                        interpolated_potential, energies_of_orbit, bin_edges):
    '''
    Calculates the analytical corrections at pericentre and 
    apocentre and adds them to the probability density of each
    orbit. It also normalises the probability (up to a factor 
    of the bin width).
    
    Returns the normalised and the unnormalised probability density.
    
    Inputs
    ----------
    snapshot: SimSnap object
    unnormalised_probs : array
    radii : array
    mass_enclosed : array
    l_ang_mom : array
    interpolated_potential : scipy.interpolate._interpolate.interp1d
    energy_of_orbits : array
    bin_edges: array
    '''
    
    probabs = []
    
    deltar = (max(radii)-min(radii))/(len(radii)-1)

    for i in tqdm(range(len(unnormalised_probs))):
        
        # calculate the derivative of the effective potential
        deriv_eff_potential = numpy.gradient(interpolated_potential(radii) + 
                                        ((l_ang_mom[i]**2)/(2*(radii**2))), 
                                        deltar, edge_order=2)
    
        
        non_zero_probs = numpy.where(unnormalised_probs[i]>0)[0]
            
        if len(non_zero_probs) == 0:
            particle_radius = numpy.sqrt(snapshot['pos'][i][0]**2 
                                         + snapshot['pos'][i][1]**2 
                                         + snapshot['pos'][i][2]**2)

            radius_index = numpy.digitize(particle_radius, bin_edges)-1
                                        # -1 since we want the bin centre
                                        
            unnormalised_probs[i][radius_index] = 1


        if len(non_zero_probs) >= 2:
            r_peri_index = numpy.where(unnormalised_probs[i]>0)[0][0]
            r_peri = radii[r_peri_index]
            
            if r_peri_index == 0:
                correction_peri = 0
            
                
            else:
                deriv_peri = deriv_eff_potential[r_peri_index]
                potential_effective_peri = interpolated_potential(r_peri) + (
                            (l_ang_mom[i]**2)/(2*(r_peri**2)))
                
                correction_peri = (2*numpy.sqrt(energies_of_orbit[i] - 
                                potential_effective_peri))/(-deriv_peri)
                
                    
                if correction_peri < 0:
                    print('The pericentre correction for particle', i, 
                          'is negative. The bin width might be too large.')
                    correction_peri = 0
                    
                if correction_peri > 0:
                    unnormalised_probs[i][r_peri_index-1] = (
                                                1/deltar)*correction_peri
                
                
            if unnormalised_probs[i][-1] == 0:
        
                r_apo_index = numpy.where(unnormalised_probs[i]>0)[-1][-1]
                r_apo = radii[r_apo_index]
                    
                deriv_apo = deriv_eff_potential[r_apo_index] 
                
                
                potential_effective_apo = interpolated_potential(r_apo) + (
                        (l_ang_mom[i]**2)/(2*(r_apo**2)))

                correction_apo = (2*numpy.sqrt(energies_of_orbit[i] - 
                                        potential_effective_apo))/deriv_apo
                
                if correction_apo < 0:
                    print('The apocentre correction for particle', i, 
                          'is negative. Turning off the correction.')
                    correction_apo = 0
                    

                if correction_apo > 0:
                    unnormalised_probs[i][r_apo_index+1] = (
                                            1/deltar)*correction_apo
    
            else:
                correction_apo = 0
            
        # integrals are normalised (*but not divided by delta r!*)
        tot_integral = numpy.sum(unnormalised_probs[i])
        
        normalised_prob = unnormalised_probs[i]/tot_integral
        probabs.append(normalised_prob)
        
        
    return probabs






def add_densities(max_r, num_bins_r, probabs, particle_mass): 
    '''
    Calculates the matter density profile from the radial probability 
    density of all the particles.
    '''
    sum_probability_density = sum(probabs)
    bin_edges = numpy.linspace(0, max_r, num_bins_r+1)

    density_profile = (particle_mass * sum_probability_density) / ((
                    4/3) * numpy.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3) )
        
    return density_profile



def bootstrap_errors(max_r, num_bins_r, probabs, snapshot, 
                    num_samples_bootstrap=100):

    '''
    Calculates the uncertainty on the density profile using 
    bootstrapping.
    
    Returns the lower and upper bound for the error.

    Inputs
    ----------
    max_r : float
    num_bins_r : array
    probabs : array / list
    snapshot : SimSnap object
    num_samples_bootstrap : int, optional. The default is 100.
    '''

    densities_repeated = []
    probs_array = numpy.array(probabs)
    
    particle_mass = snapshot['mass'][0]

    for i in tqdm(range(num_samples_bootstrap)):
        
        # Extract some probabilities from the total sample (the number of samples 
        # should be the same as the orginal sample)
        sample_idx = numpy.random.choice(numpy.arange(len(probs_array)), 
                                         size=len(probs_array), 
                                         replace=True)
        sample_y = probs_array[sample_idx]
        
        # Calculate the profile from them        
        densities_sample = add_densities(max_r, num_bins_r, sample_y, 
                                        particle_mass)
         
        rescaled_density_sample = (len(snapshot)/len(sample_idx))*(
                                                    densities_sample)
        
        densities_repeated.append(rescaled_density_sample)
        
        # Repeat


    interval_bootstrap = 95.

    error_lower_bound, error_upper_bound = numpy.percentile(densities_repeated, (
                    100-interval_bootstrap)/2., 0), numpy.percentile(
                        densities_repeated, interval_bootstrap+(
                            100-interval_bootstrap)/2., 0)


    return error_lower_bound, error_upper_bound









def dynamical_density_calculation(snapshot, maxim_radius, number_bins, 
                    interpolated_potential, 
                    new_energy, calculate_errors = False, 
                    num_samples_bootstrap=100, first_profile=True):
    '''
    Calculates the dynamical density profile for a given snapshot.
    
    
    Inputs
    ----------
    snapshot : SimSnap object
    maxim_radius : float
    number_bins : int
    num_particles_profile : int
    interpolated_potential : scipy.interpolate._interpolate.interp1d
    mass_enclosed : array
    new_energy : array
    calculate_errors : bool, optional. The default is False.
    num_samples_bootstrap : int, optional. The default is 100.
    first_profile : bool, optional. The default is True.
    '''
    
    radii_edges = numpy.linspace(0, maxim_radius, number_bins+1)
    radii = 0.5 * (radii_edges[:-1] + radii_edges[1:])

    # Assumes all the particles have the same mass (DMO simulation)
    particle_mass = snapshot['mass'][0]
    
    # runs through all the particles in the snapshot
    particle_numbers = numpy.linspace(0, len(snapshot)-1, len(snapshot), 
                                                              dtype=int)       
    
    IC_x, IC_y, IC_z, IC_vx, IC_vy, IC_vz = initial_conditions(snapshot, 
                                                        particle_numbers)

    l_ang_mom = angular_momentum(IC_x, IC_y, IC_z, IC_vx, IC_vy, IC_vz)
     
    if first_profile == True:
        pot_energies = interpolated_potential(numpy.sqrt(IC_x**2 + 
                                                    IC_y**2 + IC_z**2))
        energies_of_orbits = total_energy(IC_vx, IC_vy, IC_vz, pot_energies)
    if first_profile == False:
        energies_of_orbits = new_energy
     
    potential_list = interpolated_potential(radii)
    
    unnormalised_probs = unnormalised_probability(energies_of_orbits, 
                                    l_ang_mom, radii, interpolated_potential)
    
    probabs = calculate_corrections(snapshot, unnormalised_probs, radii, 
            l_ang_mom, interpolated_potential, 
            energies_of_orbits, radii_edges)
    
    probabs = numpy.array(probabs)
    
    densities = add_densities(maxim_radius, number_bins, 
                                              probabs, particle_mass)
    
    dynamical_density = (len(snapshot)/len(probabs))*(densities)
              
    if calculate_errors:
        lower_bound_error, upper_bound_error = bootstrap_errors(
                            maxim_radius, number_bins, probabs,
                            snapshot, num_samples_bootstrap)
        
    else:
        lower_bound_error = 0 
        upper_bound_error = 0

    return dynamical_density, lower_bound_error, upper_bound_error, energies_of_orbits, potential_list, probabs, l_ang_mom










def calculate_dynamical_density_profile(halo, max_radius, num_bins, 
                               number_of_iterations):
    '''
    Calculates the (iterated) dynamical density profile.
    
    Inputs
    ----------
    halo : SimSnap object
    max_radius : float
    num_bins : int
    num_particles_profile : int
    number_of_iterations: int
    
    
    Outputs
    ----------
    dyn_densities : array. Dynamical density profile.
    low_errs : array. Lower bound errors on the profile.
    up_errs: array. Upper bound errors on the profile.
    '''
    
    bins = numpy.linspace(0, max_radius, num_bins+1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    
    enclosed_mass = pot.calculate_mass_enclosed_with_doublesample(
                        num_bins, halo, max_radius)
    
    interpolated_potential = pot.interpolated_spherical_potential_with_doublesample(
                        max_radius, mass_enclosed = enclosed_mass)
    
    # Calculate the first density profile
    print('Calculating dynamical density profile from snapshot...')
    dyn_density, low_err, up_err, old_energies, old_potential, old_probabs, old_l_ang_mom = dynamical_density_calculation(
                        halo, max_radius, num_bins, 
                        interpolated_potential, 
                        new_energy = 0, calculate_errors=False, 
                        num_samples_bootstrap = 0, first_profile=True)

    # Iterate the profile
    dyn_densities, low_errs, up_errs = it.profile_iteration(number_of_iterations, dyn_density, halo, 
                          max_radius, num_bins, bin_centres, 
                          interpolated_potential, 
                          old_energies, old_probabs)
    
    return dyn_densities, low_errs, up_errs

