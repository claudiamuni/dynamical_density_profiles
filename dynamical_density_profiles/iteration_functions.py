#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""


import numpy
import extract_potential as pot
import dynamical_functions as dyn
import pynbody



def potential_from_dynamical_density(dynamical_profile, num_bins_r, 
                                     max_r):
    '''
    Calculate the gravitational potential inferred by the dynamical
    density profile.
    '''
    bin_edges = numpy.linspace(0, max_r, num_bins_r+1)
    
    volume_bins = (4/3) * numpy.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3) 
    
    mass_with_radius = dynamical_profile * volume_bins 
    mass_enclosed = mass_with_radius.cumsum()
    
    mass_enclosed_edges = numpy.insert(mass_enclosed, 0, 0)
    
    f_pot = pot.interpolated_spherical_potential(max_r, 
                                             mass_enclosed_edges)

    return f_pot




    
def updated_energy(old_energy, old_potential, updated_potential, 
                   old_probabilities):
    '''
    Calculate the new particle energies from the potential configuration 
    inferred by the dynamical profile.
    '''

    change_in_potential = updated_potential - old_potential
    
    avg_potential = []
    
    for i in range(len(old_probabilities)):
        avg_pots = numpy.sum(change_in_potential * old_probabilities[i]) 
        avg_potential.append(avg_pots)
    
    avg_potential = numpy.array(avg_potential)

    delta_energy = avg_potential 
    
    updated_energy = old_energy + delta_energy
    
    return updated_energy 





def profile_iteration(number_of_iterations, old_dyn_density, halo, 
                      max_radius, num_bins, bin_centres, 
                    old_interp_potential, 
                      old_energies, old_probabs):
    '''
    Iterates the dynamical density profile starting from the profile 
    obtained from the simulation snapshot.
    '''

    if len(pynbody.snapshot.SimSnap.families(halo)) == 1:

        for i in range(number_of_iterations-1):
            print('Iteration number:', i+1)
        
            new_interp_potential = potential_from_dynamical_density(
                                    old_dyn_density, num_bins, max_radius)

            new_energies = updated_energy(old_energies, 
                                    old_interp_potential(bin_centres), 
                                    new_interp_potential(bin_centres), 
                                    old_probabs)

            iterated_dyn_density, iterated_low_errs, iterated_up_errs, new_energ, new_potential, new_probabs, new_ang_mom = dyn.dynamical_density_calculation(
                                    halo, max_radius, num_bins, 
                                    new_interp_potential, new_energy = new_energies, 
                                    calculate_errors = False, num_samples_bootstrap = 0,
                                    first_profile = False)

            # update the variables and repeat
            old_interp_potential = new_interp_potential
            old_energies = new_energies
            old_probabs = new_probabs
            old_dyn_density = iterated_dyn_density

        # Final iteration (with errors)
        new_interp_potential = potential_from_dynamical_density(
                    old_dyn_density, num_bins, max_radius)
                
        new_energies = updated_energy(old_energies, 
                                            old_interp_potential(bin_centres), 
                                            new_interp_potential(bin_centres), 
                                            old_probabs)

        iterated_dyn_density, iterated_low_errs, iterated_up_errs, new_energ, new_potential, new_probabs, new_ang_mom = dyn.dynamical_density_calculation(
                        halo, max_radius, num_bins, 
                        new_interp_potential, new_energy = new_energies, 
                        calculate_errors = True, num_samples_bootstrap = 100,
                        first_profile = False)


        return iterated_dyn_density, iterated_low_errs, iterated_up_errs
    

            

    elif len(pynbody.snapshot.SimSnap.families(halo)) > 1:

        for i in range(number_of_iterations-1):
            print('Iteration number:', i+1)

            new_interp_potential = potential_from_dynamical_density(
                                    old_dyn_density, num_bins, max_radius)

            new_energies_dm = updated_energy(old_energies[0], 
                                      old_interp_potential(bin_centres), 
                                      new_interp_potential(bin_centres), 
                                      old_probabs[0])

            new_energies_stars = updated_energy(old_energies[1], 
                                      old_interp_potential(bin_centres), 
                                      new_interp_potential(bin_centres), 
                                      old_probabs[1])

            iterated_dyn_density_dm, iterated_low_errs_dm, iterated_up_errs_dm, new_energ_dm, new_potential_dm, new_probabs_dm, new_ang_mom_dm = dyn.dynamical_density_calculation(
                halo.dm, max_radius, num_bins, 
                new_interp_potential, new_energy = new_energies_dm, 
                calculate_errors = False, num_samples_bootstrap = 0,
                first_profile = False)

            iterated_dyn_density_stars, iterated_low_errs_stars, iterated_up_errs_stars, new_energ_stars, new_potential_stars, new_probabs_stars, new_ang_mom_stars = dyn.dynamical_density_calculation(
                halo.s, max_radius, num_bins, 
                new_interp_potential, new_energy = new_energies_stars, 
                calculate_errors = False, num_samples_bootstrap = 0,
                first_profile = False)

            iterated_dyn_density = iterated_dyn_density_dm + iterated_dyn_density_stars

            # update the variables and repeat
            old_interp_potential = new_interp_potential
            old_energies = [new_energies_dm, new_energies_stars]
            old_probabs = [new_probabs_dm, new_probabs_stars]
            old_dyn_density = iterated_dyn_density


        # Final iteration (with errors)
        new_interp_potential = potential_from_dynamical_density(
                    old_dyn_density, num_bins, max_radius)

        new_energies_dm = updated_energy(old_energies[0], 
                                      old_interp_potential(bin_centres), 
                                      new_interp_potential(bin_centres), 
                                      old_probabs[0])

        new_energies_stars = updated_energy(old_energies[1], 
                                      old_interp_potential(bin_centres), 
                                      new_interp_potential(bin_centres), 
                                      old_probabs[1])


        iterated_dyn_density_dm, iterated_low_errs_dm, iterated_up_errs_dm, new_energ_dm, new_potential_dm, new_probabs_dm, new_ang_mom_dm = dyn.dynamical_density_calculation(
                halo.dm, max_radius, num_bins, 
                new_interp_potential, new_energy = new_energies_dm, 
                calculate_errors = True, num_samples_bootstrap = 100,
                first_profile = False)

        iterated_dyn_density_stars, iterated_low_errs_stars, iterated_up_errs_stars, new_energ_stars, new_potential_stars, new_probabs_stars, new_ang_mom_stars = dyn.dynamical_density_calculation(
                halo.s, max_radius, num_bins, 
                new_interp_potential, new_energy = new_energies_stars, 
                calculate_errors = True, num_samples_bootstrap = 100,
                first_profile = False)


        return numpy.array([iterated_dyn_density_dm, iterated_dyn_density_stars]), numpy.array([iterated_low_errs_dm,iterated_low_errs_stars]), numpy.array([iterated_up_errs_dm,iterated_up_errs_stars])
    


 
    
    
    