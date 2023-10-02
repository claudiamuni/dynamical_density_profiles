#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""



import pynbody
import numpy
from scipy.interpolate import interp1d


# Value of G in kpc/M⊙⋅(km/s)^2
G_const = 4.30091e-6


def calculate_mass_enclosed_with_doublesample(num_bins, snapshot, max_r):
    '''
    Calculates the mass enclosed at different radii.
    (The mass is sampled at twice the rate as the number of bins 
    to make sure the correct mass distribution is captured at 
    small radii.)
    '''
    radii_edges_mass = numpy.linspace(0, max_r, (2*num_bins)+1)
    mass_enclosed = pynbody.analysis.profile.Profile(snapshot, ndim=3, 
                                            bins=radii_edges_mass)['mass_enc']
    mass_enclosed = numpy.insert(mass_enclosed, 0, 0)
        
    return mass_enclosed





def calculate_mass_enclosed(num_bins, snapshot, max_r):
    '''
    Calculates the mass enclosed at different radii.
    '''  
    radii_edges_mass = numpy.linspace(0, max_r, num_bins+1)
    mass_enclosed = pynbody.analysis.profile.Profile(snapshot, ndim=3, 
                                            bins=radii_edges_mass)['mass_enc']
    mass_enclosed = numpy.insert(mass_enclosed, 0, 0)
    
    return mass_enclosed







def interpolated_spherical_potential_with_doublesample(max_r, 
                                                    mass_enclosed):
    '''
    Calculates the spherically averaged potential from a snapshot.
    Returns the potential as an interpolation object.
    (To be used when the mass is sampled at twice the rate of the bins.)
    '''

    # radii at which we have mass enclosed
    radii = numpy.linspace(0, max_r, round(((len(mass_enclosed)-1)/2)+1))
                                                 
    # calculate the spherically averaged potential  
    phi = []
    sum_potential = 0

    to_add = 0
    phi.append(to_add)

    deltar_bins = (max(radii)-min(radii)) / (len(radii)-1) 
    
    # since the mass is sampled at twice the rate of the bins then take
    # only the odd entries
    mass_enclos_odd = mass_enclosed[1::2]

    for i in range(0, len(radii)-1, 1):
        Mass_enclos_func = mass_enclos_odd[i]
        current_radius = (radii[i]+ radii[i+1])/2
        
        sum_potential += ((G_const * Mass_enclos_func) / (current_radius**2) 
                          ) * deltar_bins 
        phi.append(to_add + sum_potential)

        
    # interpolate the potential
    f_pot = interp1d(radii, phi, kind='cubic')
    
    return f_pot






def interpolated_spherical_potential(max_r, mass_enclosed):
    '''
    Calculates the spherically averaged potential from a snapshot.
    Returns the potential as an interpolation object.
    '''
    
    # radii for which we have mass enclosed
    radii = numpy.linspace(0, max_r, len(mass_enclosed))
                                                 
    # calculate the potential 
    phi = []
    sum_potential = 0
      
    to_add = 0
    phi.append(to_add)

    deltar_bins = (max(radii)-min(radii)) / (len(radii)-1) 

    for i in range(0, len(radii)-1, 1):
        Mass_enclos_func = (mass_enclosed[i] + mass_enclosed[i+1])/2
        current_radius = ((radii[i] + radii[i+1])/2) 
        
        sum_potential += ((G_const * Mass_enclos_func) / ( (current_radius**2) 
                 ) ) * deltar_bins 
        phi.append((to_add + sum_potential))
        
    # interpolate the potential
    f_pot = interp1d(radii, phi, kind='cubic')
    
    return f_pot

