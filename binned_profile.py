#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""


import numpy
import pynbody



def load_snap_halo(file_name, halo_number, cutout_size=0):
    '''
    Loads a snapshot and centres it around a chosen halo (halo_number).
    The cutout_size argument specifies the radius of the "reflecting" 
    boundary around the halo (anything beyond this radius is cut out).
    
    Returns: entire snapshot (without boundary), snapshot cutout to 
             the desired size
    '''
    s = pynbody.load(file_name)
    h = s.halos()
    halo = h[halo_number]
    pynbody.analysis.halo.center(halo)
    s.physical_units()
    
    if cutout_size != 0:
        halo = s[pynbody.filt.Sphere("{} kpc".format(cutout_size))]
        
    pynbody.analysis.cosmology.add_hubble(s)
    pynbody.analysis.cosmology.add_hubble(halo)
        
    return s, halo
    




def calculate_binned_profile(halo, min_radius, max_radius, num_bins, 
                             bin_type='log'):
    '''
    Calculates the traditional "binned" density profile for a given halo 
    in log space from min_radius to max_radius for a given number of 
    bins (num_bins). 
    bin_type can be either 'linear' or 'log'. Default is 'log'
    
    Returns: binned density distribution, corresponding radii, 
             95% confidence Poisson errors
    '''
    
    if bin_type == 'log' or 'Log':
        bin_edges = numpy.logspace(numpy.log10(min_radius), 
                                   numpy.log10(max_radius), num_bins+1) 
    
    elif bin_type == 'linear' or 'Linear':
        bin_edges = numpy.linspace(min_radius, max_radius, 
                                   num_bins+1) 
    else:
        raise TypeError("Bin type can only be 'log' or 'linear'")
        
    volume_shell = (4/3) * numpy.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)    
    particles_radii = numpy.sqrt(halo['pos'][:,0]**2 + halo['pos'][:,1]**2 
                                     + halo['pos'][:,2]**2)
    particles_masses = halo['mass']
    mass_in_bins = numpy.histogram(particles_radii, bins=bin_edges, 
                                   weights=particles_masses)[0]

    binned_density_profile = mass_in_bins / volume_shell
    
    r_profile = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # calculate the Poisson errors 
    num_part_per_bin = numpy.histogram(particles_radii, bins=bin_edges)[0]

    y_errors = 1.96*(binned_density_profile / numpy.sqrt(num_part_per_bin))
    
    return binned_density_profile, r_profile, y_errors

