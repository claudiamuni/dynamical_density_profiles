#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudia Muni
"""

from setuptools import setup, find_packages

setup(
    name='dynamical_density_profiles',
    version='1.0.0',
    url='https://github.com/claudiamuni/dynamical_density_profiles.git',
    author='Claudia Muni',
    author_email='claudia.muni.21@ucl.ac.uk',
    description='Calculate dynamical density profiles of simulated halos',
    packages=find_packages(),    
    install_requires=['numpy >= 1.22.3', 'matplotlib >= 3.5.1', 'tqdm >= 4.64.0'],
)

