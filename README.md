# Code for simulation and data analysis of "Mixing of passive tracers at the ocean surface and implications for plastic transport modelling"
Authors: D. Wichmann, P. Delandmeter, H. A. Dijkstra and E. van Sebille

This repository contains all code needed for the simulation and analysis for the paper. For questions, please contact d.wichmann@uu.nl.

# OceanParcels version
We used OceanParcels version 1.10 with slight modifications: 

# Simulation
Use Simulations/AdvectParticles.py. The grid file has to be created beforehand, containing the initial longitudes and latitudes of 0.2 degree (entropy) and 0.1 degree (Markov chain) particles.

## Simulation for fixed depths
The following command is used to advect particles for 60 days for the transition matrix calculation. 'pos' is the index of initial particle grids.

python AdvectParticles.py -name testrun -y 2001 -m 1 -d 1 -simdays 60 -pos 0


# Analysis

## Creation of matrices
Execute the functions in Analysis/create_matrix.py

## Creation of figures
Execute functions in Analysis/paper_figures.py
