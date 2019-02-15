#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advection of particles with nemo
"""

import numpy as np
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, ErrorCode, AdvectionRK4
from argparse import ArgumentParser
from datetime import timedelta
from datetime import datetime
from glob import glob

datadir = '/data2/imau/oceanparcels/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/' #Directory for nemo data
outputdir = '/scratch/wichm003/TMSimulations/' #Directory for output files
griddir = '/home/staff/wichm003/AttractionTimeScales/ParticleGrid/Global01grid/' #Directory for initial particle distribution

def DeleteParticle(particle, fieldset, time, dt):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()

def periodicBC(particle, fieldset, time, dt):
    """
    Kernel for periodic boundaries in longitude
    """
    if particle.lon < 0.:
        particle.lon += 360.
    elif particle.lon > 360.:
        particle.lon -= 360.

def p_advect(ptype=JITParticle,outname='noname', pos=0, y=2001, m=1, d=1, simdays=90):
    """
    Main function for execution
        - outname: name of the output file. Note that all important parameters are also in the file name.
        - pos: Execution is manually parallelized over different initial position grids. These are indexed.
        - y, m, d: year, month an day of the simulation start
        - simdays: number of days to simulate
    """
    
    print '-------------------------'
    print 'Start run... Parameters: '
    print '-------------------------'
    print 'Initial time (y, m, d): ', (y, m, d)
    print 'Simulation days', simdays
    print '-------------------------'
    
    #Load grid from external file
    lons = np.load(griddir + 'Lons' + str(pos) + '.npy')
    lats = np.load(griddir + 'Lats' + str(pos) + '.npy') 
    times = [datetime(y, m, d)]*len(lons)
    print 'Number of particles: ', len(lons)
    outfile = outputdir + outname + '_y'+ str(y) + '_m' + str(m) + '_d' + str(d)  + '_simdays' + str(simdays) + '_pos' + str(pos)

    ufiles = sorted(glob(datadir+'means/ORCA0083-N06_200?????d05U.nc'))
    vfiles = sorted(glob(datadir+'means/ORCA0083-N06_200?????d05V.nc'))

    filenames = {'U': ufiles,
                 'V': vfiles,
                 'mesh_mask': datadir + 'domain/coordinates.nc'}

    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions,  allow_time_extrapolation=False)
    
    fieldset.U.vmax = 10
    fieldset.V.vmax = 10

    pset = ParticleSet(fieldset=fieldset, pclass=ptype, lon=lons, lat=lats, time=times)
    
    kernels= pset.Kernel(AdvectionRK4) + pset.Kernel(periodicBC) #Periodic boundaries (longitude)
    pset.execute(kernels, runtime=timedelta(days=simdays), dt=timedelta(minutes=10), output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(days=5)),recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

if __name__=="__main__":
    ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
    p = ArgumentParser(description="""Global advection of different particles""")
    p.add_argument('-ptype', '--ptype',choices=('scipy', 'jit'), nargs='?', default='jit',help='Execution mode for performing computation')
    p.add_argument('-name', '--name', default='noname',help='Name of output file')
    p.add_argument('-y', '--y', type=int,default=None,help='year of simulation start')
    p.add_argument('-m', '--m', type=int,default=None,help='month of simulation start')
    p.add_argument('-d', '--d', type=int,default=None,help='day of simulation start')
    p.add_argument('-simdays', '--simdays', type=int,default=None,help='Simulation days')
    p.add_argument('-pos', '--pos', type=int,default=0,help='Label of Lon/Lat initial array')
    args = p.parse_args()
    p_advect(ptype=ptype[args.ptype],outname=args.name, pos=args.posidx, y=args.y, m=args.m, d=args.d, simdays=args.simdays)
