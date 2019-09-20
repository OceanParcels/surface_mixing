"""
Advection of particles with nemo
"""

import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, AdvectionRK4
from argparse import ArgumentParser
from datetime import timedelta
from datetime import datetime
from glob import glob

datadir = '/data2/imau/oceanparcels/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/' #Directory for nemo data
outputdir = '/scratch/wichm003/surface_mixing_output/' #Directory for output files

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()

def periodicBC(particle, fieldset, time):
    """
    Kernel for periodic values in longitude
    """
    if particle.lon < 0.:
        particle.lon += 360.
    elif particle.lon >= 360.:
        particle.lon -= 360.

def p_advect(outname='noname', coordinate_file='no_file_specified', y=2001, m=1, d=1, simdays=360):
    """
    Main function for execution
        - outname: name of the output file. Note that all important parameters are also in the file name.
        - pos: Execution is manually parallelized over different initial position grids. These are indexed.
        - y, m, d: year, month an day of the simulation start
        - simdays: number of days to simulate
    """
    
    print( '-------------------------')
    print( 'Start run... Parameters: ')
    print( '-------------------------')
    print( 'Initial time (y, m, d): ', (y, m, d))
    print( 'Simulation days', simdays)
    print( '-------------------------')
    
    #Load grid from external file
    coordinates = np.load(coordinate_file)
    lons = coordinates['lons']
    lats = coordinates['lats'] 
    times = [datetime(y, m, d)]*len(lons)
    print( 'Number of particles: ', len(lons))
    outfile = outputdir + outname + '_y'+ str(y) + '_m' + str(m) + '_d' + str(d)  + '_simdays' + str(simdays)


    ufiles = sorted(glob(datadir+'means/ORCA0083-N06_200?????d05U.nc'))
    vfiles = sorted(glob(datadir+'means/ORCA0083-N06_200?????d05V.nc'))
    
    mesh_mask = datadir + 'domain/coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'data': vfiles}}
    variables = {'U': 'uo',
                 'V': 'vo'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}

    fieldset = FieldSet.from_nemo(filenames, variables, dimensions)
 
    fieldset.U.vmax = 10
    fieldset.V.vmax = 10

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, time=times)
    
    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(periodicBC)
    pset.execute(kernels, runtime=timedelta(days=simdays), dt=timedelta(minutes=10), output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(days=15)),recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})


if __name__=="__main__":
    p = ArgumentParser(description="""Global advection of different particles""")
    p.add_argument('-name', '--name', default='noname',help='Name of output file')
    p.add_argument('-y', '--y', type=int,default=None,help='year of simulation start')
    p.add_argument('-m', '--m', type=int,default=None,help='month of simulation start')
    p.add_argument('-d', '--d', type=int,default=None,help='day of simulation start')
    p.add_argument('-simdays', '--simdays', type=int,default=None,help='Simulation days')
    p.add_argument('-coords', '--coords',help='Initial coordinate file')    
    args = p.parse_args()
    p_advect(outname=args.name, coordinate_file=args.coords, y=args.y, m=args.m, d=args.d, simdays=args.simdays)
