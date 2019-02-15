"""
Mixing of passive tracers at the ocean surface and implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Script to compute Transition matrices from given simulated trajectories
--------------

"""

import numpy as np
from scipy.sparse import csr_matrix
import os
import sys
from ana_objects import TM_ParticleSet
import scipy.sparse

def matrix_powers():
    """
    To compute some matrix powers
    """
    matrix_file = '/Users/wichmann/Simulations/Proj2_MixingTime/TransitionMatrices/year2001/TransitionMatrices/TM_simdays60_ddeg2_coasts.npz'
    mfile = np.load(matrix_file)
    tm=mfile['TM']
    TM = tm.copy()   
    
    for i in range(0,100):
        print 'power: ' + str(i)
        if i%5==0:
            fname = 'tm' + str(i)
            scipy.sparse.save_npz(fname,csr_matrix(TM))
        TM=np.dot(TM,tm)

def setupTM_individual(datadir, name, ddeg, dt, Ngrids, outputdir):
    """
    Create an individual Transition matrix for a certain time step
    """
    T=TM_ParticleSet.from_nc(datadir, name, dt, Ngrids)
    Lons = np.arange(0.,360.,ddeg)
    Lats = np.arange(-90.,90.,ddeg)
    T.setup_TM(Lons=Lons, Lats=Lats)
    T.save_TM(outputdir + 'TM_' + name)

def get_annual_TM(outputdir, names):
    """
    Create annual TM from a certain number of TMs
    """
    NT=len(names)
    
    for n, i in zip(names,range(NT)):
        print 'i: ', i
        data=np.load(outputdir + 'TM_'+n+'.npz')
        if i==0:
            TM = csr_matrix(data['TM'])
            Lons = data['Lons']    
            Lats = data['Lats']
            Ninit = data['Ninit']
        else:
            tm = csr_matrix(data['TM'])
            TM=tm.dot(TM)
    
    del tm
    TM=TM.todense()
    np.savez(outputdir + 'TM_total', TM=TM, Lons=Lons, Lats=Lats, Ninit=Ninit)

def setup_annual_markov_chain():
    
    datadir='scratch/wichm003/TMSimulations_v2/year2001/simdays60/'
    outputdir = datadir + 'TMresults/simdays60_ddeg_2/'
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    else:
        print 'Output dir exists already. Stop execution.'
        sys.exit(0)

    #Get unique list of file identifiers
    files = os.listdir(datadir)
    for k in range(len(files)):
        i = files[k].find('pos')
        files[k]= files[k][0:i]
    names = np.unique(files)
    names=np.array(names)
    
    #Bring them into the correct order according to month
    month = []
    for n in names:
        i=n.find('_m')
        month.append(n[i+2:i+4])
    
    for i in range(len(month)):
        if month[i][-1]=='_':
            month[i]=month[i][0:-1]
        month[i]=int(month[i])
    names=names[np.argsort(month)]
    
    #set up individual matrixes for 60 days
    for n in names:
        setupTM_individual(datadir, n, ddeg=2., dt=-1, Ngrids=40, outputdir=outputdir)
    
    get_annual_TM(outputdir, names)


def setup_transition_matrix_10years():
    """
    P-matrix for the region definition of the mixing entropy 
    """
    pdir = '/Users/wichmann/Simulations/Proj1_SubSurface_Mixing/Layer0/' #Data directory
    filename = 'SubSurf_y2000_m1_d5_simdays3650_layer0_' #File name for many particle simulation
    outdir = './EntropyMatrix/'
    
    setupTM_individual(datadir=pdir, name=filename, ddeg=2., dt=-1, Ngrids=40, outputdir = outdir)
