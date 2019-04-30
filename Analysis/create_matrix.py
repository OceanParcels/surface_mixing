"""
Mixing of passive tracers at the ocean surface and its implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Script to compute Transition matrices from given simulated trajectories. 
Note that the computed matrix is the transpose of the one stated in the paper (i.e. we take right eigenvectors to get the stationary density).
--------------

"""

import numpy as np
from scipy.sparse import csr_matrix
import os
from ana_objects import TM_ParticleSet
import scipy.sparse

datadir = '/Users/wichmann/Simulations/Data_MixingTime/' #Data directory #directory of the data.
outdir_paper = '/Users/wichmann/surfdrive/Projects/P2_Mixing/AttractionTimeScales/Paper/Manuscript/paper_figures/' #directory for saving figures for the paper

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

def setup_annual_markov_chain(ddir, outputdir, ddeg):

    #Get unique list of file identifiers
    files = os.listdir(ddir)
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
    
    for n in names:
        setupTM_individual(ddir, n, ddeg=ddeg, dt=-1, Ngrids=40, outputdir=outputdir)
    
    get_annual_TM(outputdir, names)

#setup_annual_markov_chain(ddir = datadir +  'MarkovMixing/year2001/simdays60/',  outputdir = outdir_paper + '/MarkovMatrix/simdays60_ddeg_2/', ddeg=2.)
#setup_annual_markov_chain(ddir = datadir +  'MarkovMixing/year2001/simdays60/',  outputdir = outdir_paper + '/MarkovMatrix/simdays60_ddeg_3/', ddeg=3.)
#setup_annual_markov_chain(ddir = datadir +  'MarkovMixing/year2001/simdays90/',  outputdir = outdir_paper + '/MarkovMatrix/simdays90_ddeg_3/', ddeg=3.)
#setup_annual_markov_chain(ddir = datadir +  'MarkovMixing/year2001/simdays90/',  outputdir = outdir_paper + '/MarkovMatrix/simdays90_ddeg_4/', ddeg=4.)
#setup_annual_markov_chain(ddir = datadir +  'MarkovMixing/year2001/simdays120/',  outputdir = outdir_paper + '/MarkovMatrix/simdays120_ddeg_4/', ddeg=4.)

def setup_transition_matrix_entropy():
    """
    T-matrix for the region definition of the mixing entropy 
    """
    
    pdir = datadir +  'MixingEntropy/' #Data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_' #File name for many particle simulation
    outdir = outdir_paper + 'EntropyMatrix/'
    
    setupTM_individual(datadir=pdir, name=filename, ddeg=2., dt=-1, Ngrids=40, outputdir = outdir)

#setup_transition_matrix_entropy()

def matrix_powers(TMdir):
    """
    To compute some matrix powers. 10th power is used for determining the separate clusters in the ocean. 
    """
    matrix_file = outdir_paper + '/MarkovMatrix/'+TMdir+'/TM_total.npz'
    mfile = np.load(matrix_file)
    tm=mfile['TM']
    
    TM = tm.copy()   
    fname = 'tm0'
    scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + fname,csr_matrix(TM))
    
    for i in range(2,6):
        print 'power: ' + str(i)
        TM=np.dot(TM,tm)
        if i%5==0:
             fname = 'tm' + str(i)
             scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + fname,csr_matrix(TM))
    
    tm=TM.copy() #scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/tm5.npz').toarray()    
    TM=tm.copy()
    
    for i in range(2,22):
        TM=np.dot(TM,tm)
        fname = 'tm' + str(i * 5)
        scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + fname,csr_matrix(TM))
    
    del tm
    del TM
#    
#    tm100=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/tm100.npz').toarray()    
#    tm200=np.dot(tm100,tm100)
#    scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + 'tm200',csr_matrix(tm200))
#    tm200=np.dot(tm100,tm100)
#    scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + 'tm200',csr_matrix(tm200))
#    tm400=np.dot(tm200,tm200)
#    tm500=np.dot(tm400,tm100)
#    del tm400
#    del tm100
#    scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + 'tm500',csr_matrix(tm500))    
#    
#    tm1000=np.dot(tm500,tm500)
#    scipy.sparse.save_npz(outdir_paper + '/MarkovMatrix/'+TMdir+'/' + 'tm1000',csr_matrix(tm1000))        

#matrix_powers(TMdir = 'simdays60_ddeg_2')
#matrix_powers(TMdir = 'simdays60_ddeg_3')
#matrix_powers(TMdir = 'simdays90_ddeg_3')
#matrix_powers(TMdir = 'simdays90_ddeg_4')
#matrix_powers(TMdir = 'simdays120_ddeg_4')