"""
Mixing of passive tracers at the ocean surface and its implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Figures of paper and annex
--------------

Notes:
    - Numbering of basins is done according to (North Pacific, North Atlantic, South Pacific, South Atlantic, Indian Ocean) = (1,2,3,4,5)
"""

import numpy as np
from mixing_class import ParticleData #, square_region #, region_boundary
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import random
import datetime
import scipy.sparse.linalg as sp_linalg
from scipy import sparse
import os
from netCDF4 import Dataset

datadir = '/Users/wichmann/Simulations/Data_MixingTime/' #Data directory #directory of the data.
outdir_paper = './paper_figures/' #directory for saving figures for the paper

#create directories for output
if not os.path.isdir(outdir_paper): os.mkdir(outdir_paper)
if not os.path.isdir(outdir_paper + 'EntropyMatrix'): os.mkdir(outdir_paper + 'EntropyMatrix')
if not os.path.isdir(outdir_paper + 'MarkovMatrix'): os.mkdir(outdir_paper + 'MarkovMatrix')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays45_ddeg1'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays45_ddeg1')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg1'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg1')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg2'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg2')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg3'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg3')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg3'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg3')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg4'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg4')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2001/simdays120_ddeg4'): os.mkdir(outdir_paper + 'MarkovMatrix/year2001/simdays120_ddeg4')

if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2005'): os.mkdir(outdir_paper + 'MarkovMatrix/year2005')
if not os.path.isdir(outdir_paper + 'MarkovMatrix/year2005/simdays60_ddeg2'): os.mkdir(outdir_paper + 'MarkovMatrix/year2005/simdays60_ddeg2')


#For global plots with 2 degree binning
Lons_edges=np.linspace(-180,180,int(360/2.)+1)        
Lats_edges=np.linspace(-90,90,int(180/2.)+1)
Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)


def accumulation_zones():
    """
    Final density of initial uniform particle distribution after 10 years of advection
    """
    
    pdir = datadir + 'MixingEntropy/' #Particle data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], n_grids=40)
    pdata.remove_nans()
    pdata.set_discretizing_values(d_deg = 2.)
    
    #figure of the globe
    fig = plt.figure(figsize = (12,8))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get final distribution
    d_full=pdata.compute_distribution(t=-1) #.flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)
    
    #plot and savedistribution with colorbar
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=10, width=0.05)
    cbar.set_label('# particles per bin', size=10)
    fig.savefig(outdir_paper + 'F1_accumulation_zones.eps', dpi=300, bbox_inches='tight')


def two_colors_northpacific():
    """
    Figure illustrating mixing in the west and east north pacific on a particle level after 10 years of advection
    """
    
    pdir = datadir + 'MixingEntropy/' #Particle data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name

    #for the figure
    plt.figure(figsize = (12,8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], n_grids=40)
    pdata.remove_nans()
    pdata.set_discretizing_values(d_deg = 2.)
    
    #Select particles that start and end in the (here rectangular) north pacific
    square=[115,260,0,65]
    s=pdata.square_region(square)
    l2 = {0: s, -1: s}
    basin_data = pdata.get_subset(l2)
    del pdata

    #define colors according to initial location    
    cols = np.array(['b' if basin_data.lons[i,0]<190 else 'r' for i in range(len(basin_data.lons[:,0]))])

    #initial scatter plot
    plt.subplot(gs1[0])
    m = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=65, llcrnrlon=115,urcrnrlon=260,resolution='c')
    m.drawparallels([15,30,45,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([150,180,210,240], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(basin_data.lons[:,0], basin_data.lats[:,0])
    m.scatter(xs, ys, c=cols, s=.5)
    plt.title('a) initial', size=10, y=1.)
    
    #Randomly order the final particles for the scatter plot (otherwise one color will just be on top)
    lons=basin_data.lons[:,1]
    lats=basin_data.lats[:,1]
    indices=list(range(len(lons)))
    random.shuffle(indices)
    lons = lons[indices]
    lats = lats[indices]
    cols = cols[indices]

    #final scatter plot
    plt.subplot(gs1[1])
    m = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=65, llcrnrlon=115,urcrnrlon=260,resolution='c')
    m.drawparallels([15,30,45,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([150,180,210,240], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgray')
    xs, ys = m(lons, lats)
    m.scatter(xs, ys, c=cols, s=.5)
    plt.title('b) after 10 years', size=10, y=1.)    

    plt.savefig(outdir_paper + 'F2_two_colors_northpacific.eps', bbox_inches='tight')   
    

def setup_transition_matrix_entropy_regions():
    """
    Transport matrix for the full 10-year simulation, used for the definition of regions for the entropy method.
    Binning is 2 degree for selecting the regions, 4,5,6 degrees for the actual entropy (see below)
    """
    
    pdir = datadir +  'MixingEntropy/' #Data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #File name for many particle simulation
    outdir = outdir_paper + 'EntropyMatrix/'    
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], n_grids=40)
    pdata.compute_matrix(d_deg=2., save=True, name= outdir + 'entropy_matrix_regions')


def setup_annual_markov_chain(data_dir, output_dir, d_deg, n_grids):
    """
    Function to compute a Markov matrix from the given method.
    - data_dir: directory with the data
    - output_dir: matrices are saved there
    - d_deg: binning of matrix
    - n_grids: number of initial particle grids
    """
    
    #Get unique list of file identifiers
    files = os.listdir(data_dir)
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

    #reate individual matrices 
    for n in names:
        print('Creating matrix for ', n)
        n_open = n + 'pos'
        pdata=ParticleData.from_nc(data_dir, n_open, tload=[0,-1], n_grids=n_grids)
        pdata.compute_matrix(d_deg=d_deg, save=True, name = output_dir + 'T_' + n)

    
    #create annual matrix
    for i in range(len(names)):
        n = names[i]
        print('Loading matrix for ', n)
        t = sparse.load_npz(output_dir + 'T_' + n + '.npz')
        
        if i == 0 :
            T = t
        else:
            T = T.dot(t)
        
        sparse.save_npz(output_dir + 'Tfull', T)
   
    
def matrix_power(output_dir, power):
    
    T = sparse.load_npz(output_dir + 'Tfull.npz')
    T = sparse.csr_matrix(T)
    
    print('Computing power ', power)
    
    Tx = T**power
    sparse.save_npz(output_dir + 'T' + str(power), Tx)


def regions():
    """
    Determine the regions of the different basins. Need to create the respective matrixes (for entropy and Markov chain) first, see create_matrix.py
    """
    
    #choice of initial regions to initialize the clustering method
    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}

    fig = plt.figure(figsize = (12,8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    #regions entropy
    plt.subplot(gs1[0])
    t = sparse.load_npz(outdir_paper + 'EntropyMatrix/entropy_matrix_regions.npz')
    t2 = sparse.csr_matrix(t)

    final_regions = {}
        

    for r in initial_regions.keys():

        initial_square = initial_regions[r]
        pdata = ParticleData()
        pdata.set_discretizing_values(d_deg = 2.)
        r2=pdata.square_region(square=initial_square)
        r1=np.zeros(len(r2))
        
        #start iteration to find the regions with 50% connectivity
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=t2.dot(r1)
            
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        
        print( 'Entropy: iterations to convergence: ', i)
    
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    np.save(outdir_paper + 'EntropyMatrix/Entropy_Clusters', ocean_clusters)
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0)    

    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,90)
    
    m = Basemap(projection='robin',lon_0=-0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True, vmin=1, vmax=5)
    plt.title(r"a) Regions for Entropy")

        
    #regions markov chain
    plt.subplot(gs1[1])
    t=sparse.load_npz(outdir_paper + 'MarkovMatrix/simdays60_ddeg2/T10.npz').toarray()  
    t2 = sparse.csr_matrix(t)

    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        pdata = ParticleData()
        pdata.set_discretizing_values(d_deg = 2.)
        r2=pdata.square_region(square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=t2.dot(r1)
            
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print( 'Markov: iterations to convergence: ', i)
        
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays60_ddeg_2', ocean_clusters)

    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
    ocean_clusters=np.roll(ocean_clusters,90)
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"b) Regions for Markov Chain")

    fig.savefig(outdir_paper + 'F3_Clusters.eps', dpi=300, bbox_inches='tight')   


def entropies(figure_title, d_deg):
    """
    Function to compute for each bin the entropy of mixing of different particle species.
    
    Three functions:
        1. reduce_particleset() creates npz files with the particles that stay in a respective basin. 
           This functions requires the existence of and array Entropy_clusters.npy, created above.
        2. compute_transfer_matrix() consutructs, based on the reduced particle sets, the matrix T_ik defined in the methods section
        3. plot_spatial_entropy() plots the entropy as a global map
    """

    tload = list(range(0,730,73))+[729] #load data each year (indices, not actual times)
    time_origin=datetime.datetime(2000,1,5)
    Times = [(time_origin + datetime.timedelta(days=t*5)).strftime("%Y-%m") for t in tload]    
    
    def reduce_particleset():
        
        #Load particle data
        pdir = datadir + 'MixingEntropy/' #Data directory
        fname = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #F

        #load data
        pdata=ParticleData.from_nc(pdir, fname, tload=tload, n_grids=40)
        pdata.set_discretizing_values(d_deg = 2.)
        pdata.remove_nans()

        #Get those particles that start and end in the chosen basin
        r = np.load(outdir_paper + "EntropyMatrix/Entropy_Clusters_2deg.npy")
        
        for i_basin in range(1,6): #loop over basins as defined in figure 3a)
            
            print( '--------------')
            print( 'BASIN: ', i_basin)
            print( '--------------')
            
            #define basin region
            basin = np.array([1 if r[i]==i_basin else 0 for i in range(len(r))])
            
            #constrain to particles that are in the respective basin after each year
            l={}
            for t in range(len(tload)):
                l[t]=basin
            basin_data = pdata.get_subset(l)
            
            lons=basin_data.lons.filled(np.nan)
            lats=basin_data.lats.filled(np.nan)
            np.savez(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin), lons=lons, lats=lats)
            

    def compute_transfer_matrix():
        #deg_labels is the choice of square binning
        
        for i_basin in range(1,6):
            
            #load reduced particle data for each basin
            pdata = np.load(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin) + '.npz', 'r')
            lons=pdata['lons']
            lats=pdata['lats']

            del pdata
            pdata = ParticleData(lons=lons, lats=lats)
            
            #compute transfer matrix
            for t in range(0,len(lons[0])):     
                
                pdata.compute_matrix(d_deg=d_deg, t0=0, t1=t)            
                sparse.save_npz(outdir_paper + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '/T_matrix_t' + str(t) + '_basin' + str(i_basin), pdata.A) #, original_labels=original_labels)

            
    def plot_spatial_entropy():
        #function to get the spatial entropy
        
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
        
        fig = plt.figure(figsize = (12,8))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.15, hspace=0.)
        
        labels = ['a) ', 'b) ', 'c) ', 'd) ']
    
        for t, k in zip([1,3,6,10],range(4)):
            T=Times[t]

            S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
            
            for i_basin in range(1,6):
                #load data
                A = sparse.load_npz(outdir_paper + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '/T_matrix_t' + str(t) + '_basin' + str(i_basin) + '.npz')
                row = A.row
                col = A.col
                val = A.data                
                
                colsum = np.array(sparse.coo_matrix.sum(A, axis=0))[0]
                
                N = len(np.unique(row)) #number of labels
                S_max = np.log(N)
                
                #column-normalize                
                for c in np.unique(col):
                        val[col==c] /= colsum[c]

                for c in np.unique(col):
                    s = 0.
                    for p in val[col==c]:
                        if p!=0:
                            s-=p * np.log(p)

                    S_loc[c] = s /S_max
            

            plt.subplot(gs1[k])

            S_loc=S_loc.reshape((len(Lats_centered),len(Lons_centered)))
            S_loc=np.roll(S_loc,int(180/d_deg))
            m = Basemap(projection='robin',lon_0=0,resolution='c')
            m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawcoastlines()
            m.fillcontinents(color='lightgrey')
            
            lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)
            xs, ys = m(lon_bins_2d, lat_bins_2d)        
            assert (np.max(S_loc)<=1)
            p = plt.pcolormesh(xs, ys, S_loc,cmap='magma', vmin=0, vmax=1, rasterized=True)
            plt.title(labels[k] + str(T), size=12, y=1.01)
        
        #color bar on the right
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(p, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label(r'$S/S_{max}$',size=12)        
        fig.savefig(outdir_paper + figure_title, dpi=300, bbox_inches='tight')

    reduce_particleset()
    compute_transfer_matrix()
    plot_spatial_entropy()


def stationary_distributions(figure_name, TMname, ddeg):
    """
    Functions to:
        1. Project the global transition matrix to the basin wide matrices
        2. Compute the stationary densities
        3. plot the stationary densities
    Parameters:
        figure_name: for saving the figure
        TMname: name of file containing the global transition matrix
        ddeg: binning of transition matrix
    """
    
    def project_matrices():
        
        #Load matrix and region definitions
        A = sparse.load_npz(outdir_paper + 'MarkovMatrix/' + TMname + '/T_total.npz')
        A=sparse.coo_matrix(A)
        ocean_clusters = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/Markov_Clusters.npy')

        row = A.row
        col = A.col
        val = A.data                       
        
        #loop over basins
        for basin_number in range(1,6):
            print( 'basin: ', basin_number)

            project = np.array([i for i in range(len(ocean_clusters)) if ocean_clusters[i] == basin_number])

            
            print( 'projecting')
            
            #project matrix
            inds = [i for i in range(len(col)) if (col[i] in project and row[i] in project)]
            
            row_new = row[inds]
            col_new = col[inds]
            val_new = val[inds]
            A_new = sparse.coo_matrix((val_new, (row_new, col_new)), shape=A.shape) 
            
            print('rowsum')
            rowsum = np.array(sparse.coo_matrix.sum(A_new, axis=1))[:,0]
            
            #column-normalize                
            print('column normalize')
            for r in np.unique(row_new):
                val_new[row_new==r] /= rowsum[r]

            A_new.data = val_new

            print('projecting done')
            sparse.save_npz(outdir_paper + 'MarkovMatrix/' + TMname + '/T_basin_' + str(basin_number), A_new)

         
    def get_stationary_densities():
        
        for basin_number in range(1,6):
            print( 'basin_number: ', basin_number)
            A = sparse.load_npz(outdir_paper + 'MarkovMatrix/' + TMname + '/T_basin_' + str(basin_number) + '.npz')
            val, vec = sp_linalg.eigs(A.transpose(),k=5,which='LM')    
            vec = vec[:,np.argsort(np.abs(val))]
            val = val[np.argsort(np.abs(val))]            
            d0=np.array(vec[:,-1])
            d0/=np.sum(d0) #Normalized eigenvector with eigenvalue 1
            np.save(outdir_paper + 'MarkovMatrix/' + TMname + '/stationary_density' + str(basin_number), d0)
    
    
    def plot_stationary_densities():        
        
        Lons_edges=np.linspace(-180,180,int(360/ddeg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/ddeg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/ddeg for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/ddeg for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        
        patches = {}
        for i in range(1,6):
            patches[i] = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/stationary_density' + str(i) + '.npy')
        
        #for the figure
        fig = plt.figure(figsize = (12,12))
        gs1 = gridspec.GridSpec(3, 2)
        gs1.update(wspace=0.15, hspace=0.)
    
        #North Pacific
        plt.subplot(gs1[0])
        d0 = np.real(patches[1])
        d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
        d0=np.roll(d0,int(180/ddeg))
        d0/=np.max(d0)
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_edges_2d, lat_edges_2d)
        pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', vmin=0, vmax=1, rasterized=True)
        plt.title('a) North Pacific', size=12)
    
        #North Atlantic
        plt.subplot(gs1[1])
        d0 = np.real(patches[2])
        d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
        d0=np.roll(d0,int(180/ddeg))
        d0/=np.max(d0)
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_edges_2d, lat_edges_2d)
        pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', vmin=0, vmax=1, rasterized=True)
        plt.title('b) North Atlantic', size=12)
            
        #South Pacific
        plt.subplot(gs1[2])
        d0 = np.real(patches[3])
        d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
        d0=np.roll(d0,int(180/ddeg))
        d0/=np.max(d0)
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_edges_2d, lat_edges_2d)
        pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', vmin=0, vmax=1, rasterized=True)
        plt.title('c) South Pacific', size=12)
    
        #South Atlantic
        plt.subplot(gs1[3])
        d0 = np.real(patches[4])
        d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
        d0=np.roll(d0,int(180/ddeg))
        d0/=np.max(d0)
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_edges_2d, lat_edges_2d)
        pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', vmin=0, vmax=1, rasterized=True)
        plt.title('d) South Atlantic', size=12)

        #Indian Ocean
        plt.subplot(gs1[4])
        d0 = np.real(patches[5])
        d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
        d0=np.roll(d0,int(180/ddeg))
        d0/=np.max(d0)
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_edges_2d, lat_edges_2d)
        pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', vmin=0, vmax=1, rasterized=True)
        plt.title('e) Indian Ocean', size=12)
    
        cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(pmesh, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(r'$\rho/\rho_{max}$', size=13)
                
        fig.savefig(outdir_paper + figure_name + TMname + '.eps', dpi=300, bbox_inches='tight')
    
    project_matrices()
    get_stationary_densities()
    plot_stationary_densities()


def Markkov_mixing_times(eps, TMname, figure_name, ddeg):
    """
    Computation and plot of mixing times
    """
    
    Lons_edges=np.linspace(-180,180,int(360/ddeg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/ddeg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)

    def get_tmix():
        
        for basin_number in range(1,6):
            print( 'basin_number: ', basin_number)
            T0 = sparse.load_npz(outdir_paper + 'MarkovMatrix/' + TMname + '/T_basin_' + str(basin_number) + '.npz').toarray()
            d0 = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/stationary_density' + str(basin_number) + '.npy')
            d0 = np.real(d0)
            
            tmix = np.array([-100]*len(d0))
            
            T=T0.copy()
            
            for t in range(0,30):
                print( t)
                print( '----------')
                for i in range(T.shape[0]):
                    if tmix[i] < 0:
                        if .5 * np.sum(np.abs(d0-T[i,:]))<eps:
                            tmix[i]=t      
                T=np.dot(T0,T)
            
            np.save(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_' + 'basin_' +str(basin_number), tmix)

        
    def plot_tmix():
        
        #for the figure
        fig = plt.figure(figsize = (12,12))
        gs1 = gridspec.GridSpec(3, 2)
        gs1.update(wspace=0.15, hspace=0.)
        levels=[0,2,6,10,15,20,25,30]
        
        tmix_max=30

        #North Pacific
        plt.subplot(gs1[0])
        tmix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_' + 'basin_1.npy')
        tmix=np.array([float(t) for t in tmix])
        tmix[tmix<0]=np.nan
        tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
        tmix=np.roll(tmix,int(180/ddeg))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        xs, ys = m(lon_centered_2d, lat_centered_2d)
        pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max) #, norm=norm)
        plt.title('a) North Pacific', size=12)
    
        #North Atlantic
        plt.subplot(gs1[1])
        tmix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_' + 'basin_2.npy')
        tmix=np.array([float(t) for t in tmix])
        tmix[tmix<0]=np.nan
        tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
        tmix=np.roll(tmix,int(180/ddeg))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)    
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_centered_2d, lat_centered_2d)
        pmesh = plt.contourf(xs, ys, tmix, levels=levels , cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max) #, norm=norm)
        plt.title('b) North Atlantic', size=12)
            
        #South Pacific
        plt.subplot(gs1[2])
        tmix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_' +  'basin_3.npy')
        tmix=np.array([float(t) for t in tmix])
        tmix[tmix<0]=np.nan
        tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
        tmix=np.roll(tmix,int(180/ddeg))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)  
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_centered_2d, lat_centered_2d)
        pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max) #, norm=norm)
        plt.title('c) South Pacific', size=12)
    
        #South Atlantic
        plt.subplot(gs1[3])
        tmix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_' +  'basin_4.npy')
        tmix=np.array([float(t) for t in tmix])
        tmix[tmix<0]=np.nan
        tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
        tmix=np.roll(tmix,int(180/ddeg))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)  
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_centered_2d, lat_centered_2d)
        pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max) #, norm=norm)
        plt.title('d) South Atlantic', size=12)

        #Indian Ocean
        plt.subplot(gs1[4])
        tmix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/tmix_eps' + str(int(eps*100)) + '_'  + 'basin_5.npy')
        tmix=np.array([float(t) for t in tmix])
        tmix[tmix<0]=np.nan
        tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
        tmix=np.roll(tmix,int(180/ddeg))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)  
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_centered_2d, lat_centered_2d)
        pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max) #, norm=norm)
        plt.title('e) Indian Ocean', size=12)

        cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(pmesh, cax=cbar_ax, extend='max') #, ticks=[1,5,10,15,20])
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(r'$t_{mix}$ [years]', size=12)
        
        fig.savefig(outdir_paper + figure_name, dpi=300, bbox_inches='tight')
        
    get_tmix()
    plot_tmix()


def Markov_Entropy(TMname, figure_name, deg_labels):

        Lons_edges=np.linspace(-180,180,int(360/deg_labels)+1)        
        Lats_edges=np.linspace(-90,90,int(180/deg_labels)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        
        fig = plt.figure(figsize = (12,8))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.15, hspace=0.)
        
        labels = ['a) ', 'b) ', 'c) ', 'd) ']
    
        from numpy.linalg import matrix_power
    
        for t,k in zip([1,3,6,10],range(4)):
            
            S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
                    
            for basin_number in range(1,6):
                print( 'basin_number: ', basin_number)
                T0 = sparse.load_npz(outdir_paper + 'MarkovMatrix/' + TMname + '/T_basin_' + str(basin_number) + '.npz').toarray()

                print( 'power: ', t)
                T=matrix_power(T0,t)
                
                #column-normalize
                for i in range(len(T)):
                    s=np.sum(T[:,i])
                    if s>1e-12:
                        T[:,i]/=s
                
                #Compute entropy for each location
                S=np.zeros(len(S_loc))
                for j in range(len(T)):
                    s=0
                    for i in range(len(T)):
                        if T[i,j]>0:
                            s-=T[i,j] * np.log(T[i,j])
                    
                    S[j]=s
                
                #maximum entropy. col sum of TM0
                su=np.sum(T0, axis=1)               
                N=len(su[su!=0])
                maxS=np.log(N)
                
                S/=maxS
                
                S_loc+=S        
            
            print( 'max S_loc: ', np.max(S_loc))

            plt.subplot(gs1[k])

            S_loc=S_loc.reshape((len(Lats_centered),len(Lons_centered)))
            S_loc=np.roll(S_loc,int(180/deg_labels))
            m = Basemap(projection='robin',lon_0=0,resolution='c')
            m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawcoastlines()
            m.fillcontinents(color='lightgrey')
            
            lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)
            xs, ys = m(lon_bins_2d, lat_bins_2d)        
            p = plt.pcolormesh(xs, ys, S_loc,cmap='magma', vmin=0, vmax=1, rasterized=True)
            plt.title(labels[k] + 'Year ' + str(t), size=12, y=1.01)
        
        #color bar on the right
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(p, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label(r'$S/S_{max}$',size=12)        
        fig.savefig(outdir_paper + figure_name, dpi=300, bbox_inches='tight')
        

def initialregions():
    
    def region_boundary(region):
        """
        Function to return the boundaries of a region given by an indicator (1 or 0) array
        """
        boundary = np.zeros(region.shape)
    
        for i in range(len(region)):
            left_idx = i-1 if i>0 else len(region)-1
            right_idx = i+1 if i< len(region)-1 else 0
            for j in range(len(region[0])):
               upper_idx = j+1 if j < len(region[0])-1 else 0
               lower_idx = j-1 if j>0 else len(region[0])-1        
               if region[i, j]==1 and (region[i, upper_idx]==0 or region[i, lower_idx]==0 or region[left_idx, j]==0 or region[right_idx, j]==0):
                    boundary[i,j]=1
        
        return boundary


    pdir = '/Users/wichmann/Simulations/Proj1_SubSurface_Mixing/Layer0/' #Data directory
    filename = 'SubSurf_y2000_m1_d5_simdays3650_layer0_pos' #File name for many particle simulation
    
    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    r=np.zeros(len(Lons_centered) * len(Lats_centered))
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], n_grids=40)
    pdata.remove_nans()
    pdata.set_discretizing_values(d_deg = 2.)
    
    for ri in initial_regions.keys():
        initial_square = initial_regions[ri]
        r+=pdata.square_region(square=initial_square)
    
    r=r.reshape((len(Lats_centered),len(Lons_centered)))    
    r=np.roll(r,90)    
    b=region_boundary(r)
    b = np.ma.masked_array(b, b==0) 
    
    #for the figure
    fig = plt.figure(figsize = (12,8))
    
    #figure of the globe
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get the distribution
    d_full=pdata.compute_distribution(t=-1) #.flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)    
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full, cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=10, width=0.05)
    cbar.set_label('# particles per bin', size=10)

    plt.pcolormesh(xs, ys, b, cmap='summer', rasterized=True)
    fig.savefig(outdir_paper + 'S1_initial_regions.pdf', dpi=300, bbox_inches='tight')
    


def other_matrix_powers(output_dir):
    
    T = sparse.load_npz(output_dir + 'T_total.npz')    
    T = sparse.csr_matrix(T)
    sparse.save_npz(output_dir + 'T0', T)

    T5 = T**5
    sparse.save_npz(output_dir + 'T5', T5)

    T15 = T5**3
    sparse.save_npz(output_dir + 'T15', T15)
    
    T20 = T15.dot(T5)
    sparse.save_npz(output_dir + 'T20', T20)    
    del T15
    
    T25 = T20.dot(T5)
    sparse.save_npz(output_dir + 'T25', T25)        

    T30 = T25.dot(T5)
    sparse.save_npz(output_dir + 'T30', T30)        
    del T25
    
    T40 = T20**2
    sparse.save_npz(output_dir + 'T40', T40)        
    del T40
    
    T50 = T20.dot(T30)
    sparse.save_npz(output_dir + 'T50', T50)
    del T20
    del T30
    del T5
    
    T100 = T50**2
    sparse.save_npz(output_dir + 'T100', T100)    
    del T50
    
    T200 = T100**2
    sparse.save_npz(output_dir + 'T200', T200)
    
    T500 = T200**2
    T500 = T500.dot(T100)
    del T100
    del T200
    sparse.save_npz(output_dir + 'T500', T500)    
    
    T1000 = T500**2
    sparse.save_npz(output_dir + 'T1000', T1000)


def regions_otherpowers():

    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}
    
    figlabels = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ', 'h) ', 'i) ', 'j) ', 'k) ', 'l) ']
    
    fig = plt.figure(figsize = (12,10))
    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0.15, hspace=0.0)

    pdata = ParticleData()
    pdata.set_discretizing_values(d_deg = 2.)    

    tmno=[0,5,15,20,25,30,40,50,100,200,500,1000]
    for ki in range(len(tmno)):
   
        mfile = outdir_paper + '/MarkovMatrix/year2001/simdays60_ddeg2/T' + str(tmno[ki]) + '.npz'
        plt.subplot(gs1[ki])
        t=sparse.load_npz(mfile)
        t2 = sparse.csr_matrix(t)
        
        final_regions = {}
            
        for r in initial_regions.keys():
            initial_square = initial_regions[r]
            r2=pdata.square_region(square=initial_square)
            r1=np.zeros(len(r2))
            
            i=0
            while np.any((r1-r2)!=0):
                i+=1
                r1=r2.copy()
                d=t2.dot(r1)
                
                r2[np.argwhere(d>=.5)]=1
                r2[np.argwhere(d<.5)]=0
            
            #some tests
            assert(not np.any(r2[d<0.5]))
            final_regions[r]=r2
            
        ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))
    
        for i in range(len(ocean_clusters)):
            for k in final_regions.keys():
                if final_regions[k][i]==1:
                    ocean_clusters[i]=labels[k]
    
        ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
        
        ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
        ocean_clusters=np.roll(ocean_clusters,90)

        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color = 'grey', linewidth=1.2, size=8)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color = 'grey', linewidth=1.2, size=8)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        xs, ys = m(lon_edges_2d, lat_edges_2d) 
        plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
        plt.title(figlabels[ki] + 'Power: ' +  str(tmno[ki]), size=8)

    fig.savefig(outdir_paper + 'S2_Clusters_otherpowers.eps', dpi=300, bbox_inches='tight')
    


def regions_TMs():
    """
    Clusters based on other matrices
    """    
    
    def get_clusters(tfile, d_deg):
    
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        
        #choice of initial regions to initialize the clustering method
        initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                           'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                           'IO': (40.,100., -45., -15.)}
        labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}
    
        t = sparse.load_npz(tfile)
        t = sparse.csr_matrix(t)
        final_regions = {}
         
        pdata = ParticleData()
        pdata.set_discretizing_values(d_deg = d_deg)
    
        for r in initial_regions.keys():
            initial_square = initial_regions[r]
            r2=pdata.square_region(square=initial_square)
            r1=np.zeros(len(r2))
            
            i=0
            while np.any((r1-r2)!=0):
                i+=1
                r1=r2.copy()
                d=t.dot(r1)
                r2[np.argwhere(d>=.5)]=1
                r2[np.argwhere(d<.5)]=0
            
            #some tests
            assert(not np.any(r2[d<0.5]))
            final_regions[r]=r2
            print( 'Markov 1: iterations to convergence: ', i)
        
        ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))
    
        for i in range(len(ocean_clusters)):
            for k in final_regions.keys():
                if final_regions[k][i]==1:
                    ocean_clusters[i]=labels[k]
    
        ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
        ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
        ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))

        return ocean_clusters, lon_edges_2d, lat_edges_2d

    
    fig = plt.figure(figsize = (12,8))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    plt.subplot(gs1[0])
    ocean_clusters, lon_edges_2d, lat_edges_2d = get_clusters(outdir_paper + '/MarkovMatrix/year2001/simdays60_ddeg_3/tm10.npz', d_deg=3.)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"a) Markov chain $\Delta x = 3^\circ$, $\Delta t = 60$ days")

    plt.subplot(gs1[1])
    ocean_clusters, lon_edges_2d, lat_edges_2d = get_clusters(outdir_paper + '/MarkovMatrix/year2001/simdays60_ddeg_3/tm10.npz', d_deg=3.)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"b) Markov chain $\Delta x = 3^\circ$, $\Delta t = 90$ days")
    
    plt.subplot(gs1[2])
    ocean_clusters, lon_edges_2d, lat_edges_2d = get_clusters(outdir_paper + '/MarkovMatrix/year2001/simdays90_ddeg_4/tm10.npz', d_deg=4.)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"c) Markov chain $\Delta x = 4^\circ$, $\Delta t = 90$ days")
    
    plt.subplot(gs1[3])
    ocean_clusters, lon_edges_2d, lat_edges_2d = get_clusters(outdir_paper + '/MarkovMatrix/year2001/simdays120_ddeg_4/tm10.npz', d_deg=4.)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"d) Markov chain $\Delta x = 4^\circ$, $\Delta t = 120$ days")

    fig.savefig(outdir_paper + 'S5_Clusters_otherTMs.eps', dpi=300, bbox_inches='tight')


def waste_input():
    
    tload = list(range(0,730,73))+[729] #load data each year (indices, not actual times)    
    time_origin=datetime.datetime(2000,1,5)
    Times = [(time_origin + datetime.timedelta(days=t*5)).strftime("%Y-%m") for t in tload]    
    
    def reduce_pset():
        d_deg = 2.
        
        data    = Dataset('../waste_input/releasefunc_wasteinput.nc','r')
        Lons = np.array(data['Longitude'])
        Lats = np.array(data['Latitude'])
        waste = np.array(data['Plastic_waste_input'])
        
        from scipy.interpolate import griddata
        
        lons, lats = np.meshgrid(Lons,Lats)
        
        Lons_edges=np.linspace(0,360,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])  
    
        lon_centered_waste,lat_centered_waste = np.meshgrid(Lons_centered, Lats_centered)
        waste_points = griddata(np.array([lons.flatten(), lats.flatten()]).T, waste.flatten(), (lon_centered_waste, lat_centered_waste), method='nearest')
        waste_points[waste_points>0]=1
        
        pdir = datadir + 'MixingEntropy/' #Particle data directory
        filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name
        pdata=ParticleData.from_nc(pdir, filename, tload=tload, n_grids=40)
        pdata.remove_nans()
        pdata.set_discretizing_values(d_deg = d_deg)
        pdata_waste = pdata.get_subset({0: waste_points.ravel()})
        pdata_waste.set_discretizing_values(d_deg = d_deg)
        
        d_full = pdata_waste.compute_distribution(0)
        d_full=np.roll(d_full,int(180/d_deg))
        
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        
        plt.figure(figsize = (12,8))
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
        m.drawcoastlines()
        xs, ys = m(lon_edges_2d, lat_edges_2d) 
        plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
        cbar=plt.colorbar(orientation='vertical',shrink=0.5)
        cbar.ax.tick_params(labelsize=10, width=0.05)
        cbar.set_label('# particles per bin', size=10)
        plt.title('Remaining particles')
        plt.show()
        
        #Get those particles that start and end in the chosen basin
        r = np.load(outdir_paper + "EntropyMatrix/Entropy_Clusters_2deg.npy")
        
        #reduce to total pset that we use
        basin = np.array([1 if r[i]!=0 else 0 for i in range(len(r))])
        
        #constrain to particles that are in the respective basin after each year
        l={}
        for t in range(len(tload)):
            l[t]=basin
        waste_data = pdata_waste.get_subset(l)
        
        lons=waste_data.lons.filled(np.nan)
        lats=waste_data.lats.filled(np.nan)
        print('REMAINING WASTE: ', len(lons))
        np.savez(outdir_paper + 'EntropyMatrix/Reduced_total_particles_waste', lons=lons, lats=lats)
        
        for i_basin in range(1,6): #loop over basins as defined in figure 3a)
            
            print( '--------------')
            print( 'BASIN: ', i_basin)
            print( '--------------')
            
            #define basin region
            basin = np.array([1 if r[i]==i_basin else 0 for i in range(len(r))])
            
            #constrain to particles that are in the respective basin after each year
            l={}
            for t in range(len(tload)):
                l[t]=basin
            basin_data = pdata_waste.get_subset(l)
            
            lons=basin_data.lons.filled(np.nan)
            lats=basin_data.lats.filled(np.nan)
            np.savez(outdir_paper + 'EntropyMatrix/Reduced_particles_waste_' + str(i_basin), lons=lons, lats=lats)


    def compute_transfer_matrix():
        #deg_labels is the choice of square binning
        d_deg = 4.
        for i_basin in range(1,6):
            
            #load reduced particle data for each basin
            pdata = np.load(outdir_paper + 'EntropyMatrix/Reduced_particles_waste_' + str(i_basin) + '.npz', 'r')
            lons=pdata['lons']
            lats=pdata['lats']

            del pdata
            pdata = ParticleData(lons=lons, lats=lats)
            
            #compute transfer matrix
            for t in range(0,len(lons[0])):     
                
                pdata.compute_matrix(d_deg=d_deg, t0=0, t1=t)            
                sparse.save_npz(outdir_paper + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '/T_matrix_waste_t' + str(t) + '_basin' + str(i_basin), pdata.A) #, original_labels=original_labels)

            
    def plot_spatial_entropy():
        #function to get the spatial entropy
        
        d_deg = 4.
        
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
        
        fig = plt.figure(figsize = (12,8))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.15, hspace=0.)
        
        labels = ['a) ', 'b) ', 'c) ', 'd) ']
    
        for t, k in zip([1,3,6,10],range(4)):
            T=Times[t]

            S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
            
            for i_basin in range(1,6):
                #load data
                A = sparse.load_npz(outdir_paper + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '/T_matrix_waste_t' + str(t) + '_basin' + str(i_basin) + '.npz')
                row = A.row
                col = A.col
                val = A.data                
                
                colsum = np.array(sparse.coo_matrix.sum(A, axis=0))[0]
                
                N = len(np.unique(row)) #number of labels
                S_max = np.log(N)
                
                #column-normalize                
                for c in np.unique(col):
                        val[col==c] /= colsum[c]

                for c in np.unique(col):
                    s = 0.
                    for p in val[col==c]:
                        if p!=0:
                            s-=p * np.log(p)

                    S_loc[c] = s /S_max
            

            plt.subplot(gs1[k])

            S_loc=S_loc.reshape((len(Lats_centered),len(Lons_centered)))
            S_loc=np.roll(S_loc,int(180/d_deg))
            m = Basemap(projection='robin',lon_0=0,resolution='c')
            m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2, size=9)
            m.drawcoastlines()
            m.fillcontinents(color='lightgrey')
            
            lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)
            xs, ys = m(lon_bins_2d, lat_bins_2d)        
            assert (np.max(S_loc)<=1)
            p = plt.pcolormesh(xs, ys, S_loc,cmap='magma', vmin=0, vmax=1, rasterized=True)
            plt.title(labels[k] + str(T), size=12, y=1.01)
        
        #color bar on the right
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(p, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label(r'$S/S_{max}$',size=12)        
        fig.savefig(outdir_paper + 'entropy_waste', dpi=300, bbox_inches='tight')

#    reduce_pset()
    compute_transfer_matrix()
    plot_spatial_entropy()

waste_input()



#plot_waste_input()
#    
#
#def lambda_2(TMname='simdays60_ddeg_2', figure_name='figname'):
#    
#    for basin_number in range(1,6):
#        print 'basin_number: ', basin_number
#        TM0 = np.load(outdir_paper + 'MarkovMatrix/TM_' + TMname + '_basin_' + str(basin_number) + '.npy')
#        val, vec = sp_linalg.eigs(TM0,k=10,which='LM')    
##        vec = vec[:,np.argsort(np.abs(val))]
#        val = val[np.argsort(np.abs(val))]
#        val=val[::-1]       
#        print val
#        print np.abs(val[1])
##
##lambda_2()



"""
Figure 1
"""
#accumulation_zones()

"""
Figure 2
"""
#two_colors_northpacific()


"""
set up all Markov matrices for the study
"""

#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays60/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg2/', d_deg=2., n_grids=40) #matrix for main text
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays45/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays45_ddeg1/', d_deg=1., n_grids=40)
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays60/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg1/', d_deg=1., n_grids=40)
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays60/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg3/', d_deg=3., n_grids=40)
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays90/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg3/', d_deg=3., n_grids=40)
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays90/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays90_ddeg4/', d_deg=4., n_grids=40)
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2001/simdays120/', output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays120_ddeg4/', d_deg=4., n_grids=40)

#WHEN SIMULATION IS DONE
#setup_annual_markov_chain(data_dir = datadir + 'MarkovMixing/year2005/simdays60/', output_dir = outdir_paper + 'MarkovMatrix/year2005/simdays60_ddeg2/', d_deg=2., n_grids=5)

"""
Figure 3 (clusters)
"""
#setup_transition_matrix_entropy() #entropy
#matrix_power(output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg2/', power=10) #Markov chain
#regions()

"""
Figure 4: entropy
"""
#entropies(figure_title = 'F4_Sloc.eps', d_deg=5.)

"""
Figure 5: stationary distributions
"""
#stationary_distributions(figure_name = 'F5_GarbagePatches_', TMname = 'year2001/simdays60_ddeg2', ddeg=2.)

"""
Figure 6: mixing time
"""
#Markkov_mixing_times(eps=0.25, TMname='year2001/simdays60_ddeg2' ,figure_name = 'F6_Markov_MixingTime_simdays60_ddeg_2.eps', ddeg=2.)

"""
Figure 7: entropy from Markov chain
"""
#Markov_Entropy(TMname='year2001/simdays60_ddeg2', figure_name='F7_TMentropy_simdays60_ddeg_2.eps', deg_labels=2.)


"""
Figure S1: initial regions
"""
#initialregions()   

"""
Figure S2: clusters with other matrix powers
"""
#other_matrix_powers(output_dir = outdir_paper + 'MarkovMatrix/year2001/simdays60_ddeg2/')
#regions_otherpowers()

"""
Figures S3 and S4: entropies with different binning
"""    
#entropies(figure_title = 'S3_Sloc_deg4.eps', d_deg=4.)
#entropies(figure_title = 'S4_Sloc_deg6.eps', d_deg=6.)

"""
Figure S5: regions for other matrices
"""
#regions_TMs()

"""
Figures S6-S9: stationary distributions other Markov matrices
"""

#stationary_distributions(figure_name = 'S6_GarbagePatches_', TMname = 'simdays60_ddeg_3', ddeg=3.)
#stationary_distributions(figure_name = 'S7_GarbagePatches_', TMname = 'simdays90_ddeg_3', ddeg=3.)
#stationary_distributions(figure_name = 'S8_GarbagePatches_', TMname = 'simdays90_ddeg_4', ddeg=4.)
#stationary_distributions(figure_name = 'S9_GarbagePatches_', TMname = 'simdays120_ddeg_4', ddeg=4.)

"""
Figures S10-S14: Markov mixing times other Markov matrices and other epsilon
"""
#Markkov_mixing_times(eps=0.1, TMname='simdays60_ddeg_2' ,figure_name = 'S10_Markov_MixingTime_simdays60_ddeg_2_eps10.eps', ddeg=2.)
#Markkov_mixing_times(eps=0.25, TMname='simdays60_ddeg_3' ,figure_name = 'S11_Markov_MixingTime_simdays60_ddeg_3.eps', ddeg=3.)
#Markkov_mixing_times(eps=0.25, TMname='simdays90_ddeg_3' ,figure_name = 'S12_Markov_MixingTime_simdays90_ddeg_3.eps', ddeg=3.)
#Markkov_mixing_times(eps=0.25, TMname='simdays90_ddeg_4' ,figure_name = 'S13_Markov_MixingTime_simdays90_ddeg_4.eps', ddeg=4.)
#Markkov_mixing_times(eps=0.25, TMname='simdays120_ddeg_4' ,figure_name = 'S14_Markov_MixingTime_simdays120_ddeg_4.eps', ddeg=4.)    

"""
Figures S15-S18: Markov chain entropies for other matrices
"""
#Markov_Entropy(TMname='simdays60_ddeg_3', figure_name='S15_TMentropy_simdays60_ddeg_3.eps', deg_labels=3.)
#Markov_Entropy(TMname='simdays90_ddeg_3', figure_name='S16_TMentropy_simdays90_ddeg_3.eps', deg_labels=3.)
#Markov_Entropy(TMname='simdays90_ddeg_4', figure_name='S17_TMentropy_simdays90_ddeg_4.eps', deg_labels=4.)
#Markov_Entropy(TMname='simdays120_ddeg_4', figure_name='S18_TMentropy_simdays120_ddeg_4.eps', deg_labels=4.)