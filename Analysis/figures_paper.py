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
from ana_objects import ParticleData, square_region, region_boundary
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import random
import datetime
import scipy.sparse.linalg as sp_linalg
import scipy.sparse

datadir = '/Users/wichmann/Simulations/Data_MixingTime/' #Data directory #directory of the data.
outdir_paper = '/Users/wichmann/surfdrive/Projects/P2_Mixing/AttractionTimeScales/Paper/Manuscript/paper_figures/' #directory for saving figures for the paper

#For global plots with 2 degree binning
Lons_edges=np.linspace(-180,180,int(360/2.)+1)        
Lats_edges=np.linspace(-90,90,int(180/2.)+1)
Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)

def F1_accumulation_zones():
    pdir = datadir + 'MixingEntropy/' #Particle data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], Ngrids=40)
    
    #figure of the globe
    fig = plt.figure(figsize = (12,8))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get final distribution
    d_full=pdata.get_distribution(t=-1, ddeg=2.).flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)
    
    #plot and savedistribution with colorbar
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=10, width=0.05)
    cbar.set_label('# particles per bin', size=10)
    fig.savefig(outdir_paper + 'F1_accumulation_zones.eps', dpi=300, bbox_inches='tight')

#F1_accumulation_zones()

def F2_two_colors_northpacific():
    pdir = datadir + 'MixingEntropy/' #Particle data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name

    #for the figure
    plt.figure(figsize = (12,8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], Ngrids=40)
    pdata.remove_nans()
    
    #Select particles that start and end in the (here rectangular) north pacific
    square=[115,260,0,65]
    ddeg=1. #binning for the region definition
    s=square_region(ddeg, square)
    l2 = {0: s, -1: s}
    basin_data = pdata.get_subset(l2, ddeg)
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
    indices=range(len(lons))
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
    
#F2_two_colors_northpacific()

def Fig3_regions():
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
    data=np.load(outdir_paper + 'EntropyMatrix/TM_surfaceparticles_y2000_m1_d5_simdays3650_.npz') #created in create_matrix.py
    t=data['TM']
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=2., square=initial_square)
        r1=np.zeros(len(r2))
        
        #start iteration to find the regions with 50% connectivity
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Entropy: iterations to convergence: ', i
        
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'EntropyMatrix/Entropy_Clusters', ocean_clusters)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,90)
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"a) Regions for Entropy")
    
    #regions markov chain
    plt.subplot(gs1[1])
    t=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/simdays60_ddeg_2/tm10.npz').toarray()    
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=2., square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Markov: iterations to convergence: ', i
        
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays60_ddeg_2', ocean_clusters)
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
    
#Fig3_regions()


def Fig4_S3_S4_entropies(figure_title, deg_labels):
    """
    Function to compute for each bin the entropy of mixing of different particle species.
    
    Three functions:
        1. reduce_particleset() creates npz files with the particles that stay in a respective basin. 
           This functions requires the existence of and array Entropy_clusters.npy, created above.
        2. compute_transfer_matrix() consutructs, based on the reduced particle sets, the matrix n_ik defined in the methods section
        3. plot_spatial_entropy() plots the entropy as a global map
    """

    tload = range(0,730,73)+[729] #load data each year (indices, not actual times)
    time_origin=datetime.datetime(2000,1,5)
    Times = [(time_origin + datetime.timedelta(days=t*5)).strftime("%Y-%m") for t in tload]    
    
    def reduce_particleset():
        
        #Load particle data
        pdir = datadir + 'MixingEntropy/' #Data directory
        fname = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #F

        #load data
        pdata = ParticleData.from_nc(pdir=pdir,fname=fname,Ngrids=40, tload=tload)
        pdata.remove_nans()

        #Get those particles that start and end in the chosen basin
        r = np.load(outdir_paper + "EntropyMatrix/Entropy_Clusters.npy")
        
        for i_basin in range(1,6): #loop over basins as defined in figure 3a)
            
            print '--------------'
            print 'BASIN: ', i_basin
            print '--------------'
            
            #define basin region
            basin = np.array([1 if r[i]==i_basin else 0 for i in range(len(r))])
            
            #constrain to particles that start in the respective basin         
            l={0: basin}
            basin_data = pdata.get_subset(l, 2.)
            
            #select particles that are in the basin each subsequent year
            for t in range(len(tload)):
                l[t]=basin
            basin_data = pdata.get_subset(l, 2.)
            
            lons=basin_data.lons.filled(np.nan)
            lats=basin_data.lats.filled(np.nan)
            times=basin_data.times.filled(np.nan)
            np.savez(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin), lons=lons, lats=lats, times=times)        
            

    def compute_transfer_matrix():
        #deg_labels is the choice of square binning
        
        for i_basin in range(1,6):
            
            #load reduced particle data for each basin
            pdata = np.load(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin) + '.npz', 'r')
            lons=pdata['lons']
            lats=pdata['lats']
            times=pdata['times']
            del pdata
            pdata_ocean=ParticleData(lons=lons, lats=lats, times=times)
            
            #Define labels according to initial position
            transfer_matrix={}
            pdata_ocean.set_labels(deg_labels, 0)
            l0=pdata_ocean.label
            N=len(np.unique(l0))

            #get existing labels and translate them into labels 0, ...., N-1
            unique, counts = np.unique(l0, return_counts=True) 
            py_labels = dict(zip(unique, range(N)))
            original_labels = dict(zip(range(N), unique))

            #compute transfer matrix
            for t in range(0,len(lons[0])):
                n=np.zeros((N,N))
                pdata_ocean.set_labels(deg_labels, t)
                l=pdata_ocean.label
                
                for j in range(len(l)):
                    if l[j] in l0: #restrict to the existing labels (at t=0)
                        n[py_labels[l0[j]],py_labels[l[j]]]+=1
            
                transfer_matrix[t]=n
            
            np.savez(outdir_paper + 'EntropyMatrix/n_matrix_deg' + str(int(deg_labels)) + '/n_matrix_' + str(i_basin), n=transfer_matrix, original_labels=original_labels)

            
    def plot_spatial_entropy():
        #function to get the spatial entropy
        
        Lons_edges=np.linspace(-180,180,int(360/deg_labels)+1)        
        Lats_edges=np.linspace(-90,90,int(180/deg_labels)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/deg_labels for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/deg_labels for i in range(len(Lats_edges)-1)])        
        
        fig = plt.figure(figsize = (12,8))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.15, hspace=0.)
        
        labels = ['a) ', 'b) ', 'c) ', 'd) ']
    
        for t, k in zip([1,3,6,10],range(4)):
            T=Times[t]

            S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
            
            for i_basin in range(1,6):
                #load data
                data=np.load(outdir_paper + 'EntropyMatrix/n_matrix_deg' + str(int(deg_labels)) + '/n_matrix_' + str(i_basin) + '.npz', 'r')
                n_matrix = data['n'].tolist()
                original_labels = data['original_labels'].tolist()
                n=n_matrix[t]
                
                #row-normalize n
                for i in range(len(n)):
                    s=np.sum(n[i,:])
                    if s!=0:
                        n[i,:]/=s
                    else:
                        n[i,:]=0
                
                #column-normalize
                for i in range(len(n)):
                    s=np.sum(n[:,i])
                    if s!=0:
                        n[:,i]/=s
                    else:
                        n[:,i]=0
                
                #Compute entropy for each location
                S={}
                for j in range(len(n)):
                    s=0
                    for i in range(len(n)):
                        if n[i,j]!=0:
                            s-=n[i,j] * np.log(n[i,j])
                    
                    S[original_labels[j]]=s
                
                #maximum entropy
                N=len(np.unique(original_labels.keys()))
                maxS=np.log(N)
                
                for i in range(len(S_loc)):
                    if i in S.keys():
                        S_loc[i]=S[i]/maxS
            
            
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
    
#Fig4_S3_S4_entropies(figure_title = 'F4_Sloc.eps', deg_labels=5.)
#Fig4_S3_S4_entropies(figure_title = 'S3_Sloc_deg4.eps', deg_labels=4.)
#Fig4_S3_S4_entropies(figure_title = 'S4_Sloc_deg6.eps', deg_labels=6.)
    

def Fig5_S7_S8_S12_S13_stationary_distributions(figure_name, TMname = 'simdays60_ddeg_2', ddeg=2.):
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
        Tmatrix = np.load(outdir_paper + 'MarkovMatrix/' + TMname + '/TM_total.npz')['TM']
        ocean_clusters = np.load(outdir_paper + 'MarkovMatrix/Markov_Clusters_' + TMname + '.npy')
        
        #loop over basins
        for basin_number in range(1,6):
            print 'basin: ', basin_number

            project = [1 if ocean_clusters[i] == basin_number else 0 for i in range(len(ocean_clusters))]            
            print 'projecting'
            
            #project matrix
            TM=Tmatrix.copy()
            for i in range(len(TM)):
                if project[i]==0:
                    TM[i,:]=0
                    TM[:,i]=0
            
            #Normalize again            
            s = np.sum(TM, axis=0)    
            for i in range(len(TM)):
                if s[i]>0:
                    TM[:,i]/=s[i]
            
            print 'projecting done'    
            np.save(outdir_paper + 'MarkovMatrix/TM_'+ TMname + '_basin_' + str(basin_number), TM)
         
    def get_stationary_densities():
        patches = {}
        
        for basin_number in range(1,6):
            print 'basin_number: ', basin_number
            TM = np.load(outdir_paper + 'MarkovMatrix/TM_' + TMname + '_basin_' + str(basin_number) + '.npy')            
            val, vec = sp_linalg.eigs(TM,k=5,which='LM')    
            vec = vec[:,np.argsort(np.abs(val))]
            val = val[np.argsort(np.abs(val))]            
            d0=vec[:,-1]
            d0/=np.sum(d0) #Normalized eigenvector with eigenvalue 1
            patches[basin_number]=d0

        np.save(outdir_paper + 'MarkovMatrix/Patches_' + TMname,patches)
        
    def plot_stationary_densities():        
        
        Lons_edges=np.linspace(-180,180,int(360/ddeg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/ddeg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/ddeg for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/ddeg for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)

        patches=np.load(outdir_paper + 'MarkovMatrix/Patches_' + TMname + '.npy').tolist()
        
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
    
#    project_matrices()
#    get_stationary_densities()
    plot_stationary_densities()

#Figure 6, S9 and S10
#Fig5_S7_S8_S12_S13_stationary_distributions(figure_name = 'F5_GarbagePatches_', TMname = 'simdays60_ddeg_2', ddeg=2.)
#Fig5_S7_S8_S12_S13_stationary_distributions(figure_name = 'S6_GarbagePatches_', TMname = 'simdays60_ddeg_3', ddeg=3.)
#Fig5_S7_S8_S12_S13_stationary_distributions(figure_name = 'S7_GarbagePatches_', TMname = 'simdays90_ddeg_3', ddeg=3.)
#Fig5_S7_S8_S12_S13_stationary_distributions(figure_name = 'S8_GarbagePatches_', TMname = 'simdays90_ddeg_4', ddeg=4.)
#Fig5_S7_S8_S12_S13_stationary_distributions(figure_name = 'S9_GarbagePatches_', TMname = 'simdays120_ddeg_4', ddeg=4.)


def Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays60_ddeg_2', figure_name='figname', ddeg=2.):
    """
    Plot of mixing times
    """
    
    Lons_edges=np.linspace(-180,180,int(360/ddeg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/ddeg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)

    def get_tmix():
        
        for basin_number in range(1,6):
            print 'basin_number: ', basin_number
            TM0 = np.load(outdir_paper + 'MarkovMatrix/TM_' + TMname + '_basin_' + str(basin_number) + '.npy')
            
            val, vec = sp_linalg.eigs(TM0,k=5,which='LM')    
            vec = vec[:,np.argsort(np.abs(val))]
            val = val[np.argsort(np.abs(val))]            
            d0=vec[:,-1]
            d0/=np.sum(d0)            
            tmix = np.array([-100]*len(d0))
            
            TM=TM0.copy()
            
            for t in range(30):
                print t
                print '----------'
                for i in range(len(TM)):
                    if tmix[i] < 0:
                        if .5 * np.sum(np.abs(d0-TM[:,i]))<eps:
                            tmix[i]=t            
                TM=np.dot(TM0,TM)
            
            np.save(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_' +str(basin_number), tmix)
        
    def plot_tmix():
        
        #for the figure
        fig = plt.figure(figsize = (12,12))
        gs1 = gridspec.GridSpec(3, 2)
        gs1.update(wspace=0.15, hspace=0.)
        levels=[0,2,6,10,15,20,25,30]
        
        tmix_max=25

        #North Pacific
        plt.subplot(gs1[0])
        tmix = np.load(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_1.npy')
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
        tmix = np.load(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_2.npy')
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
        tmix = np.load(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_3.npy')
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
        tmix = np.load(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_4.npy')
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
        tmix = np.load(outdir_paper + 'MarkovMatrix/tmix_eps' + str(int(eps*100)) + '_' + TMname + 'basin_5.npy')
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
        
#    get_tmix()
    plot_tmix()

#Figure 7, S7 S11, S12
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays60_ddeg_2' ,figure_name = 'F6_Markov_MixingTime_simdays60_ddeg_2.eps', ddeg=2.)
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.1, TMname='simdays60_ddeg_2' ,figure_name = 'S10_Markov_MixingTime_simdays60_ddeg_2_eps10.eps', ddeg=2.)
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays60_ddeg_3' ,figure_name = 'S11_Markov_MixingTime_simdays60_ddeg_3.eps', ddeg=3.)
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays90_ddeg_3' ,figure_name = 'S12_Markov_MixingTime_simdays90_ddeg_3.eps', ddeg=3.)
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays90_ddeg_4' ,figure_name = 'S13_Markov_MixingTime_simdays90_ddeg_4.eps', ddeg=4.)
Fig6_S5_S9_S10_S14_15_Markkov_mixing_times(eps=0.25, TMname='simdays120_ddeg_4' ,figure_name = 'S14_Markov_MixingTime_simdays120_ddeg_4.eps', ddeg=4.)    


def Fig7_Markov_Entropy(TMname='simdays120_ddeg_4', figure_name='figname', deg_labels=4.):

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
                print 'basin_number: ', basin_number
                TM0 = np.load(outdir_paper + 'MarkovMatrix/TM_' + TMname + '_basin_' + str(basin_number) + '.npy')
                print 'power: ', t
                n=matrix_power(TM0,t).transpose()
                
                #column-normalize
                for i in range(len(n)):
                    s=np.sum(n[:,i])
                    if s>1e-12:
                        n[:,i]/=s
                    else:
                        n[:,i]=0
                
                #Compute entropy for each location
                S=np.zeros(len(S_loc))
                for j in range(len(n)):
                    s=0
                    for i in range(len(n)):
                        if n[i,j]>0:
                            s-=n[i,j] * np.log(n[i,j])
                    
                    S[j]=s
                
                #maximum entropy. col sum of TM0
                su=np.sum(TM0, axis=1)               
                N=len(su[su!=0])
                maxS=np.log(N)
                
                S/=maxS
                
                S_loc+=S        
            
            print 'max S_loc: ', np.max(S_loc)

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
        
#Fig7_Markov_Entropy(TMname='simdays60_ddeg_2', figure_name='F7_TMentropy_simdays60_ddeg_2.eps', deg_labels=2.)
#Fig7_Markov_Entropy(TMname='simdays60_ddeg_3', figure_name='S15_TMentropy_simdays60_ddeg_3.eps', deg_labels=3.)
#Fig7_Markov_Entropy(TMname='simdays90_ddeg_3', figure_name='S16_TMentropy_simdays90_ddeg_3.eps', deg_labels=3.)
#Fig7_Markov_Entropy(TMname='simdays90_ddeg_4', figure_name='S17_TMentropy_simdays90_ddeg_4.eps', deg_labels=4.)
#Fig7_Markov_Entropy(TMname='simdays120_ddeg_4', figure_name='S18_TMentropy_simdays120_ddeg_4.eps', deg_labels=4.)


"""
Further FIGURES SUPPLEMENTARY
"""

def FigS1_initialregions():
    pdir = '/Users/wichmann/Simulations/Proj1_SubSurface_Mixing/Layer0/' #Data directory
    filename = 'SubSurf_y2000_m1_d5_simdays3650_layer0_pos' #File name for many particle simulation
    
    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    r=np.zeros(len(Lons_centered) * len(Lats_centered))
    for ri in initial_regions.keys():
        initial_square = initial_regions[ri]
        r+=square_region(ddeg=2., square=initial_square)
    
    r=r.reshape((len(Lats_centered),len(Lons_centered)))    
    r=np.roll(r,90)    
    b=region_boundary(r)
    b[b==0]=np.nan
    
    #for the figure
    fig = plt.figure(figsize = (12,8))
    
    #load particle data
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], Ngrids=40)
    
    #figure of the globe
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get the distribution
    d_full=pdata.get_distribution(t=-1, ddeg=2.).flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)    
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=10, width=0.05)
    cbar.set_label('# particles per bin', size=10)
    plt.pcolormesh(xs, ys, b, cmap='summer', rasterized=True)
    fig.savefig(outdir_paper + 'S1_initial_regions.pdf', dpi=300, bbox_inches='tight')
    
#FigS1_initialregions()


def FigS2_regions_otherpowers():

    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}
    
    figlabels = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ', 'h) ', 'i) ', 'j) ', 'k) ', 'l) ']
    
    fig = plt.figure(figsize = (12,10))
    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0.15, hspace=0.0)
    
    tmno=[0,5,15,20,25,30,40,50,100,200,500,1000]
    for ki in range(len(tmno)):
   
        mfile = outdir_paper + '/MarkovMatrix/simdays60_ddeg_2/tm' + str(tmno[ki]) + '.npz'
        #regions markov chain
        plt.subplot(gs1[ki])
        t=scipy.sparse.load_npz(mfile).toarray()
        final_regions = {}
            
        for r in initial_regions.keys():
            initial_square = initial_regions[r]
            r2=square_region(ddeg=2., square=initial_square)
            r1=np.zeros(len(r2))
            
            i=0
            while np.any((r1-r2)!=0):
                i+=1
                r1=r2.copy()
                d=np.dot(r2, t)
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
    
        ocean_clusters[ocean_clusters==0]=np.nan
        
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
    
#FigS2_regions_otherpowers()

    
def FigS5_regions_TMs():
    
    Lons_edges=np.linspace(-180,180,int(360/3.)+1)        
    Lats_edges=np.linspace(-90,90,int(180/3.)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    
    #choice of initial regions to initialize the clustering method
    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}

    fig = plt.figure(figsize = (12,8))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    #regions entropy
    plt.subplot(gs1[0])
    t=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/simdays60_ddeg_3/tm10.npz').toarray()  #created in create_matrix.py
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=3., square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Markov 1: iterations to convergence: ', i
    
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays60_ddeg_3', ocean_clusters) #save for definition late
    
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
    ocean_clusters=np.roll(ocean_clusters,60)
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"a) Markov chain $\Delta x = 3^\circ$, $\Delta t = 60$ days")
    
    #regions markov chain
    plt.subplot(gs1[1])
    t=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/simdays90_ddeg_3/tm10.npz').toarray()    
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=3., square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Markov 2: iterations to convergence: ', i
        
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays90_ddeg_3', ocean_clusters) #save for definition late
    
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
    ocean_clusters=np.roll(ocean_clusters,60)
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"b) Markov chain $\Delta x = 3^\circ$, $\Delta t = 90$ days")
    
    Lons_edges=np.linspace(-180,180,int(360/4.)+1)        
    Lats_edges=np.linspace(-90,90,int(180/4.)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/4. for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/4. for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    
    #regions entropy
    plt.subplot(gs1[2])
    t=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/simdays90_ddeg_4/tm10.npz').toarray()  #created in create_matrix.py
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=4., square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Markov 1: iterations to convergence: ', i
    
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays90_ddeg_4', ocean_clusters) #save for definition late
    
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
    ocean_clusters=np.roll(ocean_clusters,int(180/4))
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"c) Markov chain $\Delta x = 4^\circ$, $\Delta t = 90$ days")
    
    #regions markov chain
    plt.subplot(gs1[3])
    t=scipy.sparse.load_npz(outdir_paper + '/MarkovMatrix/simdays120_ddeg_4/tm10.npz').toarray()    
    final_regions = {}
        
    for r in initial_regions.keys():
        initial_square = initial_regions[r]
        r2=square_region(ddeg=4., square=initial_square)
        r1=np.zeros(len(r2))
        
        i=0
        while np.any((r1-r2)!=0):
            i+=1
            r1=r2.copy()
            d=np.dot(r2, t)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print 'Markov 2: iterations to convergence: ', i
        
    ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(ocean_clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                ocean_clusters[i]=labels[k]

    ocean_clusters[ocean_clusters==0]=np.nan
    np.save(outdir_paper + 'MarkovMatrix/Markov_Clusters_simdays120_ddeg_4', ocean_clusters) #save for definition late
    
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
    ocean_clusters=np.roll(ocean_clusters,int(180/4))
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=1.2, size=10)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=1.2, size=10)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"d) Markov chain $\Delta x = 4^\circ$, $\Delta t = 120$ days")

    fig.savefig(outdir_paper + 'S5_Clusters_otherTMs.eps', dpi=300, bbox_inches='tight')
    
#FigS5_regions_TMs()
    

def lambda_2(TMname='simdays60_ddeg_2', figure_name='figname'):
    
    for basin_number in range(1,6):
        print 'basin_number: ', basin_number
        TM0 = np.load(outdir_paper + 'MarkovMatrix/TM_' + TMname + '_basin_' + str(basin_number) + '.npy')
        val, vec = sp_linalg.eigs(TM0,k=10,which='LM')    
#        vec = vec[:,np.argsort(np.abs(val))]
        val = val[np.argsort(np.abs(val))]
        val=val[::-1]       
        print val
        print np.abs(val[1])
#
#lambda_2()