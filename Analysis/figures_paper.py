"""
Mixing of passive tracers at the ocean surface and its implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Figures of paper and annex
--------------

Notes:
    - Numbering of basins is done according to (North Pacific, North Atlantic, South Pacific, South Atlantic, Indian Ocean) = (1,2,3,4,5)
    - For most figures, python arrays are created and saved first.
"""

import numpy as np
from mixing_class import ParticleData, grid_edges, transition_matrix_entropy, compute_transfer_matrices_entropy, setup_annual_markov_chain, get_matrix_power10, get_clusters, project_to_regions, stationary_densities, mixing_time, other_matrix_powers
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
outdir_plot_data = './plot_data/'

#create directories for output
if not os.path.isdir(outdir_paper): os.mkdir(outdir_paper)
if not os.path.isdir(outdir_plot_data): os.mkdir(outdir_plot_data)
if not os.path.isdir(outdir_plot_data + 'EntropyMatrix'): os.mkdir(outdir_plot_data + 'EntropyMatrix')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix'): os.mkdir(outdir_plot_data + 'MarkovMatrix')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays45_ddeg1'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays45_ddeg1')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg1'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg1')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg3'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg3')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg3'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg3')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg4'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg4')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays120_ddeg4'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2001/simdays120_ddeg4')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2005'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2005')
if not os.path.isdir(outdir_plot_data + 'MarkovMatrix/year2005/simdays60_ddeg2'): os.mkdir(outdir_plot_data + 'MarkovMatrix/year2005/simdays60_ddeg2')
if not os.path.isdir(outdir_plot_data + 'waste_input'): os.mkdir(outdir_plot_data + 'waste_input')

#For global plots with 2 degree binning
Lons_edges=np.linspace(-180,180,int(360/2.)+1)        
Lats_edges=np.linspace(-90,90,int(180/2.)+1)
Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)



def fig1_accumulation_zones():
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
    fig = plt.figure(figsize = (7,4.6))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=1.2, size=7)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get final distribution
    d_full=pdata.compute_distribution(t=-1) #.flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)
    
    #plot and savedistribution with colorbar
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=6, width=0.05)
    cbar.set_label('# particles per bin', size=7)
    fig.savefig(outdir_paper + 'F1_accumulation_zones.eps', dpi=200, bbox_inches='tight')


def fig2_two_colors_northpacific():
    """
    Figure illustrating mixing in the west and east north pacific on a particle level after 10 years of advection
    """
    
    pdir = datadir + 'MixingEntropy/' #Particle data directory
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #Particle data file name

    #for the figure
    plt.figure(figsize = (7,4.6))
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
    m.drawparallels([15,30,45,60], labels=[True, False, False, True], linewidth=1.2, size=5)
    m.drawmeridians([150,180,210,240], labels=[False, False, False, True], linewidth=1.2, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(basin_data.lons[:,0], basin_data.lats[:,0])
    m.scatter(xs, ys, c=cols, s=.1, rasterized=True)
    plt.title('a) initial', size=7, y=1.)
    
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
    m.drawparallels([15,30,45,60], labels=[True, False, False, True], linewidth=1.2, size=5)
    m.drawmeridians([150,180,210,240], labels=[False, False, False, True], linewidth=1.2, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgray')
    xs, ys = m(lons, lats)
    m.scatter(xs, ys, c=cols, s=.1, rasterized=True)
    plt.title('b) after 10 years', size=7, y=1.)    

    plt.savefig(outdir_paper + 'F2_two_colors_northpacific.eps', dpi=200, bbox_inches='tight')   


def fig3_clusters_markov_and_entropy():
    
    fig = plt.figure(figsize = (7,4.6))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.15, hspace=0.15)

    plt.subplot(gs1[0])    
    clusters = np.load(outdir_plot_data + '/EntropyMatrix/clusters.npy')
    clusters = np.ma.masked_array(clusters, clusters==0)    
    clusters=clusters.reshape((len(Lats_centered),len(Lons_centered)))
    clusters=np.roll(clusters,90)
    
    m = Basemap(projection='robin',lon_0=-0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=0.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=0.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, clusters, cmap='inferno', rasterized=True, vmin=1, vmax=5)
    plt.title(r"a) Regions for Entropy", size=7)

    #regions markov chain
    plt.subplot(gs1[1])
    clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg2/clusters.npy')

    clusters = np.ma.masked_array(clusters, clusters==0) 
    clusters=clusters.reshape((len(Lats_centered),len(Lons_centered)))
    clusters=np.roll(clusters,90)
    
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=0.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=0.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, clusters, cmap='inferno', rasterized=True)
    plt.title(r"b) Regions for Markov Chain", size=7)

    fig.savefig(outdir_paper + 'F3_Clusters.eps', dpi=200, bbox_inches='tight')   


def fig4_plot_spatial_entropy(figname, d_deg):
    #function to get the spatial entropy
    
    tload = list(range(0,730,73))+[729] #load data each year (indices, not actual times)
    time_origin=datetime.datetime(2000,1,5)
    Times = [(time_origin + datetime.timedelta(days=t*5)).strftime("%Y-%m") for t in tload]    
    
    Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
    
    fig = plt.figure(figsize = (7,4.6))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.15, hspace=0.)
    
    labels = ['a) ', 'b) ', 'c) ', 'd) ']

    for t, k in zip([1,3,6,10],range(4)):
        T=Times[t]

        S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
        
        for i_basin in range(1,6):
            #load data
            A = sparse.load_npz(outdir_plot_data + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '_t' + str(t) + '_basin' + str(i_basin) + '.npz')
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
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='w', linewidth=.7, size=5)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=.7, size=5)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        
        lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)
        xs, ys = m(lon_bins_2d, lat_bins_2d)        
        assert (np.max(S_loc)<=1)
        p = plt.pcolormesh(xs, ys, S_loc,cmap='magma', vmin=0, vmax=1, rasterized=True)
        plt.title(labels[k] + str(T), size=7)
    
    #color bar on the right
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
    cbar=fig.colorbar(p, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$S/S_{max}$',size=7)        
    fig.savefig(outdir_paper + figname + '.eps', dpi=200, bbox_inches='tight')


def fig5_plot_stationary_distributions(figure_name, matrix_dir, d_deg):
    """
    Function to plot the stationary densities
    """
        
    Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    
    patches = {}
    for i in range(1,6):
        patches[i] = np.load(matrix_dir + '/stationary_density' + str(i) + '.npy')
    
    clusters = np.load(matrix_dir + '/clusters.npy')
    #for the figure
    fig = plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.15, hspace=0.)

    vmin = 1e-8
    #North Pacific
    plt.subplot(gs1[0])
    d0 = np.real(patches[1])
    d0 = np.ma.masked_array(d0, clusters!=1)
    d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
    d0=np.roll(d0,int(180/d_deg))
    d0/=np.max(d0)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='k', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='k', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_edges_2d, lat_edges_2d)
    c_region = clusters.copy()
    c_region[c_region!=1]=0
    c_region[c_region==1]=.2
    c_region=c_region.reshape((len(Lats_centered),len(Lons_centered)))
    c_region=np.roll(c_region,int(180/d_deg))
    plt.pcolormesh(xs, ys, c_region, cmap='binary', vmin=0, vmax=1, rasterized=True)
    pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', norm=colors.LogNorm(vmin=vmin,vmax=1), rasterized=True) 
    plt.title('a) North Pacific', size=7)

    #North Atlantic
    plt.subplot(gs1[1])
    d0 = np.real(patches[2])
    d0 = np.ma.masked_array(d0, clusters!=2)
    d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
    d0=np.roll(d0,int(180/d_deg))
    d0/=np.max(d0)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='k', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='k', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_edges_2d, lat_edges_2d)
    c_region = clusters.copy()
    c_region[c_region!=2]=0
    c_region[c_region==2]=.2
    c_region=c_region.reshape((len(Lats_centered),len(Lons_centered)))
    c_region=np.roll(c_region,int(180/d_deg))
    plt.pcolormesh(xs, ys, c_region, cmap='binary', vmin=0, vmax=1, rasterized=True) #, norm=colors.LogNorm()) #  vmin=vmin,vmax=1)) 
    pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', norm=colors.LogNorm(vmin=vmin,vmax=1), rasterized=True) 
    plt.title('b) North Atlantic', size=7)
        
    #South Pacific
    plt.subplot(gs1[2])
    d0 = np.real(patches[3])
    d0 = np.ma.masked_array(d0, clusters!=3)
    d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
    d0=np.roll(d0,int(180/d_deg))
    d0/=np.max(d0)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='k', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='k', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_edges_2d, lat_edges_2d)
    c_region = clusters.copy()
    c_region[c_region!=3]=0
    c_region[c_region==3]=.2
    c_region=c_region.reshape((len(Lats_centered),len(Lons_centered)))
    c_region=np.roll(c_region,int(180/d_deg))
    plt.pcolormesh(xs, ys, c_region, cmap='binary', vmin=0, vmax=1, rasterized=True) #, norm=colors.LogNorm()) #  vmin=vmin,vmax=1)) 
    pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', norm=colors.LogNorm(vmin=vmin,vmax=1), rasterized=True) 
    plt.title('c) South Pacific', size=7)

    #South Atlantic
    plt.subplot(gs1[3])
    d0 = np.real(patches[4])
    d0 = np.ma.masked_array(d0, clusters!=4)
    d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
    d0=np.roll(d0,int(180/d_deg))
    d0/=np.max(d0)
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='k', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='k', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_edges_2d, lat_edges_2d)
    c_region = clusters.copy()
    c_region[c_region!=4]=0
    c_region[c_region==4]=.2
    c_region=c_region.reshape((len(Lats_centered),len(Lons_centered)))
    c_region=np.roll(c_region,int(180/d_deg))
    plt.pcolormesh(xs, ys, c_region, cmap='binary', vmin=0, vmax=1, rasterized=True) #, norm=colors.LogNorm()) #  vmin=vmin,vmax=1)) 
    pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', norm=colors.LogNorm(vmin=vmin,vmax=1), rasterized=True) 
    plt.title('d) South Atlantic', size=7)

    #Indian Ocean
    plt.subplot(gs1[4])
    d0 = np.real(patches[5])
    d0/=np.max(d0)
    d0 = np.ma.masked_array(d0, clusters!=5)
    d0 = np.ma.masked_array(d0, d0==0)    
    d0=d0.reshape((len(Lats_centered),len(Lons_centered)))    
    d0=np.roll(d0,int(180/d_deg))
  
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-50,-25,0,25,50], labels=[True, False, False, True], color='k', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='k', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_edges_2d, lat_edges_2d)

    c_region = clusters.copy()
    c_region[c_region!=5]=0
    c_region[c_region==5]=.2    
    c_region=c_region.reshape((len(Lats_centered),len(Lons_centered)))
    c_region=np.roll(c_region,int(180/d_deg))

    plt.pcolormesh(xs, ys, c_region, cmap='binary', vmin=0, vmax=1, rasterized=True) #, norm=colors.LogNorm()) #  vmin=vmin,vmax=1)) 
    pmesh = plt.pcolormesh(xs, ys, d0, cmap='plasma', norm=colors.LogNorm(vmin=vmin,vmax=1), rasterized=True) 

    plt.title('e) Indian Ocean', size=7)

    cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.4])
    cbar=fig.colorbar(pmesh, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$\rho/\rho_{max}$', size=7)
    
    fig.savefig(outdir_paper + figure_name + '.pdf', dpi=300, bbox_inches='tight')


def fig6_plot_Markkov_mixingtimes(matrix_dir, figure_name, d_deg, tmix_file = 'tmix_eps25'):
    """
    Computation and plot of mixing times
    """
    
    Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2 for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2 for i in range(len(Lats_edges)-1)])        
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)
        
    #for the figure
    fig = plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.15, hspace=0.)

    
    tmix_basins = {}
    for i in range(1,6):
        tmix_basins[i] = np.load(matrix_dir + '/' +  tmix_file + '_basin_' + str(i) + '.npy')

    tmix_max = np.max(list(tmix_basins.values()))
    print('tmix max: ', tmix_max)
    
    levels2 = np.arange(10,tmix_max+1,5.)
    levels1=np.array([0,2,6])
    levels = np.append(levels1, levels2)

    #North Pacific
    plt.subplot(gs1[0])

    tmix=np.array([float(t) for t in tmix_basins[1]])
    tmix[tmix<0]=np.nan
    tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
    tmix=np.roll(tmix,int(180/d_deg))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5) 
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5) 
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_centered_2d, lat_centered_2d)
    pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max)
    plt.title('a) North Pacific', size=7)

    #North Atlantic
    plt.subplot(gs1[1])
    tmix=np.array([float(t) for t in tmix_basins[2]])
    tmix[tmix<0]=np.nan
    tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
    tmix=np.roll(tmix,int(180/d_deg))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5)    
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_centered_2d, lat_centered_2d)
    pmesh = plt.contourf(xs, ys, tmix, levels=levels , cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max)
    plt.title('b) North Atlantic', size=7)
        
    #South Pacific
    plt.subplot(gs1[2])
    tmix=np.array([float(t) for t in tmix_basins[3]])
    tmix[tmix<0]=np.nan
    tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
    tmix=np.roll(tmix,int(180/d_deg))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5) 
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5) 
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_centered_2d, lat_centered_2d)
    pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max)
    plt.title('c) South Pacific', size=7)

    #South Atlantic
    plt.subplot(gs1[3])
    tmix=np.array([float(t) for t in tmix_basins[4]])
    tmix[tmix<0]=np.nan
    tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
    tmix=np.roll(tmix,int(180/d_deg))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5) 
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5) 
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_centered_2d, lat_centered_2d)
    pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max)
    plt.title('d) South Atlantic', size=7)

    #Indian Ocean
    plt.subplot(gs1[4])
    tmix=np.array([float(t) for t in tmix_basins[5]])
    tmix[tmix<0]=np.nan
    tmix=tmix.reshape((len(Lats_centered),len(Lons_centered)))    
    tmix=np.roll(tmix,int(180/d_deg))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5) 
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5) 
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
    xs, ys = m(lon_centered_2d, lat_centered_2d)
    pmesh = plt.contourf(xs, ys, tmix, levels=levels, cmap='Reds', rasterized=True, vmin=0, vmax=tmix_max)
    plt.title('e) Indian Ocean', size=7)

    cbar_ax = fig.add_axes([0.95, 0.35, 0.015, 0.4])
    cbar=fig.colorbar(pmesh, cax=cbar_ax, extend='max') #, ticks=[1,5,10,15,20])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$t_{mix}$ [years]', size=7)
    
    fig.savefig(outdir_paper + figure_name + '.eps', dpi=200, bbox_inches='tight')
        

def fig7_advect_waste_data_global(figure_name='noname'):
    
    nyears = 25
    
    d_deg = 1.
    Lons_edges=np.linspace(0,360,int(360/d_deg)+1)        
    Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
    Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
    Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])  
    
    lon_centered_waste,lat_centered_waste = np.meshgrid(Lons_centered, Lats_centered)
    
    data    = Dataset('../waste_input/releasefunc_wasteinput.nc','r')
    Lons = np.array(data['Longitude'])
    Lats = np.array(data['Latitude'])
    waste = np.array(data['Plastic_waste_input'])
    
    from scipy.interpolate import griddata
    
    lons, lats = np.meshgrid(Lons,Lats)    
    waste_points = griddata(np.array([lons.flatten(), lats.flatten()]).T, waste.flatten(), (lon_centered_waste, lat_centered_waste), method='nearest')
    waste_points = waste_points.flatten()
    
    def get_tvd_basins():

        clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/clusters.npy')        
        
        tvd = {}

        for basin in range(1,6):
            print('basin: ', basin)
            T = sparse.load_npz(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/T_basin_' + str(basin) + '.npz').toarray()
            d0 = np.real(np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/stationary_density' + str(basin) + '.npy'))

            waste_points_basin = waste_points.copy()
            waste_points_basin = waste_points_basin.flatten()
            waste_points_basin[clusters!=basin] = 0
            waste_points_basin /= np.sum(waste_points_basin)
    
            dist = []
            dist.append(.5 * np.sum(np.abs(waste_points_basin-d0)))
            
            for k in range(nyears):
                print('k: ', k)
                waste_points_basin = np.dot(waste_points_basin, T)
                dist.append(.5 * np.sum(np.abs(waste_points_basin-d0)))
                
            tvd[basin] = dist

        np.save(outdir_plot_data + 'waste_input/tvd_waste_data', tvd)

            
    def waste_advection_global():
        
        np.save(outdir_plot_data + 'waste_input/waste0', waste_points)    
        
        T = sparse.load_npz(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/T.npz').toarray()    
        
        waste_points_global = waste_points.copy()
        
        for k in range(nyears):
            print('k: ', k)
            waste_points_global = np.dot(waste_points_global, T)
            np.save(outdir_plot_data + 'waste_input/waste' + str(k+1), waste_points_global)


    def get_tvd_global():
        
        clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/clusters.npy')
        
        basin_tvds = {}

        psum = np.zeros(nyears+1)

        for basin in range(1,6):
            d0 = np.real(np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/stationary_density' + str(basin) + '.npy'))

            tvd_b = []
            
            for k in range(nyears+1):
                waste = np.load(outdir_plot_data + 'waste_input/waste' + str(k) + '.npy')
                waste_basin = waste.copy()
                waste_basin[clusters!=basin] = 0

                psum[k]+=np.sum(waste_basin)

                waste_basin/=np.sum(waste_basin)
                tvd_b.append(.5 * np.sum(np.abs(waste_basin-d0)))    
            
            basin_tvds[basin] = tvd_b
        
        np.save(outdir_plot_data + 'waste_input/TVD_global_basins', basin_tvds)

        fig = plt.figure(figsize = (7,2.3))
        gs1 = gridspec.GridSpec(1, 2, width_ratios= [3.3, 1.6])
        gs1.update(wspace=0.15, hspace=0.15)
        
        plt.subplot(gs1[0])
        m = Basemap(projection='robin',lon_0=-180,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
        m.drawmeridians([-130, -60, 0, 60, 130], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        waste = np.load(outdir_plot_data + 'waste_input/waste24.npy')
        waste[clusters!=0]=0
        waste = np.ma.masked_array(waste, waste<1)
        waste = waste.reshape(len(Lats_centered), len(Lons_centered))
        
        xs, ys = m(lon_centered_waste, lat_centered_waste) 
        plt.pcolormesh(xs, ys, waste,cmap='plasma',  norm=colors.LogNorm(vmin=1), rasterized=True)
        cbar=plt.colorbar(orientation='horizontal',shrink=0.8)
        cbar.ax.tick_params(width=0.05)
        plt.title('a) Particles outside basins after 25 years', size=7)
        cbar.set_label('# particles per bin', size=6)
        cbar.ax.tick_params(labelsize=6, width=0.05)
        
        ax = plt.subplot(gs1[1])
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        plt.plot(range(0,26),psum/np.sum(waste_points)*100, '--o', markersize = 2, linewidth=1.)
        plt.title('b) Share of particles in basins [%]', size=7)
        plt.grid()
        plt.xlabel('Time [years]', size=7)

        fig.savefig(outdir_paper + 'S24_globaltvd_particles.eps', dpi=200, bbox_inches='tight')
        
    
    def figure():
                        
        fig = plt.figure(figsize = (7,2.3))
        gs1 = gridspec.GridSpec(1, 3, width_ratios= [3.3, 1.6, 1.6])
        gs1.update(wspace=0.15, hspace=0.15)
        
        plt.subplot(gs1[0])
        m = Basemap(projection='robin',lon_0=-180,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
        m.drawmeridians([-130, -60, 0, 60, 130], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
        m.drawcoastlines(linewidth=.7)
        waste_points = np.load(outdir_plot_data + 'waste_input/waste0.npy')
        waste_points = waste_points.reshape(len(Lats_centered), len(Lons_centered))
        
        xs, ys = m(lon_centered_waste, lat_centered_waste) 
        plt.pcolormesh(xs, ys, waste_points,cmap='plasma',  norm=colors.LogNorm(), rasterized=True)
        cbar=plt.colorbar(orientation='horizontal', shrink=0.9)
        cbar.ax.tick_params(labelsize=6, width=0.05)
        plt.title('a) Initial particle distribution', size=7)
        cbar.set_label('# particles per bin', size=6)
    
        tvd = np.load(outdir_plot_data + 'waste_input/tvd_waste_data.npy', allow_pickle=True).item()
        ax = plt.subplot(gs1[1])
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        plt.plot(tvd[1], '--o', markersize = 2, label = 'North Pacific', linewidth=1.)
        plt.plot(tvd[2], '--o', markersize = 2,label = 'North Atlantic', linewidth=1.)
        plt.plot(tvd[3], '--o', markersize = 2,label = 'South Pacific', linewidth=1.)
        plt.plot(tvd[4], '--o', markersize = 2,label = 'South Atlantic', linewidth=1.)
        plt.plot(tvd[5], '--o', markersize = 2,label = 'Indian Ocean', linewidth=1.)
        plt.plot((-1, 26), (.25, .25), 'k--')
        plt.xlim([0,25])
        plt.ylim([0,1])
        plt.xlabel('Time [years]', size=6)
        plt.title('b) TVD basin wide matrices', size=7)
        plt.text(15, 0.27, r'$y = 0.25$', size=7)
        plt.legend(fontsize = 'xx-small')
        
        tvd_global = np.load(outdir_plot_data + 'waste_input/TVD_global_basins.npy', allow_pickle=True).item()
        ax = plt.subplot(gs1[2])
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_yticklabels([])
        plt.plot(tvd_global[1], '--o', markersize = 2, label = 'North Pacific', linewidth=1.)
        plt.plot(tvd_global[2], '--o', markersize = 2,label = 'North Atlantic', linewidth=1.)
        plt.plot(tvd_global[3], '--o', markersize = 2,label = 'South Pacific', linewidth=1.)
        plt.plot(tvd_global[4], '--o', markersize = 2,label = 'South Atlantic', linewidth=1.)
        plt.plot(tvd_global[5], '--o', markersize = 2,label = 'Indian Ocean', linewidth=1.)
        plt.plot((-1, 26), (.25, .25), 'k--')
        plt.xlim([0,25])
        plt.ylim([0,1])
        plt.xlabel('Time [years]', size=6)
        plt.title('c) TVD global matrix', size=7)
        plt.text(15, 0.27, r'$y = 0.25$', size=7)
        plt.legend(fontsize = 'xx-small')

        fig.savefig(outdir_paper + figure_name +'.eps', dpi=200, bbox_inches='tight')

#    get_tvd_basins()        
#    waste_advection_global()
    get_tvd_global()
    figure()



def fig8_Markov_entropy(matrix_dir, figure_name, d_deg):
        
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        
        fig = plt.figure(figsize = (7,4.6))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.15, hspace=0.)
        
        labels = ['a) ', 'b) ', 'c) ', 'd) ']
    
        from numpy.linalg import matrix_power
    
        for t,k in zip([1,3,6,10],range(4)):
            
            S_loc=np.zeros(len(Lons_centered)*len(Lats_centered)) #final entropy field
                    
            for basin_number in range(1,6):
                print( 'basin_number: ', basin_number)
                T0 = sparse.load_npz(matrix_dir + '/T_basin_' + str(basin_number) + '.npz').toarray()

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
            S_loc=np.roll(S_loc,int(180/d_deg))
            m = Basemap(projection='robin',lon_0=0,resolution='c')
            m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='w', linewidth=.7, size=5)
            m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=.7, size=5)
            m.drawcoastlines()
            m.fillcontinents(color='lightgrey')
            
            lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)
            xs, ys = m(lon_bins_2d, lat_bins_2d)        
            p = plt.pcolormesh(xs, ys, S_loc,cmap='magma', vmin=0, vmax=1, rasterized=True)
            plt.title(labels[k] + 'Year ' + str(t), size=7)
        
        #color bar on the right
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
        cbar=fig.colorbar(p, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(r'$S/S_{max}$',size=7)        
        fig.savefig(outdir_paper + figure_name + '.eps', dpi=200, bbox_inches='tight')
        

def table_tmix_parameters():
    
    with open(outdir_paper + 'tmix_table.txt', "w") as output:
        output.write('\\caption{parameter dependence}' + '\n')
        output.write('\\centering' + '\n')
        output.write('\\begin{tabular}{| l | c | c | c | c | c }' + '\n')
        output.write('\\hline' + '\n')
    
    
        for matrix_dir in ['simdays45_ddeg1', 'simdays60_ddeg1', 'simdays60_ddeg2', 'simdays60_ddeg3', 'simdays90_ddeg3', 'simdays90_ddeg4', 'simdays120_ddeg4']:
            
            output.write(matrix_dir + '&')
            
            for b in range(1,6):
                tmix = np.load('plot_data/MarkovMatrix/year2001/' + matrix_dir + '/tmix_eps25_basin_' + str(b) + '.npy')
                tmix = tmix[tmix>=0]
                
                tmix_mean = np.mean(tmix)
                tmix_max = np.max(tmix)
                tmix_min = np.min(tmix)
                
                if b<5:
                    output.write(str(round(tmix_mean,1)) + ' [' + str(round(tmix_min,1)) + ', ' + str(round(tmix_max,1)) + '] &')
                else:
                    output.write(str(round(tmix_mean,1)) + ' [' + str(round(tmix_min,1)) + ', ' + str(round(tmix_max,1)) + '] \\\\')
                    output.write('\n')


def initialregions(figure_name):
    
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
    fig = plt.figure(figsize = (7,4.6))
    
    #figure of the globe
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=.7, size=5)
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=.7, size=5)
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    #get the distribution
    d_full=pdata.compute_distribution(t=-1) #.flatten().reshape((len(Lats_centered),len(Lons_centered)))
    d_full=np.roll(d_full,90)    
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, d_full, cmap='plasma', norm=colors.LogNorm(), rasterized=True)
    
    cbar=plt.colorbar(orientation='vertical',shrink=0.5)
    cbar.ax.tick_params(labelsize=6, width=0.05)
    cbar.set_label('# particles per bin', size=7)

    plt.pcolormesh(xs, ys, b, cmap='gist_gray', rasterized=True)
    fig.savefig(outdir_paper + figure_name + '.pdf', dpi=300, bbox_inches='tight')
    
    
def regions_otherpowers(figure_name):

    initial_regions = {'NP': (180., 235., 20., 40.), 'NA': (280., 330., 15., 40.),
                       'SP': (210., 280., -45., -20.), 'SA': (320., 360., -45., -20.),
                       'IO': (40.,100., -45., -15.)}
    labels = {'NP': 1, 'NA': 2, 'SP': 3, 'SA': 4, 'IO': 5}
    
    figlabels = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ', 'h) ', 'i) ', 'j) ', 'k) ', 'l) ']
    
    fig = plt.figure(figsize = (7,7))
    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0.15, hspace=0.0)

    pdata = ParticleData()
    pdata.set_discretizing_values(d_deg = 2.)    

    tmno=[0,5,15,20,25,30,40,50,100,200,500,1000]
    for ki in range(len(tmno)):
   
        mfile = outdir_plot_data +'/MarkovMatrix/year2001/simdays60_ddeg2/T' + str(tmno[ki]) + '.npy'

        plt.subplot(gs1[ki])
        t=np.load(mfile)
        
        final_regions = {}
            
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
            
        ocean_clusters = np.zeros(len(Lats_centered)*len(Lons_centered))
    
        for i in range(len(ocean_clusters)):
            for k in final_regions.keys():
                if final_regions[k][i]==1:
                    ocean_clusters[i]=labels[k]
    
        ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
        
        ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))    
        ocean_clusters=np.roll(ocean_clusters,90)

        m = Basemap(projection='robin',lon_0=0,resolution='c')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color = 'grey', linewidth=.7, size=4)
        m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color = 'grey', linewidth=.7, size=4)
        m.drawcoastlines()
        m.fillcontinents(color='lightgrey')
        xs, ys = m(lon_edges_2d, lat_edges_2d) 
        plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
        plt.title(figlabels[ki] + 'Power: ' +  str(tmno[ki]), size=6)

    fig.savefig(outdir_paper + figure_name + '.eps', dpi=200, bbox_inches='tight')
    
    
def deleted_particles_endup_entropy(figname):
    
    def get_deleted_particles():
    
        #Load particle data
        fname = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #F
        tload = [0,-1]
    
        #load data
        pdata=ParticleData.from_nc(datadir + 'MixingEntropy/', fname, tload=tload, n_grids=40)
        pdata.set_discretizing_values(d_deg = 2.)
        pdata.remove_nans()
    
        #Get those particles that start and end in the chosen basin
        r = np.load(outdir_plot_data +  'EntropyMatrix/clusters.npy')
        
        for i_basin in range(1,6): #loop over basins as defined in figure 3a)
            
            print( '--------------')
            print( 'BASIN: ', i_basin)
            print( '--------------')
            
            #define basin region
            basin = np.array([1 if r[i]==i_basin else 0 for i in range(len(r))])
            basin_complement = np.array([0 if r[i]==i_basin else 1 for i in range(len(r))])
            
            #constrain to particles that are in the respective basin after each year
            l={}
            l[0] = basin
            l[1] = basin_complement
            
            leaving_data = pdata.get_subset(l)
            
            lons=leaving_data.lons.filled(np.nan)
            lats=leaving_data.lats.filled(np.nan)
            np.savez(outdir_plot_data + 'EntropyMatrix/particles_left_basin' + str(i_basin), lons=lons, lats=lats)


    def plot_deleted_particles():

        
        fig = plt.figure(figsize = (6,6))
        gs1 = gridspec.GridSpec(3, 2)
        gs1.update(wspace=0.15, hspace=0.)
    
        basins = ['North Pacific', 'North Atlantic', 'South Pacific', 'South Atlantic', 'Indian Ocean']
        labels = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ']

        
        for i_basin in range(1,6):
            
            plt.subplot(gs1[i_basin-1])
            data = np.load(outdir_plot_data + 'EntropyMatrix/particles_left_basin' + str(i_basin) + '.npz')
            lons = data['lons']
            lats = data['lats']
            
            pdata = ParticleData(lons=lons, lats=lats)
            pdata.set_discretizing_values(d_deg = 2.)
    
            #figure of the globe
            plt.subplot(gs1[i_basin-1])
            m = Basemap(projection='robin',lon_0=0,resolution='c')
            m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], color='gray', linewidth=.7, size=5)
            m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='gray', linewidth=.7, size=5)
            m.drawcoastlines()
            m.fillcontinents(color='lightgrey')
            
            #get final distribution
            d_full=pdata.compute_distribution(t=1) #.flatten().reshape((len(Lats_centered),len(Lons_centered)))
            d_full=np.roll(d_full,90)
            
            #plot and savedistribution with colorbar
            xs, ys = m(lon_edges_2d, lat_edges_2d) 
            plt.pcolormesh(xs, ys, d_full,cmap='plasma', norm=colors.LogNorm(), rasterized=True)
            plt.title(labels[i_basin-1] + basins[i_basin-1], size=7)
            cbar=plt.colorbar(orientation='vertical',shrink=0.5)
            cbar.ax.tick_params(labelsize=6, width=0.05)
            cbar.set_label('# particles per bin', size=5)

        fig.savefig(outdir_paper + figname + '.eps', dpi=200, bbox_inches='tight')

#    get_deleted_particles()
    plot_deleted_particles()


def deleted_particles_Markov(matrix_dir, figure_name):
    
    A = sparse.load_npz(matrix_dir + 'T.npz')
    A = sparse.coo_matrix(A)
    clusters = np.load(matrix_dir + 'clusters.npy')
    
    row = A.row
    col = A.col
    val = A.data                       
    
    fig = plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(wspace=0.15, hspace=0.)
    
    rowsum = np.zeros(A.shape[0])
    
    #loop over basins
    for basin_number in range(1,6):
        
        print( 'basin: ', basin_number)
    
        project = np.array([i for i in range(len(clusters)) if clusters[i] == basin_number])
    
        print( 'projecting')
        
        #project matrix
        inds = [i for i in range(len(col)) if (col[i] in project and row[i] in project)]
        
        row_new = row[inds]
        col_new = col[inds]
        val_new = val[inds]
        A_new = sparse.coo_matrix((val_new, (row_new, col_new)), shape=A.shape) 
        
        r = np.array(sparse.coo_matrix.sum(A_new, axis=1))[:,0]
        rowsum += r
    
    rowsum=rowsum.reshape((len(Lats_centered),len(Lons_centered)))
    rowsum=np.roll(rowsum,int(180/2.))
    rowsum = (1. - rowsum) * 100.
    
    fig = plt.figure(figsize = (9,4.5))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    p = plt.pcolormesh(xs, ys, rowsum, cmap='Reds', vmin=0., vmax=100., rasterized=True)
    cbar=plt.colorbar(p, shrink=.7)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('%', size=7)        
    fig.savefig(outdir_paper + figure_name + '.eps', dpi=200, bbox_inches='tight')


def clusters_other_Matrkov_matrices(figure_name):
    """
    Clusters based on other matrices
    """    
    
    fig = plt.figure(figsize = (6,8))
    gs1 = gridspec.GridSpec(4, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    
    plt.subplot(gs1[0])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays45_ddeg1/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 1.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"a) $\Delta x = 1^\circ$, $\Delta t = 45$ days", size=7)
        
    plt.subplot(gs1[1])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg1/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 1.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"b) $\Delta x = 1^\circ$, $\Delta t = 60$ days", size=7)        
    
    plt.subplot(gs1[2])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays60_ddeg3/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 3.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"c) $\Delta x = 3^\circ$, $\Delta t = 60$ days", size=7)

    plt.subplot(gs1[3])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays90_ddeg3/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 3.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"d) $\Delta x = 3^\circ$, $\Delta t = 90$ days", size=7)

    plt.subplot(gs1[4])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays90_ddeg4/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 4.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"e) $\Delta x = 4^\circ$, $\Delta t = 90$ days", size=7)

    plt.subplot(gs1[5])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2001/simdays120_ddeg4/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')

    d_deg = 4.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))        

    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"f) $\Delta x = 4^\circ$, $\Delta t = 120$ days", size=7)

    plt.subplot(gs1[6])
    ocean_clusters = np.load(outdir_plot_data + '/MarkovMatrix/year2005/simdays60_ddeg2/clusters.npy')
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], linewidth=.7, size=5, color='grey')
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey')
    
    d_deg = 2.
    ocean_clusters = np.ma.masked_array(ocean_clusters, ocean_clusters==0) 
    (Lons_centered, Lats_centered, lon_edges_2d,lat_edges_2d ) = grid_edges(d_deg)
    ocean_clusters=ocean_clusters.reshape((len(Lats_centered),len(Lons_centered)))
    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))     
    
    xs, ys = m(lon_edges_2d, lat_edges_2d) 
    plt.pcolormesh(xs, ys, ocean_clusters, cmap='inferno', rasterized=True)
    plt.title(r"g) $\Delta x = 2^\circ$, $\Delta t = 60$ days, 2005", size=7)

    fig.savefig(outdir_paper + figure_name + '.eps', dpi=200, bbox_inches='tight')


def lambda_2(matrix_dir):
    
    for basin_number in range(1,6):
        print( 'basin_number: ', basin_number)
        TM0 = sparse.load_npz(matrix_dir + 'T_basin_' + str(basin_number) + '.npz').toarray()
        val, vec = sp_linalg.eigs(TM0, k=5, which='LM')    
        val = val[np.argsort(np.abs(val))]
        val=val[::-1]       
        print( val)
        print( np.abs(val[1]))


def mixing_time_cumdistr(figure_name):
    
    
    def create_cumdistr(matrix_dir, tmix_file, d_deg):
    
        Lons_edges=np.linspace(-180,180,int(360/d_deg)+1)        
        Lats_edges=np.linspace(-90,90,int(180/d_deg)+1)
        Lons_centered=np.array([(Lons_edges[i]+Lons_edges[i+1])/2. for i in range(len(Lons_edges)-1)])
        Lats_centered=np.array([(Lats_edges[i]+Lats_edges[i+1])/2. for i in range(len(Lats_edges)-1)])        
        lon_edges_2d,lat_edges_2d = np.meshgrid(Lons_edges,Lats_edges)
        lon_centered_2d,lat_centered_2d = np.meshgrid(Lons_centered,Lats_centered)

        dy = np.diff(np.sin(Lats_edges * np.pi/180.))

        r = 6371.
        a = d_deg * np.pi/180. * r**2 * dy
        
        A = np.zeros(lon_centered_2d.shape)
        
        for i in range(len(A[0])):
            A[:,i] = a
            
        A = A.flatten()
    
        h = np.array([])
        w = np.array([])
        
        tmix = np.load(matrix_dir + tmix_file + '_basin_1.npy')
        h = np.append(h, tmix[tmix>=0])
        w = np.append(w, A[tmix>=0])
        
        for i in range(2,6):
            tmix = np.load(matrix_dir + tmix_file + '_basin_' + str(i) + '.npy')
            h = np.append(h, tmix[tmix>=0])
            w = np.append(w, A[tmix>=0])
    
        bins = np.arange(0,int(np.max(h))+2)
        
        hist_area, bin_edges = np.histogram(h, weights = w, bins=bins, density=True)
        hist_area_sum = np.cumsum(hist_area)
        
        return (bin_edges[:-1], hist_area_sum, hist_area)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].grid(True)
    axs[1].grid(True)

    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/'
    tmix_file = 'tmix_eps25'
    d_deg = 2.    
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 2^\circ, \Delta t = 60$ days')
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)
    
    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2005/simdays60_ddeg2/'
    tmix_file = 'tmix_eps25'
    d_deg = 2.    
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2005: \Delta x = 2^\circ, \Delta t = 60$ days')
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)
    
    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays45_ddeg1/'
    tmix_file = 'tmix_eps25'
    d_deg = 1.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 1^\circ, \Delta t = 45$ days')
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)
    
    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg1/'
    tmix_file = 'tmix_eps25'
    d_deg = 1.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 1^\circ, \Delta t = 60$ days')    
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)

    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg3/'
    tmix_file = 'tmix_eps25'
    d_deg = 3.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 3^\circ, \Delta t = 90$ days')
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)

    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/'
    tmix_file = 'tmix_eps10'
    d_deg = 2.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 2^\circ, \Delta t = 60$ days, $\epsilon=1/10$')    
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)

    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays120_ddeg4/'
    tmix_file = 'tmix_eps25'
    d_deg = 4.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1., label = r'$2001: \Delta x = 4^\circ, \Delta t = 120$ days')    
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)
    
    matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg4/'
    tmix_file = 'tmix_eps25'
    d_deg = 4.
    (tmix, cumul, hist) = create_cumdistr(matrix_dir, tmix_file, d_deg)
    axs[0].plot(tmix, hist, 'o--', markersize = 2, linewidth=1.,  label = r'$2001: \Delta x = 4^\circ, \Delta t = 90$ days')    
    axs[1].plot(tmix, cumul, 'o--', markersize = 2, linewidth=1.)
    
    axs[0].set_title('a) area weighted share', size=7)
    axs[0].set_xlabel(r'$t_{mix}$ [years]', size=7)
    axs[1].set_title('b) cumulative', size=7)
    axs[1].set_xlabel(r'$t_{mix}$ [years]', size=7)
    axs[0].tick_params(axis="x", labelsize=6)
    axs[1].tick_params(axis="x", labelsize=6)
    axs[0].tick_params(axis="y", labelsize=6)
    axs[1].tick_params(axis="y", labelsize=6)
    
    handles, labels = axs[0].get_legend_handles_labels()
    plt.subplots_adjust(bottom=0.3)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 0.17), ncol=3,  fancybox=True, prop={'size': 6})
    
    fig.savefig(outdir_paper + figure_name + '.eps', dpi=200)
    

"""
Create npy and npz arrays with the data to plot
------------------------------------------------
"""

"""
Set up matrix for entropy (10 year transition matrix) and compute clusters
"""

#transition_matrix_entropy(pdir = datadir + 'MixingEntropy/', entropy_dir = outdir_plot_data + 'EntropyMatrix/')
#get_clusters(matrix_dir = outdir_plot_data + 'EntropyMatrix/', d_deg=2., matrix_file = 'T.npz')
#compute_transfer_matrices_entropy(pdir= datadir + 'MixingEntropy/', entropy_dir=outdir_plot_data + 'EntropyMatrix/', d_deg = 5.)
#compute_transfer_matrices_entropy(pdir= datadir + 'MixingEntropy/', entropy_dir=outdir_plot_data + 'EntropyMatrix/', d_deg = 6.)
#compute_transfer_matrices_entropy(pdir= datadir + 'MixingEntropy/', entropy_dir=outdir_plot_data + 'EntropyMatrix/', d_deg = 4.)


"""
Set up matrices for Markov chain. The compute 10th power, compute the clusters, project on basins, get stationary distribution and compute mixing time
"""

data_dirs       = ['year2001/simdays45/', 'year2001/simdays60/', 'year2001/simdays60/', 'year2001/simdays60/', 
                   'year2001/simdays90/', 'year2001/simdays90/', 'year2001/simdays120/', 'year2005/simdays60/']
data_dirs       = [datadir + 'MarkovMixing/' + d for d in data_dirs]

matrix_dirs     = ['year2001/simdays45_ddeg1/', 'year2001/simdays60_ddeg1/', 'year2001/simdays60_ddeg2/', 'year2001/simdays60_ddeg3/', 
                   'year2001/simdays90_ddeg3/', 'year2001/simdays90_ddeg4/', 'year2001/simdays120_ddeg4/', 'year2005/simdays60_ddeg2/']
matrix_dirs     = [outdir_plot_data + 'MarkovMatrix/' +  d for d in matrix_dirs]

d_degs          = [1., 1., 2., 3., 3., 4., 4., 2.]
ns_grids        = [40, 40, 40, 40, 40, 40, 40, 5]

#eps = 1/4
#for (d, matrix_dir, d_deg, n_grids) in zip(data_dirs, matrix_dirs, d_degs, ns_grids):
#    print('Create data for: ', d)
#    
#    
#    setup_annual_markov_chain(data_dir = d, matrix_dir = matrix_dir, d_deg=d_deg, n_grids=n_grids)
#    get_matrix_power10(matrix_dir = matrix_dir, power = 10)
#    get_clusters(matrix_dir = matrix_dir, d_deg = d_deg)
#    project_to_regions(matrix_dir)
#    stationary_densities(matrix_dir)
#    mixing_time(matrix_dir=matrix_dir, eps = .25)

#eps = 1/10
#mixing_time(matrix_dir=outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/', eps = .1)
#other_matrix_powers(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/')
 

"""
Paper figures
------------------------------------------
------------------------------------------
"""

"""
#Accumulation of particles in the subtropical gyres
#"""
#fig1_accumulation_zones()
#
#"""
#Mixing of colors in the North Pacific
#"""
#fig2_two_colors_northpacific()
#
#"""
#Results of regions for Entropy and Markov methods
#"""
#fig3_clusters_markov_and_entropy()
#
#"""
#Plot of entropy
#"""
#fig4_plot_spatial_entropy(figname = 'F4_Sloc', d_deg=5.)
#
#"""
#Stationary distributions for the Markov chain
#"""
#fig5_plot_stationary_distributions(figure_name = 'F5_GarbagePatches_simdays60_ddeg2', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2', d_deg=2.)
#
#"""
#Markov mixing time
#"""
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2', figure_name = 'F6_Markov_MixingTime_simdays60_ddeg_2', d_deg=2)

#"""
#Mixing time for realistic input scenario
#"""
#fig7_advect_waste_data_global(figure_name = 'F7_tvd_realistic_waste')
#
#"""
#Entropy from the Markov matrix
#"""
#fig8_Markov_entropy(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2', figure_name = 'F8_TMentropy_simdays60_ddeg_2', d_deg=2)



"""
Supplementary figures
---------------------------------
---------------------------------
"""

#"""
#Initial regions as initialization for the clustering algorithm
#"""
#initialregions(figure_name = 'S1_supplementary_initial_regions')
#
#"""
#Definition of the regions if we used other matrix powers
#"""
#regions_otherpowers(figure_name = 'S2_supplementary_clusters_otherpowers')
#
#"""
#Final locations of particles that leave their initial basin after 10 years
#"""
#deleted_particles_endup_entropy(figname = 'S3_particles_left_basin')
#
#"""
#Share of deleted particles per bin, coming from the projection of the Markov matrix
#"""
#deleted_particles_Markov(matrix_dir=outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/', figure_name='S4_supplementary_deleted_particles_Markov')
#
#"""
#Definition of the regions if we used the other 7 Markov matrices
#"""
#clusters_other_Matrkov_matrices(figure_name = 'S5_supplementary_clusters_other_matrices')
#
#"""
#Spatial entropies for 4 and 6 degree
#"""
#fig4_plot_spatial_entropy(figname = 'S6_supplementary_Sloc_deg4', d_deg=4.)
#fig4_plot_spatial_entropy(figname = 'S7_supplementary_Sloc_deg6', d_deg=6.)
#
#
#"""
#Stationary distributions other 7 matrices
#"""
#
#fig5_plot_stationary_distributions(figure_name = 'S8_supplementary_garbagePatches_simdays45_ddeg1', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays45_ddeg1', d_deg=1.)
#
#fig5_plot_stationary_distributions(figure_name = 'S9_supplementary_garbagePatches_simdays60_ddeg1', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg1', d_deg=1.)
#
#fig5_plot_stationary_distributions(figure_name = 'S10_supplementary_garbagePatches_simdays60_ddeg3', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg3', d_deg=3.)
#
#fig5_plot_stationary_distributions(figure_name = 'S11_supplementary_garbagePatches_simdays90_ddeg3', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg3', d_deg=3.)
#
#fig5_plot_stationary_distributions(figure_name = 'S12_supplementary_garbagePatches_simdays90_ddeg4', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg4', d_deg=4.)
#
#fig5_plot_stationary_distributions(figure_name = 'S13_supplementary_garbagePatches_simdays120_ddeg4', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays120_ddeg4', d_deg=4.)
#
#fig5_plot_stationary_distributions(figure_name = 'S14_supplementary_garbagePatches_simdays60_ddeg2_y2005', matrix_dir = outdir_plot_data + 'MarkovMatrix/year2005/simdays60_ddeg2', d_deg=2.)
#
#
#"""
#Markov mixing times other 7 matrices and epsilon = 1/10
#"""
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays45_ddeg1', figure_name = 'S15_supplementary_Markov_MixingTime_simdays45_ddeg_1', d_deg=1.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg1', figure_name = 'S16_supplementary_Markov_MixingTime_simdays60_ddeg_1', d_deg=1.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg3', figure_name = 'S17_supplementary_Markov_MixingTime_simdays60_ddeg_3', d_deg=3.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg3', figure_name = 'S18_supplementary_Markov_MixingTime_simdays90_ddeg_3', d_deg=3.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays90_ddeg4', figure_name = 'S19_supplementary_Markov_MixingTime_simdays90_ddeg_4', d_deg=4.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays120_ddeg4', figure_name = 'S20_supplementary_Markov_MixingTime_simdays120_ddeg_4', d_deg=4.)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2005/simdays60_ddeg2', figure_name = 'S21_supplementary_Markov_MixingTime_simdays60_ddeg_2_y2005', d_deg=2)
#
#fig6_plot_Markkov_mixingtimes(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2', figure_name = 'S22_supplementary_Markov_MixingTime_simdays60_ddeg_2_eps01', d_deg=2., tmix_file = 'tmix_eps10')

#"""
#Probability and cumulative distribution of area weighted mixing times
#"""
#mixing_time_cumdistr(figure_name = 'S23_distribution_tmix')
#

"""
Particle concentrations outside the basins for real plastic input scenario (fig. S24);
created in fig7_advect_waste_data_global() -> get_tvd_global()
"""

"""
Table with mixing times for different matrices
"""
#table_tmix_parameters()


"""
Other
-------------------
-------------------
"""

"""
Display second largest eigenvalue modulus (for separation to 1)
"""
#lambda_2(matrix_dir = outdir_plot_data + 'MarkovMatrix/year2001/simdays60_ddeg2/')d