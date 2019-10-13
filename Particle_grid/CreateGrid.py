"""
Create a uniformly spaced (lon,lat) grid of initial particle locations based on nemo bathymetry
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap

spacing = 0.1 #spacing between particles
plotspacing = 1. #For binning of final plot
outdir = './initial_coordinates/'
name = 'coordinates_ddeg01'


def create_particles():
    #Create uniform grid of particles
    filename='./nemo_bathymetry/bathy_level.nc'
    data = Dataset(filename,'r')
    bathy=np.array(data['Bathy_level'][0])
    lon=np.array([data['nav_lon']][0])
    lat=np.array([data['nav_lat']][0])
    
    print('Data loaded')
    grid=np.mgrid[-180:180:spacing,-90:90:spacing]
    n=grid[0].size;
    lons=np.reshape(grid[0],n)
    lats=np.reshape(grid[1],n)
      
    print('Interpolating')

    bathy_points = griddata(np.array([lon.flatten(), lat.flatten()]).T, bathy.flatten(), (lons, lats), method='nearest')
    
    lons_new=np.array([lons[i] for i in range(len(lons)) if bathy_points[i]!=0])
    lats_new=np.array([lats[i] for i in range(len(lats)) if bathy_points[i]!=0])
    
    lons_new[lons_new<0.] += 360.
    
    np.savez(outdir + str(name), lons = lons_new, lats = lats_new)

#create_particles()


def Plot_particles():
    #Plot to check if everything went well
    data = np.load(outdir + str(name) + '.npz')
    lons = data['lons']
    lats = data['lats']
#    lons=np.load(outdir + 'Lons_full' + str(name) + '.npy')
#    lats=np.load(outdir + 'Lats_full' + str(name) + '.npy')
    
    assert (len(lons)==len(lats))
    
    print('Number of particles: ', len(lons))
    fig = plt.figure(figsize=(25, 30))
    ax = fig.add_subplot(211)
    ax.set_title("Particles")
    
    m = Basemap(projection='robin',lon_0=-180,resolution='c')
    m.drawcoastlines()
    xs, ys = m(lons, lats)
    m.scatter(xs,ys)
    
    ax = fig.add_subplot(212)
    ax.set_title("Particles per bin. Should be constant everywhere but on land.")

    m = Basemap(projection='robin',lon_0=-180,resolution='c')
    m.drawcoastlines()

    lon_bin_edges = np.arange(0, 360+spacing, plotspacing)
    lat_bins_edges = np.arange(-90, 90+spacing, plotspacing)
    
    density, _, _ = np.histogram2d(lats, lons, [lat_bins_edges, lon_bin_edges])
    
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bin_edges, lat_bins_edges)
    xs, ys = m(lon_bins_2d, lat_bins_2d)

    plt.pcolormesh(xs, ys, density,cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(orientation='vertical', shrink=0.625, aspect=20, fraction=0.2,pad=0.02)
    cbar.set_label('Particles per bin',size=8)

#Plot_particles()

def split_grid():
    
    data = np.load(outdir + str(name) + '.npz')
    lons = data['lons']
    lats = data['lats']
       
    print('Total number of particles: ', len(lons))
    
    N=5 #Number of sub-grids
    
    k = len(lons)//N+1 #Number of particles per file
    print(k)
    
    for i in range(0,len(lons)//k+1):
        lo = lons[i*k:(i+1)*k]
        la = lats[i*k:(i+1)*k]
        np.savez(outdir + 'coordinates' + str(i), lons = lo, lats=la)
        print('shape: ', lo.shape)

#split_grid()


def plot_grid():
    ##For testing the distribution
    
    plt.figure(figsize=(15,15))
    m = Basemap(projection='mill', llcrnrlat=-89., urcrnrlat=89., llcrnrlon=0., urcrnrlon=360., resolution='l')
    m.drawcoastlines()
    
    p_tot=0
    for i in range(5):

        data = np.load(outdir + 'coordinates' + str(i) + '.npz')
        lons = data['lons']
        print('min: ', np.min(lons))
        print('max: ', np.max(lons))
        lats = data['lats']
        p_tot+=len(lons)
        print(len(lons))
        xs, ys = m(lons, lats)
        m.scatter(xs,ys)
    
    print('total number of particles: ', p_tot)

plot_grid()