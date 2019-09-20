"""
Mixing of passive tracers at the ocean surface and implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Objects and functions for data analysis
--------------
"""

import numpy as np
from netCDF4 import Dataset
from scipy import sparse
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as colors

(minlon, maxlon, minlat, maxlat) = (0., 360., -90., 90.) #maximum of bathymetry file is 10565 m


class ParticleData(object):
    """
    Explain
    """

    def __init__(self, lons=[], lats=[]):
        """
        :param lons, lats, z, times: arrays containing the data
        """
        print('Particle data created')
        print('---------------------')
        print('Particles: ', len(lons))
        print('---------------------')
        self.lons               = lons
        self.lats               = lats
        self.n_particles        = len(lons)
        self.nans_removed       = False
        self.discrete           = False


    @classmethod
    def from_nc(cls, pdir, fname, tload=None, n_grids=40):
        """
        Load 2D data from netcdf particle output files
        :param pdir: directory of files
        :param fname: file name in pdir
        :param tload: vector of times to be loaded
        :Ngrids: number of different grid output files
        """

        print('Loading data from files: ', pdir + fname)
        
        #Load data from first grid-array
        i = 0
        print('Load grid no: ', i)

        pfile   = pdir + fname + str(i)+'.nc'     
        data    = Dataset(pfile,'r')
        lons    = data.variables['lon'][:,tload]
        lats    = data.variables['lat'][:,tload]
        
        #Load data from other grid-arrays        
        for i in range(1,n_grids):
            print('Load grid no: ', i)
            pfile   = pdir + fname + str(i)+'.nc'  
            data    = Dataset(pfile,'r')
            lons    = np.vstack((lons, data.variables['lon'][:,tload]))
            lats    = np.vstack((lats, data.variables['lat'][:,tload]))

        return cls(lons=lons, lats=lats)
        
    
    def __del__(self):
        print('Particle Data deleted')


    def remove_nans(self):
        
        print('Removing deleted particles...')
        
        if np.any(np.ma.getmask(self.lons)):
            inds = np.bitwise_or.reduce(np.ma.getmask(self.lons), 1)
            self.lons   = self.lons[~inds]
            self.lats   = self.lats[~inds]
            print('Number of removed particles: ', self.n_particles - len(self.lons))
            self.n_particles  = len(self.lons)

        elif np.any(np.isnan(self.lons)):
            inds = np.bitwise_or.reduce(np.isnan(self.lons), 1)
            self.lons = self.lons[~inds]
            self.lats = self.lats[~inds]    
            print('Number of removed particles: ', self.n_particles - len(self.lons))
            self.n_particles  = len(self.lons)

        else:
            print('Data does not contain any masked elements')

        print('---------------------')  
        self.nans_removed = True

    
    def coords_to_matrixindex(self, coords):
        """
        Return the matrix index from a (lon,lat,z) coordinate
        """

        (lon, lat) = coords
        index_2D    = int((lat - minlat)//self.d_deg * self.n_lons + (lon - minlon)//self.d_deg)
        return index_2D

    
    def matrixindex_to_coord(self, index_2D):
        """
        Return the central coordinate of a cell
        """
        
        lon = (index_2D%self.n_lons)*self.d_deg + minlon
        lat = (index_2D - (lon - minlon)//self.d_deg) / self.n_lons * self.d_deg + minlat        
        lon += self.d_deg/2.
        lat += self.d_deg/2.
        
        return (lon, lat)
        
    
    def set_discretizing_values(self, d_deg):
        """
        Set cell size for matrix computation
        """
        self.d_deg  = d_deg
        self.n_lons  = int((maxlon - minlon)/d_deg)
        self.n_lats  = int((maxlat - minlat)/d_deg)
        self.n_total   = self.n_lons * self.n_lats        
        self.discrete = True
    
    
    def compute_matrix(self, d_deg, save=False, name='noname', t0=0, t1=-1): #, remove_land=False, re_normalize=False):
        
        if not self.nans_removed:
            self.remove_nans()
            
        if not self.discrete:
            self.set_discretizing_values(d_deg)
            
        print('computing indices')
        initial_index_2D    = np.array([self.coords_to_matrixindex((lon, lat)) for (lon, lat) in zip(self.lons[:,t0], self.lats[:,t0])])
        final_index_2D      = np.array([self.coords_to_matrixindex((lon, lat)) for (lon, lat) in zip(self.lons[:,t1], self.lats[:,t1])])
        
        assert(np.min(initial_index_2D)>=0)
        assert(np.min(final_index_2D)>=0)
        assert(np.max(initial_index_2D)<self.n_total)
        assert(np.max(final_index_2D)<self.n_total)

        initial_indices, initial_counts = np.unique(initial_index_2D, return_counts=True)

        #Create list of initial and final tuples
        index_pairs = [[(initial_index_2D[i], final_index_2D[i])] for i in range(len(final_index_2D))]
        del final_index_2D

        #count the tuple occurences
        import collections 
        Output = collections.defaultdict(float) 
        for elem in index_pairs:
            Output[elem[0]] += 1.
        
        coords = np.array(list(Output.keys()))

        #for the sparse coo matrix
        rows = coords[:,0]
        columns = coords[:,1]
        vals = np.array(list(Output.values()))
        #normalize the vals

        for i in range(len(initial_indices)):
            vals[rows == initial_indices[i]] /= initial_counts[i]

#        if remove_land:
#            m = np.max(initial_counts)
#            I = initial_indices[np.argwhere(initial_counts == m)]
#            r = [i for i in range(len(rows)) if rows[i] in I]
#            rows = rows[r]
#            columns = columns[r]
#            vals = vals[r]
#            
#            r = [i for i in range(len(rows)) if columns[i] in I]
#            rows = rows[r]
#            columns = columns[r]
#            vals = vals[r]
#
#            del I
#            del r
#            
#            if re_normalize:
#                for r in np.unique(rows):
#                    s = np.sum(vals[rows == r])
#                    vals[rows == r] = vals[rows == r]/s
#                                
        del initial_indices
        del initial_counts

        self.A = sparse.coo_matrix((vals, (rows, columns)), shape=(self.n_total, self.n_total))

        #check if non-zero rows add up to 1
#        if not (remove_land and not re_normalize):
        print('Check normalization')
        s = sparse.coo_matrix.sum(self.A, axis=1)
        s = s[s!=0]
        assert(np.all(abs(s-1) < 1e-13))
        
        print('saving matrix')
        if save:
            sparse.save_npz(name, self.A)


    def plot_horizontal_distribution(self, v, title = 'noname'):
        """
        Plot distribution on horizontal (lon-lats) for tests. Summed over z_range
        """
        if v.ndim == 1:
            d2d = v.reshape(self.n_lats, self.n_lons)
        else:
            d2d = v

        Lons_edges = np.linspace(minlon,maxlon,int((maxlon-minlon)/self.d_deg)+1) 
        Lats_edges = np.linspace(minlat,maxlat,int((maxlat-minlat)/self.d_deg)+1) 
       
        lon_bins_2d,lat_bins_2d = np.meshgrid(Lons_edges,Lats_edges)        
        
        fig = plt.figure(figsize = (14,11))
        m = Basemap(projection='robin',lon_0=-0,resolution='l')
        m.drawparallels([-60,-30,0,30,60], labels=[True, False, False, True], linewidth=.8, size=7)
        m.drawmeridians([50,150,250,350], labels=[False, False, False, True], linewidth=.8, size=7)
        m.drawcoastlines()
        xs, ys = m(lon_bins_2d, lat_bins_2d)         
        p=plt.pcolormesh(xs, ys, d2d, cmap='plasma', rasterized=True)
        plt.title(title, size=10, y=1.)
        fig.colorbar(p, shrink=.5)
        plt.show()


    def compute_distribution(self, t):
            
        if not self.discrete:
            print('Please specify discretization values first.')
            sys.exit()
        else:
            index_2D    = np.array([self.coords_to_matrixindex((lon, lat)) for (lon, lat) in zip(self.lons[:,t], self.lats[:,t])])
            grid_indices, counts = np.unique(index_2D, return_counts=True)
            
            d = np.zeros(self.n_total)
            
            for i in range(len(grid_indices)):
                d[grid_indices[i]] = counts[i]
            
            d2d = d.reshape(self.n_lats, self.n_lons)
            
            return d2d


    def get_subset(self, subset_list):
        
        
        if not self.discrete:
            print('Please specify discretization values first.')
            sys.exit()
        else:

            print('Retrieving subset...')
            print('---------------------')
            
            ind = []
            for t, c in subset_list.items():
                print( 'subset t: ', t)
                keep = np.array(range(len(c)))
                keep = np.multiply(keep,c)
                
                particle_gridindex=np.array([self.coords_to_matrixindex((lon, lat)) for (lon, lat) in zip(self.lons[:,t], self.lats[:,t])])
    #            particle_gridindex=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])
                ind.append([i for i in range(len(self.lons)) if particle_gridindex[i] in keep])
            
            print( 'Get set intersections...')
            print( '---------------------')
            
            s=set(ind[0])
            for k in range(1,len(ind)):
                s2=ind[k]
                s=s.intersection(s2)
            indices=list(s)
            return ParticleData(lons=self.lons[indices], lats=self.lats[indices]) #, times=self.times[indices])
        


    def square_region(self, square):
        
        Lons = np.arange(minlon, maxlon, self.d_deg)
        Lats = np.arange(minlat, maxlat, self.d_deg)
        
        (minlo, maxlo, minla, maxla)=square    
    
        region = np.zeros((len(Lons),len(Lats)))
    
        for i in range(len(Lats)):
            for j in range(len(Lons)):
    
                la=Lats[i]
                lo=Lons[j]
    
                if (lo<maxlo and lo>=minlo and la<maxla and la>=minla):
                    region[j,i]=1
    
        region=region.T
        region=region.ravel()
        return region

    
#    def set_labels(self, ddeg, t):
#        """
#        labeling of particles according to rectilinear boxes with spacing ddeg
#        """
#        
#        self.label_ddeg=ddeg
#        N=360//ddeg
#        self.label=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])