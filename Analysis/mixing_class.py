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
import os
import scipy.sparse.linalg as sp_linalg
import datetime

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


"""
For Entropy
"""

def transition_matrix_entropy(pdir, entropy_dir):
    """
    Transport matrix for the full 10-year simulation, used for the definition of regions for the entropy method.
    Binning is 2 degree for selecting the regions, 4,5,6 degrees for the actual entropy (see below)
    """
    
    filename = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #File name for many particle simulation
    pdata=ParticleData.from_nc(pdir, filename, tload=[0,-1], n_grids=40)
    pdata.compute_matrix(d_deg=2., save=True, name = entropy_dir + 'T')


def compute_transfer_matrices_entropy(pdir, entropy_dir, d_deg):
    
    #Load particle data
    fname = 'surfaceparticles_y2000_m1_d5_simdays3650_pos' #F
    tload = list(range(0,730,73))+[729] #load data each year (indices, not actual times)

    #load data
    pdata=ParticleData.from_nc(pdir, fname, tload=tload, n_grids=40)
    pdata.set_discretizing_values(d_deg = 2.)
    pdata.remove_nans()

    #Get those particles that start and end in the chosen basin
    r = np.load(entropy_dir + 'clusters.npy')
    
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
        
#        lons=basin_data.lons.filled(np.nan)
#        lats=basin_data.lats.filled(np.nan)
#        pdata_basin = ParticleData(lons=lons, lats=lats)
        #compute transfer matrix
        for t in range(0,len(basin_data.lons[0])):     
            
            basin_data.compute_matrix(d_deg=d_deg, t0=0, t1=t)            
            sparse.save_npz(entropy_dir + 'transfer_matrix_deg' + str(int(d_deg)) + '_t' + str(t) + '_basin' + str(i_basin), basin_data.A)
        
#        np.savez(entropy_dir + '/reduced_particles_' + str(i_basin), lons=lons, lats=lats)
        

#def compute_transfer_matrix():
#    #deg_labels is the choice of square binning
#    
#    for i_basin in range(1,6):
#        
#        #load reduced particle data for each basin
#        pdata = np.load(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin) + '.npz', 'r')
#        lons=pdata['lons']
#        lats=pdata['lats']
#
#        del pdata
#        pdata = ParticleData(lons=lons, lats=lats)
#        
#        #compute transfer matrix
#        for t in range(0,len(lons[0])):     
#            
#            pdata.compute_matrix(d_deg=d_deg, t0=0, t1=t)            
#            sparse.save_npz(outdir_paper + 'EntropyMatrix/transfer_matrix_deg' + str(int(d_deg)) + '/T_matrix_t' + str(t) + '_basin' + str(i_basin), pdata.A) #, original_labels=original_labels)


"""
For Markov mixing time
"""

def setup_annual_markov_chain(data_dir, matrix_dir, d_deg, n_grids):
    """
    Function to compute a Markov matrix from the given method.
    - data_dir: directory with the data
    - output_dir: matrices are saved there
    - d_deg: binning of matrix
    - n_grids: number of initial particle grids
    """
    print('data dir: ', data_dir)
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
        pdata.compute_matrix(d_deg=d_deg, save=True, name = matrix_dir + 'T_' + n)
    
    #create annual matrix
    for i in range(len(names)):
        n = names[i]
        print('Loading matrix for ', n)
        t = sparse.load_npz(matrix_dir + 'T_' + n + '.npz')
        
        if i == 0 :
            T = t
        else:
            T = T.dot(t)
        
        sparse.save_npz(matrix_dir + 'T', T)


def get_matrix_power(matrix_dir, power):
    
    T = sparse.load_npz(matrix_dir + 'T.npz')
    T = sparse.csr_matrix(T)
    
    print('Computing power ', power)
    
    Tx = T**power
    sparse.save_npz(matrix_dir + 'T' + str(power), Tx)


def get_clusters(matrix_dir, d_deg, matrix_file = 'T10.npz'):

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

    T = sparse.load_npz(matrix_dir + matrix_file)
    T = sparse.csr_matrix(T)
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
            d=T.dot(r1)
            r2[np.argwhere(d>=.5)]=1
            r2[np.argwhere(d<.5)]=0
        
        #some tests
        assert(not np.any(r2[d<0.5]))
        final_regions[r]=r2
        print( 'iterations to convergence: ', i)
    
    clusters = np.zeros(len(Lats_centered)*len(Lons_centered))

    for i in range(len(clusters)):
        for k in final_regions.keys():
            if final_regions[k][i]==1:
                clusters[i]=labels[k]

#    clusters = np.ma.masked_array(clusters, clusters==0) 
#    clusters=clusters.reshape((len(Lats_centered),len(Lons_centered)))    
#    ocean_clusters=np.roll(ocean_clusters,int(180/d_deg))
    
    np.save(matrix_dir + 'clusters', clusters)


def project_to_regions(matrix_dir):
    
    #Load matrix and region definitions
    A = sparse.load_npz(matrix_dir + 'T.npz')
    A = sparse.coo_matrix(A)
    clusters = np.load(matrix_dir + 'clusters.npy')

    row = A.row
    col = A.col
    val = A.data                       
    
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
        
        print('rowsum')
        rowsum = np.array(sparse.coo_matrix.sum(A_new, axis=1))[:,0]
        
        #column-normalize                
        print('column normalize')
        for r in np.unique(row_new):
            val_new[row_new==r] /= rowsum[r]

        A_new.data = val_new

        print('projecting done')
        sparse.save_npz(matrix_dir + 'T_basin_' + str(basin_number), A_new)


def stationary_densities(matrix_dir):
    
    for basin_number in range(1,6):
        print( 'basin_number: ', basin_number)
        A = sparse.load_npz(matrix_dir + 'T_basin_' + str(basin_number) + '.npz')
        val, vec = sp_linalg.eigs(A.transpose(),k=5,which='LM')    
        vec = vec[:,np.argsort(np.abs(val))]
        val = val[np.argsort(np.abs(val))]            
        d0=np.array(vec[:,-1])
        d0/=np.sum(d0) #Normalized eigenvector with eigenvalue 1
        np.save(matrix_dir + 'stationary_density' + str(basin_number), d0)


def mixing_time(matrix_dir, eps):
    
    for basin_number in range(1,6):
        print( 'basin_number: ', basin_number)
        T0 = sparse.load_npz(matrix_dir + 'T_basin_' + str(basin_number) + '.npz').toarray()
        d0 = np.load(matrix_dir + 'stationary_density' + str(basin_number) + '.npy')
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
        
        np.save(matrix_dir + 'tmix_eps' + str(int(eps*100)) + '_' + 'basin_' +str(basin_number), tmix)


def other_matrix_powers(matrix_dir):
    
    T = sparse.load_npz(matrix_dir + 'T.npz')    
    T = sparse.csr_matrix(T)
    sparse.save_npz(matrix_dir + 'T0', T)

    T5 = T**5
    sparse.save_npz(matrix_dir + 'T5', T5)

    T15 = T5**3
    sparse.save_npz(matrix_dir + 'T15', T15)
    
    T20 = T15.dot(T5)
    sparse.save_npz(matrix_dir + 'T20', T20)    
    del T15
    
    T25 = T20.dot(T5)
    sparse.save_npz(matrix_dir + 'T25', T25)        

    T30 = T25.dot(T5)
    sparse.save_npz(matrix_dir + 'T30', T30)        
    del T25
    
    T40 = T20**2
    sparse.save_npz(matrix_dir + 'T40', T40)        
    del T40
    
    T50 = T20.dot(T30)
    sparse.save_npz(matrix_dir + 'T50', T50)
    del T20
    del T30
    del T5
    
    T100 = T50**2
    sparse.save_npz(matrix_dir + 'T100', T100)    
    del T50
    
    T200 = T100**2
    sparse.save_npz(matrix_dir + 'T200', T200)
    
    T500 = T200**2
    T500 = T500.dot(T100)
    del T100
    del T200
    sparse.save_npz(matrix_dir + 'T500', T500)    
    
    T1000 = T500**2
    sparse.save_npz(matrix_dir + 'T1000', T1000)



#    def set_labels(self, ddeg, t):
#        """
#        labeling of particles according to rectilinear boxes with spacing ddeg
#        """
#        
#        self.label_ddeg=ddeg
#        N=360//ddeg
#        self.label=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])