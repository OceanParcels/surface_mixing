"""
Mixing of passive tracers at the ocean surface and implications for plastic transport modelling

David Wichmann, Philippe Delandmeter, Henk A Dijkstra and Erik van Sebille

--------------
Objects and functions for data analysis
--------------
"""

import numpy as np
from netCDF4 import Dataset

minlon=0.
maxlon=360.
minlat=-90.
maxlat=90

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


def square_region(ddeg, square):
    Lons = np.arange(0.,360.,ddeg)
    Lats = np.arange(-90.,90.,ddeg)
    
    region = np.zeros((len(Lons),len(Lats)))

    for i in range(len(Lats)):
        for j in range(len(Lons)):

            la=Lats[i]
            lo=Lons[j]
            
            (minlon, maxlon, minlat, maxlat)=square
            if (lo<maxlon and lo>=minlon and la<maxlat and la>=minlat):
                region[j,i]=1

    region=region.T
    region=region.ravel()
    return region


class ParticleData(object):
    """
    Class that containing 2D particle data and functions to analyse it
    """

    def __init__(self, lons, lats, times):
        """
        :param lons, lats, times: arrays containing the data
        """
        print '---------------------'
        print 'Particle data created'
        print '---------------------'
        print 'Particles: ', len(lons)
        print 'Snapshots: ', len(lons[0])
        print '---------------------'        
        self.lons=lons
        self.lats=lats
        self.times=times
        
    def __del__(self):
        print "Particle Data deleted"
        
    def remove_nans(self):
        print 'Removing NaNs...'
        nan_entries = np.argwhere(np.isnan(self.lons))[:,0]
        indices = [i for i in range(len(self.lons)) if i not in nan_entries]
        print 'Removed number of NaN values: ', len(self.lons)-len(indices)
        self.lons = self.lons[indices]
        self.lats = self.lats[indices]
        self.times = self.times[indices]

        print 'NaNs are removed'
        print '---------------------'


    def get_distribution(self, t, ddeg):
        """
        Calculate the particle distribution at time t. t is the integer time from loaded particles
        """
        
        lon_edges=np.linspace(minlon,maxlon,int((maxlon-minlon)/ddeg)+1)        
        lat_edges=np.linspace(minlat,maxlat,int((maxlat-minlat)/ddeg)+1)  
        d , _, _ = np.histogram2d(self.lats[:,t], self.lons[:,t], [lat_edges, lon_edges])
        return d


    @classmethod
    def from_nc(cls, pdir, fname, tload=None, Ngrids=40):
        """
        Load 2D data from netcdf particle output files
        :param pdir: directory of files
        :param fname: file name in pdir
        :param tload: vector of times to be loaded
        :Ngrids: number of different grid output files
        """

        print 'Loading data from files: ', pdir + fname
        
        #Load data from first grid-array
        i = 0
        print 'Load grid no: ', i
        pfile = pdir + fname + str(i)+'.nc'     
        data = Dataset(pfile,'r')
        times=data.variables['time'][:,tload]
        lons=data.variables['lon'][:,tload]
        lats=data.variables['lat'][:,tload]
        
        #Load data from other grid-arrays        
        for i in range(1,Ngrids):
            print 'Load grid no: ', i
            pfile = pdir + fname + str(i)+'.nc'  
            data = Dataset(pfile,'r')
            times=np.vstack((times, data.variables['time'][:,tload]))
            lons=np.vstack((lons, data.variables['lon'][:,tload]))
            lats=np.vstack((lats, data.variables['lat'][:,tload]))
        times/=86400. #Convert to days

        return cls(lons=lons, lats=lats, times=times)


    def get_subset(self, subset_list, ddeg):
        
        print 'Retrieving subset...'
        print '---------------------'
        
        N=360//ddeg
        
        ind = []
        for t, c in subset_list.iteritems():
            print 'subset t: ', t
            keep = np.array(range(len(c)))
            keep = np.multiply(keep,c)
            
            particle_gridindex=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])
            ind.append([i for i in range(len(self.lons)) if particle_gridindex[i] in keep])
        
        print 'Get set intersections...'
        print '---------------------'
        
        s=set(ind[0])
        for k in range(1,len(ind)):
            s2=ind[k]
            s=s.intersection(s2)
        indices=list(s)
        return ParticleData(lons=self.lons[indices], lats=self.lats[indices], times=self.times[indices])
        
    
    def set_labels(self, ddeg, t):
        """
        labeling of particles according to rectilinear boxes with spacing ddeg
        """
        
        self.label_ddeg=ddeg
        N=360//ddeg
        self.label=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])


class Particle(object):
    def __init__(self,time,lon,lat):
        self.time=time/86400. #Convert to days
        self.lon=lon
        self.lat=lat
        self.surv= [False if (np.ma.is_masked(time[j]) or np.isnan(time[j]) or np.isnan(lon[j])) else True for j in range(len(lat))] #Label for non-deleted particles


class TM_ParticleSet(object):
    """
    Class for particle sets used to compute transition matrices. Contains only data of two times (initial and final)
    """
    
    def __init__(self,time,lon,lat):
        print 'Total number of particles: ', len(lon)
        self.all_particles = np.empty(len(lon),dtype = Particle) #Container for all particiles
        self.survived =[]
        for i in range(len(self.all_particles)):
            if (i%10000==0):
                print 'Set up particles: ', i
            self.all_particles[i]=Particle(time[i],lon[i],lat[i])
            self.survived.append(self.all_particles[i].surv)
        
        self.lons_I=[lon[i][0] for i in range(len(lon))] #For computing initial particle number per cell later (we do not only consider those that survive the entire period)
        self.lats_I=[lat[i][0] for i in range(len(lon))]
    
    def setup_TM(self,Lons,Lats):
        ddeg=Lons[1]-Lons[0]
        
        surv = [self.survived[i][1] for i in range(len(self.all_particles))]
        self.particles = self.all_particles[surv] #Take only particles that are not deleted

        self.Lons = Lons
        self.Lats = Lats
        self.N_initial_I=np.zeros(self.Lons.size*self.Lats.size)         
        
        #Initial and final points for non-deleted particles
        lons_i = np.array([p.lon[0] for p in self.particles])
        lats_i = np.array([p.lat[0] for p in self.particles])
        lons_f = np.array([p.lon[1] for p in self.particles])
        lats_f = np.array([p.lat[1] for p in self.particles])
        print 'Number of non-deleted particles: ', len(lons_f)

        print 'Computing TM indices'
        self.TMindex_i=np.array([int(((la-np.min(self.Lats))//ddeg)*len(self.Lons)+(lo-np.min(self.Lons))//ddeg) for la,lo in zip(lats_i,lons_i)])
        self.TMindex_f=np.array([int(((la-np.min(self.Lats))//ddeg)*len(self.Lons)+(lo-np.min(self.Lons))//ddeg) for la,lo in zip(lats_f,lons_f)])
        self.TMindex_I=np.array([int(((la-np.min(self.Lats))//ddeg)*len(self.Lons)+(lo-np.min(self.Lons))//ddeg) for la,lo in zip(self.lats_I,self.lons_I)])

        print 'Computing initial position'
        for i in range(len(self.lons_I)):
            if not np.isnan(self.lons_I[i]) or np.ma.is_masked(self.lons_I[i]):            
                self.N_initial_I[self.TMindex_I[i]]+=1.       

        print 'Compute TM'
        self.TM = np.zeros((len(self.N_initial_I),len(self.N_initial_I)))
        for i in range(len(self.particles)):
            i_start = self.TMindex_i[i]
            i_finish = self.TMindex_f[i]
            self.TM[i_finish,i_start]+=1./self.N_initial_I[self.TMindex_i[i]]
        
    def save_TM(self,name):
        np.savez(name, TM=self.TM, Lons=self.Lons, Lats=self.Lats, Ninit = self.N_initial_I)
    
    @classmethod
    def from_nc(cls, path, name, dt, Ngrids):
        print 'Setting up TM: ', name
        g = 0
        pfile = path + name + 'pos' + str(g) + '.nc'
        data = Dataset(pfile, 'r')
        time=data['time'][:,[0,dt]]
        lon=data['lon'][:,[0,dt]]
        lat=data['lat'][:,[0,dt]]
        
        for g in range(1,Ngrids):
            print 'g: ', g
            pfile = path + name + 'pos' + str(g) + '.nc'
            data = Dataset(pfile, 'r')
            time=np.vstack((time, data['time'][:,[0,dt]]))
            lon=np.vstack((lon, data['lon'][:,[0,dt]]))
            lat=np.vstack((lat, data['lat'][:,[0,dt]]))

        return cls(time=time,lon=lon,lat=lat)