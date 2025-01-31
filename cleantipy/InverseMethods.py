# -*- coding: utf-8 -*-
"""
Library computing different aspects of beamforming, PSF or other tools

Created on Fri Apr  1 14:33:53 2022
@author: rleiba
"""

import numpy as np
import pylab as pl
import sys
from joblib import Parallel, delayed
from scipy.signal import butter, lfilter, filtfilt, sosfilt
import time
import multiprocessing as mp
from scipy.interpolate import interp1d
from .Rotations_3D import angle_to_pos
from .CommonFunctions import computeDistance, InterpolateTimeTrajectory

import psutil
n_CPU_Threads = mp.cpu_count()



class PointSpreadFunction:
    """
    Class to compute the Point Spread Fucntion of a microphone array based on 
    its geometry, the grid and the source position in the grid plane
    """
    def __init__(self, geom, grid, f, isMicActive):
        self.c = 343 #m/s
        self.geom = geom
        self.grid = grid
        self.f = f
        self.isMicActive = isMicActive
        self.omega = 2*np.pi*f
        self.k = self.omega/self.c
        self.Nf = f.shape[0]
        self.Ni = grid.shape[0]
        self.PSF = np.zeros((self.Ni,self.Nf),dtype='complex128')
        self.indFreqUsed, = np.where(np.sum(isMicActive,axis=0))

    def compute(self,parrallel=True):
        if parrallel:
            PSF = Parallel(n_jobs=4)(delayed(self.compute_core)(ff) for ff in range(self.Nf))
            for i in self.indFreqUsed:
                self.PSF[:,i] = PSF[i]
        else:
            for ff in range(len(self.f)):
                self.PSF[:,ff] = self.compute_core(ff)

    
    def compute_r_mi(self,geom=None,grid=None):
        if grid is None:
            grid = self.grid
        if geom is None:
            geom = self.geom 
            
        r_mi = np.zeros((geom.shape[0],grid.shape[0]))
        for m in range(geom.shape[0]):
            geom_tiled = np.tile(geom[m,:],grid.shape[0]).reshape((grid.shape[0],3))
            r_mi[m,:] = pl.norm(geom_tiled-grid, axis=1)
        return r_mi

    def compute_core(self,ff):
        indMicActive, = np.where(self.isMicActive[:,ff])
        if len(indMicActive)!=0:
            geom_new = self.geom[indMicActive,:]
            R_0 = self.compute_r_mi(geom=geom_new,grid=np.array([[0,0,self.grid[0,2]]]))
            r_mi = self.compute_r_mi(geom=geom_new)
            w_n = np.matrix(np.exp(-1j*self.k[ff]*r_mi))/r_mi
            G = np.matrix(np.exp(-1j*self.k[ff]*R_0))/R_0
            PSF = np.array(np.abs(np.dot(w_n.H, G))).squeeze()**2/ \
                np.sum(np.abs(np.array(G).squeeze())**2)
            return PSF
    
    def plot(self,f0=[1000]):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])

        ax0.scatter(self.grid[:,0],self.grid[:,1],self.grid[:,2])
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        ax0.set_title('Setup (mic in blue and image points in orange)')
        pl.tight_layout()
        
        
        
        
        fig = pl.figure(figsize=(12,8))
        for ff in range(len(f0)):
            fig.add_subplot(2,int(len(f0)/2)+1,ff+1)
            ind_f = np.argmin(abs(f0[ff]-self.f))

            pl.scatter(self.geom[self.isMicActive[:,ind_f],0],\
                       self.geom[self.isMicActive[:,ind_f],1])
            
            pl.title(str(f0[ff]) + ' Hz')
            if np.mod(ff+1,2):
                pl.ylabel('y (m)')
            if ff in [4,5]:
                pl.xlabel('x (m)')
            pl.tight_layout()
    
    
class Beamforming_f:
    """
    Class to compute the classical frequency domain beamforming

    """
    def __init__(self, geom, grid, f, p_f, isMicActive):
        self.c = 343 #m/s
        self.geom = geom
        self.grid = grid
        self.f = f
        self.p_f = p_f # (Nmic, freq) data
        self.isMicActive = isMicActive
        self.omega = 2*np.pi*f
        self.k = self.omega/self.c
        self.Nf = f.shape[0]
        self.Ni = grid.shape[0]
        self.BF_f = np.zeros((self.Ni,self.Nf),dtype='complex128')
        self.indFreqUsed, = np.where(np.sum(isMicActive,axis=0))
        self.Nm = geom.shape[0]

    def compute(self,parrallel=True):
        t1 = time.time()
        if parrallel:
            BF_f = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_core)(ff) for ff in range(self.Nf))
            for i in self.indFreqUsed:
                self.BF_f[:,i] = BF_f[i]
        else:
            for ff in range(len(self.f)):
                self.BF_f[:,ff] = self.compute_core(ff)
        t2 = time.time()
        print("Beamforming computation took %.1f s"%(t2-t1))

    
    def compute_r_mi(self,geom=None,grid=None):
        if grid is None:
            grid = self.grid
        if geom is None:
            geom = self.geom 
            
        r_mi = np.zeros((geom.shape[0],grid.shape[0]))
        for m in range(geom.shape[0]):
            geom_tiled = np.tile(geom[m,:],grid.shape[0]).reshape((grid.shape[0],3))
            r_mi[m,:] = pl.norm(geom_tiled-grid, axis=1)
        return r_mi

    def compute_core(self,ff):
        indMicActive, = np.where(self.isMicActive[:,ff])
        if len(indMicActive)!=0:
            geom_new = self.geom[indMicActive,:]
            r_mi = self.compute_r_mi(geom=geom_new)
            W = np.matrix(np.exp(-1j*self.k[ff]*r_mi))/self.Nm
            BF_f = np.dot(W.H, self.isMicActive[indMicActive,ff]*self.p_f[ff,indMicActive]).squeeze()
            return BF_f
    
    def plot(self,f0=[1000]):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])

        ax0.scatter(self.grid[:,0],self.grid[:,1],self.grid[:,2])
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        ax0.set_title('Setup (mic in blue and image points in orange)')
        pl.tight_layout()
        
        
        
        
        fig = pl.figure(figsize=(12,8))
        for ff in range(len(f0)):
            fig.add_subplot(2,int(len(f0)/2)+1,ff+1)
            ind_f = np.argmin(abs(f0[ff]-self.f))

            pl.scatter(self.geom[self.isMicActive[:,ind_f],0],\
                       self.geom[self.isMicActive[:,ind_f],1])
            
            pl.title(str(f0[ff]) + ' Hz')
            if np.mod(ff+1,2):
                pl.ylabel('y (m)')
            if ff in [4,5]:
                pl.xlabel('x (m)')
            pl.tight_layout()

    
class Beamforming_t:
    """
    Class to compute temporal domain beamforming

    """
    def __init__(self, geom, grid, t, p_t):
        self.c = 343 #m/s
        self.geom = geom
        self.grid = grid
        self.t = t
        self.p_t = p_t # (Nmic, time) data
        self.Nt = t.shape[0]
        self.Ni = grid.shape[0]
        self.BF_t = np.zeros((self.Ni,self.Nt))
        self.Nm = geom.shape[0]
        self.fs = 1/(t[1] - t[0])

    def compute(self,parrallel=True):
        t1 = time.time()
        ind_end = np.zeros((self.Ni,))
        if parrallel:
            BF_t = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_core)(i) for i in range(self.Ni))
            for i in range(self.Ni):
                self.BF_t[i,:] = BF_t[i]
                ind_end[i] = np.min(np.where(self.BF_t[i,:]==0))
        else:
            for i in range(self.Ni):
                self.BF_t[i,:] = self.compute_core(i)
                ind_end[i] = np.min(np.where(self.BF_t[i,:]==0))
        
        # Removing extra zeros at the end of the signals
        self.BF_t = self.BF_t[:,0:np.max(ind_end).astype(np.int32)]

        t2 = time.time()
        print("Beamforming computation took %.1f s"%(t2-t1))
    
    def compute_r_mi(self,geom=None,grid=None):
        if grid is None:
            grid = self.grid
        if geom is None:
            geom = self.geom 
            
        r_mi = np.zeros((geom.shape[0],grid.shape[0]))
        for m in range(geom.shape[0]):
            geom_tiled = np.tile(geom[m,:],grid.shape[0]).reshape((grid.shape[0],3))
            r_mi[m,:] = pl.norm(geom_tiled-grid, axis=1)
        return r_mi

    def compute_core(self,ii):
        r_mi = self.compute_r_mi(grid=np.array([self.grid[ii,:]]))
        tau = r_mi/self.c
        BF_t = np.zeros(self.Nt)
        # tau -= np.min(tau)
        K = np.sum((1/r_mi)**2,axis=0)
        for mm in range(self.Nm):
            inds = np.arange(self.Nt)+int(tau[mm]*self.fs)
            inds = inds[np.where(inds<self.Nt)]
            # BF_t[inds] += self.p_t[mm,inds]/r_mi[mm]/K
            BF_t += np.interp(np.arange(self.Nt),inds-inds[0] , self.p_t[mm,inds]/r_mi[mm,:]/K, right=0)
        return BF_t

    def plot(self):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])

        ax0.scatter(self.grid[:,0],self.grid[:,1],self.grid[:,2])
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        ax0.set_title('Setup (mic in blue and image points in orange)')
        pl.tight_layout()


class Beamforming_t_traj:
    """
    Class to compute temporal domain beamforming over a trajectory

    """
    def __init__(self, geom, grid, traj, t, p_t, angles=None, t_traj=None,\
                 debug=False,QuantitativeComputation=True,internVar_dtype=np.float64,\
                 c=343,Mach_w=None):
        """
        Parameters
        ----------
        geom : numpy array
            geometry of the array (Nmic x 3)
        grid : numpy array
            grid points (Ni x 3)
        traj : numpy array
            trajectory of the grid (Nt_ x 3)
        t : numpy array
            time vector (Nt)
        p_t : numpy array
            microphone signals (Nmic x Nt)
        angles : numpy array
            rotations around x,y and z in rad (Nt x 3)
        t_traj : numpy array
            time vector for the trajectory (Nt_)
        c : float
            Speed of sound
        Mach_w : numpy array (3,)
            Mach number of the wind along x, y, z
        """
        self.c = c #m/s
        self.Mach_w = Mach_w
        self.geom = geom
        self.grid = grid
        self.t = t
        self.p_t = p_t.copy().astype(np.float64) # (Nmic, time) data
        self.Nt = t.shape[0]
        self.Ni = grid.shape[0]
        self.Nm = geom.shape[0]
        self.fs = 1/(t[1] - t[0])
        self.traj = traj
        self.internVar_dtype=internVar_dtype
        self.angles = angles
        self.debug = debug
        self.QuantitativeComputation = QuantitativeComputation

        
        if self.debug:
            print("Number of CPU available : %d" %(n_CPU_Threads))
        
        if t_traj is None:
            if self.Nt != self.traj.shape[0]:
                print("[BF init] Trajectory and time sampling are not equivalent, please indicate a time vector for the trajectory")
                sys.exit()
            else:
                print("Beamforming : assuming trajectory has the same sampling frequency and time origin as mic signals")
                self.t_traj = np.arange(traj.shape[0])/self.fs
                self.Nt_traj = self.Nt
                self.fs_traj = self.fs
                # self.samplingRatio = 1
        else:
            self.t_traj = t_traj
            self.Nt_traj = t_traj.shape[0]
            self.fs_traj = 1/(t_traj[1] - t_traj[0])

        self.t_traj_interp = InterpolateTimeTrajectory(self.t_traj,self.fs)
        
        # limit the computation to temporal points corresponding to the selected part of the trajectory
        self.Nt = self.t_traj_interp.size


        self.BF_t = np.zeros((self.Ni,self.Nt))

        info = psutil.virtual_memory()
        memory_needed = self.Nt_traj*self.Nm*self.Ni*8*2 + self.Nt*self.Ni*8 + self.Nt*self.Nm*8  # 3 variables (r_mi_t & mag) + BF_t + p_t
        if memory_needed > info.available:
            if self.debug:
                print("Not enough RAM available (%.1f GB), approx. memory needed: %.1f GB" %(info.available/1024/1024/1024, memory_needed/1024/1024/1024))
            if self.Nt_traj*self.Nm*self.Ni*4*2 + self.Nt*self.Ni*8 + self.Nt*self.Nm*8 < info.available: # 3 variables (r_mi_t & mag in float 32) + BF_t + p_t
                if self.debug:
                    print("Passing temporary variables in float 32")
                self.internVar_dtype=np.float32
        
        # Initiate the mic-grid distance computation
        self.compute_r_mi_t(parrallel=True)

        


    def grid_pos_rotated(self,grid,traj,ang,parrallel=True):
        self.grid_t = np.zeros((grid.shape[0],self.Nt_,3),dtype=self.internVar_dtype) 
        
        # if self.angles is None:
        #     if self.debug:
        #         print('No angle considered') 
            
        # for i in range(grid.shape[0]):
        #     grid_tiled = np.tile(grid[i,:],traj.shape[0]).reshape((traj.shape[0],3)).astype(np.float64)
        #     if self.angles is not(None):
        #         grid_tiled = angle_to_pos(grid_tiled,ang[:,0],ang[:,1],ang[:,2])
        #     grid_tiled += self.traj_.astype(np.float64)
            
        #     # Storing the position of sources along the trajectory
        #     self.grid_t[i,:,:] =  grid_tiled 
 
        if not(parrallel):
            for i in range(grid.shape[0]):
                self.grid_t[i,:,:] = self.grid_pos_rotated_core(i,grid,traj,ang)

        else:
            tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(self.grid_pos_rotated_core)(i,grid,traj,ang) for i in range(grid.shape[0]))
            for i in range(grid.shape[0]):
                self.grid_t[i,:,:] = tmp[i]

    def grid_pos_rotated_core(self,i,grid,traj,ang):
        grid_tiled = np.tile(grid[i,:],traj.shape[0]).reshape((traj.shape[0],3)).astype(np.float64)
        if self.angles is not(None):
            grid_tiled = angle_to_pos(grid_tiled,ang[:,0],ang[:,1],ang[:,2])
        grid_tiled += self.traj_.astype(np.float64)
        return grid_tiled


    def compute(self,parrallel=True, interpolation='linear'):
        """
        Function computing the beamforming
        
        Parameters
        ----------
        parrallel : bool
            Parrallelize the computation (default=True)
        interpolation : str
            select the interpolation type from scipy.interpolate.interp1d : 
            ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’,
            ‘cubic’, ‘previous’, or ‘next’. 
            
            ‘quadratic’ recomended for better results and 'linear' for good
            compromize (computation time vs quality)


        """
        self.interp = interpolation        
       
        t1 = time.time()
        if parrallel:
            BF_t = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_core)(i) for i in range(self.Ni))
            for i in range(self.Ni):
                self.BF_t[i,:] = BF_t[i]
                # ind_end[i] = np.min(np.where(self.BF_t[i,:]==0))
        else:
            for i in range(self.Ni):
                self.BF_t[i,:] = self.compute_core(i)
                # ind_end[i] = np.min(np.where(self.BF_t[i,:]==0))
        
        # Removing extra zeros at the end of the signals
        # self.BF_t = self.BF_t[:,0:np.max(ind_end).astype(np.int32)]
        
        t2 = time.time()
        if self.debug:
            print("Time domain beamforming over trajectory computation took %.1f s"%(t2-t1))


    def compute_r_mi_t_core_i(self,i):    
        r_m_t = np.zeros((self.Nm,self.Nt_),dtype=self.internVar_dtype)
        Mcostheta = np.zeros((self.Nm,self.Nt_),dtype=self.internVar_dtype)  
        for m in range(self.Nm):
            r_m_t[m,:], Mcostheta[m,:] = computeDistance(self.geom_[m,:3],self.grid_t[i,:,:3],self.M,self.Mach_w)
        
        return r_m_t, (r_m_t*(1-Mcostheta)**2)  
    
    def compute_r_mi_t_core(self,m):    
        r_i_t = np.zeros((self.Ni,self.Nt_),dtype=self.internVar_dtype)
        Mcostheta = np.zeros((self.Ni,self.Nt_),dtype=self.internVar_dtype)  
        for i in range(self.Ni):
            r_i_t[i,:], Mcostheta[i,:] = computeDistance(self.geom_[m,:3],self.grid_t[i,:,:3],self.M,self.Mach_w)
        
        return r_i_t, (r_i_t*(1-Mcostheta)**2) 
        
    def compute_r_mi_t(self,geom=None,grid=None,traj=None,t_traj=None,parrallel=True,ang=None):
        if grid is None:
            grid = self.grid
        if geom is None:
            geom = self.geom 
        if traj is None:
            traj = self.traj 
        if t_traj is None:
            t_traj = self.t_traj 
        if ang is None and self.angles is not None:
            ang = self.angles
            
        # self.grid = grid
        self.geom_ = geom
        self.traj_ = traj
        self.t_traj_ = t_traj
        self.Nt_ = traj.shape[0]  
        self.fs_traj_ = 1/(t_traj[1] - t_traj[0])
        self.ang_ = ang 
        
        t0 = time.time() 
        self.grid_pos_rotated(grid,traj,ang,parrallel)

        t1 = time.time()   
        if self.debug:
            print("grid_pos_rotated function computation took %.1f s"%(t1-t0))

        self.r_mi_t = np.zeros((geom.shape[0],grid.shape[0],self.Nt_),dtype=self.internVar_dtype)
        self.mag = np.zeros((geom.shape[0],grid.shape[0],self.Nt_),dtype=self.internVar_dtype)

        # M = np.diff(traj,axis=0)*self.fs_traj/self.decimFactor/self.c
        M = np.diff(traj,axis=0)*self.fs_traj_/self.c
        M = np.append(M,np.array([M[-1,:]]), axis = 0)
        self.M = M        

        if not(parrallel):
            print("No Parrallel calculation for r_mi_t")
            for m in range(self.Nm):
                r_i_t, mag = self.compute_r_mi_t_core(m)
                self.r_mi_t[m,:,:] = r_i_t
                self.mag[m,:,:] = mag

        else:
            tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_r_mi_t_core)(m) for m in range(self.Nm))
            for m in range(self.Nm):
                self.r_mi_t[m,:,:] = tmp[m][0]
                self.mag[m,:,:] = tmp[m][1]        
        t2 = time.time()
        if self.debug:
            print("r_mi computation took %.1f s"%(t2-t1))
        

    def compute_core(self,ii):
        BF_t = np.zeros(self.Nt,dtype=self.internVar_dtype)
        if self.Nt_ == self.Nt:
            tau = np.squeeze(self.r_mi_t[:,ii,:])/self.c
            if self.QuantitativeComputation:
                # not needed for CLEAN-T algorithm
                A = 1/np.squeeze(self.mag[:,ii,:])
        else:
            f_tau = interp1d(self.t_traj, \
                              np.squeeze(self.r_mi_t[:,ii,:]), \
                          kind=self.interp,bounds_error=False,fill_value="extrapolate")
            tau = f_tau(self.t_traj_interp)/self.c            

            
            if self.QuantitativeComputation:
                # not needed for CLEAN-T algorithm
                f_mag = interp1d(self.t_traj, \
                                 np.squeeze(self.mag[:,ii,:]), \
                             kind=self.interp,bounds_error=False,fill_value="extrapolate")
                    
                A = 1/f_mag(self.t_traj_interp)                
        if self.QuantitativeComputation: 
            # not needed for CLEAN-T algorithm
            Q_i = 1/np.sum((A)**2,axis=0)  
            
        for mm in range(self.Nm):
            taus = self.t_traj_interp+tau[mm,:]
            tau0 = tau[mm,0]
            ind_0 = int(np.ceil(taus[0]*self.fs))

            f_interp = interp1d(self.t[ind_0:ind_0+self.Nt], self.p_t[mm,ind_0:ind_0+self.Nt], \
                          kind=self.interp,bounds_error=False,fill_value=0)

            if self.QuantitativeComputation: 
                # not needed for CLEAN-T algorithm
                BF_t += A[mm,:]*f_interp(taus)*Q_i
            else:
                BF_t += f_interp(taus)

        return BF_t

    def plot(self):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])

        ax0.scatter(self.grid[:,0]+self.traj[0,0], \
                    self.grid[:,1]+self.traj[0,1], \
                    self.grid[:,2]+self.traj[0,2])
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        ax0.set_title('Setup (mic in blue and image points in orange)')
        pl.tight_layout()


def get_weight(r,freq=[1000]):
    
    # Option 1
    # w = np.ones((r.size,))
    # func = 1000/freq*5000/r**4
    # w[np.where(func<1)] = func[func<1]

    # Option 2
    window = np.kaiser(51, 14)
    w = np.zeros((len(r),len(freq)))
    for ff, f in enumerate(freq):
        w[:,ff] = np.interp(r,np.arange(25)*max(r)/25*1000/f,window[25:-1])
    
    # pl.scatter(geom[:,0],geom[:,1],alpha=w)
    return w.squeeze() # in case of len(freq)==1

def frequencyBand(data,f,fc=[1000],type='octave'):
    '''
    Compute SPL from data on octave bands

    Parameters
    ----------
    data  : numpy array
        data in shape (freq, n_data)
    f  : numpy array
        frequency axis
    fc : list
        Center frequencies of octave bands to process (Hz).
    type : string
        chose between 'octave' for octave bands or 'third' for third-octave bands
    '''
    if type=='third':
        fminmax = np.array([[fck/2**(1./6), fck*2**(1./6)] for fck in fc])

    elif type=='octave':
        fminmax = np.array([[fck/2**0.5, fck*2**0.5] for fck in fc])

    else:
        sys.exit('Frequency band not coded')
    
    p_ref = 2*10**-5
    
    data_octave = np.zeros((len(fc),data.shape[-1]))
    for ff in range(len(fc)):
        inds = np.where((f>=fminmax[ff,0]) & (f<fminmax[ff,1]))[0]
        num = 2*np.sum(np.abs(data[inds,:])**2,axis=0) 
        data_octave[ff,:] = 10*np.log10(num/(len(f)**2 * p_ref**2))
    return data_octave

def butter_bandpass(lowcut, highcut, fs, order=3):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, highcut, fs, order=3):
    b, a = butter(order, highcut, fs=fs, btype='lowpass')
    y = filtfilt(b, a, data)
    return y
    

def interp_nb(x_vals, x, y):
    y_vals = np.zeros((y.shape[0],x_vals.size))
    for mm in range(y.shape[0]):
        y_vals[mm,:] = np.interp(x_vals, x, y[mm,:])
    return y_vals
