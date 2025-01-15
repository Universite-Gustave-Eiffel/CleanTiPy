# -*- coding: utf-8 -*-
"""
Library generating simulated data

Created on Fri Apr  1 15:41:51 2022
@author: rleiba
"""
import numpy as np
import pylab as pl
from joblib import Parallel, delayed 
import time
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from Rotations_3D import angle_to_pos
import sys 
from CommonFunctions import computeDistance, InterpolateTimeTrajectory


import multiprocessing as mp
n_CPU_Threads = mp.cpu_count()

class StaticSrcSimu_t ():
    """
    Class to compute the simulated signal received by the array microphones for
    a static source. Computation in time domain.

    """
    def __init__(self, geom, pos, t, sig, SNR=None):
        self.c = 343 #m/s
        self.geom = geom
        self.pos = pos
        self.t = t
        self.Nt = t.shape[0]
        self.Nm = geom.shape[0]
        self.Ns = pos.shape[0]
        self.sig = sig
        self.fs = 1/(t[1] - t[0])
        self.SNR = SNR
        
    def compute_r_ms(self,geom=None,pos=None):
        if pos is None:
            pos = self.pos
        if geom is None:
            geom = self.geom 
            
        r_ms = np.zeros((geom.shape[0],pos.shape[0]))
        for m in range(geom.shape[0]):
            geom_tiled = np.tile(geom[m,:],pos.shape[0]).reshape((pos.shape[0],3))
            r_ms[m,:] = pl.norm(geom_tiled-pos, axis=1)
        return r_ms
    
    def compute(self,src=None):
        if src is None:
            src = np.arange(self.pos.shape[0])
        self.r_ms = self.compute_r_ms()
        tau = self.r_ms/self.c
        p_t = np.zeros((self.Nm,self.Nt))
        # tau -= np.min(tau)
        for ii in src:
            for mm in range(self.Nm):
                # inds = np.arange(self.Nt)-int(tau[mm,ii]*self.fs)
                # p_t[mm,:] += self.sig[ii,inds]/self.r_ms[mm,ii]
                inds = np.arange(self.Nt)+int(tau[mm,ii]*self.fs)
                tmp = np.interp(np.arange(self.Nt), inds, self.sig[ii,:]/self.r_ms[mm,ii], left=0)
                p_t[mm,:] += tmp
        if self.SNR is not(None):
            for mm in range(self.Nm):
                rms = np.std(p_t[mm,:])
                p_t[mm,:] += np.random.randn(self.Nt)*rms*10**(-self.SNR/20)               
        self.p_t = p_t
        

class MovingSrcSimu_t ():
    """
    Class to compute the simulated signal received by the array microphones for
    a static source. Computation in time domain.

    """
    def __init__(self, geom, pos, traj, t, sig, t_traj=None, angles=None, SNR=None, \
                 timeOrigin='source', debug=False,\
                        c=343,Mach_w=None):
        self.c = c #m/s
        self.geom = geom
        self.pos = pos
        self.t = t
        self.Nt = t.shape[0]
        self.Nm = geom.shape[0]
        self.Ns = pos.shape[0]
        self.sig = sig
        self.fs = 1/(t[1] - t[0])
        self.SNR = SNR
        self.traj = traj
        self.timeOrigin = timeOrigin
        self.angles = angles
        self.debug = debug
        self.Mach_w = Mach_w
        if t_traj is None:
            if self.Nt != self.traj.shape[0]:
                print("[Simu init] Trajectory and time sampling are not equivalent, please indicate a time vector for the trajectory")
                sys.exit()
            else:
                print("Propagation : assuming trajectory has the same sampling frequency and time origin as mic signals")
                self.t_traj = np.arange(traj.shape[0])/self.fs
                self.Nt_traj = self.Nt
                self.fs_traj = self.fs
        else:
            self.t_traj = t_traj
            self.Nt_traj = t_traj.shape[0]
            self.fs_traj = 1/(t_traj[1] - t_traj[0])
            
        if self.debug:
            if self.angles is None:
                print('No angle considered') 
            else:
                print("Rotations considered for propagation")
        
        self.source_pos_rotated()

        self.compute_r_ms_t(parrallel=True)
        
        


    def source_pos_rotated(self):
        self.pos_t = np.zeros((self.pos.shape[0],self.Nt_traj,3)) 
            
        for s in range(self.pos.shape[0]):
            pos_tiled = np.tile(self.pos[s,:],self.traj.shape[0]).reshape((self.traj.shape[0],3)).astype(np.float64)
            if self.angles is not(None):
                pos_tiled = angle_to_pos(pos_tiled,self.angles[:,0],self.angles[:,1],self.angles[:,2])
            pos_tiled += self.traj.astype(np.float64)
            
            # Storing the position of sources along the trajectory
            self.pos_t[s,:,:] =  pos_tiled 

    def compute_r_ms_t_core(self,mm,s):
        return computeDistance(self.geom_[mm,:3],self.pos_t[s,:,:3],self.M,self.Mach_w)
        
    def compute_r_ms_t(self,geom=None,pos=None,t_traj=None,traj=None,ang=None,parrallel=True):
        if pos is None:
            pos = self.pos
        if geom is None:
            geom = self.geom 
        if t_traj is None:
            t_traj = self.t_traj 
        if traj is None:
            traj = self.traj 
        if ang is None:
            ang = self.angles 
            
        self.pos_ = pos
        self.geom_ = geom
        self.t_traj_ = t_traj        
        self.traj_ = traj        
        self.ang_ = ang
        
        t1 = time.time()   
        self.r_ms_t = np.zeros((geom.shape[0],pos.shape[0],self.Nt_traj))
        self.Mcostheta = np.zeros((geom.shape[0],pos.shape[0],self.Nt_traj))

        M = np.diff(traj,axis=0)*self.fs_traj/self.c
        M = np.append(M,np.array([M[-1,:]]), axis = 0)
        self.M = M 
            
        for s in range(pos.shape[0]):
           
            if not(parrallel):
                for m in range(geom.shape[0]):
                    r_ms_t, Mcostheta = self.compute_r_ms_t_core(m,s)
                    self.r_ms_t[m,s,:] = r_ms_t
                    self.Mcostheta[m,s,:] = Mcostheta
            else:
                tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_r_ms_t_core)(mm,s) for mm in range(geom.shape[0]))
                for m in range(geom.shape[0]):
                    self.r_ms_t[m,s,:] = tmp[m][0]
                    self.Mcostheta[m,s,:] = tmp[m][1]
        
        self.mag_ = np.multiply(self.r_ms_t,(1-self.Mcostheta)**2)
        t2 = time.time()
        if self.debug:
            print("r_ms computation took %.1f s"%(t2-t1))

    def compute_core(self,mm,ii):
        # with interpolation
        taus = self.t_traj_interp+self.tau[mm,ii,:] 
        Nt_ = self.sig[ii,:].size
        
        # No 4*pi = considering that the signal is the source power
        f = interp1d(taus[:Nt_], self.sig[ii,:]/self.mag[mm,ii,:Nt_],kind=self.interp,bounds_error=False,fill_value=0)
        p_t = f(self.t) 

        return p_t
        
    def compute(self,src=None,parrallel=True,interpolation='quadratic'):
        self.interp = interpolation
        
        if src is None:
            src = np.arange(self.pos.shape[0])        
        
        t1 = time.time()
        
        if self.Nt_traj == self.Nt: #No decimation, computation on all trajectory points
            tau = self.r_ms_t/self.c
            Nt = self.Nt
            min_tau = np.min(tau[:,:,0])
            self.tau = tau #- min_tau
            if self.timeOrigin == 'array':
                p_t = np.zeros((self.Nm,Nt))
                for ii in src:
                    for mm in range(self.Nm):
                        inds = np.arange(Nt)-(tau[mm,ii,:]*self.fs).astype(int)
                        p_t[mm,:] += self.sig[ii,inds]/(self.r_ms_t[mm,ii,:]* \
                                            (1-self.Mcostheta[mm,ii,:])**2) # No 4*pi = considering that the signal is the source power
                self.t += min_tau
            elif self.timeOrigin == 'source':

                self.p_t = np.zeros((self.Nm,Nt))
                self.mag = self.mag_
                
                try:
                    del(self.Mcostheta,self.r_ms_t,self.mag_)
                except:
                    del(self.r_ms_t,self.mag_)
                
                
                for ii in src:
                    if not(parrallel):
                        for mm in range(self.Nm):
                            self.p_t[mm,:] += self.compute_core(mm,ii)
                    else:
                        tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_core)(mm,ii) for mm in range(self.Nm))
                        for mm in range(self.Nm):
                            self.p_t[mm,:] += tmp[mm]

        else:
            self.t_traj_interp = InterpolateTimeTrajectory(self.t_traj,self.fs)
            self.Nt_traj_interp = self.t_traj_interp.size
            
            if self.timeOrigin == 'source':
                self.p_t = np.zeros((self.Nm,self.Nt))

                t2 = time.time()
                if self.debug:
                    print("matrix multiplication took %.1f s"%(t2-t1))
                
                t1 = time.time()
                
                self.tau = np.zeros((self.Nm,self.r_ms_t.shape[1],self.Nt_traj_interp))
                self.mag = np.zeros((self.Nm,self.r_ms_t.shape[1],self.Nt_traj_interp))
                for ii in src:
                    self.tau[:,ii,:], self.mag[:,ii,:] = self.interp_core(ii)                
                try:
                    del(self.Mcostheta,self.r_ms_t,self.mag_)
                except:
                    del(self.r_ms_t,self.mag_)
                
                t2 = time.time()
                if self.debug:
                    print("distance interpolation took %.1f s"%(t2-t1))
                
                t1 = time.time()
                for ii in src:
                    if not(parrallel):
                        for mm in range(self.Nm):
                            self.p_t[mm,:] += self.compute_core(mm,ii)
                    else:
                        tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(self.compute_core)(mm,ii) for mm in range(self.Nm))
                        for mm in range(self.Nm):
                            self.p_t[mm,:] += tmp[mm]
                            

            elif self.timeOrigin == 'array':
                SystemError("timeOrigin == 'array' : Not coded yet")
            
        t2 = time.time()
        if self.debug:
            print("signal computation took %.1f s"%(t2-t1))
                        
        if self.SNR is not(None):
            for mm in range(self.Nm):
                rms = np.std(self.p_t[mm,:])
                self.p_t[mm,:] += np.random.randn(self.Nt)*rms*10**(-self.SNR/20)               
        # self.p_t = p_t

    def interp_core(self,ii):
        f_tau = interp1d(self.t_traj, np.squeeze(self.r_ms_t[:,ii,:]), \
                     kind='quadratic',bounds_error=False,fill_value="extrapolate")
        f_mag = interp1d(self.t_traj, np.squeeze(self.mag_[:,ii,:]), \
                     kind='quadratic',bounds_error=False,fill_value="extrapolate")
        return f_tau(self.t_traj_interp)/self.c, f_mag(self.t_traj_interp)

        
    def plot(self):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.set_aspect('auto')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])
        TrajPnt = int(self.fs_traj/2) #2 traj points per seconds
        if hasattr(self, 'pos_t'):
            ax0.scatter(self.pos_t[:,::TrajPnt,0],self.pos_t[:,::TrajPnt,1],self.pos_t[:,::TrajPnt,2])
        else:
            ax0.scatter(self.traj[::TrajPnt,0],self.traj[::TrajPnt,1],self.traj[::TrajPnt,2])
            
            ax0.scatter(self.pos[:,0]+self.traj[0,0], \
                        self.pos[:,1]+self.traj[0,1], \
                        self.pos[:,2]+self.traj[0,2])
            ax0.set_title('Setup (mic in blue , trajectory in orange and source position at t = 0 in green)')
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        pl.tight_layout()
        

