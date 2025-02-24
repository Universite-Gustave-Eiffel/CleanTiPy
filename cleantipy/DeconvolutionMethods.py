# -*- coding: utf-8 -*-
"""
CLEAN-T core functions

Created on Mon Jun 19 2023
@author: rleiba


for gathering the requirements :
run "pipreqs ." in a terminal running in the current folder
"""

import numpy as np
import pylab as pl
import time
import sys

from .InverseMethods import Beamforming_t_traj, butter_bandpass_filter, butter_lowpass_filter
from .CommonFunctions import Doppler, InterpolateTimeTrajectory
from .Propagation import MovingSrcSimu_t
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline, interp1d, RegularGridInterpolator
from scipy import signal
import multiprocessing as mp
n_CPU_Threads = mp.cpu_count()
from simplespectral import welch, _spectral_helper
from joblib import Parallel, delayed
from pathlib import Path

class CleanT:
    """
    Class to compute CLEAN-T, time-domain variant of the CLEAN deconvolution
    algorithm
    """
    def __init__(self, geom, grid, traj, t, Sig, angles=None, t_traj=None,
                 N_iter=50,alpha=1,E_criterion=0.05,E_convergence_criterion=0.5,angleSelection=None,theta=None,\
                debug=False,fc=None,c=343,Mach_w=None,findTonal=True,monitor=False):
        """
        Parameters
        ----------
        geom : numpy array
            geometry of the array (Nmic x 3)
        grid : numpy array
            grid points (Ni x 3)
        traj : numpy array
            trajectory of the grid (Nt_traj x 3)
        t : numpy array
            time vector of the microphone signals (Nt)
        p_t : numpy array
            microphone signals (Nmic x Nt)
        angles : numpy array
            rotations around x,y and z in rad (Nt x 3)
        t_traj : numpy array
            time vector for the trajectory (Nt_traj)
        N_iter : int
            maximum number of iteration
        alpha : float
            proportion of source amplitude to be removed at each iteration
        E_criterion : float
            Stop criterion base on the ratio of remaining energy (default: 0.05 for 5%)
        E_convergence_criterion : float
            Stop criterion base on the convergence of remaining energy (default: 0.5%)
            
            If the reduction in energy between two iteration is less than *E_convergence_criterion*
            execution stops
        angleSelection : numpy array
            Limits of angle windows in degree (Nwin x 2). The CLEAN-T process 
            will be performed for each angle window
        debug : boolean
            if true displays debugging prints
        fc : float (optional)
            central frequency of analysis (interesting for monitoring, not
            needed otherwise) 
        c : float, scalar
            Speed of sound
        Mach_w : numpy array (3,)
            Mach number of the wind along x, y, z
        findTonal : boolean
            default True, il false, dont look for tonal source and only process
            sources broadband
        monitor : boolean
            if true displays monitoring prints and graphs
        """
        self.c = c #m/s
        self.p_ref = 2e-5 #Pascal
        self.geom = geom
        self.grid = grid
        self.t = t
        self.Sig = Sig # (Nmic, time) data
        self.Nt = t.shape[0]
        self.Ni = grid.shape[0]
        self.BF_t = np.zeros((self.Ni,self.Nt))
        self.Nm = geom.shape[0]
        self.fs = 1/(t[1] - t[0])
        self.traj = traj
        self.internVar_dtype=np.float64
        self.angles = angles
        self.N_iter = N_iter
        self.alpha = alpha
        self.E_criterion = E_criterion
        self.E_convergence_criterion = E_convergence_criterion
        self.debug = debug
        self.angleSelection = angleSelection
        self.theta = theta
        self.nx = np.unique(self.grid[:,0]).size
        self.ny = np.max([np.unique(self.grid[:,1]).size,np.unique(self.grid[:,2]).size])
        self.fc = fc
        self.Mach_w = Mach_w
        self.findTonal = findTonal
        self.monitor = monitor

        
        if t_traj is None:
            # Checking trajectory dimentions with regards to time vector
            if self.Nt != self.traj.shape[0]:
                print("[Clean-T init] Trajectory and time sampling are not equivalent, please indicate a time vector for the trajectory")
                sys.exit()
            else:
                print("Beamforming : assuming trajectory has the same sampling frequency and time origin as mic signals")
                self.t_traj = np.arange(traj.shape[0])/self.fs
                self.Nt_traj = self.Nt
                self.fs_traj = self.fs
        else:
            self.t_traj = t_traj
            self.Nt_traj = t_traj.shape[0]
            self.fs_traj = 1/(t_traj[1] - t_traj[0])
            
        if self.angleSelection is not None :
            self.computeAngleWindows_traj()
            
            # Limit computation to trajectory in the angular selection, adding -2 and + 2
            inds, = np.where(np.sum(self.TrajMask,axis=0)!=0)
            Inds = inds.tolist()
            Inds.insert(0,Inds[0]-1)
            Inds.insert(0,Inds[0]-1) 
            Inds.append(Inds[-1]+1)
            Inds.append(Inds[-1]+1)
            ind_total_masks = np.array(Inds)
            self.t_traj = self.t_traj[ind_total_masks]
            self.traj = self.traj[ind_total_masks,:]
            self.Nt_traj = len(ind_total_masks)
            self.angles = self.angles[ind_total_masks,:]
            
            self.t_traj_interp = InterpolateTimeTrajectory(self.t_traj,self.fs)
            self.Nt_interp = self.t_traj_interp.size

            self.computeAngleWindows()
        else:
            self.TemporalMask = np.ones((1,self.Nt))
            # self.ind_total_masks = np.arange(self.Nt,dtype=np.int64)
            self.indsMask,= np.where(self.TemporalMask[0,:]!=0)
        self.ind_total_masks, = np.where(np.sum(self.TemporalMask,axis=0)!=0)
        

        
        self.set_envs()
        
    def set_envs(self):
        """
        Sets the propagation and beaforming environements that will be used 
        during the iterative process

        """

        # Define propagation object
        self.propa = MovingSrcSimu_t(self.geom, np.array([self.grid[0,:]]),\
                    self.traj, self.t, self.Sig, self.t_traj, self.angles,\
                        timeOrigin='source', debug=self.debug,\
                        c=self.c,Mach_w=self.Mach_w)
        if self.debug:
            print('Propagation object initiated')
            
        # Define beaforming object
        self.bf = Beamforming_t_traj(self.geom,self.grid,self.traj,self.t,\
                                     self.Sig,self.angles, self.t_traj,\
                                     QuantitativeComputation=False,\
                                     debug=self.debug, internVar_dtype=self.internVar_dtype,\
                                     c=self.c,Mach_w=self.Mach_w)
        if self.debug:
            print('Beamforming object initiated')
        
        self.bf_single_pt = Beamforming_t_traj(self.geom,\
                        np.array([self.grid[self.Ni//2,:]]), self.traj,\
                        self.t,\
                        self.bf.p_t,self.angles, self.t_traj, debug=self.debug,\
                        QuantitativeComputation=True,\
                        internVar_dtype=self.internVar_dtype,\
                        c=self.c,Mach_w=self.Mach_w)
        if self.debug:
            print('Single point beamforming object initiated')
       
    
    def find_max(self,dB_,aa=None,parrallel=True):
        """
        find_max take the beamformed signals over the grid in order to localise
        the main source of the grid
                
        Returns
        -------
        i_max: int
            index of the source (assumed to be the most energetic bin of the 
            beamforming)
        src_type: int
            type of source: 0 (broadband) or 1 (tonal)

        """
        Nt_ = np.min([len(self.indsMask), self.bf.BF_t.shape[1]])

        if aa is not None:
            Mask = self.TemporalMask[aa,self.indsMask[:Nt_]]
        else:
            Mask = np.ones((self.Nt,))
        
        i_bb = np.argmax(dB_)
        
        if self.findTonal:
            if self.fc is not None:
                nfft = 2**int(np.ceil(np.log2(256*(1+np.log10(self.bf.fs/1500)))))
                N_av = nfft//10
                freq_resolution = 1024//nfft
            else:
                nfft = 2**int(np.ceil(np.log2(1024*(1+np.log10(self.bf.fs/1500)))))
                N_av = nfft//40                
                freq_resolution = 4096//nfft              
    
    
            t1 = time.time()
            spec = np.zeros((self.bf.Ni,nfft//2))
            Spec = np.zeros((self.bf.Ni,nfft//2))
            bgNoise = np.zeros((self.bf.Ni,nfft//2))
            if parrallel:
                tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(compute_spectrum_core)(self.bf.BF_t[ii,self.indsMask[:Nt_]],Mask,N_av,nfft,self.bf.fs) for ii in range(self.bf.Ni))
                for ii in range(self.bf.Ni):
                    f = tmp[ii][0]
                    spec[ii,N_av//2:nfft//2-N_av//2] = tmp[ii][1][N_av//2:nfft//2-N_av//2]
                    bgNoise[ii,N_av//2:nfft//2-N_av//2] = tmp[ii][2][N_av//2:nfft//2-N_av//2]
            else:
                for ii in range(self.bf.Ni):
                    tmp = compute_spectrum_core(self.bf.BF_t[ii,:self.indsMask[:Nt_]],Mask[:Nt_],N_av,nfft,self.bf.fs)#[N_av//2:nfft//2-N_av//2]
                    f = tmp[0]
                    spec[ii,N_av//2:nfft//2-N_av//2] = tmp[1][N_av//2:nfft//2-N_av//2]
                    bgNoise[ii,N_av//2:nfft//2-N_av//2] = tmp[2][N_av//2:nfft//2-N_av//2]
            Spec[:,N_av//2:nfft//2-N_av//2] = spec[:,N_av//2:nfft//2-N_av//2]/bgNoise[:,N_av//2:nfft//2-N_av//2] #removing background noise 
            
            if self.debug:
                t2 = time.time()
                print("Spectrum computation took %.1f s"%(t2-t1))
            
            ind_fmax = np.argmax(Spec,axis=-1)
            tonalIndicator = np.zeros((len(ind_fmax),))
            
            for ii in range(len(ind_fmax)):
                tonalIndicator[ii] = np.mean(spec[ii,ind_fmax[ii]-freq_resolution:ind_fmax[ii]+freq_resolution]**2)
            
            i_ton = np.argmax(tonalIndicator)
            self.tonal_crit = np.max(Spec[i_ton,:])
        
            self.MaxTonalMap = tonalIndicator.reshape((self.ny,self.nx))

            tonal_threshold = 20*(np.floor(np.log10(self.fs_traj))+1)
            # if self.fc is None:
            #     tonal_threshold *= self.fs

            if self.monitor:
                gs_kw = dict(width_ratios=[1.4, 1.4,1], height_ratios=[1, 2])
                fig = pl.figure(num="Monitor", figsize=(12, 4.5))
                fig.clf()
                fig, axd = pl.subplot_mosaic([['left', 'center', 'upper right'],
                                               ['left', 'center', 'lower right']],
                                              gridspec_kw=gs_kw, figsize=(12, 4.5),
                                              layout="constrained",num="Monitor")
                fig.tight_layout()
                # pl.semilogy(f[N_av//2:nfft//2],spec[N_av//2:nfft//2])
                # pl.plot(f[:nfft//2-N_av//2],bgNoise[:nfft//2-N_av//2])
                axd['upper right'].semilogy(f[:nfft//2],spec[i_ton,:nfft//2])
                axd['upper right'].plot(f[N_av//2:nfft//2-N_av//2],bgNoise[i_ton,N_av//2:nfft//2-N_av//2])
                axd['lower right'].semilogy(f[:nfft//2],Spec[i_ton,:])
                axd['lower right'].plot([f[0], f[nfft//2-1]],[np.median(Spec[i_bb,:]), np.median(Spec[i_bb,:])])
                
                axd['center'].imshow(dB_.reshape((self.ny,self.nx)),\
                        origin='lower',cmap='hot_r',\
                            extent=[self.grid[0,0],self.grid[-1,0],self.grid[0,1],self.grid[-1,1]],\
                                vmax=self.debug_max_disp,vmin=self.debug_max_disp-30)
                axd['center'].scatter(self.grid[i_ton,0],self.grid[i_ton,1])
                axd['center'].scatter(self.grid[i_bb,0],self.grid[i_bb,1])
                axd['center'].legend(['Max Tonal','MAX Broadband'],loc='lower center',ncol=2)
                axd['left'].imshow(self.MaxTonalMap,\
                        origin='lower',cmap='hot_r',\
                            extent=[self.grid[0,0],self.grid[-1,0],self.grid[0,1],self.grid[-1,1]])
                fig.suptitle(self.tonal_crit)
                directory = "monitor"
                Path(directory).mkdir(parents=True, exist_ok=True)
                if aa is None:
                    if self.fc is None:
                        fig.savefig(directory+"/FullSpectrum_noWin_%d" %(len(self.E)),transparent=True)
                    else:
                        fig.savefig(directory+"/%dHz_noWin_%d" %(self.fc,len(self.E)),transparent=True)
                else:
                    if self.fc is None:
                        fig.savefig(directory+"/FullSpectrum_Win%d_%d" %(aa,len(self.E[aa])),transparent=True)
                    else:
                        fig.savefig(directory+"/%dHz_Win%d_%d" %(self.fc,aa,len(self.E[aa])),transparent=True)
                pl.pause(0.01)
                
            if self.debug:
                print(np.max(Spec[i_ton,:]),np.median(Spec[i_ton,:]),self.tonal_crit,i_ton,i_bb)

            if self.tonal_crit < tonal_threshold:
                i_max = i_bb
                src_type = 0
                return i_max, (src_type,)
            else:
                i_max = i_ton
                src_type = 1
                ind_f_ton = int(ind_fmax[i_max])
                f_ton = f[ind_f_ton]
                return i_max, (src_type, f_ton)
        else:
            i_max = i_bb
            src_type = 0
            return i_max, (src_type,)





    def computeAngleWindows(self):
        """
        compute self.TemporalMask: a  smooth mask in the interpolated 
        (microphone signals sample frequency) trajectory time scale for each 
        angular window
        """            
        self.TemporalMask = np.zeros((len(self.angleSelection),self.Nt_interp))
        TrajMask = np.zeros((len(self.angleSelection),self.Nt_interp))
        
        geom_center = np.mean(self.geom,axis=0)
        centeredTraj = self.traj[:,:]-geom_center
        
        # finding the axis in which the difference is greater
        main_direction = np.argmax(np.abs(centeredTraj[-1,:]-centeredTraj[0,:]))
        
        # finding the axis orthogonal to the array
        orth_direction = np.argmin([np.std(self.geom[:,ii]) for ii in range(3)])
        
        
        # Interpolate the trajectory        
        interpolator_x = RegularGridInterpolator((self.t_traj,), centeredTraj[:, 0], method='linear',bounds_error=False,fill_value=None)
        interpolator_y = RegularGridInterpolator((self.t_traj,), centeredTraj[:, 1], method='linear',bounds_error=False,fill_value=None)
        interpolator_z = RegularGridInterpolator((self.t_traj,), centeredTraj[:, 2], method='linear',bounds_error=False,fill_value=None)

        t_new = self.t_traj_interp
        centeredTraj_interp = np.array([interpolator_x(t_new),interpolator_y(t_new),interpolator_z(t_new)]).T
        

        for aa in range(len(self.angleSelection)):
            limits = self.angleSelection[aa,:]/180*np.pi
            if np.diff(np.abs(limits)) == 0 or np.sign(limits[0])==-1 and np.sign(limits[1])==-1:
                i_max = np.argmax(limits)
                i_min = np.argmin(limits)
            else:
                i_max = np.argmax(np.abs(limits))
                i_min = np.argmin(np.abs(limits))

            for tt in range(len(self.t_traj_interp)):
                xyz_min = np.tan(limits[i_min])*centeredTraj_interp[tt,orth_direction]
                xyz_max = np.tan(limits[i_max])*centeredTraj_interp[tt,orth_direction]
                
                #to take into account the direction of the trajectory (left to right or right to left)
                xyz_min *= np.sign(np.mean(np.diff(self.traj[:,main_direction])))
                xyz_max *= np.sign(np.mean(np.diff(self.traj[:,main_direction])))
                
                if centeredTraj_interp[tt,main_direction] <= xyz_max and centeredTraj_interp[tt,main_direction] >= xyz_min :
                    TrajMask[aa,tt] = 1  # binary mask : 0 or 1
            
            
            # Number of temporal points for smoothing the angular window
            smoothing_pt = np.floor(self.fs/25)*2+1 # needs to be odd and to vary with fs
            
            # Triangular window
            win = signal.windows.triang(smoothing_pt)\
                /np.sum(signal.windows.triang(smoothing_pt))
            
            #Smoothing the masks
            self.TemporalMask[aa,:] = np.convolve(TrajMask[aa,:], \
                                                  win, mode='same')
            
        self.theta = np.arctan(centeredTraj_interp[:,main_direction]/centeredTraj_interp[:,2])*180/np.pi


    def computeAngleWindows_traj(self):
        """
        compute self.TrajMask: a binary mask in the trajectory time scale for 
        each angular window:
            
        0: trajectory point not in angular selection
        
        1: trajectory point in angular selection

        """
        self.TrajMask = np.zeros((len(self.angleSelection),len(self.traj)))
        geom_center = np.mean(self.geom,axis=0)
        centeredTraj = self.traj[:,:]-geom_center
        
        # finding the axis in which the difference is greater
        main_direction = np.argmax(np.abs(centeredTraj[-1,:]-centeredTraj[0,:]))
        
        # finding the axis orthogonal to the array
        orth_direction = np.argmin([np.std(self.geom[:,ii]) for ii in range(3)])
        
        for aa in range(len(self.angleSelection)):
            limits = self.angleSelection[aa,:]/180*np.pi
            if np.diff(np.abs(limits)) == 0 or np.sign(limits[0])==-1 and np.sign(limits[1])==-1:
                i_max = np.argmax(limits)
                i_min = np.argmin(limits)
            else:
                i_max = np.argmax(np.abs(limits))
                i_min = np.argmin(np.abs(limits))

            for tt in range(len(self.traj)):
                xyz_min = np.tan(limits[i_min])*centeredTraj[tt,orth_direction]
                xyz_max = np.tan(limits[i_max])*centeredTraj[tt,orth_direction]
                
                #to take into account the direction of the trajectory (left to right or right to left)
                xyz_min *= np.sign(np.mean(np.diff(self.traj[:,main_direction])))
                xyz_max *= np.sign(np.mean(np.diff(self.traj[:,main_direction])))
                
                if centeredTraj[tt,main_direction] <= xyz_max and centeredTraj[tt,main_direction] >= xyz_min :
                    self.TrajMask[aa,tt] = 1  # binary mask : 0 or 1


                
    def compute(self,parrallel=True,QuantFirstIter=False):
        self.Sources = list()
        self.E = list() #Energy for each iteration
        
        
        if self.angleSelection is not None :
            for aa in range(len(self.angleSelection)):
                # Set environements for each angular window
                self.Sources.append(list())
                self.E.append(list())

                self.compute_core_windowed(parrallel,QuantFirstIter,aa)
        else:
            self.compute_core(parrallel,QuantFirstIter)

    def compute_core_windowed(self,parrallel,QuantFirstIter,aa):
        print("****** Angular window: %.1f-%.1f° ******" %(self.angleSelection[aa,0],self.angleSelection[aa,1]))
        # Mic_E = list()
        
        # # Propagate the temporal (angular) mask for grid center
        # TemporalMaskPropagated = np.zeros((self.Nm,self.Nt))
        # f_tau = interp1d(self.t_traj, \
        #                   (np.squeeze(self.bf.r_mi_t[:,self.Ni//2,:]))/self.c+self.t_traj, \
        #               kind='linear',bounds_error=False,fill_value="extrapolate")
        # tau = f_tau(self.t-self.t[0]) #interp in source-related coordonate
        # for mm in range (self.Nm):
        #     f_propaWinAng = interp1d(tau[mm,:], self.TemporalMask[aa,:], \
        #                   kind='linear',bounds_error=False,fill_value="extrapolate")
        #     TemporalMaskPropagated[mm,:] = f_propaWinAng(self.t-self.t[0])
        
        # Mic_E.append(np.mean(np.mean((self.bf.p_t*TemporalMaskPropagated)**2)))
        
        t_rec = self.t-self.t[0]
        # self.indsMask,= np.where(np.bitwise_and(t_rec>self.t_traj[0], t_rec<self.t_traj[-1]))
        self.indsMask,= np.where(self.TemporalMask[aa,:]!=0)

        for nn in range(int(self.N_iter//self.alpha)):
            # Compute the beaforming on the grid
            if QuantFirstIter and nn==0:
                # if first BF map needs actual noise levels
                self.bf.QuantitativeComputation=True
                self.bf.compute(parrallel=parrallel, interpolation='linear')
                self.bf.QuantitativeComputation=False
            else:
                self.bf.compute(parrallel=parrallel, interpolation='linear')

                
            # Compute the acoustic map (over the angular window)
            p_eff = np.std(self.bf.BF_t[:,self.indsMask]*self.TemporalMask[aa,self.indsMask],axis=-1)
            dB_ = 20*np.log10(p_eff/self.p_ref)
            
            
            if nn == 0:
                self.debug_max_disp = np.max(dB_)
            
            self.E[aa].append(np.sum(p_eff**2))
            
            print("%d - Residual energy: %.1f%%" %(nn, self.E[aa][-1]*100/self.E[aa][0]))
            if nn>3 :
                convergenceRate = np.abs(moving_average(np.diff(self.E[aa][:]/self.E[aa][0]*100),3))[-1]
                if self.debug:
                    print("%d - Convergence Rate: %.2f%%" %(nn, convergenceRate))
           
            if self.E[aa][-1]/self.E[aa][0]<=self.E_criterion:
                print("Residual energy inferior to stop criterion : %d%%" %(self.E_criterion*100))
                break
            if nn>3 \
                and convergenceRate <self.E_convergence_criterion \
                and self.E[aa][-1]/self.E[aa][0]<=(self.E_criterion*3):
                print("Residual energy inferior to %d%% and decreasing by less than %.1f%% : stoping computation"\
                      %(self.E_criterion*3*100,self.E_convergence_criterion))
                break                
            
            # find the source and its position
            t1 = time.time()
            i_max, src_type = self.find_max(dB_,aa,parrallel=True)
            if self.debug:
                t2 = time.time()
                print("Finding source position took %.1f s"%(t2-t1))


            if len(src_type)==2:
                f_ton = src_type[1]
            src_type = src_type[0]

            # Compute BF again properly on single point : source location
            self.bf_single_pt.r_mi_t[:,0,:] = self.bf.r_mi_t[:,i_max,:].copy()
            self.bf_single_pt.mag[:,0,:] = self.bf.mag[:,i_max,:].copy()
            self.bf_single_pt.p_t = self.bf.p_t.copy()
            
            self.bf_single_pt.compute(parrallel=parrallel, interpolation='quadratic')
            
            # define the signal to propagate and store it
            self.propa.pos = np.array([self.grid[i_max,:]]) 
            del(self.propa.sig)
            if src_type:
                lowcut = f_ton - 30
                if lowcut <= 0:
                    lowcut = 0.5
                highcut = f_ton + 30
                if self.debug:
                    print("Tonal source: Filtering between %d and %d Hz - Position : %.1f, %.1f, %.1f"\
                          %(lowcut,highcut,self.grid[i_max,0],self.grid[i_max,1],self.grid[i_max,2]))
                filteredSig = np.array([butter_bandpass_filter(self.bf_single_pt.BF_t[0,:], \
                                                lowcut, highcut, self.fs, order=3)]) 
                self.propa.sig = filteredSig*self.TemporalMask[aa,:]
            else:
                if self.debug:
                    print("Broadband source: no filtering - Position : %.1f, %.1f, %.1f"\
                          %(self.grid[i_max,0],self.grid[i_max,1],self.grid[i_max,2]))
                self.propa.sig = np.array([self.bf_single_pt.BF_t[0,:]])*self.TemporalMask[aa,:]
            
            if self.debug:
                pl.figure(num='Windowing check')
                pl.clf()
                pl.plot(self.bf.t_traj_interp,self.bf.BF_t[i_max,:])
                pl.plot(self.bf.t_traj_interp[self.indsMask],self.bf.BF_t[i_max,self.indsMask])
                pl.plot(self.bf.t_traj_interp[self.indsMask],self.bf.BF_t[i_max,self.indsMask]*self.TemporalMask[aa,self.indsMask])
            
                # verifying results after recomputing the beamforming with quadratic interpolation
                pl.plot(self.bf_single_pt.t_traj_interp[self.indsMask],\
                        self.bf_single_pt.BF_t[0,self.indsMask]*self.TemporalMask[aa,self.indsMask])
                pl.legend(['Signal of selected source - linear interpolation',\
                           'Portion corresponding to angular window - linear interpolation',\
                           'portion with windowing - linear interpolation',\
                           'portion with windowing - quadratic interpolation'])
             
            
            self.propa.r_ms_t = self.bf_single_pt.r_mi_t.copy()
            self.propa.mag_ = self.bf_single_pt.mag.copy()
            self.propa.pos_t = self.bf_single_pt.grid_t.copy()
            # self.propa.source_pos_rotated()
            
            
            
            # Propagate the signal for the identified source
            self.propa.compute(src=[0],parrallel=parrallel,interpolation="quadratic")
            
            # Propagate the temporal (angular) masks
            TemporalMaskPropagated = np.zeros((self.Nm,self.Nt))
            tau = self.propa.tau[:,0,:]+self.propa.t_traj_interp

            for mm in range (self.Nm):
                f_propaWinAng = interp1d(tau[mm,:], self.TemporalMask[aa,:], \
                              kind='linear',bounds_error=False,fill_value="extrapolate")
                TemporalMaskPropagated[mm,:] = f_propaWinAng(self.t)

                ind_prop_masks,= np.where(TemporalMaskPropagated[mm,:]>0)
                nt = np.min([len(ind_prop_masks), self.propa.p_t[mm,:].size])
                if self.debug and mm==0:
                    pl.figure(num='Mask Propagation check')
                    pl.clf()
                    pl.plot(self.t_traj_interp,self.TemporalMask[aa,:])
                    pl.plot(self.t,TemporalMaskPropagated[mm,:])
                    pl.legend(['source related mask','Propagated mask for mic 0'])


                
                # Substrating the propagated signals into the microphonic signals
                if self.debug and mm==0:
                    pl.figure(num='Propagation check')
                    pl.clf()
                    # pl.plot(self.bf.p_t[mm,ind_prop_masks[:nt]])
                    # pl.plot(self.propa.p_t[mm,:nt])
                    pl.plot(self.bf.t,self.bf.p_t[mm,:])
                    pl.plot(self.propa.t_traj_interp,self.propa.sig[0]/self.propa.sig[0].max()*self.bf.p_t[mm,:].max())
                    pl.plot(self.t_traj_interp,self.TemporalMask[aa,:]*self.bf.p_t[mm,:].max())
                    pl.plot(self.propa.t,self.propa.p_t[mm,:])
                    pl.plot(self.t,TemporalMaskPropagated[mm,:]*self.bf.p_t[mm,:].max())

                    pl.legend(['initial microphone signal','source signal to propagate',\
                               'source related mask','Propagated source to microphone',\
                                   'Propagated mask for mic 0'])

                    pl.figure(num='Propagation check - time selection')
                    pl.clf()
                    pl.plot(self.bf.t[ind_prop_masks[:nt]],self.bf.p_t[mm,ind_prop_masks[:nt]])
                    pl.plot(self.propa.t[ind_prop_masks[:nt]],self.propa.p_t[mm,ind_prop_masks[:nt]])    
                    print(20*np.log10(np.std(self.bf.p_t[mm,:])/2e-5))
                    print(20*np.log10(np.std(self.bf.p_t[mm,:] - self.propa.p_t[mm,:] * self.alpha )/2e-5))

                    print(20*np.log10(np.std(self.bf.p_t[mm,ind_prop_masks[:nt]])/2e-5))
                    print(20*np.log10(np.std(self.bf.p_t[mm,ind_prop_masks[:nt]] - self.propa.p_t[mm,ind_prop_masks[:nt]] * self.alpha )/2e-5))


                self.bf.p_t[mm,ind_prop_masks[:nt]] -= self.propa.p_t[mm,ind_prop_masks[:nt]] * self.alpha   
                

            
            # Mic_E.append(np.mean(np.mean((self.bf.p_t*TemporalMaskPropagated)**2)))
            # print("Mic energy remaining : %.3f%%" %(Mic_E[-1]*100/Mic_E[0]))
            
            # Store the data
            if self.findTonal:
                self.Sources[aa].append({'SourceSignal':self.propa.sig.T, \
                                    'SourceIndex':i_max, \
                                    'SourcePosition':self.propa.pos,\
                                    'AcousticMap':dB_,\
                                    'Type':src_type,\
                                    'SourceRelatedTimeVector':self.propa.t_traj_interp,\
                                    'MaxTonalMap':self.MaxTonalMap,\
                                    'TonalCriterion':self.tonal_crit,\
                                    'RemainingEnergy':self.E[aa][-1]/self.E[aa][0],\
                                    'TemporalMask':self.TemporalMask[aa,:]})
            else:        
                self.Sources[aa].append({'SourceSignal':self.propa.sig.T, \
                                   'SourceIndex':i_max, \
                                   'SourcePosition':self.propa.pos,\
                                   'AcousticMap':dB_,\
                                   'Type':src_type,\
                                   'SourceRelatedTimeVector':self.propa.t_traj_interp,\
                                   'RemainingEnergy':self.E[aa][-1]/self.E[aa][0],\
                                   'TemporalMask':self.TemporalMask[aa,:]})

    def compute_core(self,parrallel,QuantFirstIter):
        
        for nn in range(int(self.N_iter//self.alpha)):
            # Compute the beaforming on the grid
            if QuantFirstIter:
                # if first BF map needs actual noise levels
                self.bf.QuantitativeComputation=True
                self.bf.compute(parrallel=parrallel, interpolation='linear')
                self.bf.QuantitativeComputation=False
            else:
                self.bf.compute(parrallel=parrallel, interpolation='linear')
                
            # Compute the acoustic map (over the angular window)
            p_eff = np.std(self.bf.BF_t[:,self.indsMask]*self.TemporalMask[0,self.indsMask],axis=-1)
            dB_ = 20*np.log10(p_eff/self.p_ref)
            
            if nn == 0:
                self.debug_max_disp = np.max(dB_)
            
            self.E.append(np.sum(p_eff**2))
            
            print("%d - Residual energy: %.1f%%" %(nn, self.E[-1]*100/self.E[0]))
            
            if self.E[-1]/self.E[0]<=self.E_criterion:
                print("Residual energy inferior to stop criterion : %d%%" %(self.E_criterion*100))
                break
            
            # find the source and its position
            i_max, src_type = self.find_max(dB_)
            if len(src_type)==2:
                f_ton = src_type[1]
            src_type = src_type[0]
                

            
            # Compute BF again properly on single point : source location
            self.bf_single_pt.compute_r_mi_t(grid=np.array([self.grid[i_max,:]]))
            self.bf_single_pt.p_t = self.bf.p_t.copy()
            
            self.bf_single_pt.compute(parrallel=parrallel, interpolation='quadratic')
            
            # define the signal to propagate and store it
            self.propa.pos = np.array([self.grid[i_max,:]])   
            if src_type:
                lowcut = f_ton - 20 # Ask quentin what parameter he uses
                highcut = f_ton + 20
                if self.debug:
                    print("Tonal source: Filtering between %d and %d Hz - Position : %.1f, %.1f, %.1f"\
                          %(lowcut,highcut,self.grid[i_max,0],self.grid[i_max,1],self.grid[i_max,2]))
                filteredSig = butter_bandpass_filter(self.bf_single_pt.BF_t[0,:], \
                                                lowcut, highcut, self.fs, order=3)
                self.propa.sig = np.array([filteredSig])
            else:
                if self.debug:
                    print("Broadband source: no filtering - Position : %.1f, %.1f, %.1f"\
                          %(self.grid[i_max,0],self.grid[i_max,1],self.grid[i_max,2]))
                self.propa.sig = np.array([self.bf_single_pt.BF_t[0,:]])
            
            self.propa.r_ms_t = self.bf_single_pt.r_mi_t
            self.propa.mag_ = self.bf_single_pt.mag
            self.propa.pos_t = self.bf_single_pt.grid_t
            # self.propa.source_pos_rotated()
            # del self.bf_single_pt
            
            
            # Propagate the signal of the identified source
            self.propa.compute(src=[0],parrallel=parrallel,interpolation="quadratic")
            
            # Substrating the propagated signals into the microphonic signals
            self.bf.p_t -= self.propa.p_t*self.alpha
            
            # Store the data
            if self.findTonal:
                self.Sources.append({'SourceSignal':self.propa.sig.T, \
                                    'SourceIndex':i_max, \
                                    'SourcePosition':self.propa.pos,\
                                    'AcousticMap':dB_,\
                                    'Type':src_type,\
                                    'SourceRelatedTimeVector':self.propa.t_traj_interp,\
                                    'MaxTonalMap':self.MaxTonalMap,\
                                    'TonalCriterion':self.tonal_crit,\
                                    'RemainingEnergy':self.E[-1]/self.E[0]})
            else:        
                self.Sources.append({'SourceSignal':self.propa.sig.T, \
                                   'SourceIndex':i_max, \
                                   'SourcePosition':self.propa.pos,\
                                   'AcousticMap':dB_,\
                                   'Type':src_type,\
                                   'SourceRelatedTimeVector':self.propa.t_traj_interp,\
                                   'RemainingEnergy':self.E[-1]/self.E[0]})

            
    def CleantMap(self,gauss=True,dyn=30,sameDynRange=True,adym=False,sig=0.5):
        CleantMap(self,gauss=gauss,dyn=dyn,sameDynRange=sameDynRange,adym=adym,sig=sig)
          
                
    def printSourceData(self):
        Types = ['Broadband','Tonal    ']
        if self.angleSelection is None :
            for i, source in enumerate(self.Sources):
                level = 20*np.log10(np.std(source['SourceSignal'])/self.p_ref)
                
                print('%s - %.1f dB - Pos.: x:%2.1f\ty:%2.1f\tz:%2.1f (rel. traj.) - E: %.1f%%'\
                      %(Types[source['Type']],level,\
                        source['SourcePosition'][0][0],\
                          source['SourcePosition'][0][1],\
                            source['SourcePosition'][0][2],\
                                source['RemainingEnergy']*100))
            
        else:
            for ww in range(len(self.Sources)):
                print("****** Angular window: %.1f-%.1f° ******" %(self.angleSelection[ww,0],self.angleSelection[ww,1]))
                for i, source in enumerate(self.Sources[ww]):
                    level = 20*np.log10(np.std(source['SourceSignal'])/self.p_ref)
                        
                    print('%s - %.1f dB - Pos.: x:%2.1f\ty:%2.1f\tz:%2.1f (rel. traj.) - E: %.1f%%'\
                          %(Types[source['Type']],level,\
                            source['SourcePosition'][0][0],\
                              source['SourcePosition'][0][1],\
                                source['SourcePosition'][0][2],\
                                    source['RemainingEnergy']*100))

    def plot(self):
        fig=pl.figure(figsize=(8,5))
        ax0 = fig.add_subplot(projection='3d')
        ax0.set_aspect('auto')
        ax0.scatter(self.geom[:,0],self.geom[:,1],self.geom[:,2])
        TrajPnt = int(self.fs_traj/2) #2 traj points per seconds
        
        ax0.scatter(self.traj[::TrajPnt,0],self.traj[::TrajPnt,1],self.traj[::TrajPnt,2])
        ax0.set_title('Setup (mic in blue , trajectory in orange)')
        
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_zlabel('z')
        pl.tight_layout()


class MultiFreqCleanT:
    """
    Class to compute CLEAN-T, time-domain variant of the CLEAN deconvolution
    algorithm
    """
    def __init__(self, geom, grid, traj, t, Sig, angles=None, t_traj=None, N_iter=50, alpha=1,\
                  E_criterion=0.05,E_convergence_criterion=0.5,debug=False,fc=[500,1000],bandtype='octave',\
                  isMicActive=None,angleSelection=None,c=343,Mach_w=None,findTonal=True,monitor=False):
        """
        Parameters
        ----------
        geom : numpy array
            geometry of the array (Nmic x 3)
        grid : numpy array
            grid points (Ni x 3)
        traj : numpy array
            trajectory of the grid (Nt_traj x 3)
        t : numpy array
            time vector (Nt)
        p_t : numpy array
            microphone signals (Nmic x Nt)
        angles : numpy array
            rotations around x,y and z in rad (Nt x 3)
        t_traj : numpy array
            time vector for the trajectory (Nt_traj)
        N_iter : int
            maximum number of iteration
        alpha : float
            proportion of source amplitude to be removed at each iteration
        E_criterion : float
            Stop criterion base on the ratio of remaining energy (default: 5%)
        E_convergence_criterion : float
            Stop criterion base on the convergence of remaining energy (default: 0.5%)
            
            If the reduction in energy between two iteration is less than *E_convergence_criterion*
            execution stops
        debug : boolean
            if true displays debugging prints
        fc : list
            central frequencies of frequency band
        bandtype : string
            type of frequency band: can be 'octave' or 'thirdoctave'
        isMicActive : numpy array
            indicates the activation of microphones for each frequency 
            (Nmic x len(fc)), default: all microphones are active
        angleSelection : numpy array
            Limits of angle windows (Nwin x 2). The CLEAN-T process will be 
            performed for each angle window
        c : float
            Speed of sound
        Mach_w : numpy array (3,)
            Mach number of the wind along x, y, z
        findTonal : boolean
            default True, if false, dont look for tonal source and only process
            sources broadband
        monitor : boolean
            if true displays monitoring prints and graphs
        """
        self.c = c #m/s
        self.p_ref = 2e-5 #Pascal
        self.geom = geom
        self.grid = grid
        self.t = t
        self.Sig = Sig # (Nmic, time) data
        self.Nt = t.shape[0]
        self.Ni = grid.shape[0]
        # self.BF_t = np.zeros((self.Ni,self.Nt))
        self.Nm = geom.shape[0]
        self.fs = 1/(t[1] - t[0])
        self.traj = traj
        self.internVar_dtype=np.float64
        self.angles = angles
        self.N_iter = N_iter
        self.alpha = alpha
        self.E_criterion = E_criterion
        self.E_convergence_criterion = E_convergence_criterion
        self.debug = debug
        self.fc = fc
        self.bandtype = bandtype
        self.angleSelection = angleSelection
        self.Mach_w = Mach_w
        self.findTonal = findTonal
        self.monitor = monitor
        if t_traj is None:
            # Checking trajectory dimentions with regards to time vector
            if self.Nt != self.traj.shape[0]:
                print("[Multifreq Clean-T init] Trajectory and time sampling are not equivalent, please indicate a time vector for the trajectory")
                sys.exit()
            else:
                print("Beamforming : assuming trajectory has the same sampling frequency and time origin as mic signals")
                self.t_traj = np.arange(traj.shape[0])/self.fs
        else:
            self.t_traj = t_traj
        
        if bandtype not in ['thirdoctave','octave']:
            sys.exit("bandtype must be 'thirdoctave' or 'octave'")
        
        if isMicActive is None:
            self.isMicActive = np.ones((geom.shape[0],len(fc)),dtype=int)
        else:
            self.isMicActive = isMicActive
        
        if self.angles is not None:
            AnglesSaved = self.angles
        else:
            AnglesSaved = np.zeros_like(self.traj)
            
        self.Header = {"Grid":self.grid,"AngularWindow":self.angleSelection,\
                       "CentralFrequencies":self.fc,"LoopFactor":self.alpha,\
                    "Trajectory":self.traj,"GridInclinationAngles_rad":AnglesSaved,\
                    "SpeedOfSound":self.c,"MicrophonePositions":self.geom,\
                    "ActivatedMicrophones":self.isMicActive,\
                    "MicrophonesTimeVector":self.t,"TrajectoryTimeVector":self.t_traj}

                
    def ComputeCleanT(self,dyn=30,parrallel=False):
        """
        Compute CLEAN-T for each frequency band

        """
        self.decimFactor = np.zeros((len(self.fc),),dtype=int)
        
        pos = np.array([np.mean(self.grid,axis=0)])
        
        self.Results = list()
        self.CleanTObjects = list()
        
        for ff,fc in enumerate(self.fc):
            
            print('\n')
            print("** Starting CLEAN-T computation over the %d Hz %s band **" %(fc, self.bandtype))

            t1 = time.time()
            
            if self.bandtype == 'octave':
                lowcut = fc/(2**.5)
                highcut = fc*(2**.5)
            elif self.bandtype == 'thirdoctave': 
                lowcut = fc/(2**(1/6))
                highcut = fc*(2**(1/6))
            ind_activeMics, = np.where(self.isMicActive[:,ff])    
            geom = self.geom[ind_activeMics,:]
            sig = self.Sig[ind_activeMics,:]
            
            # Beamforming object 
            bf = Beamforming_t_traj(geom, pos, self.traj, self.t, sig,\
                                    self.angles,self.t_traj,\
                                    debug=self.debug,c=self.c,Mach_w=self.Mach_w)
            if self.debug:
                pl.figure(num='initial sig')
                pl.specgram(sig[0,:],2048,self.fs,noverlap=1024)
            
            # De-dopplerisation stage  
            Sig_mov_src = Doppler(sig,bf.r_mi_t,self.t,self.t_traj,self.c,mode="back",debug=self.debug)
            
                    
            # Filtering stage
            Order=3
            # Band-pass filter in one stage
            # Filtered_sig_mov_src = butter_bandpass_filter(Sig_mov_src, \
            #                                 lowcut, highcut, self.fs,order=Order)
            
            # Band-pass filter in two stages: two low-pass filters successively
            Filtered_sig_mov_src = butter_lowpass_filter(Sig_mov_src, \
                                            highcut, self.fs,order=Order)    
            Filtered_sig_mov_src -= butter_lowpass_filter(Sig_mov_src, \
                                                lowcut, self.fs,order=Order)   
            if self.debug:
                pl.figure(num='filtered sig')
                pl.specgram(Filtered_sig_mov_src[0,:],2048,self.fs,noverlap=1024,\
                            vmin=20*np.log10(np.std(Filtered_sig_mov_src[0,:]))-60)  
            
            # Re-dopplerisation stage  
            Sig = Doppler(Filtered_sig_mov_src,bf.r_mi_t,self.t,self.t_traj,self.c,mode="forth",debug=self.debug)
            
            
            # Defining the decimation to optimise computation time
            dopfmax=(1-np.max(bf.M)/2)**-1;
            aliasingfactor = 3 # 2 for respecting Shanon criterion
            fs_decim = aliasingfactor*highcut*dopfmax
            self.decimFactor[ff] = np.max([1, np.floor(self.fs/fs_decim)])
            

            # decimation on temporal variables (except for trajectory if specified)
            Sig = Sig[:,::self.decimFactor[ff]]
            t = self.t[::self.decimFactor[ff]]
            
            # Localising central microphone array for original signal storage
            ind_central = np.argmin(pl.norm(geom-pl.mean(geom,axis=0),axis=-1))
            
            if self.debug:
                print("Frequency band filtering took %.1f s"%(time.time()-t1))
            
            # cleant object for the curent frequency band
            cleant = CleanT(geom, self.grid, self.traj, t, Sig, self.angles, self.t_traj,\
                N_iter=self.N_iter,alpha=self.alpha, E_criterion=self.E_criterion,\
                E_convergence_criterion=self.E_convergence_criterion,angleSelection=self.angleSelection,\
                debug=self.debug,fc=fc,c=self.c,Mach_w=self.Mach_w,findTonal=self.findTonal,\
                monitor=self.monitor)
            
            if self.debug:
                cleant.plot()
            
            t1 = time.time()
            cleant.compute(parrallel=parrallel,QuantFirstIter=True)
            t2 = time.time()
            
            CleantMap(cleant,gauss=True,dyn=dyn)     
            
            cleant.printSourceData()
            
            self.CleanTObjects.append(cleant)
            
            self.Results.append({'fc':fc, 'SampleRate':fs_decim, 'Sources':cleant.Sources,\
                                 'CLEANT_Map':cleant.q_disp,\
                                'Central_Microphone_Signal':Sig[ind_central,:],\
                                'Central_Microphone_Time_Vector':t,\
                                'Angular_Windows':self.angleSelection,\
                                'Central_Microphone_Index':ind_central+1,\
                                'ComputationTime':t2-t1})
            del cleant

def CleantMap(CleantObj,gauss=True,dyn=30,sameDynRange=True,adym=False,reverse=False,sig=0.5):

    nx = np.unique(CleantObj.grid[:,0]).size
    ny = np.unique(CleantObj.grid[:,1]).size
    nz = np.unique(CleantObj.grid[:,2]).size
    
    if CleantObj.angleSelection is not None :
        # CleantObj.Sources is a list of Sources in this case

        # CLEAN-T Maps q_disp are stored in lists for each angular window
        CleantObj.q_disp = list()
        CleantObj.qmax_ton = list()
        CleantObj.qmax_bb = list()
        
        # Loop over angular windows
        for ww, Sources in enumerate(CleantObj.Sources):
            q_disp, qmax_bb, qmax_ton = CleantMap_core(Sources,nx,ny,nz,\
                                        gauss,sameDynRange,dyn,CleantObj.p_ref,adym,reverse,sig)
            
            CleantObj.q_disp.append(q_disp)
            CleantObj.qmax_bb.append(qmax_bb)
            CleantObj.qmax_ton.append(qmax_ton)
    else:
        q_disp, qmax_bb, qmax_ton = CleantMap_core(CleantObj.Sources,nx,ny,nz,\
                                    gauss,sameDynRange,dyn,CleantObj.p_ref,adym,reverse,sig)
        CleantObj.q_disp = q_disp
        CleantObj.qmax_bb = qmax_bb
        CleantObj.qmax_ton = qmax_ton
        

def CleantMap_core(Sources,nx,ny,nz,gauss,sameDynRange,dyn,p_ref,adym,reverse=False,sig=0.1):
    Ni = nx*ny*nz
    if reverse:
        n2=nx
        n1 = np.max([ny,nz])
    else:
        n1=nx
        n2 = np.max([ny,nz])

    q_ton = np.zeros((Ni,))
    q_bb = np.zeros((Ni,))

    
    for ii, source in enumerate(Sources):
        if len(source['SourceSignal']) > 1:
            Energy = np.std(source['SourceSignal'])**2
            if source['Type'] == 1:
                q_ton[source['SourceIndex']] += Energy
            else:
                q_bb[source['SourceIndex']] += Energy
        else: # When charging data from Matlab format, adding extra empty dimentions
            Energy = np.std(source['SourceSignal'][0,0])**2
            if source['Type'] == 1:
                q_ton[source['SourceIndex'][0,0]] += Energy
            else:
                q_bb[source['SourceIndex'][0,0]] += Energy

    if gauss:
        if nx >= 60:
            target_x = int(200)
        else:
            target_x = int(100)
        target_y = int(n2/n1*target_x)
        gauss_sig = sig*target_x/n1
        
        q_disp=np.zeros((target_x,target_y))
        
        xx = np.linspace(0,1,n1)
        yy = np.linspace(0,1,n2)
        f_bb = RegularGridInterpolator((xx,yy),\
                                       np.reshape(q_bb,(n1,n2)),method="linear")
        f_ton = RegularGridInterpolator((xx,yy),\
                                        np.reshape(q_ton,(n1,n2)),method="linear")
        X, Y = np.meshgrid(np.linspace(0,1,target_x), np.linspace(0,1,target_y), indexing='ij')
        q_bb_mat = gaussian_filter(f_bb((X,Y)),gauss_sig, order=0)
        q_ton_mat = gaussian_filter(f_ton((X,Y)),gauss_sig, order=0)
    else:
        q_disp=np.zeros((n1,n2))
        q_bb_mat = np.reshape(q_bb,(n1,n2))
        q_ton_mat = np.reshape(q_ton,(n1,n2))

            
    # set the dynamic and set the displayed map to have broadband and tonal
    # noises on the same map
    if sameDynRange:
        MAX_disp = np.max([np.max(q_ton_mat),np.max(q_bb_mat)])    
        MIN_disp = MAX_disp/10**(dyn/10)
        inds_bb_mat = q_bb_mat>MIN_disp
        inds_ton_mat = q_ton_mat>MIN_disp  
    
        q_disp[inds_bb_mat] = -10*np.log10(q_bb_mat[inds_bb_mat]/MAX_disp)-dyn;
        q_disp[inds_ton_mat] = 10*np.log10(q_ton_mat[inds_ton_mat]/MAX_disp)+dyn;
        
        if not(adym):
            MAX_disp = p_ref**2
        qmax_bb = 10*np.log10(np.max([np.max(q_ton),np.max(q_bb)])/MAX_disp)
        qmax_ton = qmax_bb 
                
    else:
        MAX_disp_ton = np.max(q_ton_mat)
        MAX_disp_bb = np.max(q_bb_mat)
        MIN_disp_ton = MAX_disp_ton/10**(dyn/10)
        MIN_disp_bb = MAX_disp_bb/10**(dyn/10)  
        inds_bb_mat = q_bb_mat>MIN_disp_bb
        inds_ton_mat = q_ton_mat>MIN_disp_ton    

        q_disp[inds_bb_mat] = -10*np.log10(q_bb_mat[inds_bb_mat]/MAX_disp_bb)-dyn;
        q_disp[inds_ton_mat] = 10*np.log10(q_ton_mat[inds_ton_mat]/MAX_disp_ton)+dyn;                

        if not(adym):
            MAX_disp_ton = p_ref**2
            MAX_disp_bb = p_ref**2
        qmax_bb = 10*np.log10(np.max(q_bb)/MAX_disp_bb)
        qmax_ton = 10*np.log10(np.max(q_ton)/MAX_disp_ton)
    
    return q_disp, qmax_bb, qmax_ton


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def compute_spectrum_core(sig,Mask,N_av,nfft,fs):
    f, spec = welch((sig-np.mean(sig))*Mask,fs,nfft,noverlap=0,scaling='spectrum')        
    bgNoise = signal.medfilt(spec, N_av)
    
    return (f,spec,bgNoise)