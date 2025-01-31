# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:57:33 2024

@author: rleiba
"""
import numpy as np
import pylab as pl
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy.interpolate import interp1d

n_CPU_Threads = mp.cpu_count()

def computeDistance(mic_pos,source_traj,M,Mach_w=None):
    """
    Function computing the distance between a microphone and a position 
    following a trajectory
    
    
    //!\\\  Add reference for wind considerations
    
    Parameters
    ----------
    mic_pos : numpy aray (3,)
        (x,y,z) position of the microphone
    source_traj : numpy aray (Nt,3)
        Source position along the trajectory 
    M : numpy aray (Nt,3)
        Mach number projected over (x,y,z) folling the trajectory
    c : float
        Speed of sound
    Mach_w : numpy array (3,)
        Mach number of the wind along x, y, z
        
    """
        
    geom_tiled = np.tile(mic_pos,source_traj.shape[0]).reshape((source_traj.shape[0],3))
    r_ms_t = pl.norm(geom_tiled-(source_traj[:,:]), axis=1)
    
    if Mach_w is not(None) :
        b2 = (1 - np.sum(Mach_w**2) )
        Mr0 = pl.sum((geom_tiled-(source_traj[:,:]))*Mach_w, axis=1)
        
        r02 = r_ms_t**2
        r_ms_t=(-Mr0+np.sqrt(Mr0**2+r02*b2))/b2;
 
    Mcostheta = np.zeros((source_traj.shape[0],))    
    for tt in range(source_traj.shape[0]):   
        Mcostheta[tt] = np.dot(M[tt,:], (geom_tiled[tt,:]-source_traj[tt,:]))
    Mcostheta = Mcostheta/r_ms_t        
    return r_ms_t, Mcostheta


def Doppler(Sig,distances,t,t_traj,c,mode="back",debug=False,parrallel=True):
    """

    Parameters
    ----------
    Sig : numpy array (Nm x Nt)
        Array containing the signals.
    distances : numpy array (Nm x Nt_)
        Array containing the distance between the mics and the destination or 
        origin (depending on "mode").
    t : numpy array (Nt)
        time vector for the signals
    t_traj : numpy array (Nt_)
        time vector for the trajectory
    c : float
        speed of sound.
    mode : str, optional
        Determine the type of action: "back" stands for dedpllerrisation and 
        "forth" stands for dopplerisation. The default is "back".

    Returns
    -------
    Doppler_Sig : numpy array (Nm x Nt)
        Array containing the dedopplerised (or dopplerised) signals.

    """
    Nt_ = distances.shape[-1]
    Nm = Sig.shape[0]
    Nt = Sig.shape[-1]
    fs = 1/(t[1] - t[0])
    fs_traj = 1/(t_traj[1] - t_traj[0])
    t_traj_sig = np.arange(Nt)/fs
    
    if Nt_ != Nt:
        f_dist = interp1d(t_traj, distances, \
                        kind="quadratic",bounds_error=False,fill_value="extrapolate")
        distances = f_dist(t-t[0]+t_traj[0])          
    
    tau = distances/c 
    
    Doppler_Sig = np.zeros((Nm,Nt))
    if not(parrallel):
        for mm in range(Nm):
            Doppler_Sig[mm,:] = Doppler_core(mm,Sig,distances,tau,t_traj_sig,mode)

    else:
        tmp = Parallel(n_jobs=n_CPU_Threads)(delayed(Doppler_core)(mm,Sig[mm,:],distances,tau,t_traj_sig,mode) for mm in range(Nm))
        for mm in range(Nm):
            Doppler_Sig[mm,:] = tmp[mm]
                                
    # if debug:
    #     pl.figure(num=mode)
    #     pl.specgram(Doppler_Sig[0,:],2048,fs,noverlap=1024,\
    #                 vmin=20*np.log10(np.std(Sig[0,:]))-60)

    return Doppler_Sig

def Doppler_core(mm,sig,dist,tau,t_traj_sig,mode):
    taus = t_traj_sig + tau[mm,:][0] # [0] to avoid using np.squeeze()
    if mode == "back": #dedopplerisation
        f = interp1d(t_traj_sig, sig,kind="quadratic",bounds_error=False,fill_value=0)
        Doppler_Sig = f(taus)
   
    elif mode == "forth": #dopplerisation
        f = interp1d(taus, sig,kind="quadratic",bounds_error=False,fill_value=0)
        Doppler_Sig = f(t_traj_sig)
    return Doppler_Sig

def InterpolateTimeTrajectory(t_traj,fs_target):
    Nt_traj = len(t_traj)
    fs_traj = 1/(t_traj[1]-t_traj[0])
    f_t = interp1d(np.arange(Nt_traj)/fs_traj,t_traj, \
                  kind='linear',bounds_error=False,fill_value="extrapolate")
    t_traj_interp = f_t(np.arange(np.floor(Nt_traj/fs_traj*fs_target))/fs_target)
    return t_traj_interp