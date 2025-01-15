# -*- coding: utf-8 -*-
"""
This script is an example of use of InverseMethods library using time domain 
generated signal simulated using the Propagation library.
--Moving source--


-------------------------------------
Created on Apr 05 2022
@author: rleiba
"""
import sys
sys.path.insert(0, '..')

from InverseMethods import Beamforming_t_traj
from Propagation import MovingSrcSimu_t
import numpy as np
import pylab as pl
# import pandas as pd
import scipy.io as io
from Sarradj_2016_array import MicArrayGeom

pl.close('all')
# pl.ion()


pref = 2*10**-5 #Pa
fs = 18000
T = 3
Nt = int(fs*T)
t = np.arange(T*fs)/fs 

# Trajectory setup
v = 100   # m/s
Z = 150 # m
Y = 0   # m
X = np.arange(Nt)*v/fs - T*v/2    # m

traj=np.array([X,Y*np.ones(Nt),Z*np.ones(Nt)]).T

# Micropophone array geometry setup
Nmic = 250
x, y = MicArrayGeom(Nmic,h=2)
z_array = 0

geom = np.array([x,y,z_array*np.ones((Nmic))]).T

# Source definition (position and signal)
sig = np.array([np.sin(2*np.pi*440*t)]) # 440 Hz sine
pos = np.array([[10,0,0]]) # relative position to the trajectory


# Define simulation object
simu = MovingSrcSimu_t(geom, pos, traj, t, sig, SNR=60, timeOrigin='source')

simu.plot()
ax = pl.gca()        
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)


#%% Compute simulated pressures

print("** Computing microphone signals **")
simu.compute(parrallel=True,interpolation="quadratic")
Sig = simu.p_t

# Check Dopplerization
pl.figure()
pl.specgram(simu.p_t[0,:],2048,fs,noverlap=1024)

# Check Dopplerization
pl.figure()
pl.plot(simu.p_t[0,:])


del simu
#%% Define and compute the Beamforming on one point (the source location)
print('\n')
print(52*'*')
print("** Beamforming on one point (the source location) **")
print(52*'*')

bf_t = Beamforming_t_traj(geom,np.array([[10,0,0]]),traj,t,Sig)

bf_t.plot()
ax=pl.gca()
ax.set_aspect('equal')

bf_t.compute(parrallel=True, interpolation='quadratic')

# Check de-Dopplerization
pl.figure()
pl.plot(bf_t.BF_t[0,:])

pl.figure()
pl.specgram(bf_t.BF_t[0,:],2048,fs,noverlap=1024,vmin=-20*np.log10(2)-100)


#%% define image plan relatively to the trajectory
print('\n')
print(54*'*')
print("** Beamforming on the grid following the trajectory **")
print(54*'*')

Lx = 80
Ly = 60
x_F = np.arange(0,Lx,.5)-Lx/2
y_F = np.arange(0,Ly,.5)-Ly/2
z_F = 0 
X_F, Y_F, Z_F = np.meshgrid(x_F,y_F,z_F)
grid = np.array([X_F.reshape(-1),Y_F.reshape(-1),Z_F.reshape(-1)]).T


#%% Define and compute the Beamforming on a grid
bf_t = Beamforming_t_traj(geom,grid,traj,t,Sig)

bf_t.plot()
ax=pl.gca()
ax.set_aspect('equal')

bf_t.compute(parrallel=True)


#%% Display results on grid along the trajectory
fig = pl.figure(figsize=(12,8))
BF_dB = (20*np.log10(np.std(bf_t.BF_t,axis=1)/pref)).reshape((y_F.size,x_F.size))        
mx = np.max(BF_dB)
pl.imshow(BF_dB, vmax=mx, vmin=mx-15, \
        origin='lower',cmap='hot_r',\
            extent=[x_F[0],x_F[-1],y_F[0],y_F[-1]])
pl.ylabel('y (m)')
pl.colorbar()
pl.xlabel('x (m)')
pl.tight_layout()

pl.show()









