# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:39:26 2022

@author: rleiba
"""

import numpy as np
import math as m
import os
  
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, -m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def angle_to_pos(pos, theta, phi, cap):
    """

    Parameters
    ----------
    pos : numpy array
        position of points to be shigted in space, dim = (Npoints, 3).
    theta : numpy array, 
        angle around the axis x, (=roll for planes), dim = (Nt,).
    phi : numpy array
        angle around the axis y, (=pitch for planes), dim = (Nt,).
    cap : numpy array
        angle around the axis z, (=yaw or drift for planes), dim = (Nt,).

    Returns
    -------
    rotated_pos : numpy array
        position modified using the 3 angles, dim = (Npoints, 3, Nt).

    """
    if theta.size == phi.size == cap.size :
        if pos.shape[0] == theta.size :
            rotated_pos = np.zeros((theta.size,3))
            for aa in range(theta.size):
                # rotated_pos[aa,:] = np.array(pos[aa,:]*(Rx(theta[aa])*Ry(phi[aa])*Rz(cap[aa])))
                rotated_pos[aa,:] = np.array(pos[aa,:]*(Rz(cap[aa])*Ry(phi[aa])*Rx(theta[aa])).T)
        else: 
            rotated_pos = np.zeros((pos.shape[0],3,theta.size))
            for aa in range(theta.size):
                rotated_pos[:,:,aa] = np.array(pos*(Rx(theta[aa])*Ry(phi[aa])*Rz(cap[aa])))
    else:
        os.error('Angles have different sizes')
    return rotated_pos