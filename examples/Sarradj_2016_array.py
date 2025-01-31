# -*- coding: utf-8 -*-
"""

Created on Thu Jun 15 2023
@author: rleiba
"""
import numpy as np
import scipy.special as sp

def MicArrayGeom(M,h):
    """
    Function generating the array geometry proposed in Sarradj 2016 Bebec paper
    """
    V = 5
    R = 2
    
    m = np.arange(M)+1
    
    # Eq. (8) and (9)
    phi = 2*np.pi*m*(1+np.sqrt(V))/2
    r=R*np.sqrt(m/M)

    # Eq. (11)  
    H = np.matrix([np.arange(-2,3,0.5)]).T
    rho = np.matrix([r])/R
    fH = np.power(sp.i0(np.pi*np.dot(H,np.sqrt(1-np.square(rho)))),np.sign(H))
    
    integral = np.trapz(fH,dx = np.sqrt(1/M),axis=1)
    r_H = np.zeros((len(H),M))
    for mm, mic in enumerate(m):
        r_H[:,mm] = np.squeeze(R*np.sqrt(np.array(integral)*np.array(np.sum(1/(M*fH[:,:mic]),axis=1))))
    
    ind = int(np.where(np.array(H).squeeze()==h)[0])
    x=r_H[ind,:]*np.cos(phi)
    y=r_H[ind,:]*np.sin(phi)
    return x,y
