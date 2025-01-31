# -*- coding: utf-8 -*-
"""
Print Data from CLEAN-T Export (in MAT files)

Created on Mon Jul 22 08:51:21 2024
@author: rleiba
"""

import numpy as np
import pylab as pl
pl.close('all')
import scipy.io as io

import sys
sys.path.insert(0, '..')
from DeconvolutionMethods import CleantMap_core

# q_disp, qmax_bb, qmax_ton = CleantMap_core(Sources,nx,ny,nz,\
#                             gauss,sameDynRange,dyn,CleantObj.p_ref,adym,reverse)

#%%

tmp = io.loadmat("../Validation/CleanT_results_A351LikeSimulation_multiFreq_highRes.mat",struct_as_record=True)
Results = tmp['Results'][0]
Informations = tmp['Informations'][0]

AngleWindows= Informations['AngularWindow'][0]
Grid = Informations['Grid'][0]
x_F = np.unique(Grid[:,0])
y_F = np.unique(Grid[:,1])
z_F = np.unique(Grid[:,2])

SourcePos = np.array([[30,0,0],\
                [6,10,0],\
                [6,10,0],\
                [6,-10,0],\
                [6,-10,0],\
                [-5,-5.5,0],\
                [-5,5.5,0],\
                [-1.5,-5.5,0],\
                [-1.5,5.5,0]])
#%% Plot array Geometry

geom = Informations['MicrophonePositions'][0]        
ActivatedMics = Informations['ActivatedMicrophones'][0]  



pl.figure()
strm = ['o', 'x', '^', 'd', '.']
fbc=Informations['CentralFrequencies'][0][0]
for ii, fb in enumerate(fbc):
    ind_activeMics, = np.where(ActivatedMics[:,ii])    
    geom_fc = geom[ind_activeMics,:]
    pl.scatter(geom_fc[:,0], geom_fc[:,1], marker=strm[ii])      
pl.legend(fbc)
    
#%% Display results on grid along the trajectory
dyn = 15

TemporalMasks = list()
for ww in range(len(AngleWindows)):
    TemporalMasks.append(Results[0]['Sources'][0][0][0][ww][0][0]['TemporalMask'][0,0][0])
TemporalMasks = np.array(TemporalMasks)

pl.figure(num="Summation of Temporal Masks")
pl.plot(np.sum(TemporalMasks,axis=0))
pl.plot(np.sum(TemporalMasks**2,axis=0))
# pl.plot(TemporalMasks.T)
pl.legend(['pressure','energy'])


for ww in range(len(AngleWindows)):

    fig, axs = pl.subplots(2,len(Results), 
                           num='CLEAN-T vs BF - [%.1f° , %.1f°]' \
                               %(AngleWindows[ww,0],AngleWindows[ww,1]), 
                           constrained_layout=True,figsize=(3.5*len(Results),8.5))

    
    for ff in range(len(Results)):
        
        if len(Results)>1:
            ax = axs[0,ff]
        else:
            ax = axs[0]
        Sources_ang = Results[ff]['Sources'][0][0][0]
        BF_dB = Sources_ang[ww][0][0]['AcousticMap'][0,0].reshape((y_F.size,x_F.size))        
        mx = np.max(BF_dB)
        ax.imshow(BF_dB, vmax=mx, vmin=mx-dyn, \
                origin='lower',cmap='hot_r',\
                    extent=[x_F[0],x_F[-1],y_F[0],y_F[-1]])
        pl.ylabel('y (m)')
        # fig.colorbar(ax=ax)
        pl.xlabel('x (m)')
        ax.set_title('f_c = %d Hz' %(Results[ff]['fc']))
        ax.scatter(SourcePos[:,0],SourcePos[:,1],marker='o',facecolor='none',edgecolor='w',linestyle='--')
        pl.tight_layout()
        
        if len(Results)>1:
            ax = axs[1,ff]
        else:
            ax = axs[1]

        q_disp, qmax_bb, qmax_ton = CleantMap_core(Sources_ang[ww][0],x_F.size,y_F.size,z_F.size,\
                                    gauss=True,sameDynRange=False,dyn=dyn,p_ref=2e-5,adym=False,\
                                        reverse=True,sig=1)
        
        img = ax.imshow(q_disp, origin='lower',
                  extent=[x_F[0],x_F[-1],y_F[0],y_F[-1]], cmap='RdBu',
                  vmin=-dyn,vmax=dyn,interpolation_stage='data')
        ax.scatter(SourcePos[:,0],SourcePos[:,1],marker='o',facecolor='none',edgecolor='gray',linestyle='--')
        cbar = fig.colorbar(img, ax=ax,\
                            ticks=[-dyn, 0, dyn],location="bottom")
        cbar.ax.set_xticklabels(['%.1f' %(qmax_bb),\
                                  '%d' %(-dyn), '%.1f' %(qmax_ton)])
        # cbar.ax.set_title('[dB]')
        cbar.set_label('Broadband               Tonal       ', fontstyle='italic', labelpad=-13)