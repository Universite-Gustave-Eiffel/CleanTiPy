=======================
Principle of the method
=======================

The schematics bellow presents the CLEAN-T algorithm implemented in this repository.

.. note::
    
    The subject is supposed in this example to be a plane but it can be 
    any object as long as you provide its trajectory and a possible source grid sourounding it.

As depicted in the central part of the schematics, CLEAN-T is based on the used of the time-domain formulation of moving source
propagation and beaforming to iteratively :

* compute the beamforming over the given grid (or focal plan)
* localise the maximum
* isolate the temporal signal comming from the maximum location
* propagate the signal to the microphones 
* substract the propagated signals to the previously stored signals

.. image:: _static/CLEAN-T_Schematic.*
  :width: 100%
  :alt: CLEAN-T Schematics




If a multi-frequency analysis is performed, the left block of the schematics is performed in order to filter each 
microphone signal in the moving-source related domain (thus dedopplerising, filtering and re-dopplering the signals). 
