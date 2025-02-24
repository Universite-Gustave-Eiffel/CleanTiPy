Structure
*********

Overall structure
=================

The project is based on the use of two main classes:

* the ``Propagation.MovingSrcSimu_t`` class that computes the propagation of the sound from a moving source.
* the ``InverseMethods.Beamforming_t_traj`` class that computes the beamforming over a moving source in the time domain.

They are both used to build the CLEAN-T class :

* the ``DeconvolutionMethods.CleanT`` class computes CLEAN-T algorithm for a moving source.
* the ``DeconvolutionMethods.MultiFreqCleanT`` class is a wrap of ``DeconvolutionMethods.CleanT`` to separate the analysis by frequency band (octave or third-octave bands)

All sources are assumed to be monopolar and the sound propagation occurs in a homogeneous medium with homogeneous wind.

The class links to implement the CLEAN-T method are detailed in the figure below:

.. image:: _static/SchemaBlockCleanT.*
  :width: 80%
  :alt: CleanTiPy modules organisation




.. .. autofunction:: Propagation.MovingSrcSimu_t
.. 	:noindex:

.. .. autofunction:: InverseMethods.Beamforming_t_traj
.. 	:noindex:

.. .. autofunction:: DeconvolutionMethods.CleanT
.. 	:noindex:

.. .. autofunction:: DeconvolutionMethods.MultiFreqCleanT
.. 	:noindex:

In details, CLEAN-T class is initialised with theses parameters:

.. autofunction:: cleantipy.DeconvolutionMethods.CleanT.__init__
	:noindex:



File structure
==============

The project is structured over three modules :

* the ``Propagation`` module that hold the classes for the direct path : the propagation of the sound.
* the ``InverseMethods`` module that hold the classes for the inverse methods : the back-propagation of the sound.
* the ``DeconvolutionMethods`` module that hold the classes for CLEAN-T algorithm.

Propagation
-----------

.. automodule:: cleantipy.Propagation
	:members:

InverseMethods
--------------

.. automodule:: cleantipy.InverseMethods
	:members:

DeconvolutionMethods
--------------------

.. automodule:: cleantipy.DeconvolutionMethods
	:members:
