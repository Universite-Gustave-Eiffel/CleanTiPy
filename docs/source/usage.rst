Usage
*****

Requirement
===========

To use CleanTiPy, first install the required packages :

* numpy
* matplotlib
* scipy
* pyfftw (optional, to speed up the discrete Fourier transforms (DFT) in ``DeconvolutionMethods.CleanT.find_max()``).
* simplespectral (uses pyfftw, scipy.fft or numpy.fft seamlessly)
* joblib

.. note::
    
    pyfftw requires FFTW3 to function. FFTW3 is available under two licenses, the free GPL and a non-free license that allows it to be used in proprietary program

Installation
============

This code is developed in Python 3.11 and therefore back-compatibility is not guaranteed.

Install the required packages with

.. code-block:: console

    pip install -r requirements.txt

.. _Examples:

Examples
========

For a multi-frequency analysis you can run this example:

.. code-block:: console

    cd ./Examples/
    python computeCleanT_multiFreq.py


For a multi-frequency analysis with different angular selection windows you can run this example:

.. code-block:: console

    cd ./Examples/
    python computeCleanT_multiFreq_multiAngles.py



