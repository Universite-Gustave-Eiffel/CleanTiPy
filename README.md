![CleanTiPy logo](https://raphael.leiba.fr/assets/img/CLEAN-T_Logo_1_bw_white_bg.svg)


CleanTiPy is a python package allowing users to realise acoustic images of moving sources. It implements the CLEAN-T algorithm, with some additional features such as:

 * Angular selection of the trajectory (to realise the analysis on portions of the trajectory)
 * Frequency band filtering (on third-octave or octave band) to do the analysis for specific frequency band (can speed up the computation in low frequency by decimating the signal)
 * Implementation of (an homogeneous) wind effect on propagation and back-propagation.

As implementing CLEAN-T, this package can thus be used for computing classical Beamforming computation and propagation simulations (being constitutive blocks of iterative CLEAN-T algorithm).

It is based on the work published in [Cousson *et al.*](https://doi.org/10.1016/j.jsv.2018.11.026) and in [Leiba *et al.*](https://www.bebec.eu/fileadmin/bebec/downloads/bebec-2022/papers/BeBeC-2022-D06.pdf)

## Installation

This code is developed in Python 3.11 and therefore back-compatibility is not guaranteed.

Install the required packages with pip:

```
pip install cleantipy
```

## Usage

Examples can be found in the examples directory on [github](https://github.com/Universite-Gustave-Eiffel/CleanTiPy/tree/main/examples)

An exemple (CLEAN-T over trajectory, for multiple frequency bands) can be run this way:

```
cd ./examples/
python computeCleanT_multiFreq.py
```

An exemple of CLEAN-T over trajectory, for multiple frequency bands, and for multiple angular windows can be run this way:

```
cd ./examples/
python computeCleanT_multiFreq_multiAngles.py
```

Same example can be found as notebook [here](https://github.com/Universite-Gustave-Eiffel/CleanTiPy/blob/main/examples/computeCleanT_multiFreq_multiAngles.ipynb)

## Documentation

The full documentation of the project is available in pdf [here](https://github.com/Universite-Gustave-Eiffel/CleanTiPy/blob/main/docs/build/latex/cleantipy.pdf). It can also be built locally using

```
sphinx-build -b html docs/source/ docs/build/html
```

## Support

Contact Raphaël LEIBA : raphael.leiba@univ-eiffel.fr


## Contributing

Not open for contribution yet

## Authors and acknowledgment

Raphaël Leiba<sup>1,2</sup>, with the help of Quentin Leclère<sup>2</sup>

<sup>1</sup>Joint Research Unit in Environmental Acoustics [UMRAE](https://www.umrae.fr/), [Gustave Eiffel University](https://www.univ-gustave-eiffel.fr/en/) <BR>
<sup>2</sup>Laboratory of Vibration and Acoustics [LVA](https://lva.insa-lyon.fr/), [INSA Lyon](https://www.insa-lyon.fr/en/)

## License

CleanTiPy is licensed under the EUPL-1.2. See [LICENSE](LICENSE.txt)

## Project status

Ready for production release but still in active development

