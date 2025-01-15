![CleanTiPy logo](docs/source/_static/CLEAN-T_Logo_1_bw_white_bg.svg)

This package implements CLEAN-T algorithm in Python.

It is based on the work published in [Cousson *et al.*](https://doi.org/10.1016/j.jsv.2018.11.026) and in [Leiba *et al.*](https://www.bebec.eu/fileadmin/bebec/downloads/bebec-2022/papers/BeBeC-2022-D06.pdf)

## Installation

This code is developed in Python 3.11 and therefore back-compatibility is not guaranteed.

Install the required packages with

```
pip install -r requirements.txt
```

## Usage

An exemple (CLEAN-T over trajectory, for multiple frequency bands) can be run this way:

```
cd ./Examples/
python computeCleanT_multiFreq.py
```

An exemple of CLEAN-T over trajectory, for multiple frequency bands, and for multiple angular windows can be run this way:

```
cd ./Examples/
python computeCleanT_multiFreq_multiAngles.py
```

## Documentation

The full documentation of the project is available in pdf [here](docs/build/latex/cleantipy.pdf). It can also be built locally using

```
sphinx-build -b html docs/source/ docs/build/html
```

## Support

Contact Raphaël LEIBA : raphael.leiba@univ-eiffel.fr


## Contributing

Not open for contribution yet

## Authors and acknowledgment

Raphaël Leiba, with the help of Quentin Leclère and Marley Nejmi

## License

CleanTiPy is licensed under the EUPL. See [LICENSE](LICENSE.txt)

## Project status

early stage
