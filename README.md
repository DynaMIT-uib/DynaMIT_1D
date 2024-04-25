# 1D simulation of dynamic MIT coupling

Python code for 1-dimensional simulation of dynamic magnetosphere-ionosphere-thermosphere coupling. 

To replicate the figures from Laundal et al. (2024), do this:
1. Run `dbdt.py`. This runs simulations with settings defined in `cases.py`, and places datafiles in the `output` directory
2. Run `cartoon.py`. This reads the datafiles and produces figures in the `figures` directory

The repository also contains `diffmatrix.py`, which has code to produce sparse differentiation matrices based on finite differences. It is highly recommended to test it properly before using it for other applications. 

### Installation and dependencies:

There is no installation, just copy the scripts. To run them, you need
- `numpy`
- `scipy`
- `xarray`
- `matplotlib`

### References
- _K. M. Laundal, S. M. Hatch, J. P. Reistad, A. Ohma, P. Tenfjord, M. Madelaire, How the ionosphere responds dynamically to magnetospheric forcing, Geophysical Research Letters (2024, accepted)_

### Funding
This work was funded by
- Trond Mohn Research Foundation (StG, _What Shapes Space?_)
- The European Research Council (ERC _DynaMIT_, 101086985)
