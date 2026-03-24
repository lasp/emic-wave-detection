# EMIC Wave Detection
This Python code detects EMIC waves from Van Allen Probe, Magnetospheric Multiscale (MMS), Time History of Events and Macroscale Interactions during Substorms (THEMIS), and Geostationary Operational Environmental Satellite (GOES) magnetic field data. The algorithm follows that described in Bortnik et al (2007). Magnetic field and ephemeris data are pulled using the PySPEDAS package. It focuses on geomagnetic storm times throughout the overlap of all four missions (September 2015 - October 2019). Geomagnetic storms are identified by Pedersen et al (2024), from which a list is uploaded into the code. 

## Getting Started
This code uses the PySPEDAS module. Documentation and installation guides can be found at: https://pyspedas.readthedocs.io/en/latest/getting_started.html. Similarly, the documentation for PyTplot variables and plotting can be found at: https://pytplot.readthedocs.io/en/latest/index.html.
The code also imports a list of geomagnetic storms from Pedersen et al (2024): https://doi.org/10.1029/2024JA032656.

## Contibution guide

## Contributors
