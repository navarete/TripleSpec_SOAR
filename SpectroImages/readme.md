Overview

Here I present an extra feature of the official data reduction pipeline for the TripleSpec4.1 NIR Imaging Spectrograph at the SOAR telescope. 

The IDL data reduction pipeline can be downloaded at https://noirlab.edu/science/observing-noirlab/observing-ctio/observing-soar/data-reduction/triplespec-data

If you wish to know more about the instrument please check the SOAR website ((https://noirlab.edu/science/programs/ctio/instruments/triplespec41-nir-imaging-spectrograph))


The code uses the products from the IDL data reduction pipeline to produce linearized 2d spectro images for all the spectral orders ranging from n=3 to n=7.

In the Python version, you can correct for the heliocentric velocity but this feature is not fully implemented on IDL yet.
There is a Python Notebook example for the code (triplespec_spectroimages_example.ipynb). Both Python and IDL versions are documented.

