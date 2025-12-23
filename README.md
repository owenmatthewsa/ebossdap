# The eBOSS - Data Analysis Pipeline
The eBOSS Data Analysis Pipeline (eBOSS-DAP) is a wrapper for the MaNGA - data-analysis pipeline (MaNGA DAP), which is set up to analyze eBOSS spectra.

Download the output catalogs here: https://datalab.noirlab.edu/data/sdss#sdss-iv-eboss-dap-value-added-catalog

## Citation

If you use the DAP software and/or its output products, please cite the
following paper:

 - Matthews Acuña et al. 2025
  
 - https://ui.adsabs.harvard.edu/abs/2025arXiv251218076M/exportcitation
 

## Requirements:
This project depends on the following Python packages:



numpy >= 1.26.4

pandas >= 2.0

astropy >= 6.0

ipython >= 8.0

mangadap >= 3.4

dust_extinction >= 1.0

dustmaps >= 1.0

You can install them with:

pip install numpy pandas astropy ipython mangadap dust_extinction dustmaps

## Installation
To install and use the eBOSS-DAP, simply install all requirements and download the files in this repository.

## Start Guide
The eBOSS-DAP expects a certain data structure to run.

First, all spectra must be contained in a subdirectory of the directory where the files in this repository are (hereafter called a Bin).  

The eBOSS-DAP expects spectra to be eBOSS spectra and, as such, have the naming scheme spec-PLATE-MJD-FIBER.fits 
For example spec-3650-55244-0067 where:

PLATE = 3650

MJD = 55244

FIBER = 0067

It then expects all spectra to be within subdirectories named the PLATE.

Additionally, it expects all spectra to have a premade galactic extinction helper CSV file named ebv-PLATE-MJD-FIBER.csv
with the following structure:

ebv,ra,dec,tag,z

##,##,##,PLATE-MJD-FIBER,#

For example:

ebv,ra,dec,tag,z

0.0254465521499514,39.997546,-0.000498,3650-55244-0067,0.48071203

This leaves two key files with the following paths:

bin\PLATE\spec-PLATE-MJD-FIBER.fits

bin\PLATE\ebv-PLATE-MJD-FIBER.csv

For example:

\bin_001\3650\ebv-3650-55244-0067.csv

\bin_001\3650\spec-3650-55244-0067.fits



The call should have the form eBOSS-DAP.py Bin-name EW-Selection Redshift-Selection.

Where Bin-Name is the name of the folder containing the desired plates,

EW-Selection is either 'high' or 'low' which determines whether or not the full or reduced line list is used respectively.

Redshift-Selection is either 'high' or 'low' which determines whether or not the lines are tied to H-Beta or H-alpha respectively.

For Example

eBOSS-DAP.py Bin_001 high high

## File Guide
bin_001: a directory containing two sample plates for testing.

eBOSSDAP: a directory containing all spectral templates as well as emission line and spectral index definition files.

fits: a directory containing the fits found by the eBOSS-DAP for the files in bin_001.

eBOSS-DAP.py: the main executable file for the eBOSS-DAP.

eBOSS-DAP-Fit-Plotter.ipynb: a Jupyter notebook meant to plot the results of the eBOSS-DAP overlayed on the initial spectrum.
