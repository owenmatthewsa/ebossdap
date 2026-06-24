# The eBOSS - Data Analysis Pipeline
The eBOSS Data Analysis Pipeline (eBOSS-DAP) is a wrapper for the MaNGA - Data Analysis Pipeline (MaNGA DAP), set up to analyze eBOSS spectra.

Download the output catalogs here: https://datalab.noirlab.edu/data/sdss#sdss-iv-eboss-dap-value-added-catalog

## Citation

If you use the DAP software and/or its output products, please cite the following paper:

 - Matthews Acuña et al. 2025
 
 - https://ui.adsabs.harvard.edu/abs/2025arXiv251218076M/exportcitation

## Requirements

### eBOSS-DAP Pipeline (eBOSS-DAP.py)

numpy >= 1.26.4, pandas >= 2.0, astropy >= 6.0, ipython >= 8.0, mangadap >= 3.4, dust_extinction >= 1.0, dustmaps >= 1.0

```
pip install numpy pandas astropy ipython mangadap dust_extinction dustmaps
```

### Master Notebook (eBOSSDAP_master_notebook.ipynb)

All of the above, plus: scipy >= 1.0, seaborn >= 0.12, tqdm >= 4.0

```
pip install scipy seaborn tqdm
```

Python 3.10 or later is recommended for both.

## Installation
To install and use the eBOSS-DAP, simply install all requirements and download the files in this repository.

## Start Guide
The eBOSS-DAP expects a certain data structure to run.

First, all spectra must be contained in a subdirectory of the directory where the files in this repository are (hereafter called a Bin).

The eBOSS-DAP expects spectra to be eBOSS spectra and, as such, have the naming scheme spec-PLATE-MJD-FIBER.fits.
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

EW-Selection is either 'high' or 'low', which determines whether or not the full or reduced line list is used, respectively.

Redshift-Selection is either 'high' or 'low', which determines whether or not the lines are tied to H-Beta or H-alpha, respectively.

For Example

eBOSS-DAP.py Bin_001 high high

## Master Notebook

eBOSSDAP_master_notebook.ipynb reproduces all figures and tables from Matthews Acuña et al. 2025. It requires the DR2 catalogs from NOIRLab (linked above) as well as the companion files included in this repository. All user-configurable paths are set at the top of Section 0 — point WORKING_DIR at your local clone and set DATA_DIR, C3K_DIR, and the catalog paths accordingly.

The bowtie and pair-histogram figures (Sections 5, 6, 7) additionally require a local copy of the individual eBOSS spectrum FITS files. Set MATCH_SPECS_DIR in Section 0 to point at your spectra directory and set RUN_SPECTRA_PLOTS = True. These figures can be skipped without affecting any other section.

Sections requiring the Drake et al. (in prep.) stellar mass catalog are gated behind HAS_SUMMARY = False by default and will be skipped cleanly until that catalog is released.

The notebook caches the MZR metallicity computation to data/mzr_measured.csv on first run. Subsequent runs load from cache automatically.

## Patch Notes

### eBOSS-DAP.py V2
- For high-EW runs, stellar kinematics are now fit simultaneously with the emission lines by passing stellar_tied=['free', 'free'] to emlfit.fit(). The resulting free stellar velocity and dispersion are written to the fit file header as free_sc_vel, free_sc_disp, free_sc_vel_err, and free_sc_disp_err. Low-EW runs are unchanged.

### Repository Changes (Paper 1 release)
- Added eBOSSDAP_master_notebook.ipynb, spectra/, templates/ssps/, and data/Bin_hbew_approx_10_*.fits
- Moved choice consistency table FITS files from choice_consistency_tables/ to data/ for clean out-of-box notebook execution; choice_consistency_tables/ has been removed
- Removed eBOSSDAP/c3k_scl/ (unused scaled template set)

## File Guide

bin_001: A directory containing two sample plates for testing.

data: Data files required by the master notebook, including the pipeline survey choice consistency tables (Bin_hbew_approx_10_*.fits, used in Appendix Figs A3 & A4).

eBOSSDAP: A directory containing all spectral templates as well as emission line and spectral index definition files.

Figure_Tables: CSV tables released alongside the paper containing the per-bin spectrophotometric error statistics and repeat-spectra trumpet plot statistics.

fits: A directory containing the fits found by the eBOSS-DAP for the files in bin_001.

spectra: Companion spectra required by the master notebook, including the stacked test spectra and fit results for the nebular continuum comparison figure (Fig 11) and the cutoff spectrum figure (Fig 9).

templates/ssps: Dn4000 vs HδA model tracks required by the master notebook (Fig 24).

eBOSS-DAP.py: The main executable file for the eBOSS-DAP.

eBOSSDAP_master_notebook.ipynb: Jupyter notebook reproducing all figures and tables from Matthews Acuña et al. 2025.

eBOSS-DAP-Fit-Plotter.ipynb: A Jupyter notebook meant to plot the results of the eBOSS-DAP overlayed on the initial spectrum.

eBOSS-DAP_BPT_Maker.ipynb: Creates a PDF of the Baldwin, Phillips, and Terlevich (BPT) diagram (J. A. Baldwin et al. 1981) using the updated classification lines from D. R. Law et al. (2021c) for the eBOSS-DAP Catalog.