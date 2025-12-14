import sys
if sys.version > '3':
    long = int
import os
import gzip
import shutil
import logging
import numpy as np
import pandas as pd
import pdb
from IPython import embed
from astropy.io import fits
from astropy.io import ascii

from mangadap.util import sampling
from mangadap.util import fileio

## Need FWHM of resolution element in Ang
## Vacuum or air?


select_ages = np.arange(-1.2,1.2,0.1)
#allZ = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040])
select_Z = np.array([0.008, 0.02, 0.03])

#bpass_fil = './v2.2.1_imf135_300/spectra-bin-imf135_300.z020.dat'
#bpass_root = '/Users/rubin/Research/MaNGA/NaImcmc/Analysis/BPASS/v2.2.1_imf135_300/spectra-bin-imf135_300.z'
#out_root = '/Users/rubin/Repo/manga/mangadap/data/spectral_templates/bpass/'

## Set up info common to all files
ncol = 52
col_arr = np.arange(1,ncol+1)
age_arr = 10.0**(6.0 + 0.1*(col_arr-2.0))
logage_arr = np.log10(age_arr / 1.0e9)

#pdb.set_trace()

import fnmatch
import os
import pandas as pd
ssps = fnmatch.filter(os.listdir('.'), '*.csv')[1:]
sres_df = pd.read_csv(r"C:\Users\owenm\Documents\catalog_paper\eBOSSDAP\c3k\c3k_sres.csv")
#print(ssps)
def write_template(ofile, wave, flux, ivar, sres):

    # Construct the WCS
    hdr = fits.Header()
    hdr['CTYPE1'] = 'WAVE-LOG'
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = wave[0]
    hdr['CDELT1'] = np.median(wave - np.roll(wave,1))
    hdr['CD1_1'] = np.median(wave - np.roll(wave,1))

    fits.HDUList([fits.PrimaryHDU(data=flux, header=hdr),
                  fits.ImageHDU(name='IVAR', data=ivar),
                  fits.ImageHDU(name='SRES', data=sres)]).writeto(ofile, overwrite=True)



for ssp in ssps:
    data = pd.read_csv(ssp)
    Z_str = ssp[16:19].replace('-','.')
    #zval = float(Z_str)
    #zvalfix = np.round((0.2 * zval), 3)
    zvalfix = Z_str

    wave = data['lam'][2687:9277] #
    wavemin = wave[2687] # 2687
    wavemax = wave[9276] # 9276
    wavedif = np.mean(np.diff(np.log10(wave)))
    n = int(np.ceil((np.log10(wavemax) - np.log10(wavemin)) / wavedif))
    newwave = np.logspace(np.log10(wavemin),np.log10(wavemax),num = n , base=10.0)




    #down_sample_factor = 1.0

    dlam = newwave - np.roll(newwave,1)
    #sres = newwave / (10**((np.log10(newwave) - np.log10(np.roll(newwave,1)))*down_sample_factor+np.log10(np.roll(newwave,1))) - np.roll(newwave,1)) 
    #sres[0] = sres[1]

    sres = sres_df['sres'][2687-1400:9277-1400]


    #embed()

    ivar = np.ones_like(newwave)

    npix = len(dlam)

    crval = min(newwave)
    crpix = 1.0
    cdelt = np.median(dlam)
    #cdelt = sres

    out_flux = data['flux'][2687:9277]

    age_str = z = ssp[16:19].replace('-','.')
    #age_flt = float(age_str)
    #age_fix = np.round(np.log10(age_flt),2)
    age_fix = age_str
    tag = ssp[7:-4]
    #embed()
    '''if tag[1] == '-':
        tag = '0'+tag
    else:
        tag = tag[:6]+'0'+tag[6:]'''
    out_fil = 'SpecAge'+tag+'.fits'
    print(out_fil)

    ## Maybe can just use fileio.writefits_1dspec
    write_template(out_fil, np.log10(newwave), out_flux, ivar, sres)

    #step = sampling.spectral_coordinate_step(wave)

    #pdb.set_trace()
