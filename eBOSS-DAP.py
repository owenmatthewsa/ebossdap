# Imports
import sys
from pathlib import Path
import os
import traceback
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import astropy.constants as const
import astropy.units as u
import glob
import time
from astropy.io import fits
from IPython import embed
from mangadap.util.pixelmask import SpectralPixelMask
from mangadap.par.artifactdb import ArtifactDB
from mangadap.par.emissionmomentsdb import EmissionMomentsDB
from mangadap.par.emissionlinedb import EmissionLineDB
from mangadap.par.absorptionindexdb import AbsorptionIndexDB
from mangadap.par.bandheadindexdb import BandheadIndexDB
from mangadap.proc.emissionlinemoments import EmissionLineMoments
from mangadap.proc.sasuke import Sasuke
from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModel, StellarContinuumModelBitMask
from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
from mangadap.proc.spectralfitting import EmissionLineFit
from mangadap.proc.spectralindices import SpectralIndices
from mangadap.util.sampling import angstroms_per_pixel
from mangadap.util.filter import interpolate_masked_vector
from mangadap.util.bitmask import BitMask
from mangadap.proc.templatelibrary import TemplateLibrary, TemplateLibraryDef
from mangadap.config import defaults
from mangadap.proc.bandpassfilter import emission_line_equivalent_width
from dust_extinction.averages import GCC09_MWAvg as dext
import mangadap


#----------------------------------------------------------------------------
def get_redshift(initdir,plate, ifu):
    """
    Get the redshift of a galaxy from the eboss file.
    Args:
        initdir (:obj:`str`, optional):
            Directory with the spectra fits file. If None, uses the
            default directory path based on the environmental

        plate (:obj:`str`):
            Plate number

        ifu (:obj:`str`):
            IFU identifier
       
    Returns:
        :obj:`tuple`: Returns 3 floats: The redshift, Right Ascension, and Declination to the galaxy observed by the
        provided PLATEIFU. 
    """
    hdu = fits.getdata((initdir+'/'+plate+'/'+ifu),2)
    return hdu['Z'][0], hdu['RA'][0], hdu['DEC'][0]


#-----------------------------------------------------------------------------
def get_spectra(initdir,plate, ifu, ew):
    """
    Extract spectra from an eBOSS observation.
    Args:
        initdir (:obj:`str`, optional):
            Directory with the spectra fits file and dust csv. If None, uses the
            default directory path based on the environmental
            variables.

        plate (:obj:`str`):
            Plate number

        ifu (:obj:`str`):
            IFU identifier
    Returns:
        :obj:`tuple`: Returns 8 numpy vectors: 
        The wavelength in log units, 
        flux,
        flux inverse variance with a mask from 3706.3 > lambda > 3662 for use in flux and continuum fits,
        flux inverse variance with no mask for use in spectral index fitting,  
        spectral resolution, 
        E(B-V), 
        AND mask,
        and the SPAll redshift 
        extracted from the spectra and helper file.
    """

    
    f = fits.getdata((initdir+'/'+plate+'/'+ifu),1)

    tag = ifu.split('spec-')[1].split('.fits')[0]


    #Reads in the previously measured E(B-V) based on Milky-Way dust extinction as well as the SPAll redshift. 
    try:
        helperdf = pd.read_csv(initdir+'/'+plate+'/'+'ebv--'+tag+'.csv')
    except:
        helperdf = pd.read_csv(initdir+'/'+plate+'/'+'ebv-'+tag+'.csv')
    
    try:
        ebv = helperdf['ebv'][0]
        z = helperdf['z'][0]
    except:
        ebv = 0.0
        z = 0.0
    ext = dext()

    #Reads in the spectra and performs dust correction, creating loglam, flux, and ivars
    loglam = f['loglam']
    lam = 10 ** loglam
    lamu = lam * u.AA

    corr = ext.extinguish(lamu,Ebv=ebv)
    flux = f['flux'] / corr
    
    lam2 = 10 ** (np.arange(0,loglam.size)  * 1E-4 + loglam[0])
    

    #Creates seperate ivars for use in spectral index measurements and emission line fitting
    ivar = f['ivar'] * corr**2
    ivar_spind =  f['ivar'] * corr**2



    restlam = lam/(1+z)

    if ew == 'high':
        var_balmer = np.where((restlam < 3706.3) & (restlam > 3662),True,False)
        ivar[var_balmer] = 0 
    

    #Creates resolution vector
    dlam = angstroms_per_pixel(lam2,log=True)
    
    res = np.ma.divide(np.ma.divide(lam2, f['wdisp']), dlam) / 2
    res = interpolate_masked_vector(res)

    #Masks out all places with errors <= 0
    ivar[ivar < 0] = 0.0
    ivar_spind[ivar_spind < 0] = 0.0
    
    #initializes AND_MASK
    and_mask =  f['and_mask']
    
    return loglam, flux, ivar, ivar_spind, res, ebv, and_mask ,z
#-----------------------------------------------------------------------------


'''
Reads in the bin to run the main pipeline on. 
The call should have the form eBOSS-DAP.py Bin-name EW-Selection Redshift-Selection.
Where Bin-Name is the name of the folder containing the desired plates,
EW-Selection is either 'high' or 'low' which determines whether or not the full or reduced line list is used respectively.
Redshift-Selection is either 'high' or 'low' which determines whether or not the lines are tied to H-Beta or H-alpha respectively.
'''

d = sys.argv[1]
plates = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
print(type(plates[0]))
print('Plates:',plates)
if len(sys.argv) > 2:
    el_ew = sys.argv[2]
else:
    el_ew = 'high' # high, low

if len(sys.argv) > 3:
    el_z = sys.argv[3]
else:
    el_z = 'high' # high, low



#-----------------------------------------------------------------------------
print(defaults.dap_data_root())


'''
Read in and creates the Stellar tenmplate library. Flexible for either having it in a nested or single directory.

tpl is the template library used for fitting the emission lines and stellar continuum
with a broad range in Lambda with R ~= 1180 between 1281 -- 2750 Angstroms, 
R ~= 7060 between 2750 -- 9100 Angstroms, and R ~= 1180 between 9100 - 10616 Angstroms. 

scl_tpl is the template library used for fitting the stellar continuum kinematics,
with a more narrow range in Lambda with R ~= 7060 between 2750 -- 9100 Angstroms.

This portion runs once per BIN, allowing for reduced computation time while running many plates



You also need to specify the sampling for the template spectra.
The templates must be sampled with the same pixel step as the spectra to be fit, up to an integer factor. 
The critical thing for the sampling is that you do not want to undersample the
spectral resolution element of the template spectra. 
Here, we set the sampling for the C3K templates to be a factor of 4 smaller
than the eBOSS spectrum to be fit (vel_scale_ratio).



Note that *no* matching of the spectral resolution to the galaxy spectrum is performed.
'''

nested = False

vel_scale_ratio = 4

try:

    file_search = str(Path().absolute() / 'eBOSSDAP' / 'c3k' / 'SpecAge*.fits')
    
    tpllib = TemplateLibraryDef('c3k',
                                file_search=file_search,
                                sres_ext='SRES',
                                in_vacuum=True,
                                log10=True)
    tpl = TemplateLibrary(tpllib, match_resolution=False, velscale_ratio=vel_scale_ratio, spectral_step=1e-4,
                          log=True, hardcopy=False)


    sc_file_search = str(Path().absolute() / 'eBOSSDAP' / 'c3k_scl' / 'SpecAge*.fits')

    sc_tpllib = TemplateLibraryDef('c3k_scl',
                                file_search=sc_file_search,
                                sres_ext='SRES',
                                in_vacuum=True,
                                log10=True)
    sc_tpl = TemplateLibrary(sc_tpllib, match_resolution=False, velscale_ratio=vel_scale_ratio, spectral_step=1e-4,
                          log=True, hardcopy=False)
    os.listdir(str(Path().absolute() / 'eBOSSDAP'))

except:
    file_search = str(Path().absolute() / 'eBOSSDAP' / 'c3k' / 'SpecAge*.fits')


    tpllib = TemplateLibraryDef('c3k',
                                file_search=file_search,
                                sres_ext='SRES',
                                in_vacuum=True,
                                log10=True)
    tpl = TemplateLibrary(tpllib, match_resolution=False, velscale_ratio=vel_scale_ratio, spectral_step=1e-4,
                          log=True, hardcopy=False)

    sc_file_search = str(Path().absolute() / 'eBOSSDAP' / 'eBOSSDAP' / 'c3k_scl' / 'SpecAge*.fits')

    sc_tpllib = TemplateLibraryDef('c3k_scl',
                                file_search=sc_file_search,
                                sres_ext='SRES',
                                in_vacuum=True,
                                log10=True)
    sc_tpl = TemplateLibrary(sc_tpllib, match_resolution=False, velscale_ratio=vel_scale_ratio, spectral_step=1e-4,
                          log=True, hardcopy=False)


    os.listdir(str(Path().absolute() / 'eBOSSDAP'/ 'eBOSSDAP'))
    nested = True




#-----------------------------------------------------------------------------
def main(initdir,plate,el_ew,el_z):

    #-------------------------------------------------------------------
    # Read spectra to fit. The following reads a single eBOSS plate.
    # All spectrum contained within the plate directory will be run simultaniously.

    file_search = str( initdir+'/'+ plate + '/spec*.fits')
    files = sorted(glob.glob(file_search))
    use_datamask = True
    print(file_search)

    plat = plate
    ifus = []
    for file in files:
        ifus.append(os.path.basename(file))

    # Read a spectrum and create a vector of shape (N,M) for all key spectral variables.
    # where N is the number of spectra in the plate and M is the number of unique wavelengths in the plate.
    waves = []
    fluxes = []
    ivars = []
    ivar_spinds = []
    sress = [] 
    zs = []
    ras = []
    decs = []
    ebvs = []
    and_masks = []
    tags = []
    for _i,ifu in enumerate(ifus):
        lwav, flux, ivar, ivar_spind, sres, ebv, and_mask, spallz = get_spectra(initdir,plat, ifu,el_ew)
        waves.append(lwav.tolist())
        fluxes.append(flux.tolist())
        ivars.append(ivar.tolist())
        ivar_spinds.append(ivar_spind.tolist())
        sress.append(sres.tolist())#
        ebvs.append(ebv)
        and_masks.append(and_mask)
        dataz, ra, dec = get_redshift(initdir,plat,ifu)
        if spallz != 0.0:
            zs.append(spallz)
        else:
            print('No spAll redshift detected for '+str(ifu))
            zs.append(dataz)
        ras.append(ra)
        decs.append(dec)
        tags.append(ifu.split('spec-')[1].split('.fits')[0])
    from itertools import zip_longest
    wavearr =  np.array(list(zip_longest(*waves,fillvalue=np.nan))).T
    zarr = np.array(zs)
    raarray = np.array(ras)
    decarray = np.array(decs)
    ebvarr = np.array(ebvs)
    

    # In general, the DAP fitting functions expect data to be in 2D
    # arrays with shape (N-spectra,N-wave). So if you only have one
    # spectrum, you need to expand the dimensions:
    if np.logical_not(np.isfinite(np.unique(wavearr)[-1])):
        wave_uni = np.unique(wavearr)[:-1]
    else:
        wave_uni = np.unique(wavearr)

    flux = np.ma.masked_all((len(ifus),len(wave_uni)))
    ivar = np.zeros((len(ifus),len(wave_uni)))
    ivar_spind = np.zeros((len(ifus),len(wave_uni)))
    sres = np.zeros((len(ifus),len(wave_uni)))
    and_mask = np.zeros((len(ifus),len(wave_uni)))

    for i in range(len(ifus)):
        ind = np.where(wave_uni>=waves[i][0])[0][0]
        flux[i,ind:ind+len(fluxes[i][:-1])] = fluxes[i][:-1]
        ivar[i,ind:ind+len(ivars[i][:-1])] = ivars[i][:-1]
        ivar_spind[i,ind:ind+len(ivar_spinds[i][:-1])] = ivar_spinds[i][:-1]
        sres[i,ind:ind+len(sress[i][:-1])] = sress[i][:-1]  
        and_mask[i,ind:ind+len(and_masks[i][:-1])] = and_masks[i][:-1]
        sres[i] = interpolate_masked_vector(np.ma.MaskedArray(sres[i],mask=sres[i]==0))

    
    ''' 
    Initializes the SDSS bitmask to unmask certain bits within each AND_MASK. This is modular and can be adjusted 
    per the user's specifications.
    '''
    if nested: 
        bitmask_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'sdssMaskbits.par')
    else:
        bitmask_path = str(Path().absolute() / 'eBOSSDAP'/'sdssMaskbits.par')

    targ1bm = BitMask.from_par_file(bitmask_path, 'SPPIXMASK')

    fibermasks = ['NOPLUG', 'BADTRACE', 'BADFLAT', 'BADARC', 'MANYBADCOLUMNS', 'MANYREJECTED', 
                    'LARGESHIFT', 'BADSKYFIBER','NEARWHOPPER', 'WHOPPER', 'SMEARIMAGE', 'SMEARHIGHSN', 'SMEARMEDSN']

    and_mask = and_mask.astype(int)
    ignore_bits = targ1bm.flagged(and_mask, flag=fibermasks)


    and_mask[ignore_bits] = targ1bm.turn_off(and_mask[ignore_bits], fibermasks)


    '''
    Create a composite Mask of three masks, 
    var_and: the positions in the spectra that are masked in the spectra's AND_MASK ; ONLY USED IF datamask IS True(Default)
    var_inf: the positions in the spectra where either flux or ivar is not finite
    var_neg: the positions in the spectra where ivar is less than or equal to zero
    '''
    var_and = targ1bm.flagged(targ1bm.turn_off(and_mask, 'BRIGHTSKY'))
    var_inf = np.logical_not(np.isfinite(flux)) | np.logical_not(np.isfinite(ivar))
    var_neg = np.where(ivar <= 0,True,False)

    var_and_spind = targ1bm.flagged(targ1bm.turn_off(and_mask, 'BRIGHTSKY'))
    var_inf_spind = np.logical_not(np.isfinite(flux)) | np.logical_not(np.isfinite(ivar_spind))
    var_neg_spind = np.where(ivar_spind <= 0,True,False)

    data_mask = var_and | var_inf | var_neg
    data_mask_spind = var_and_spind | var_inf_spind | var_neg_spind


    #Masking areas where either flux or ivar is not finite
    flux[var_inf] = 0
    flux[var_inf] = np.ma.masked
    ivar[var_inf] = 0
    ivar[var_inf] = np.ma.masked
    ivar_spind[var_inf_spind] = 0
    ivar_spind[var_inf_spind] = np.ma.masked
    #Masking areas where Ivar is less than or equal to zero
    flux[var_neg] = np.ma.masked
    ivar[var_neg] = np.ma.masked
    ivar_spind[var_neg_spind] = np.ma.masked
    if use_datamask:
        #masking areas found in the AND mask from the spectra
        flux[var_and] = np.ma.masked
        ivar[var_and] = np.ma.masked
        ivar_spind[var_and_spind] = np.ma.masked

    #Create a uniform wavelength grid containing all unique values found in the wavelengths of all spectra in the plate.
    wave =  10 ** (np.arange(0,wave_uni.size)  * 1E-4 + wave_uni[0]) 
    ferr = np.ma.power(ivar, -0.5)
    ferr_spind = np.ma.power(ivar_spind, -0.5)
    for i,z in enumerate(zarr):
        if z > 1.75:
            flux[i] = np.ma.masked
            zarr[i] = -0.1
        
    init_mask = flux.mask
    
    # The majority (if not all) of the DAP methods expect that your
    # spectra are binned logarithmically in wavelength (primarily
    # because this is what pPXF expects). You can either have the DAP
    # function determine this value (commented line below) or set it
    # directly. The value is used to resample the template spectra to
    # match the sampling of the spectra to fit (up to some integer; see
    # velscale_ratio).
    # spectral_step = spectral_coordinate_step(wave, log=True)
    spectral_step = 1e-4


    # Hereafter, the methods expect a wavelength vector, a flux array
    # with the spectra to fit, an ferr array with the 1-sigma errors in
    # the flux, and sres with the wavelength-dependent spectral
    # resolution, R = lambda / Dlambda
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    # The DAP needs a reasonable guess of the redshift of the spectrum
    # (within +/- 2000 km/s). In the eBOSS-DAP, we pull the redshift
    # from the SPAll file. There must be one redshift estimate per
    # spectrum to fit.
    
    z = zarr
    print('Redshifts: {0}'.format(z))

    # The DAP also requires an initial guess for the velocity
    # dispersion. A guess of 100 km/s is usually robust, but this may
    # depend on your spectral resolution.
    dispersion = np.ones_like(z) * 100
    #-------------------------------------------------------------------


    #-------------------------------------------------------------------
    # Templates used in the emission-line modeling and the stellar continuum fits 
    # were created outside of the main function and are pulled in, 
    # however we can still adjust the vel_scale_ratio for each template library as needed.


    # Template pixel scale a factor of 4 smaller than galaxy data
    sc_velscale_ratio = vel_scale_ratio
    # Template pixel scale a factor of 4 smaller than galaxy data
    el_velscale_ratio = vel_scale_ratio


    # You then need to identify the database that defines the
    # emission-line passbands for the non-parametric
    # emission-line moment calculations (elmom), and the emission-line
    # parameters (elfit) for the Gaussian emission-line modeling.
    # See Section 3.2.3 of Matthews-Acuna et all 2026a.
    # If high/low were chosen for equivalent width the line list will have 78/27 lines
    # If high/low were chosen for redshift the majority of lines in the line list will be tied to H-Beta/H-Alpha

    if nested: 
        if el_ew == 'high' and el_z == 'high':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elbedr2_hew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elpedr2_hew_hz.par')
        elif el_ew == 'high' and el_z == 'low':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elbedr2_hew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elpedr2_hew_lz.par')
        elif el_ew == 'low' and el_z == 'high':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elbedr2_lew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elpedr2_lew_hz.par')
        else:
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elbedr2_lew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'elpedr2_lew_lz.par')
    else:
        if el_ew == 'high' and el_z == 'high':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'elbedr2_hew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'elpedr2_hew_hz.par')
        elif el_ew == 'high' and el_z == 'low':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'elbedr2_hew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'elpedr2_hew_lz.par')
        elif el_ew == 'low' and el_z == 'high':
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'elbedr2_lew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'elpedr2_lew_hz.par')
        else:
            elmom_path = str(Path().absolute() / 'eBOSSDAP'/'elbedr2_lew.par')
            elfit_path = str(Path().absolute() / 'eBOSSDAP'/'elpedr2_lew_lz.par')

    # If you want to also calculate the spectral indices, you can
    # provide a keyword that indicates the database with the passband
    # definitions for both the absorption-line and bandhead/color
    # indices to measure. 
    if nested:
        absindx_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'extindx.par')
        bhdindx_path = str(Path().absolute() / 'eBOSSDAP'/'eBOSSDAP'/'bhedr1.par')
    else:
        absindx_path = str(Path().absolute() / 'eBOSSDAP'/'extindx.par')
        bhdindx_path = str(Path().absolute() / 'eBOSSDAP'/'bhedr1.par')



    # Now we want to construct a pixel mask that excludes regions with
    # known artifacts and emission lines. The 'BADSKY' artifact
    # database only masks the 5577, which can have strong left-over
    # residuals after sky-subtraction. The list of emission lines (set
    # by the ELPMPL8 keyword) can be different from the list of
    # emission lines fit below.
    sc_pixel_mask = SpectralPixelMask(artdb=ArtifactDB.from_key('BADSKY'),
                                      emldb=EmissionLineDB(elfit_path))
    # Mask the 5577 sky line
    el_pixel_mask = SpectralPixelMask(artdb=ArtifactDB.from_key('BADSKY'))

    # Finally, you can set whether or not to show a set of plots.
    #
    # Show the ppxf-generated plots for each fit stage.
    fit_plots = False
    # Show summary plots
    usr_plots = False
    #-------------------------------------------------------------------


    #-------------------------------------------------------------------
    # Fit the stellar continuum kinematics
    # We use the pre-constructed template library above.  
    
    # Instantiate the fitting class, including the mask that it should
    # use to flag the data. [[This mask should just be default...]]
    ppxf = PPXFFit(StellarContinuumModelBitMask())

    # The following call performs the fit to the spectrum. Specifically
    # note that the code only fits the first two moments, uses an
    # 8th-order additive polynomial, and uses the 'no_global_wrej'
    # iteration mode. See
    # https://sdss-mangadap.readthedocs.io/en/latest/api/mangadap.proc.ppxffit.html#mangadap.proc.ppxffit.PPXFFit.fit

    # Read the database that define the emission lines and passbands
    momdb = EmissionMomentsDB(elmom_path)

    # This calculation of the mean spectral resolution is a kludge. The
    # template library should provide spectra that are *all* at the
    # same spectral resolution. Otherwise, one cannot freely combine
    # the spectra to fit the Doppler broadening of the galaxy spectrum
    # in a robust (constrained) way (without substantially more
    # effort). There should be no difference between what's done below
    # and simply taking the spectral resolution to be that of the first
    # template spectrum (i.e., sc_tpl['SPECRES'].data[0])
    sc_tpl_sres = np.mean(sc_tpl['SPECRES'].data, axis=0).ravel()

    cont_wave, cont_flux, cont_mask, cont_par \
        = ppxf.fit(sc_tpl['WAVE'].data.copy(), sc_tpl['FLUX'].data.copy(), wave, flux, ferr,
                   z, dispersion, iteration_mode='no_global_wrej', reject_boxcar=101,
                   ensemble=False, velscale_ratio=sc_velscale_ratio, mask=sc_pixel_mask,
                   matched_resolution=False, tpl_sres=sc_tpl_sres, obj_sres=sres, mdegree=-1, degree=8,
                   moments=2, plot=fit_plots)


    # The returned objects from the fit are the wavelength, model, and
    # mask vectors and the record array with the best-fitting model
    # parameters. The datamodel of the best-fitting model parameters is
    # set by:
    # https://sdss-mangadap.readthedocs.io/en/latest/api/mangadap.proc.spectralfitting.html#mangadap.proc.spectralfitthttps://sdss-mangadap.readthedocs.io/en/latest/api/mangadap.proc.spectralfitting.html#mangadap.proc.spectralfitting.StellarKinematicsFit._per_stellar_kinematics_dtypeing.StellarKinematicsFit._per_stellar_kinematics_dtype

    # Remask the continuum fit
    sc_continuum = StellarContinuumModel.reset_continuum_mask_window(
                    np.ma.MaskedArray(cont_flux, mask=cont_mask>0))



    #-------------------------------------------------------------------
    # Get the emission-line moments using the fitted stellar continuum

    # Measure the moments: if line fits are bad bring this back in as better redshift estimate
    #elmom = EmissionLineMoments.measure_moments(momdb, wave, flux, continuum=sc_continuum,
    #                                            redshift=z)
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    # Fit the emission-line model

    # Set the emission-line continuum templates if different from those
    # used for the stellar continuum (default)
    # If the template sets are different, we need to match the
    # spectral resolution to the galaxy data ...

    corr_inf = np.logical_not(np.isfinite(cont_par['SIGMACORR_SRES']))
    cont_par['SIGMACORR_SRES'][corr_inf] = 0.0


    el_tpl = tpl
    el_tpl_sres = np.mean(el_tpl['SPECRES'].data, axis=0).ravel()

    # ... and use the corrected velocity dispersions.
    stellar_kinematics = cont_par['KIN']
    stellar_kinematics[:,1] = np.ma.sqrt(np.square(cont_par['KIN'][:,1]) -
                                    np.square(cont_par['SIGMACORR_SRES'])).filled(0.0)


    stellar_kinematics_err = cont_par['KINERR']
    stellar_kinematics_err[:,1] = (cont_par['KIN'][:,1] / stellar_kinematics[:,1]) * cont_par['KINERR'][:,1]
    

    sigma_err_inf = np.logical_not(np.isfinite(stellar_kinematics_err[:,1]))
    stellar_kinematics_err[:,1] [sigma_err_inf] = -999.0


    #Next we mask out any velocity dispersions greater than 450 km/s as these are likely non-physical.
    stellar_kinematics[stellar_kinematics[:,1] >= 450.0,1] = 450.0

    print()
    print('CORRECTIONS')
    print(cont_par['SIGMACORR_SRES'])
    print()
    print('sigmas')
    print(cont_par['KIN'][:,1])
    print()

    # Read the emission line fitting database

    emldb = EmissionLineDB(elfit_path)
    # Instantiate the fitting class
    emlfit = Sasuke(EmissionLineModelBitMask())

    # Perform the fit
    efit_t = time.perf_counter()


    eml_wave, model_flux, eml_flux, eml_mask, eml_fit_par, eml_eml_par \
            = emlfit.fit(emldb, wave, flux, obj_ferr=ferr, obj_mask=el_pixel_mask, obj_sres=sres,
                         guess_redshift=z, guess_dispersion=dispersion, reject_boxcar=101,
                         stpl_wave=el_tpl['WAVE'].data, stpl_flux=el_tpl['FLUX'].data,
                         stpl_sres=el_tpl_sres, stellar_kinematics=stellar_kinematics,
                         etpl_sinst_mode='offset', etpl_sinst_min=10.,
                         velscale_ratio=el_velscale_ratio, mdegree=8,  degree=-1,
                         plot=fit_plots,ensemble=False,sigma_rej=5)
    print('TIME: ', time.perf_counter() - efit_t)

    # Line-fit metrics
    eml_eml_par = EmissionLineFit.line_metrics(emldb, wave, flux, ferr, model_flux, eml_eml_par,
                                               model_mask=eml_mask, bitmask=emlfit.bitmask)
    EmissionLineFit.measure_equivalent_width(wave, flux, emldb, eml_eml_par,
                                               bitmask=emlfit.bitmask)


    c = const.c.to('km/s').value
    init_redshift = eml_eml_par['KIN'][...,0]/c
    redshift = np.where(eml_eml_par['KIN'][...,0] == -999.0,0,init_redshift)
    redshift_mask = np.where(redshift == 0,1,0)

    line_center = emldb['restwave'][None,:] * (1+redshift)
    eml_eml_par['BMED'], eml_eml_par['RMED'], pos, eml_eml_par['EWCONT'], \
                eml_eml_par['EW'], eml_eml_par['EWERR'] \
                        = emission_line_equivalent_width(wave, flux, emldb['blueside'],
                                                         emldb['redside'], line_center,
                                                         eml_eml_par['FLUX'], mask=eml_mask,
                                                         redshift=redshift,
                                                         line_flux_err=eml_eml_par['FLUXERR'])


    
    # Get the stellar continuum that was fit for the emission lines, Masks bad pixels
    elcmask = eml_mask > 0
    for i,msk in enumerate(elcmask):  
        goodpix = np.arange(msk.size)[np.invert(msk)]
        if len(goodpix) != 0:
            start, end = goodpix[0], goodpix[-1]+1
        else:
            print('NO GOOD PIXELS: ',tags[i])
            start = 0
            end = len(msk)
        elcmask[i,start:end] = False
    el_continuum = np.ma.MaskedArray(model_flux - eml_flux, mask=elcmask)


    # Remeasure the emission-line moments with the new continuum



    print(z)

    new_elmom = EmissionLineMoments.measure_moments(momdb, wave, flux, continuum=el_continuum, redshift=z)

    line_center = momdb['restwave'][None,:] * (1+redshift)
    new_elmom['BMED'], new_elmom['RMED'], pos, new_elmom['EWCONT'], \
                new_elmom['EW'], new_elmom['EWERR'] \
                        = emission_line_equivalent_width(wave, flux, momdb['blueside'],
                                                         momdb['redside'], line_center,
                                                         new_elmom['FLUX'], mask=elcmask,
                                                         redshift=redshift,
                                                         line_flux_err=new_elmom['FLUXERR'])

    
    # Compare the summed flux and Gaussian-fitted flux for all the
    # fitted lines, This will help determine how good the quality of your fit is


    #-------------------------------------------------------------------
    # Measure the spectral indices
    if absindx_path is None or bhdindx_path is None:
        # Neither are defined, so we're done
        print('Elapsed time: {0} seconds'.format(time.perf_counter() - t))
        return

    # Setup the databases that define the indices to measure
    absdb = None if absindx_path is None else AbsorptionIndexDB(absindx_path)
    bhddb = None if bhdindx_path is None else BandheadIndexDB(bhdindx_path)

    # Remove the modeled emission lines from the spectra
    flux_noeml = flux - eml_flux
    redshift = stellar_kinematics[:,0] / const.c.to('km/s').value

    # Initialize the spectral indicies. these are measured first on the masked spectra, which is improper,
    # this is resolved by the following steps. 
    sp_indices = SpectralIndices.measure_indices(absdb, bhddb, wave, flux_noeml, ivar=ivar_spind,
                                                 redshift=redshift)


    #---------------------------------------------------------------------------------------------------------------------

    
    # Find all emission lines that are more than 50% masked and flag them as unfit.
    # The user can modify this masked threshold by modifying the masked_eml_percent varible 

    the masked_eml_percent = 0.5
    for i,tag in enumerate(tags):
        for j in range(len(emldb)):
            wavemin = emldb['blueside'][j][1]
            wavemax = emldb['redside'][j][0]
            wavelet = wave
            data_masklet = data_mask[i]
            masklet = np.where(eml_mask[i]!=0,1,0)
            fluxlet = flux[i]
            index = (wavelet>wavemin) & (wavelet<wavemax)
            if np.sum((index & fluxlet.mask) | (index & masklet))/np.sum(index) > the masked_eml_percent:
                eml_eml_par['FLUX'][i][j] = -999.0
                eml_eml_par['FLUXERR'][i][j] = -999.0
            if use_datamask:
                if np.sum(index & data_masklet)/np.sum(index) > the masked_eml_percent:
                    eml_eml_par['FLUX'][i][j] = -999.0
                    eml_eml_par['FLUXERR'][i][j] = -999.0
            

    #-------------------------------------------------------------------------------------------------------------------

    # Find all spectral indices that are more than 10% masked and flag them as unfit.
    # The user can modify this masked threshold by modifying the masked_spind_percent varible 

    masked_spind_percent = 0.1

    ivar_mask = np.where(ivar_spind<= 0.0, 1, 0)
    
    for i,tag in enumerate(tags):
        for j in range(len(absdb)):
            wavemin = absdb['blueside'][j][0]
            wavemax = absdb['redside'][j][1]
            wavelet = wave
            data_masklet = np.where(ivar_mask[i]!=0,1,0)
            masklet = np.where(eml_mask[i]!=0,1,0)
            fluxlet = flux[i]
            index = (wavelet>wavemin) & (wavelet<wavemax)
            if np.sum((index & fluxlet.mask) | (index & masklet))/np.sum(index) > masked_spind_percent:
                sp_indices['INDX'][i][j] = -999.0
                sp_indices['INDX_ERR'][i][j] = -999.0
            if use_datamask:
                if np.sum(index & data_masklet)/np.sum(index) > masked_spind_percent:
                    sp_indices['INDX'][i][j] = -999.0
                    sp_indices['INDX_ERR'][i][j] = -999.0

        for j in range(len(absdb),len(absdb)+len(bhddb)):
            wavemin = bhddb['blueside'][j-len(absdb)][0]
            wavemax = bhddb['redside'][j-len(absdb)][1]
            wavelet = wave
            data_masklet = np.where(ivar_mask[i]!=0,1,0)
            masklet = np.where(eml_mask[i]!=0,1,0)
            fluxlet = flux[i]
            index = (wavelet>wavemin) & (wavelet<wavemax)
            if np.sum((index & fluxlet.mask) | (index & masklet))/np.sum(index) > masked_spind_percent:
                sp_indices['INDX'][i][j] = -999.0
                sp_indices['INDX_ERR'][i][j] = -999.0
            if use_datamask:
                if np.sum(index & data_masklet)/np.sum(index) > masked_spind_percent:
                    sp_indices['INDX'][i][j] = -999.0
                    sp_indices['INDX_ERR'][i][j] = -999.0


    # remeasure the spectral indicies with an unmasked version of the spectra
    sp_indices_new = SpectralIndices.measure_indices(absdb, bhddb, wave, flux_noeml.data, ivar=ivar_spind,
                                                 redshift=redshift)


    # Replace only the spectral indicies that are less than 
    # 10% masked with the versions measured on the unmasked spectra.
    index = sp_indices['INDX'] != -999.0
    index_names = np.append(absdb['name'], bhddb['name'])


    for key in sp_indices.keys:
        try:
            sp_indices[key][index] = sp_indices_new[key][index]
        except:
            pass


    # replace stellar kinematics that are infinite with -999.0 the DAP error Code.
    stellar_kinematics = np.where(np.isfinite(stellar_kinematics), stellar_kinematics,-999.0)
    cont_par['SIGMACORR_SRES'] = np.where(np.isfinite(cont_par['SIGMACORR_SRES']), cont_par['SIGMACORR_SRES'],-999.0)

    print('Finished ' + plat)




    #-------------------------------------------------------------------------------------------------------------------
    # Save the fit to our custom eBOSS-DAP data structure fit files.
    with open('fits/'+d+'_spec_failed.txt','a') as f:
        for i,tag in enumerate(tags):

            if eml_fit_par['MASK'][i] != 0:
                f.write(tag + ' FAILED \n')
                continue
            npixfit = eml_fit_par['NPIXFIT'][i] #Number of pixels used by the fit.
            tplwgt = eml_fit_par['TPLWGT'][i] #Optimal weight of each template.
            multcoef = eml_fit_par['MULTCOEF'][i] #'Multiplicative polynomal coefficients.'
            rchi2 = eml_fit_par['RCHI2'][i] #Reduced chi-square of the fit
            fit_index = eml_eml_par['FIT_INDEX'][i] #The index in the fit database associated with each emission line.
            mask = eml_eml_par['MASK'][i] #Maskbit value for each emission line.
            emlflux = eml_eml_par['FLUX'][i] #The best-fitting flux of the emission line.
            emlfluxerr = eml_eml_par['FLUXERR'][i] #The error in the best-fitting emission-line flux
            emlkin = eml_eml_par['KIN'][i] #The best-fitting kinematics in each emission line
            emlkinerr = eml_eml_par['KINERR'][i] #The error in the best-fitting emission-line kinematics
            sigmacorr = eml_eml_par['SIGMACORR'][i] #Quadrature correction in the emission-line velocity dispersion
            sigmainst = eml_eml_par['SIGMAINST'][i] #Dispersion of the instrumental line-spread function at the location of each emission line.
            sigmatpl = eml_eml_par['SIGMATPL'][i] #Dispersion of the instrumental line-spread function of the emission-line templates.
            contmply = eml_eml_par['CONTMPLY'][i] #The value of any multiplicative polynomial included in the fit at the location of each emission line
            contrfit = eml_eml_par['CONTRFIT'][i] #The value of any extinction curve included in the fit at the location of each emission line
            line_pixc = eml_eml_par['LINE_PIXC'][i] #The integer pixel nearest the center of each emission line.
            amp = eml_eml_par['AMP'][i] #The best-fitting amplitude of the emission line.
            anr = eml_eml_par['ANR'][i] #The amplitude-to-noise ratio defined as the model amplitude divided by the median noise in the two (blue and red) sidebands defined for the emission line.
            line_nstat = eml_eml_par['LINE_NSTAT'][i] #The number of pixels included in the fit metric calculations (LINE_RMS, LINE_FRMS, LINE_CHI2) near each emission line.
            line_rms = eml_eml_par['LINE_RMS'][i] #The root-mean-square residual of the model fit near each emission line.
            line_frms = eml_eml_par['LINE_FRMS'][i] #The root-mean-square of the fractional residuals of the model fit near each emission line.
            line_chi2 = eml_eml_par['LINE_CHI2'][i] #The chi-square of the model fit near each emission line.
            bmed = eml_eml_par['BMED'][i] #The median flux in the blue sideband of each emission line
            rmed = eml_eml_par['RMED'][i] #The median flux in the red sideband of each emission line
            ewcont = eml_eml_par['EWCONT'][i] #The continuum value interpolated at the emission-line center (in the observed frame) used for the equivalent width measurement.
            ew = eml_eml_par['EW'][i] #The equivalent width of each emission line
            ewerr = eml_eml_par['EWERR'][i] #The error in the equivalent width of each emission line
            ew_mask = redshift_mask[i] #The equivalent width mask for no redshift
            int_bmed = new_elmom['BMED'][i] #The integrated median flux in the blue sideband of each emission line
            int_rmed = new_elmom['RMED'][i] #The integrated median flux in the red sideband of each emission line
            int_ewcont = new_elmom['EWCONT'][i] #The integrated continuum value interpolated at the emission-line center (in the observed frame) used for the equivalent width measurement.
            int_ew = new_elmom['EW'][i] #The integrated equivalent width of each emission line
            int_ewerr = new_elmom['EWERR'][i] #The integrated error in the equivalent width of each emission line

            sc_kin = stellar_kinematics[i]
            sc_vel = sc_kin[0]
            sc_sig = sc_kin[1]
            if sc_sig == -999.0:
                sc_vel = -999.0
            elif sc_vel == -999.0:
                sc_sig = -999.0

            sc_kin_err = stellar_kinematics_err[i]
            sc_vel_err = sc_kin_err[0]
            sc_sig_err = sc_kin_err[1]
            if sc_sig_err == -999.0:
                sc_vel_err = -999.0
            elif sc_vel_err == -999.0:
                sc_sig_err = -999.0

            sc_corr = cont_par['SIGMACORR_SRES'][i]

            fluxlet = flux[i]
            errlet = ferr[i]
            try:
                snr_arr = np.ma.masked_where(np.isnan(fluxlet/errlet), fluxlet/errlet)
                snr = np.ma.median(snr_arr)
            except:
                snr = -999.0




            flaglet = ignore_bits[i]
            bitmasked = any(flaglet)

            hdr = fits.Header()
            col9 = fits.Column(name='fit_index',format='D',array=fit_index) 
            col10 = fits.Column(name='mask',format='D',array=mask)
            col11 = fits.Column(name='emlflux',format='D',array=emlflux)
            col12 = fits.Column(name='emlfluxerr',format='D',array=emlfluxerr) 
            col15 = fits.Column(name='sigmainst',format='D',array=sigmainst) 
            col16 = fits.Column(name='sigmatpl',format='D',array=sigmatpl) 
            col18 = fits.Column(name='contmply',format='D',array=contmply) 
            col19 = fits.Column(name='contrfit',format='D',array=contrfit) 
            col20 = fits.Column(name='line_pixc',format='D',array=line_pixc) 
            col21 = fits.Column(name='amp',format='D',array=amp)
            col22 = fits.Column(name='anr',format='D',array=anr) 
            col23 = fits.Column(name='line_nstat',format='D',array=line_nstat) 
            col24 = fits.Column(name='line_rms',format='D',array=line_rms) 
            col25 = fits.Column(name='line_frms',format='D',array=line_frms) 
            col26 = fits.Column(name='line_chi2',format='D',array=line_chi2) 
            col27 = fits.Column(name='bmed',format='D',array=bmed)
            col28 = fits.Column(name='rmed',format='D',array=rmed) 
            col29 = fits.Column(name='ewcont',format='D',array=ewcont) 
            col30 = fits.Column(name='ew',format='D',array=ew)
            col31 = fits.Column(name='ewerr',format='D',array=ewerr) 
            col38 = fits.Column(name='ewmask',format='D',array=ew_mask) 
            col39 = fits.Column(name='int_bmed',format='D',array=int_bmed)
            col40 = fits.Column(name='int_rmed',format='D',array=int_rmed) 
            col41 = fits.Column(name='int_ewcont',format='D',array=int_ewcont) 
            col42 = fits.Column(name='int_ew',format='D',array=int_ew)
            col43 = fits.Column(name='int_ewerr',format='D',array=int_ewerr) 

            coldefs2 = fits.ColDefs([col9,col10,col11,col12,col15,col16,col18,col19,col20,
                                    col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col38,col39,col40,col41,col42,col43])             
            col13_l = fits.Column(name='eml_velocity',format='D',array=np.rot90(emlkin)[1])
            col13_r = fits.Column(name='eml_dispersion',format='D',array=np.rot90(emlkin)[0])
            col14_l = fits.Column(name='eml_velocity_err',format='D',array=np.rot90(emlkinerr)[1])
            col14_r = fits.Column(name='eml_dispersion_err',format='D',array=np.rot90(emlkinerr)[0])
            col_corr = fits.Column(name='dispersion_correction',format='D',array=sigmacorr)
            coldefs3 = fits.ColDefs([col13_l,col13_r,col14_l,col14_r,col_corr])   
            
            col32 = fits.Column(name='index_names',format='A15',array=index_names)
            col33 = fits.Column(name='INDX',format='D',array=sp_indices["INDX"][i])
            col34 = fits.Column(name='INDX_ERR',format='D',array=sp_indices["INDX_ERR"][i])
            col35 = fits.Column(name='INDX_CORR',format='D',array=sp_indices["INDX_CORR"][i])
            coldefs4 = fits.ColDefs([col32,col33,col34,col35])


            hdr['npixfit']=np.round(npixfit,decimals=3)
            hdr['rchi2']=np.round(rchi2,decimals=3)
            hdr['redshift'] = z[i]
            hdr['guess_dispersion'] = dispersion[i]
            hdr['RA'] = raarray[i]
            hdr['DEC'] = decarray[i]
            hdr['E(B-V)'] = ebvarr[i]
            hdr['sc_velocity'] = np.round(sc_vel,decimals=3)
            hdr['sc_dispersion'] = np.round(sc_sig,decimals=3)
            hdr['sc_velocity_err'] = np.round(sc_vel_err,decimals=3)
            hdr['sc_dispersion_err'] = np.round(sc_sig_err,decimals=3)
            hdr['sc_correction'] = np.round(sc_corr,decimals=3)
            try:
                hdr['SNR_median'] = np.round(snr,decimals=3)
            except:
                hdr['SNR_median'] = np.round(-999.0,decimals=3)
            hdr['tag'] = tag.zfill(16)
            hdr['bitmasked'] = bitmasked
            
            
            restwave = wave/(1+z[i])

            data_flux_ob = np.zeros(len(fluxes[i]),dtype=float)
            model_flux_ob = np.zeros(len(fluxes[i]),dtype=float)
            model_mask_ob = np.zeros(len(fluxes[i]),dtype=bool)
            eml_flux_ob = np.zeros(len(fluxes[i]),dtype=float)

            ind = np.where(wave_uni>=waves[i][0])[0][0]
            model_flux_ob[:-1] = model_flux.data[i,ind:ind+len(fluxes[i][:-1])] 
            eml_flux_ob[:-1] = eml_flux.data[i,ind:ind+len(fluxes[i][:-1])] 
            model_mask_ob[:-1] = model_flux.mask[i,ind:ind+len(fluxes[i][:-1])] #data_mask[i,ind:ind+len(fluxes[i][:-1])]
            model_mask_ob[0] = True 
            data_flux_ob[:-1] = flux.data[i,ind:ind+len(fluxes[i][:-1])] 

            
            hdul = fits.HDUList([fits.PrimaryHDU(data=model_flux_ob.data, header=hdr),
                          fits.ImageHDU(name='eml_flux',data = eml_flux_ob),
                          fits.ImageHDU(name='model_mask',data = model_mask_ob.astype(int)), 
                          fits.ImageHDU(name='eml_mask', data=eml_mask[i]),
                          fits.ImageHDU(name='tplwgt', data=tplwgt),
                          fits.ImageHDU(name='multcoef', data=multcoef),                          
                          fits.TableHDU.from_columns(coldefs2,name='eml'),
                          fits.TableHDU.from_columns(coldefs3,name='eml_kin'),
                          fits.TableHDU.from_columns(coldefs4,name='sp_ind'),
                          fits.ImageHDU(name='restwave', data=restwave)])

            hdul.writeto('fits/'+tag+'_fit.fits', overwrite=True)


#-----------------------------------------------------------------------------------------------



with open('fits/'+d+'_plate_failed.txt','a') as f2:
    for plate in plates:
        if len(plate.split('/')) > 1:
            plate = plate.split('/')[-1]
        else:
            plate = plate.split('\\')[-1]
        print('Running: '+plate)
        print(sys.argv[1],plate)


        try:
            main(sys.argv[1],plate,el_ew,el_z)
            print('SUCCESS')
        except Exception as e: 
            print(e)
            print(traceback.format_exc())
            f2.write(plate + ' FAILED: '+str(e)+' \n')
            print('FAILED')
        #'''