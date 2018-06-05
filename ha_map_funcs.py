import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
import numpy as np
import astropy.table as t
import matplotlib.image as img
from scipy.optimize import newton
from pathlib import Path
import math
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import numpy.random as rnd
from matplotlib import patches
import sys
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

import re
import csv

import os

import requests

import numpy as np
from scipy.stats import chi2

def get_line_ew(maps,key,sn=3.):
    sew_hdu=maps['EMLINE_GFLUX']
    sew_ivar_hdu=maps['EMLINE_GFLUX_IVAR']

    # get a mapping from eline key to channel key
    v2k={v:k for(k,v)in sew_hdu.header.items()}
    # get a mapping from channel key to channel
    cstring2ix=lambda s:int(s[1:])-1

    ix=cstring2ix(v2k[key])

    ew=sew_hdu.data[ix,...]
    ew_ivar=sew_ivar_hdu.data[ix,...]
    snr= ew*np.sqrt(ew_ivar)
    ew_mask=(snr<sn)

    return np.ma.array(ew,mask=ew_mask)
    

def pp04_o3n2_w_errs(n2,ha,o3,hb,n2_err,ha_err,o3_err,hb_err):
    o3n2=(o3/hb) / (n2/ha)
    lo3n2=np.log10(o3n2)
    o3hb_err=(o3/hb)*np.sqrt((o3_err/o3)**2 + (hb_err/hb)**2)
    n2ha_err=(n2/ha)*np.sqrt((n2_err/n2)**2 + (ha_err/ha)**2)
    o3n2_err=o3n2*np.sqrt((o3hb_err/(o3/hb))**2 + ((n2ha_err)/(n2/ha))**2)
    lo3n2_err=o3n2_err/(o3n2*np.log(10))
    met=8.73-0.32*lo3n2
    met_err=0.32*lo3n2_err
    return met,met_err
    
def is_sf_array(n2,ha,o3,hb):
    '''
    Checks whether arrays of line fluxes come from star formation based on BPT diagnostics
    returns 1 if spaxel is star-forming, 0 if non-star forming and nan if not determinable.
    '''
    issf=np.zeros(n2.shape)
    x=np.log10(n2/ha)
    y=np.log10(o3/hb)
    issf[np.where(x>0)]=0
    goodpix=np.where((y<(0.61/(x-0.47))+1.19) & (y<(0.61/(x-0.05))+1.3) & (x<0.0))
    badpix=np.where((np.isnan(x)==True) | (np.isnan(y)==True) | (np.isinf(x)==True) | (np.isinf(y)==True))
    issf[badpix]=np.nan
    issf[goodpix]=1
    return issf
    
def n2s2_dopita16_w_errs(ha,n22,s21,s22,ha_err,n22_err,s21_err,s22_err):
    '''
    N2S2 metallicity diagnostic from Dopita et al. (2016)
    includes a calculation of the errors
    '''
    y=np.log10(n22/(s21+s22))+0.264*np.log10(n22/ha)
    s2=s21+s22
    s2_err=np.sqrt(s21_err**2 + s22_err**2)
    met_err=(1.0/np.log(10)) * np.sqrt( (1+0.264**2)*(n22_err/n22)**2 + (s2_err/s2)**2 + (ha_err/ha)**2 )
    met=8.77+y
    return met, met_err



##Takes an input array and returns the values if a condition is met. Basically a glorified call to numpy.where

def array_if(array,condition=None):
    array1=np.array(array)
    if condition is None:
        return array1
    ret=array1[np.where(condition)]
    return ret
    
def get_filenames(url):
    file_names = np.genfromtxt(url, usecols = (0), skip_header = 1, dtype = str, delimiter = ',')
    return file_names
    
def get_ha_map(iden):
    print('Doing file ' + str(iden) + '.') 
    hdulist = get_hdu(iden)
    logcube = get_logcube(iden)
    plate_id = hdulist['PRIMARY'].header['PLATEIFU']
    plate_number = hdulist['PRIMARY'].header['PLATEID']
    fiber_number = hdulist['PRIMARY'].header['IFUDSGN']
    Ha, Hb, snmap, fluxes, Ha_err, OIII, o3_err, H_beta, Hb_err, n2_err, NII, s21, s22, s21_err, s22_err, velocity, velocity_err = get_buncha_data(hdulist, logcube)
    
    
    contours_i = logcube['IIMG'].data
    contours_i_same = contours_i
    ew_cut = hdulist['EMLINE_GEW'].data[18,...]
    
    logOH12, logOH12error = n2s2_dopita16_w_errs(Ha, NII, s21, s22, Ha_err, n2_err, s21_err, s22_err)
    is_starforming = is_sf_array(NII,Ha,OIII, H_beta)
    
    shape = (Ha.shape[1])
    shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]
    matplotlib.rcParams.update({'font.size': 20})
    drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
    r_Re = hdulist['SPX_ELLCOO'].data[1]
    
    final_flux = get_final_flux(Ha, Ha_err, r_Re, contours_i, shapemap)
    plot_figure(final_flux)
    plt.show()
    print("finished with this one")
    plt.close('all')
    
def plot_figure(final_flux):
    plt.gca().invert_yaxis()
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    #Adds a colorbar
    cb = plt.colorbar(shrink = .7)
    cb.set_label('H-alpha flux [$10^{17} {erg/s/cm^2/pix}$]', rotation = 270, labelpad = 25)
    
def get_final_flux(Ha_2d, Ha_err, r_Re,contours_i, shapemap):
    badpix = ((Ha_2d/Ha_err) < 3)
    Ha_2d[badpix]=np.nan
    contours_i[badpix]=np.nan
    imgplot = plt.imshow(Ha_2d, cmap = "viridis", extent = shapemap, zorder = 1)
    csss=plt.gca().contour(contours_i, 8, colors = 'k', alpha = 0.6, extent = shapemap, origin = 'upper', zorder = 4)
    one_eff = plt.gca().contour(r_Re*2,[2],extent=shapemap, colors='r', origin = 'upper', zorder = 2, z = 2)
    return imgplot, csss, one_eff
    
def get_buncha_data(hdulist, logcube):
    ##gets the hydrogen alpha and hydrogen beta data, and all other necessary lines
    Ha = hdulist['EMLINE_GFLUX'].data[18,...]
    Hb = hdulist['EMLINE_GFLUX'].data[1,...]
    snmap = hdulist['SPX_SNR'].data
    fluxes = hdulist['EMLINE_GFLUX'].data
    #Errors is raised to the -1/2 power
    errs=(hdulist['EMLINE_GFLUX_IVAR'].data)**-0.5
    H_alpha = fluxes[18,:,:]
    Ha = H_alpha
    Ha_err = errs[18,:,:]
    OIII = fluxes[13,:,:]
    o3_err = errs[13,:,:]
    H_beta = fluxes[11,:,:]
    Hb_err = errs[11,:,:]
    n2_err = errs[19,:,:]
    NII = fluxes[19,:,:]
    s21 = fluxes[20,:,:]
    s22 = fluxes[21,:,:]
    s21_err = errs[20,:,:]
    s22_err = errs[21,:,:]
    velocity = hdulist['EMLINE_GVEL'].data[18,...]
    velocity_err = (hdulist['EMLINE_GVEL_IVAR'].data[18,...])**-0.5
    return Ha, Hb, snmap, fluxes, Ha_err, OIII, o3_err, H_beta, Hb_err, n2_err, NII, s21, s22, s21_err, s22_err, velocity, velocity_err
    
    
def get_hdu(iden):
    try:
        hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-' + str(iden) + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    except FileNotFoundError:
        print("failed on the MAPS file.")
        print("------------------------------------------")
    return hdulist
    
def get_logcube(iden):
    try:
        logcube = fits.open('/media/celeste/Hypatia/MPL7/LOGCUBES/manga-'+ str(iden) + '-LOGCUBE.fits.gz')
    except FileNotFoundError:
        print("failed on the LOGCUBE file.")
        print(failed_logcube)
        print("------------------------------------------")
    return logcube

    
filename = '/home/celeste/Documents/astro_research/thesis_git/Good_Galaxies_SPX_3_N2S2.txt'
files = get_filenames(filename)

print(files)

for x in range(0, len(files)):
    get_ha_map(files[x])
