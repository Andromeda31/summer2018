##lots of imports. Some are unnecessary but I left a lot just to be safe...
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
from astropy.table import Table, Column

import re
import csv

import os

import requests

import numpy as np
from scipy.stats import chi2


#Simply sets all the backgrounds of the plots to be white. I imagine this was necessary for one plot but we may as well do it for all of them
plt.rcParams['axes.facecolor'] = 'white'

#fun galaxy: 8332-12701
##ON:
##7977-12705

#Check later: 
#7977-9102


#Good examples of boring, normal galaxies:
#7990-6101

#Place-IFU (plate # - number of fibers, number bundle that is out
#of the number of bundles with that number of fibers)
##Code from Zach, do not touch!!

##Bunch of definitions for use later. Most are from Adam or Zach
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
    
#It will take an image and radius array (distarr; same size as the image) as arguments. Because of the way it was coded up, it works best when the distarr array is in units of pixels, i.e. don't use the R_re array.


def radial_profile(image,centre=None,distarr=None,mask=None,binwidth=2,radtype='unweighted'):
    '''
    image=2D array to calculate RP of.
    centre=centre of image in pixel coordinates. Not needed if distarr is given.
    distarr=2D array giving distance of each pixel from the centre.
    mask = 2D array, 1 if you want to include given pixels, 0 if not.
    binwidth= width of radial bins in pixels.
    radtype='weighted' or 'unweighted'. Weighted will give you the average radius of pixels in the bin. Unweighted will give you the middle of each radial bin.
    '''
    distarr=distarr/binwidth
    if centre is None:
        centre=np.array(image.shape,dtype=float)/2
    if distarr is None:
        y,x=np.indices(image.shape)
        distarr=np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    if mask is None:
        mask=np.ones(image.shape)
    rmax=int(np.max(distarr))
    r_edge=np.linspace(0,rmax,rmax+1)
    rp=np.zeros(len(r_edge)-1)*np.nan
    nums=np.zeros(len(r_edge)-1)*np.nan
    sig=np.zeros(len(r_edge)-1)*np.nan
    r=np.zeros(len(r_edge)-1)*np.nan
    for i in range(0,len(r)):
        rp[i]=np.nanmean(image[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))])
        nums[i]=len(np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False) & (np.isnan(image)==False))[0])
        sig[i]=np.nanstd((image[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))]))
        if radtype=='unweighted':
            r[i]=(r_edge[i]+r_edge[i+1])/2.0
        elif radtype=='weighted':
            r[i]=np.nanmean(distarr[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))])
    r=r*binwidth
    return r,rp,nums,sig


def scatter_if(x_in,y_in,condition=None,**kwargs):
    """
    scatter_if:
    Creates a scatter plot from two arrays, but only plots points if a condition is met. All other plt.scatter functionality should be retained.
    """
    x_ret=array_if(x_in,condition)
    y_ret=array_if(y_in,condition)
    if 'c' in kwargs and type(kwargs['c'])==np.ndarray:
        kwargs['c']=array_if(kwargs['c'],condition)
    ax=plt.scatter(x_ret,y_ret,**kwargs)
    return ax
    
def find_slope(X, Y):
    slope = ((X*Y).mean(axis=1) - X.mean()*Y.mean(axis=1)) / ((X**2).mean() - (X.mean())**2)
    return slope
    
    ##After hoefully downloading all the required fits files, this will read all the names
#file_names = os.listdir("/home/celeste/Documents/astro_research/keepers")
#file_names = os.listdir("/home/celeste/Documents/astro_research/fits_files")
#The filename with the list of all galaxies Adam found during our cuts. First line is the list, second line splits the list
filename = '/home/celeste/Documents/astro_research/thesis_git/Good_Galaxies_SPX_3_N2S2.txt'
file_names = np.genfromtxt(filename, usecols=(0), skip_header=1, dtype=str, delimiter=',')

#Back when we needed the mass data not from the data. These masses no longer match up with our list of galaxies
with open('/home/celeste/Documents/astro_research/thesis_git/mass_data.txt') as f:
    mass_data=[]
    for line in f:
        mass_data.append(line)


##creates the empty arrays to append the names of the files in the folder
plate_num = []
fiber_num = []
split = []


##Goes through all files in the folder. Splits up the galaxy names
for ii in range(0, len(file_names)):
    ##Removes all non alphanumeric characters and only leaves numbers and periods
    file_names[ii] = re.sub("[^0-9-]", "", file_names[ii])
    #print(file_names[ii])
    #print(file_names[ii][4:])
    #print(file_names[ii][:4])
    ##splits the two numbers into a plate number and fiber number
    one, two = (str(file_names[ii]).split('-'))
    ##splits the two numbers into a plate number and fiber number
    plate_num.insert(ii, one)
    fiber_num.insert(ii, two)
    
count_continue1 = 0
count_continue2 = 0
count_continue3 = 0
failed_maps = "failed maps\n"
failed_logcube = "failed_logcube\n"
failed_other = "failed_TYPERROR\n"

expec = []
calc = []
names = []
calc_err = []
expec_err = []
y_calc = []
y_expec = []
y_mass = []
mass_arr = []
mass_grad = []

for i in range(0, len(plate_num)): ##len(plate_num)
        #Just prints out the id of the galaxy. Prints out to the console so you can see what galaxy the code is on
        print(plate_num[i] + '-' + fiber_num[i])
        print("Index: " + str(i))

        
        #Tries to get the big fits file for all the data. Gets the specific file for the galaxy
        try:
            hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-' + plate_num[i] + '-' + fiber_num[i] + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
        except FileNotFoundError:
            failed_maps = failed_maps + str(plate_num[i]) + "-" + str(fiber_num[i]) + "\n"
            print("failed on the MAPS file.")
            print(failed_maps)
            print("------------------------------------------")
            continue
        
        
        #Gets the logcube fits file for the specific galaxy. 
        try:
            logcube = fits.open('/media/celeste/Hypatia/MPL7/LOGCUBES/manga-'+ str(plate_num[i])+ '-' + str(fiber_num[i]) + '-LOGCUBE.fits.gz')
        except FileNotFoundError:
            failed_logcube = failed_logcube + str(plate_num[i]) + "-" + str(fiber_num[i]) + "\n"
            print("failed on the LOGCUBE file.")
            print(failed_logcube)
            print("------------------------------------------")
            continue


        ##assigns the plate id based on what is in the data cube
        plateifu = hdulist['PRIMARY'].header['PLATEIFU']

        ##gets official plate number
        plate_number = hdulist['PRIMARY'].header['PLATEID']
        fiber_number = hdulist['PRIMARY'].header['IFUDSGN']
        
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
        r_Re = hdulist['SPX_ELLCOO'].data[1]
        drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
        obj = drpall[drpall['plateifu']==plateifu][0]
        Re = obj['nsa_elpetro_th50_r']
        
        mass = math.log10(obj['nsa_elpetro_mass'])-np.log10(.49)
        
        if mass > 10.75:
            m = -0.16878698011761817
            b = 8.92174257450408
        if mass > 10.50 and mass <= 10.75:
            m = -0.19145937059393828
            b = 8.898917413495317
        if mass > 10.25 and mass <= 10.50:
            m = -0.16938127151421675
            b = 8.825998835583249
        if mass > 10.00 and mass <= 10.25:
            m = -0.1762907767970223
            b = 8.713865209075324
        if mass > 9.75 and mass <= 10.00:
            m = -0.14756252418062643
            b = 8.59167993089605
        if mass > 9.50 and mass <= 9.75:
            m = -0.07514461331863775
            b = 8.36144939226056
        elif mass > 9.25 and mass <= 9.50:
            m = -0.05300368644036175
            b = 8.26602769508888
        else:
            m = -0.05059620593888811
            b = 8.147647436306206
        
        logOH12, logOH12error = n2s2_dopita16_w_errs(H_alpha, NII, s21, s22, Ha_err, n2_err, s21_err, s22_err)
        
        #finds the indices where the arrays are infinity
        idx = np.isfinite(r_Re.flatten()) & np.isfinite(logOH12.flatten())
        idx_err = np.isfinite(r_Re.flatten()) & np.isfinite(logOH12error.flatten())
        #Sorts the array
        indarr=np.argsort(r_Re.flatten()[idx])
        indarr_err = np.argsort(r_Re.flatten()[idx_err])
        
        #Fits a line to the sorted effective radius array
        yfit = [b + m * xi for xi in r_Re.flatten()[idx][indarr]]
        yfit_err = [b + m * xi for xi in r_Re.flatten()[idx_err][indarr_err]]
        
        slope_expected = (yfit[1]-yfit[0])/(r_Re.flatten()[idx][indarr][1]-r_Re.flatten()[idx][indarr][0])
        slope_expected_err = (yfit_err[1]-yfit_err[0])/(r_Re.flatten()[idx_err][indarr_err][1]-r_Re.flatten()[idx_err][indarr_err][0])
        
        y_int_expected = yfit[1]-slope_expected*r_Re.flatten()[idx][indarr][1]
        
        p_grad=np.poly1d([0.04477852,-1.32279522,12.93676181,-42.02882191])
        p_int=np.poly1d([ -0.03036036,0.81782123,-6.83415102,25.60820575])
        
        expec_mass = p_grad(mass)
        expec_mass_int = p_int(mass)
        
        
        def func(x, m, b):
            return m*x+b
        
        
        #Code from Adam. We use this to create our line fit
        rad_pix=hdulist['SPX_ELLCOO'].data[0,:,:]*2.0 #since there are 2 pixels/arcsec
        rad, rp, n, sig =radial_profile(image=logOH12,distarr=rad_pix, radtype = 'weighted')
        rad=rad/(2*Re) #This is now in units of Re.
        rad_err, rp_err, n_err, sig_err =radial_profile(image=logOH12error,distarr=rad_pix, radtype = 'weighted')
        rad_err = rad_err/(2*Re) #This is now in units of Re.
        valid = ~(np.isnan(rad) | np.isnan(rp) | np.isinf(rad) | np.isinf(rp) | ((rad < .5) | (rad > 2) ) | (n < 5))
        
        try:
            #creating the line fit
            popt, pcov = curve_fit(func, rad[valid], rp[valid], check_finite = True)
        except TypeError:
            print("Improper input: N=2 must not exceed M=0")
            failed_other = failed_other + str(plate_num[i]) + "-" + str(fiber_num[i]) + "\n"
            print("failed on the TYPE ERROR.")
            print("==========================================================================================")
            plt.close('all')
            count_continue=count_continue3+1
            continue
            
        try:
            #creating the line fit
            popt_err, pcov_err = curve_fit(func, rad_err[valid], rp_err[valid], check_finite = True)
        except TypeError:
            print("Improper input: N=2 must not exceed M=0")
            failed_other = failed_other + str(plate_num[i]) + "-" + str(fiber_num[i]) + "\n"
            print("failed on the TYPE ERROR.")
            print("==========================================================================================")
            plt.close('all')
            count_continue=count_continue3+1
            continue
        
        slope_calculated = (func(rad[valid], *popt)[1]-func(rad[valid], *popt)[0])/(rad[valid][1]-rad[valid][0])
        slope_calculated_err = (func(rad_err[valid], *popt_err)[1]-func(rad_err[valid], *popt_err)[0])/(rad_err[valid][1]-rad_err[valid][0])
        
        y_int_calculated = func(rad[valid], *popt)[1]-(slope_calculated*rad[valid][1])
        
        names.append(plateifu)
        calc.append(slope_calculated)
        expec.append(slope_expected)
        calc_err.append(slope_calculated_err)
        expec_err.append(slope_expected_err)
        y_calc.append(y_int_calculated)
        y_expec.append(y_int_expected)
        mass_arr.append(mass)
        mass_grad.append(expec_mass)
        y_mass.append(expec_mass_int)
        
        
#create the table
names = np.array(names)
expec = np.array(expec)
calc = np.array(calc)
calc_err = np.array(calc_err)
y_calc = np.array(y_calc)
y_expec = np.array(y_expec)
expec_err = np.array(expec_err)
print(names)
print(expec)
print(calc)
t=Table()

t['NAMES'] = Column(names, description = 'MaNGA PlateIFU' )
t['EXPECTED'] = Column(expec, description = 'gradient expected')
t['CALCULATED'] = Column(calc, description = 'gradient calculate')
t['CALCULATED_ERR'] = Column(calc_err, description = 'error on the calculated gradient')
t['EXPECTED_ERR'] = Column(expec_err, description = 'error on the expected gradient')
t['Y_INT_CALC'] = Column(y_calc, description = 'y intercept of the calculated gradient')
t['Y_INT_MASS'] = Column(y_expec, description = 'y intercept with the mass from Adam')
t['Y_INT_EXPEC'] = Column(y_expec, description = 'y intercept of the expected gradient')
t['MASS'] = Column(y_expec, description = 'just the mass of the galaxy')
t['MASS_GRAD'] = Column(y_expec, description = 'gradient from the mass from the polynomial from Adam')

t.write('/home/celeste/Documents/astro_research/summer_2018/slopesv4.fits')
