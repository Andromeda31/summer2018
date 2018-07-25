from pafit.fit_kinematic_pa import fit_kinematic_pa
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
import scipy
from astropy.table import Table, Column

import re
import csv

import os

import requests

import numpy as np
from scipy.stats import chi2

def reject_invalid(variables,bad_flag=None):
    '''
    This function takes in a list of variables stored in numpy arrays and returns these arrays where there are no nans.
    variables=[variable1,variable2,variable3...]
    bad_flag=a value that denotes bad data e.g. -999
    '''
    if type(variables)!=list:
        print("please input a list of numpy arrays")
        return
    good=np.ones(variables[0].shape)
    for var in variables:
        if type(var[0])!=str and type(var[0])!=np.str_:
            bad=np.where((np.isnan(var)==True) | (np.isinf(var)==True) | (var==bad_flag))
            good[bad]=0
    var_out=[]
    for var in variables:
        var_out.append(var[good==1])
    return var_out

def plot_point(point, angle, length=100):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     Will plot the line on a 10 x 10 plot.
     '''

     # unpack the first point
     x, y = point
     #angle = 0
     
     print(point)
     print('angle: ' +  str(angle))   
     # find the end point
     endx = length * math.sin(math.radians(angle)) + x
     endy = length * math.cos(math.radians(angle)) + y
     beginx = -length * math.sin(math.radians(angle)) + x
     beginy = -length * math.cos(math.radians(angle)) + y
     
     print(endx)
     print(endy)
     print(beginx)
     print(beginy)
     
     return beginx, endx, beginy, endy

def get_filenames(url):
    file_names = np.genfromtxt(url, usecols = (0), skip_header = 1, dtype = str, delimiter = ',')
    return file_names
    
def get_hdu(iden):
    try:
        hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-' + str(iden) + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    except FileNotFoundError:
        print("failed on the MAPS file.")
        print("------------------------------------------")
        hdulist = 0
    return hdulist
    
def get_logcube(iden):
    try:
        logcube = fits.open('/media/celeste/Hypatia/MPL7/LOGCUBES/manga-'+ str(iden) + '-LOGCUBE.fits.gz')
    except FileNotFoundError:
        print("failed on the LOGCUBE file.")
        print(failed_logcube)
        print("------------------------------------------")
    return logcube
    
def get_buncha_data(hdulist, logcube):
    vel = hdulist['EMLINE_GVEL'].data[18,...]
    vel_err = (hdulist['EMLINE_GVEL_IVAR'].data[18,...])**-0.5
    return vel, vel_err
    
def get_plot(iden):
    global shapemap
    global r_Re
    global fig
    print('Doing file ' + str(iden) + '.') 
    hdulist = get_hdu(iden)
    should_save = True
    if hdulist == 0:
        should_save = False
        return 0, 0, 0, 0, 0, 0, 0, should_save
    logcube = get_logcube(iden)
    plate_id = hdulist['PRIMARY'].header['PLATEIFU']
    velocity, velocity_err = get_buncha_data(hdulist, logcube)
    velocity_mask = hdulist['EMLINE_GFLUX_MASK'].data[18,...]
    velocity[velocity_mask != 0] = np.nan
    stel_vel = hdulist['STELLAR_VEL'].data
    stel_vel_err = (hdulist['STELLAR_VEL_IVAR'].data)**(-0.5)
    star_v_mask = hdulist['STELLAR_VEL_MASK'].data
    stel_vel[star_v_mask != 0] = np.nan
    

    
    errs=(hdulist['EMLINE_GFLUX_IVAR'].data)**-0.5
    fluxes = hdulist['EMLINE_GFLUX'].data
    Ha = fluxes[18,:,:]
    Ha_err = errs[18,:,:]
    
    contours_i = logcube['IIMG'].data
    
    
    shape = (Ha.shape[1])
    shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]
    matplotlib.rcParams.update({'font.size': 20})
    drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
    r_Re = hdulist['SPX_ELLCOO'].data[1]
    pa = drpall[drpall['plateifu']==plate_id][0]['nsa_elpetro_phi']
    
    stellar_kin_pa, gas_kin_pa, pa_from_data, stellar_kin_pa_err, gas_kin_pa_err = plot_kinematics(plate_id, velocity, velocity_err, contours_i, pa, (Ha/Ha_err), stel_vel, stel_vel_err)
    
    if pa_from_data != np.nan:
        pa_from_data = pa_from_data + 90
    else:
        should_save = False
        return 1,1,1,1,1,1, plate_id

    pa_err = np.nan
    
    print('stellar kin: ' + str(stellar_kin_pa))
    print('gas kin: ' + str(gas_kin_pa))
    print('data kin: ' + str(pa_from_data))
    print("finished with this one")
    plt.close('all')
    
    
    
    return stellar_kin_pa, gas_kin_pa, pa_from_data, stellar_kin_pa_err, gas_kin_pa_err, pa_err, plate_id, should_save
    

    
def plot_kinematics(plateifu, velocity, velocity_err, contours_i, pa, err, stel_vel, stel_vel_err):
    global shapemap
    global r_Re
    global fig
    global ylim
    global xlim
    
    
    
    badpix = err < 3
    contours_i[badpix] = np.nan
    
    badpix_stelvel = ((stel_vel_err) > 25)
    stel_vel[badpix_stelvel] = np.nan
    badpix_vel = ((velocity_err) > 25)
    velocity[badpix_vel]=np.nan
    more_bad = (velocity / velocity_err) < 3
    #velocity[more_bad] = np.nan
    
    
    #Finds the 95th and 5th percentiles
    vel_min = np.nanpercentile(velocity, 5)
    vel_max = np.nanpercentile(velocity, 95)
    
    #Want to find the largest absolute value of min or max. We will use this to create a consistent velocity map
    if abs(vel_min) > abs(vel_max):
        vel_final = abs(vel_min)
        want = vel_max
    else:
        vel_final = abs(vel_max)
        want = vel_min
        
    dist = np.where(r_Re == np.min(r_Re))
    
    yzero, xzero = find_new_center(shapemap, velocity, dist)
    x, endx, y, endy = plot_point((xzero,yzero), pa-90)
    
    
    #plots the velocity map

    #adds the colorbar

    #Adds a contour line for the one effective radius

    #adds the contors from the i band image

    
    #If all the velocities are less than zero, we make sure to get all of the correct velocities on the plot. 
    '''
    print('--------------2-----------------')
    '''
    x2, endx2, y2, endy2, bestAng, bestAng_err2 = fit_kin(velocity, r_Re, offset = -90)
    '''
    print('--------------3-----------------')
    #print(stel_vel)
    #print(stel_vel.shape)
    #print(r_Re)
    '''
    x3, endx3, y3, endy3, bestAng_stelvel, bestAng_stelvel_err = fit_kin(stel_vel, r_Re, offset = 90)
    
    if x == np.nan:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if x2 == np.nan:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if x3 == np.nan:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    
    return bestAng_stelvel, bestAng, pa-90, bestAng_stelvel_err, bestAng_err2
    
def fit_kin(velocity, r_Re, offset = 0):
    global shapemap
    dist = np.where(r_Re == np.min(r_Re))
    
    
    ybin, xbin = np.indices(velocity.shape)
    
    ybin = ybin - dist[0]
    xbin = xbin - dist[1]
    
    ybin = ybin.ravel()
    xbin = xbin.ravel()
    velocity_notravel = velocity
    velocity = velocity.ravel()
    
    [ybin, xbin, velocity] = reject_invalid([ybin, xbin, velocity])
    
    yzero, xzero = find_new_center(shapemap, velocity_notravel, dist)

    try:
        angBest, angErr, vSyst = fit_kinematic_pa(xbin, ybin, velocity - velocity_notravel[(dist[1][0])][(dist[0][0])], nsteps = 361, plot = False)
        angErr = angErr/3.
    except ValueError:
        print("Have empty xbin and ybins. Must continue")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    except scipy.spatial.qhull.QhullError:
        print("not enough points(2) to construct initial simplex (need 4)blah2")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    

    '''
    print("left: " + str(left))
    print("right: " + str(right))
    print("size_of_vel: " + str(size_of_vel))
    print("width_of_shapemap: " + str(width_of_shapemap))
    print("xzero: " + str(xzero))
    print("yzero: " + str(yzero))
    print("dist0: " + str(dist[0]))
    print("dist1: " + str(dist[1]))
    '''
    

    x2, endx2, y2, endy2 = plot_point((xzero, yzero), angBest+offset)

    #return x2, endx2, y2, endy2
    return x2, endx2, y2, endy2, angBest, angErr
    
def find_new_center(shapemap, velocity, dist):
    left = shapemap[0]
    right = shapemap[3]
    size_of_vel = velocity.shape[0]
    width_of_shapemap = shapemap[1]-shapemap[0]
    
    yzero = left + width_of_shapemap/size_of_vel*(dist[0])
    xzero = right - width_of_shapemap/size_of_vel*(dist[1])
    
    #yzero = left + width_of_shapemap/size_of_vel*(dist[0]+.5)
    #xzero = right - width_of_shapemap/size_of_vel*(dist[1]+.5)
    
    return xzero, yzero
   
    

def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    return int(round(x) - .5) + (x > 0)
    
    
shapemap = [0,0,0,0]
fig = plt.figure(figsize=(35,11), facecolor='white')
r_Re = []
ylim = 0
xlim = 0
stel = []
gas = []
data = []
gal_id = []
gas_error = []
stel_error = []
    

filename = '/home/celeste/Documents/astro_research/thesis_git/Good_Galaxies_SPX_3_N2S2.txt'
files = get_filenames(filename)

how_many_bad = 0
bad_gala = []

print(len(files))
asdfsa


for x in range(0, len(files)):
    print("number " + str(x+1))
    fig = plt.figure(figsize=(35,11), facecolor='white')
    stel_pa, gas_pa, pa, stel_err, gas_err, pa_err, plate_id, should_save = get_plot(files[x])
    '''
    if stel_pa == 0:
        if gas_pa == 0:
            if pa == 0:
                stel_pa = np.nan
                gas_pa = np.nan
                pa = np.nan
                stel_pa_err = np.nan
                gas_pa_err = np.nan
                pa_err = np.nan
                print("failed to get hdulist.")
                continue
    if stel_pa == 1:
        if gas_pa ==1:
            if pa == 1:
                stel_pa = np.nan
                gas_pa = np.nan
                pa = np.nan
                stel_pa_err = np.nan
                gas_pa_err = np.nan
                pa_err = np.nan
                print("failed on that weird error on line 257.")
                continue
    '''
    if should_save == False:
        print('maaaaaaaaaaaaaaaaaaaaaaaaaa')
        how_many_bad = how_many_bad + 1
        bad_gala.append(plate_id)
        continue
    print("stellar_pa: " + str(stel_pa))
    plt.close('all')
    stel.append(stel_pa)
    gas.append(gas_pa)
    data.append(pa)  
    gas_error.append(gas_err)
    stel_error.append(stel_err)
    gal_id.append(plate_id)
        
        
stel = np.array(stel)
gas = np.array(gas)
data = np.array(data)
gal_id = np.array(gal_id)
stel_error = np.array(stel_error)
gas_error = np.array(gas_error)
bad = np.array([how_many_bad])
bad_gals = np.array(bad_gals)
        
t = Table()
t['STEL_PA'] = Column(stel, description = 'Stellar position angle' )
t['STEL_PA_ERR'] = Column(stel_error, description = 'Stellar position angle error' )
t['GAS_PA'] = Column(gas, description = 'Position angle calculated from the gas')
t['GAS_PA_ERR'] = Column(gas_error, description = 'Position angle calculated from the gas error')
t['PA'] = Column(data, description = 'Position angle from the MaNGA data')
t['GALAXY_ID'] = Column(gal_id, description = 'galaxy ID')
t['NUM_BAD_GALS'] = Column(bad, description = "For Celeste's testing, ignore")
t['BAD_GALS_NAMES'] = Column(bad_gals, description = "For Celeste's testing, ignore")

t.write('/home/celeste/Documents/astro_research/summer_2018/pa_datav5.fits')
