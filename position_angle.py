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

     # find the end point
     endx = length * math.sin(math.radians(angle)) + x
     endy = length * math.cos(math.radians(angle)) + y
     x = -length * math.sin(math.radians(angle)) + x
     y = -length * math.cos(math.radians(angle)) + y
     
     return x, endx, y, endy

def get_filenames(url):
    file_names = np.genfromtxt(url, usecols = (0), skip_header = 1, dtype = str, delimiter = ',')
    return file_names
    
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
    logcube = get_logcube(iden)
    plate_id = hdulist['PRIMARY'].header['PLATEIFU']
    plate_number = hdulist['PRIMARY'].header['PLATEID']
    fiber_number = hdulist['PRIMARY'].header['IFUDSGN']
    velocity, velocity_err = get_buncha_data(hdulist, logcube)
    
    errs=(hdulist['EMLINE_GFLUX_IVAR'].data)**-0.5
    fluxes = hdulist['EMLINE_GFLUX'].data
    Ha = fluxes[18,:,:]
    Ha_err = errs[18,:,:]
    
    contours_i = logcube['IIMG'].data
    contours_i_same = contours_i
    ew_cut = hdulist['EMLINE_GEW'].data[18,...]
    
    
    shape = (Ha.shape[1])
    shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]
    matplotlib.rcParams.update({'font.size': 20})
    drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
    r_Re = hdulist['SPX_ELLCOO'].data[1]
    pa = drpall[drpall['plateifu']==plate_id][0]['nsa_elpetro_phi']
    
    plot_image(plate_number, fiber_number)
    plot_kinematics(plate_id, velocity, velocity_err, contours_i, pa, (Ha/Ha_err))
    plot_iband(plate_number, fiber_number, contours_i, (Ha/Ha_err), pa, velocity)
    
    #plt.show()
    plt.savefig('/home/celeste/Documents/astro_research/position_angle/pa_' + str(plate_id) + '.png')
    print("finished with this one")
    plt.close('all')
    
    
def plot_kinematics(plateifu, velocity, velocity_err, contours_i, pa, err):
    global shapemap
    global r_Re
    global fig
    global ylim
    global xlim
    
    a = fig.add_subplot(1, 3, 3)
    
    badpix = err < 3
    contours_i[badpix] = np.nan
    
    badpix_vel = ((velocity_err) > 25)
    velocity[badpix_vel]=np.nan
    more_bad = (velocity / velocity_err) < 3
    
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
    imgplot = plt.imshow(velocity, origin = "lower", cmap = "RdYlBu_r", extent = shapemap, vmin = -vel_final, vmax = vel_final, zorder = 2)
    #adds the colorbar
    cb = plt.colorbar(shrink = .7, mappable = imgplot)
    #Adds a contour line for the one effective radius
    css = plt.gca().contour(r_Re*2,[2], extent=shapemap, colors='springgreen', origin = 'lower', zorder = 5)
    #adds the contors from the i band image
    csss=plt.gca().contour(contours_i, 8, colors = 'k', alpha = 0.6, extent = shapemap, zorder = 3)
    axes = plt.gca()
    ylim = axes.get_ylim()
    xlim = axes.get_xlim()
    
    #If all the velocities are less than zero, we make sure to get all of the correct velocities on the plot. 
    if ((vel_min <=0) and (vel_max <=0)):
        plt.clim(-vel_final, want)
    else:
        plt.clim(-vel_final,vel_final)
    cb.set_label('km/s', rotation = 270, labelpad = 25)
    a.set_facecolor('white')
    
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    
    x2, endx2, y2, endy2, bestAng = fit_kin(velocity, r_Re, offset = -90)
    
    print(x2, endx2)
    print(y2, endy2)
    
    plt.plot([y, endy], [x, endx], color = 'darkorchid', zorder = 4, label = 'PA from data')
    #plt.plot([-x, -endx], [-y, -endy], color = 'darkorchid', zorder = 5)
    plt.plot([-endy2, endy2],[-endx2, endx2], color = 'turquoise', zorder = 6, label = 'PA from kinematics')
    #plt.plot([x2], [y2], color = 'turquoise', marker = '.', zorder = 6, label = 'PA from kinematics')
    #plt.plot([-x2, -endx2], [-y2, -endy2], color = 'turquoise', zorder = 7)
    
    plt.legend(prop={'size': 12})
    
    plt.title("Gas Velocity")
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    
def fit_kin(velocity, r_Re, offset = 0):
    global shapemap
    dist = np.where(r_Re == np.min(r_Re))
    
    
    ybin, xbin = np.indices(velocity.shape)
    
    ybin = ybin - dist[1]
    xbin = xbin - dist[0]
    
    ybin = ybin.ravel()
    xbin = xbin.ravel()
    velocity_notravel = velocity
    velocity = velocity.ravel()
    
    [ybin, xbin, velocity] = reject_invalid([ybin, xbin, velocity])
    
    angBest, angErr, vSyst = fit_kinematic_pa(xbin, ybin,velocity-np.median(velocity), nsteps = 30, plot = False)
    
    yzero, xzero = find_new_center(shapemap, velocity_notravel, dist)

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
    print(x2, endx2)
    print(y2, endy2)
    return xzero, endx2, yzero, endy2, angBest
    
def find_new_center(shapemap, velocity, dist):
    left = shapemap[0]
    right = shapemap[3]
    size_of_vel = velocity.shape[0]
    width_of_shapemap = shapemap[1]-shapemap[0]
    
    yzero = left + width_of_shapemap/size_of_vel*(dist[0])
    xzero = right - width_of_shapemap/size_of_vel*(dist[1])
    
    print('xzero ' + str(xzero))
    print('yzero ' + str(yzero))
    
    #yzero = left + width_of_shapemap/size_of_vel*(dist[0]+.5)
    #xzero = right - width_of_shapemap/size_of_vel*(dist[1]+.5)
    
    return xzero, yzero
   
    
def plot_iband(plate_num, fiber_num, iband, err, pa, velocity):
    global shapemap
    global r_Re
    global fig
    global xlim
    global ylim
    a = fig.add_subplot(1, 3, 2)
    badpix = err < 3
    iband[badpix] = np.nan
    print(iband.shape)
    #iband[5][0] = 1000
    imgplot = plt.imshow(iband, cmap = "viridis", extent = shapemap, zorder = 1, origin = 'lower')
    css = plt.gca().contour(r_Re*2,[2], extent=shapemap, colors='r', origin = 'lower', zorder = 2, z = 2)
    #csss=plt.gca().contour(iband, 8, colors = 'black', alpha = 0.6, extent = shapemap, zorder = 3)
    #plt.gca().invert_yaxis()
    
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    
    x2, endx2, y2, endy2, bestAng = fit_kin(velocity, r_Re, offset = -90)
    
    print(x2, endx2)
    print(y2, endy2)
    
    dist = np.where(r_Re == np.min(r_Re))
    print(dist[0])
    print(dist[1])
    yzero, xzero = find_new_center(shapemap, velocity, dist)
    x, endx, y, endy = plot_point((xzero,yzero), pa+90)
    
    
    #data = plt.plot([x, endx], [y, endy], color = 'darkorchid', label = "PA from data", zorder = 5)
    data = plt.plot([y, endy], [x, endx], color = 'darkorchid', label = "PA from data: " + str(pa))
    #kinematics = plt.plot([x2, endx2],[y2, endy2], color = 'turquoise', zorder = 5, label = "PA from kinematics")
    #kinematics = plt.plot([y2],[x2], color = 'turquoise', marker = '.', zorder = 5, label = "PA from kinematics")
    kinematics = plt.plot([-endy2, endy2],[-endx2, endx2], color = 'turquoise', zorder = 6, label = "PA from kinematics: " + str(bestAng))
    #axes.invert_yaxis()
    plt.legend(prop={'size': 12})

    


    plt.title("i-band Image")
    
def plot_image(plate_num, fiber_num):
    r = requests.get('https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_4_3/' + str(plate_num) + '/stack/images/' + str(fiber_num) + '.png', auth=('sdss', '2.5-meters'))

    ##Saves the image
    with open('/home/celeste/Documents/astro_research/astro_images/marvin_images/' + str(plate_num) + '-' + str(fiber_num) + '.png', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
	        fd.write(chunk)
    a = fig.add_subplot(1,3,1)
    try:
        image = img.imread('/home/celeste/Documents/astro_research/astro_images/marvin_images/' + str(plate_num) + '-' + str(fiber_num) + '.png')
    except ValueError:
        print("No image.")
    #plt.subplot(121)
    
    
    imgplot = plt.imshow(image)
    
shapemap = [0,0,0,0]
fig = plt.figure(figsize=(28,7), facecolor='white')
r_Re = []
ylim = 0
xlim = 0
    
filename = '/home/celeste/Documents/astro_research/thesis_git/Good_Galaxies_SPX_3_N2S2.txt'
files = get_filenames(filename)

#files = ['7443-12702']

for x in range(0, len(files)):
    fig = plt.figure(figsize=(30,9), facecolor='white')
    get_plot(files[x])
    plt.close('all')
