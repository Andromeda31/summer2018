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
     endy = length * math.sin(math.radians(angle))
     endx = length * math.cos(math.radians(angle))
     
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
    plot_iband(plate_number, fiber_number, contours_i, (Ha/Ha_err), pa)
    
    plt.show()
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
        
    x, endx, y, endy = plot_point((0,0), pa-90)
    
    
    #plots the velocity map
    imgplot = plt.imshow(velocity, origin = "lower", cmap = "RdYlBu_r", extent = shapemap, vmin = -vel_final, vmax = vel_final, zorder = 2)
    #Adds a contour line for the one effective radius
    css = plt.gca().contour(r_Re*2,[2],extent=shapemap, colors='springgreen', origin = 'lower', zorder = 5)
    #adds the contors from the i band image
    csss=plt.gca().contour(contours_i, 8, colors = 'k', alpha = 0.6, extent = shapemap, zorder = 3)
    axes = plt.gca()
    ylim = axes.get_ylim()
    xlim = axes.get_xlim()
    
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    
    xbin, ybin = np.random.uniform(low=[-want, -want+10], high=[want, want - 10], size = velocity.shape).T
    
    angBest, angErr, vSyst = fit_kinematic_pa(xbin, ybin,velocity-np.median(velocity))
    x2, endx2, y2, endy2 = plot_point((0,0), angBest-90)
    
    plt.plot([x, endx], [y, endy], color = 'darkorchid', zorder = 4)
    plt.plot([x2, endx2], [y2, endy2], color = 'turquoise', zorder = 5)
    
    plt.title("Gas Velocity")
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    
    #adds the colorbar
    cb = plt.colorbar(shrink = .7)
    
    #If all the velocities are less than zero, we make sure to get all of the correct velocities on the plot. 
    if ((vel_min <=0) and (vel_max <=0)):
        plt.clim(-vel_final, want)
    else:
        plt.clim(-vel_final,vel_final)
    cb.set_label('km/s', rotation = 270, labelpad = 25)
    a.set_facecolor('white')
    
def plot_iband(plate_num, fiber_num, iband, err, pa):
    global shapemap
    global r_Re
    global fig
    global xlim
    global ylim
    a = fig.add_subplot(1, 3, 2)
    badpix = err < 3
    iband[badpix] = np.nan
    print(shapemap)
    imgplot = plt.imshow(iband, cmap = "viridis", extent = shapemap, zorder = 1)
    css = plt.gca().contour(r_Re*2,[2], extent=shapemap, colors='r', origin = 'upper', zorder = 2, z = 2)
    #plt.gca().invert_yaxis()
    
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    
    x, endx, y, endy = plot_point((0,0), pa)
    
    plt.plot([y, endy], [x, endx], color = 'darkorchid')
    
    axes.invert_yaxis()

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
    lum_img = image[:,:,0]
    #plt.subplot(121)
    
    
    imgplot = plt.imshow(image)
    
shapemap = [0,0,0,0]
fig = plt.figure(figsize=(30,9), facecolor='white')
r_Re = []
ylim = 0
xlim = 0
    
filename = '/home/celeste/Documents/astro_research/thesis_git/Good_Galaxies_SPX_3_N2S2.txt'
files = get_filenames(filename)

for x in range(0, len(files)):
    fig = plt.figure(figsize=(30,9), facecolor='white')
    get_plot(files[x])
    plt.close('all')
