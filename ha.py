import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import requests
import matplotlib.image as img

def get_filenames(url):
    file_names = np.genfromtxt(url, usecols = (0), skip_header = 1, dtype = str, delimiter = ',')
    return file_names

def get_ha_map(files):
    hdulist = get_hdu(files)
    logcube = get_logcube(files)
    plate = hdulist['PRIMARY'].header['PLATEID']
    fiber = hdulist['PRIMARY'].header['IFUDSGN']
    Ha = hdulist['EMLINE_GFLUX'].data[18,...]
    errs=(hdulist['EMLINE_GFLUX_IVAR'].data)**-0.5
    Ha_err = errs[18,:,:]
    badpix = ((Ha/Ha_err) < 3)
    Ha[badpix] = np.nan
    eff_rad = hdulist['SPX_ELLCOO'].data[1]
    contours_i = logcube['IIMG'].data
    contours_i[badpix] = np.nan
    shape = Ha.shape[1]
    shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]
    
    #marvin(plate, fiber)
    
    a = fig.add_subplot(1,2,2)
    
    plt.imshow(Ha, cmap = "viridis", extent = shapemap, zorder = 1)
    plt.gca().invert_yaxis()
    plt.gca().contour(eff_rad*2,[2], extent = shapemap, colors='r', origin = 'upper', zorder = 3)
    plt.gca().contour(contours_i, 8, colors = 'black', alpha = 0.6, extent = shapemap, origin = 'upper', zorder = 2)
    cb = plt.colorbar(shrink = .7)
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    cb.set_label('Ha flux', rotation = 270, labelpad = 10)
    plt.show()
    #plt.savefig('')
    plt.close('all')
    
def get_logcube(iden):
    logcube = fits.open('/media/celeste/Hypatia/MPL7/LOGCUBES/manga-'+ str(iden) + '-LOGCUBE.fits.gz')
    return logcube

def get_hdu(iden):
    hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-' + str(iden) + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    return hdulist

filename = '/home/celeste/Documents/astro_research/summer_2018/test.txt'
files = get_filenames(filename)
print(files)

fig = plt.figure(figsize=(20,9), facecolor='white')

for x in range(0, 1):
    get_ha_map(files)
