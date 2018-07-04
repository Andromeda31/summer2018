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
    
#print(plate_num[0] + "-" + fiber_num[0])
#print(file_names[0])


    ##Main loop over all the plates
    
"""
Bad Plots?

8445-3701
8332-1902
8309-3703

"""
    
#plate_num=['9194']
#fiber_num = ['12701']
#plate_num = ['8252', '8338', '8568', '9865']
#fiber_num = ['12705', '12701', '12702', '12705']

##Just some counts I implemented to keep track of galaxies that failed. These don't come into play until some exception calls and then are just printed out immediately at the very end of the run
count_continue1 = 0
count_continue2 = 0
count_continue3 = 0
failed_maps = "failed maps\n"
failed_logcube = "failed_logcube\n"
failed_other = "failed_TYPERROR\n"

iden = []
slope1 = []
slope2 = []


#####################################################################################

###The Big Loop

#####################################################################################



for i in range(500, len(plate_num)): ##len(plate_num)
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
        plate_id = hdulist['PRIMARY'].header['PLATEIFU']

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
        
        
        
        #I band for contours
        contours_i = logcube['IIMG'].data
        contours_i_same = contours_i
        
        #Gets the velocity values
        velocity = hdulist['EMLINE_GVEL'].data[18,...]
        velocity_err = (hdulist['EMLINE_GVEL_IVAR'].data[18,...])**-0.5
        
        #Equivalent width line call
        ew_cut = hdulist['EMLINE_GEW'].data[18,...]
        

        ##Imports the thingy we need to get the images of the galaxy without having to download directly all the pictures. This also bypasses the password needed
        
        import requests
        
        #These next commented out lines can be uncommented IF all the images of the galaxies are not already downloaded. It slows down the code a lot to run this for every galaxy. It gets the image from the sdss website and then save the image on the computer.
        
        
        
        r = requests.get('https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_3_1/' + str(plate_num[i]) + '/stack/images/' + str(fiber_num[i]) + '.png', auth=('sdss', '2.5-meters'))
        

        ##Saves the image
        with open('/home/celeste/Documents/astro_research/astro_images/marvin_images/' + str(plate_id) + '.png', 'wb') as fd:
	        for chunk in r.iter_content(chunk_size=128):
		        fd.write(chunk)
		        
		 
        

        ##Calculates the line ratios 
        O_B = OIII/H_beta

        N_A = NII/H_alpha

        R = O_B/N_A	
        logR = np.log10(R)
        
        


        #Very early indicator work
        c0 = 0.281
        c1 = (-4.765)
        c2 = (-2.268)

        cs = np.array([c0, c1, c2])


        ##A lambda function, do not touch!
        x2logR = lambda x, cs: np.sum((c*x**p for p,c in zip(np.linspace(0, len(cs)-1, len(cs)), cs)))

        x2logR_zero = lambda x, cs, logR: x2logR(x, cs)-logR-0.001

        ##takes the log of the OH12 array
        #logOH12 = np.ma.array(np.zeros(logR.shape),mask=logR.mask)
        
        logOH12_old = 8.73-0.32*np.log10(R)
        
        #Calculates the NEW indicator values. Gives you back the logOH+12 values and their error. Function is from Adam. 
        logOH12, logOH12error = n2s2_dopita16_w_errs(H_alpha, NII, s21, s22, Ha_err, n2_err, s21_err, s22_err)
        #Also gets the spaxels that are star forming. Another function from Adam.
        is_starforming = is_sf_array(NII,H_alpha,OIII, H_beta)
        
	    
		
        ##Finds the standard deviation and mean for future use	
        std_dev = np.std(Ha)
        mean = np.mean(Ha)	
		
        ##if Ha deviates too much from the mean it is removed		
        for j in range(len(Ha)):
	        for k in range(len(Ha[0])):
		        if (Ha[j][k] > std_dev*20+mean):
			        Ha[j][k] = np.nan
		

        ##Creates a shape that is the same size as the h-alpha array	
        shape = (Ha.shape[1])
        shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]

        ##Changes the font size
        matplotlib.rcParams.update({'font.size': 20})
        #Creates the figure, sets the figure size, also sets the facecolor
        fig = plt.figure(figsize=(30,18), facecolor='white')


        ##places text on the plot
        plateifu = plate_id
        
        
        
        #Gets the image saved earlier or from the computer
        image = img.imread('/home/celeste/Documents/astro_research/astro_images/marvin_images/' + str(plateifu) + '.png')
        
        #Reads the drpall file. Update this file as new ones are released
        drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
        #Gets the radius of the galaxy
        r = hdulist['SPX_ELLCOO'].data[0, ...]
        #Gets the data from the object that corresponds to the current plate ID
        obj = drpall[drpall['plateifu']==plateifu][0]
        Re = obj['nsa_elpetro_th50_r']
        pa = obj['nsa_elpetro_phi']
        ba = obj['nsa_elpetro_ba']
        #radius of each spec. Calculates the effective radius
        r_Re = r/Re	
        r_Re = hdulist['SPX_ELLCOO'].data[1]

        print(plateifu)
        #gets the mass of the galaxy. Need to edit the mass a bit to make it accurate. the petrosian mass
        mass = math.log10(obj['nsa_elpetro_mass'])-np.log10(.49)
        print("mass from data", mass)
        axis=obj['nsa_sersic_ba']

        
        #################################################################################
        #
        #   Mass of galaxy to get slope of average profile from Belfiore
        #  
        #################################################################################

        ##Just uses the data from Belfiore to get a slope and y intercept based on the mass of the galaxy
        
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
        


        #If there are any values in our O_B array that aren't NaN or less than zero, we keep the array.    
        zeros= False
        for element in range(0, len(O_B)):
            for item in range(0, len(O_B[element])):
                if O_B[element][item] >= 0:
                    zeros = True

        #################################################################################
        #
        #   BPT Diagram Creator
        #  
        #################################################################################

#Sum the fluxes over a 3" center of the galaxy, put into is starforming
        
        #Starts the creation of the BPT diagram
        ax_bpt = fig.add_subplot(2, 3, 4)
        if zeros == True:
            
            total=0
            sfr=0
            nsfr=0
            
            ax_bpt.set_aspect(1)
            ax_bpt.set_title("BPT Diagram")

        #Creates the lines for each BPT diagram line
        
        #Kewley
        X = np.linspace(-1.5, 0.3)
        Y = ((0.61/(X-0.47))+1.19)
        
        #Kauffmann
        Xk = np.linspace(-1.5,0.)
        Yk= (0.61/(Xk-0.05)+1.3)
       
        #plots the lines
        ax_bpt.plot(X, Y, '--', color = "red", lw = 1, label = "Kewley+01")
        ax_bpt.plot(Xk, Yk, '-', color = "blue", lw = 1, label = "Kauffmann+03")
        
        #plots the seyfert line
        x=np.linspace(-0.133638005,0.75,100)
        y=2.1445*x+0.465
        ax_bpt.plot(x, y, '--', color = "green", lw = 1, label = "Seyfert/LINER")

        #creates the arrays that will be used to plot the BPT diagram
        bpt_n2ha = np.log10(NII/H_alpha)
        bpt_o3hb = np.log10(OIII/H_beta)
        
        #badpix selects the points where the error is too high
        badpix = ((Ha/Ha_err) < 5) | ((H_beta/Hb_err) < 5) | ((OIII/o3_err) < 3) | ((NII/n2_err) < 3) | np.isinf(bpt_n2ha) | np.isinf(bpt_o3hb)
        #removes the bad points from both BPT arrays
        bpt_n2ha[badpix] = np.nan
        bpt_o3hb[badpix] = np.nan
        
        #finds the 98th percentile and 2nd percentile of the O3 HB lines
        bpt_o3hb95 = np.nanpercentile(bpt_o3hb, 98)
        bpt_o3hb5 = np.nanpercentile(bpt_o3hb, 2)
        
        #gets the minimum and maximum of each array, and increases/decreases them each a bit. These are to make the xlim and ylim of the plots look nicer.
        xmin = np.nanmin(bpt_n2ha) - 0.1
        xmax = np.nanmax(bpt_n2ha) + 0.1
        ymin = np.nanmin(bpt_o3hb) - 0.1
        ymax = np.nanmax(bpt_o3hb) + 0.1
        
        #Adds a legend to the plot to indicate the lines plotted
        plt.legend()
        
        #Scatters the points if they are star forming. Colors them based on their effective radius
        scatter_if(bpt_n2ha, bpt_o3hb, (is_starforming == 1) | (is_starforming == 0), c=r_Re, marker = ".", s = 65, alpha = 0.5, cmap = 'jet_r')
        #scatter_if(bpt_n2ha, bpt_o3hb, is_starforming == 0, c=r_Re, marker = ".", s = 65, alpha = 0.5, cmap = 'jet')
        
        #Sets the limits on the plots
        ax_bpt.set_xlim(xmin, xmax)
        ax_bpt.set_ylim(ymin, ymax)
        ax_bpt.set_aspect((xmax-xmin)/(ymax-ymin))
        
        #Sets a colorbar
        cb_bpt = plt.colorbar(shrink = .7)
        cb_bpt.set_label('r/$R_e$', rotation = 270, labelpad = 25)

        
        plt.tight_layout()
        
        try:
            plt.tight_layout()
        except ValueError:
            print("all NaN")
            print("==========================================================================================")

            print("value error, all NaN")
            count_continue1=count_continue1+1
            continue

        #Gets rid of all the points in Ha and logOH12 if they are not starforming.
        Ha[is_starforming==0]=np.nan
        logOH12[is_starforming==0]=np.nan


        #################################################################################
        #
        #   Halpha Flux 2-D
        #  
        #################################################################################

        a = fig.add_subplot(2,3,2)
        
        #sets the badpix where our error is less than 3 of the ratio
        badpix = ((Ha/Ha_err) < 3)
        #We want to save the un masked Ha array for later
        Ha_2d = Ha
        #Gets rid of the badpix
        Ha_2d[badpix]=np.nan
        #Also removes the badpix from the plotted contours
        contours_i[badpix]=np.nan
        #Plots the image
        imgplot = plt.imshow(Ha_2d, cmap = "viridis", extent = shapemap, zorder = 1)
        plt.title("H-alpha Flux")
        #Plots the contours based on the i band image
        csss=plt.gca().contour(contours_i, 8, colors = 'k', alpha = 0.6, extent = shapemap, origin = 'upper', zorder = 4)
        #Plots 1 effective radius
        css = plt.gca().contour(r_Re*2,[2],extent=shapemap, colors='r', origin = 'upper', zorder = 2, z = 2)
        
        #Need to invert the y axis to make the plot match the plotted image
        plt.gca().invert_yaxis()
        

        #Just plots the labels
        plt.xlabel('Arcseconds')
        plt.ylabel('Arcseconds')
        #Adds a colorbar
        cb = plt.colorbar(shrink = .7)
        cb.set_label('H-alpha flux [$10^{17} {erg/s/cm^2/pix}$]', rotation = 270, labelpad = 25)

        
        mask = hdulist['EMLINE_GVEL_MASK'].data[18,:,:]
        
        #make white be zero
        #find min and max of velocity, whichever one's absolute value is larger, use as min and max

        velocity[np.isinf(velocity)==True]=np.nan
        #velocity[(np.isnan(snmap)==True)|(snmap==0)]=np.nan
        velocity[mask != 0]=np.nan
        velocity_err[np.isinf(velocity_err)==True]=np.nan

        size=50

        #################################################################################
        #
        #   Velocity Map
        #  
        #################################################################################
        
        a = fig.add_subplot(2,3,3)
        ##Makes  a scatter plot of the data
        
        #gets rid of the bad velocity points where the error is too high
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
        
        
        #plots the velocity map
        imgplot = plt.imshow(velocity, origin = "lower", cmap = "RdYlBu_r", extent = shapemap, vmin = -vel_final, vmax = vel_final)
        #Adds a contour line for the one effective radius
        css = plt.gca().contour(r_Re*2,[2],extent=shapemap, colors='green', origin = 'lower', zorder = 2, z = 1)
        #adds the contors from the i band image
        csss=plt.gca().contour(contours_i, 8, colors = 'k', alpha = 0.6, extent = shapemap, zorder = 4)
        
        plt.title("Gas Velocity")
        
        #adds the colorbar
        cb = plt.colorbar(shrink = .7)
        
        #If all the velocities are less than zero, we make sure to get all of the correct velocities on the plot. 
        if ((vel_min <=0) and (vel_max <=0)):
            plt.clim(-vel_final, want)
        else:
            plt.clim(-vel_final,vel_final)
        cb.set_label('km/s', rotation = 270, labelpad = 25)
        a.set_facecolor('white')
        
        #################################################################################
        #
        #   Plots Galaxy Image
        #  
        #################################################################################


        ##Adds another subplot with the plateifu
        a = fig.add_subplot(2,3,1)
        print("plate ifu for plotting image" + plateifu)
        #Adds the galaxy image, as long as it actually exists
        try:
            image = img.imread('/home/celeste/Documents/astro_research/astro_images/marvin_images/' + plateifu + '.png')
        except ValueError:
            print("No image.")
            print("========================================================================================")
        lum_img = image[:,:,0]
        #plt.subplot(121)
        imgplot = plt.imshow(image)
        plt.title("Galaxy "  + str(plate_number) + "-" + str(fiber_number))
        

        #################################################################################
        #
        #   Metallicity Gradient with fitted Lines
        #  
        #################################################################################
       
			        

        a = fig.add_subplot(2, 3, 6)
        plt.xlabel("Effective Radii r/$R_e$")
        plt.ylabel('12+log(O/H)')

        #finds the indices where the arrays are infinity
        idx = np.isfinite(r_Re.flatten()) & np.isfinite(logOH12.flatten())
        #Sorts the array
        indarr=np.argsort(r_Re.flatten()[idx])
        
        #Fits a line to the sorted effective radius array
        yfit = [b + m * xi for xi in r_Re.flatten()[idx][indarr]]
        try:
            plt.tight_layout()
        except ValueError:
            print("all NaN")
            print("==========================================================================================")
            count_continue2=count_continue2+1
            
        #plots the line, is the expected profile for stellar mass
        plt.plot(r_Re.flatten()[idx][indarr], yfit, color = "red", zorder = 3, label = 'Expected profile for stellar mass')
        
        
        slope_expected = (yfit[1]-yfit[0])/(r_Re.flatten()[idx][indarr][1]-r_Re.flatten()[idx][indarr][0])
     
        #defines a line fitting function
        def func(x, m, b):
            return m*x+b
        
        #redefined abundance, we want to keep the original logOH12 not edited
        abundance = logOH12
        radius = r_Re
            
        #creates an x array
        trialX = np.linspace(np.amin(radius.ravel()), np.amax(radius.ravel()), 1000)
        
        #errors where our logOH12 errors are less than the 95th percentile
        cond_err = logOH12error.ravel()<np.nanpercentile(logOH12error.ravel(), 95)
        #our maximum error is decided to be the 95th percentile of the logOH12 errors. We authomatically set it to 0.1
        max_err = np.nanpercentile(logOH12error.ravel(), 95)
        max_err = 0.1
        #Our conditional indicies are going to be listed here:
        condition = (logOH12error.flatten() < max_err) & ((Ha/Ha_err).flatten() > 3) & ((s22/s22_err).flatten() >3) & ((s21/s21_err).flatten() > 3) & ((NII/n2_err).flatten() >3)
        #We scatter the points if the condition is met
        scatter_if(r_Re.flatten(), logOH12.flatten(), condition, s= 10, edgecolors = "black", color = "gray", zorder = 1)
        #Error bar plot. We only plot every 45 points, otherwise the errorbars become too ubiquitous and its hard to read the plot
        plt.errorbar(r_Re.ravel()[condition], logOH12.ravel()[condition], yerr=logOH12error.ravel()[condition], fmt=None, errorevery = 45, capsize = 15, color = "black", zorder = 2)
        
        #new condition
        condition2 = (logOH12error < max_err) & ((Ha/Ha_err) > 3) & ((s22/s22_err) >3) & ((s21/s21_err) > 3) & ((NII/n2_err) >3)
        logOH12_2=logOH12.copy()
        #sets our new array to be nan if the condition is not met
        logOH12_2[condition2==False]=np.nan
        
        #Code from Adam. We use this to create our line fit
        rad_pix=hdulist['SPX_ELLCOO'].data[0,:,:]*2.0 #since there are 2 pixels/arcsec
        rad, rp, n, sig =radial_profile(image=logOH12_2,distarr=rad_pix, radtype = 'weighted')
        rad=rad/(2*Re) #This is now in units of Re.
        
        #valid indicies
        valid = ~(np.isnan(rad) | np.isnan(rp) | np.isinf(rad) | np.isinf(rp) | ((rad < .5) | (rad > 2) ) | (n < 5))
        
        #plots the binned median
        plt.plot(rad, rp, '.m', label = 'binned median', markersize =7, marker = 'D')
        
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
        
        #plots the line of best fit between 0.5 and 2 R_e
        plt.plot(rad[valid], func(rad[valid], *popt), 'cyan', label = '0.5-2 $R_e$ fit', linewidth = 5)
        
        slope_calculated = (func(rad[valid], *popt)[1]-func(rad[valid], *popt)[0])/(rad[valid][1]-rad[valid][0])
        print("slope expected: " + str(slope_expected))
        print("slope calculated: " + str(slope_calculated))
        plt.close("all")
        

        slopefile = open('/home/celeste/Documents/astro_research/summer_2018/slopes_summer2018.txt', 'a')

        slopefile.write(str(plateifu) + '/' + str(slope_expected) + '/' + str(slope_calculated) + '\n')

        slopefile.close()
        continue

        plt.legend()
        plt.title("Metallicity Gradient")
        plt.xlim(xmin = 0)
        
        #################################################################################
        #
        #   Metallicity Radial Plot 3-D
        #  
        #################################################################################
     
        

        shape = (logOH12.shape[1])
        shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]

			
        matplotlib.rcParams.update({'font.size': 20})

        #if logOH12 is infinity, we set that value to NaN
        logOH12[np.isinf(logOH12)==True]=np.nan
        
        #again, got rid of the bad pixels
        badpix = (logOH12error > max_err) | ((Ha/Ha_err) < 3) | ((s22/s22_err) < 3) | ((s21/s21_err) < 3) | ((NII/n2_err) < 3) |  (ew_cut < 3)
        logOH12[badpix]=np.nan
        
        
        minimum = np.nanpercentile(logOH12, 5)
        maximum = np.nanpercentile(logOH12, 95)
        median = np.nanpercentile(logOH12, 50)
        
        #masks the same points on all the contours
        Ha_contour = Ha
        Ha_contour[badpix]=np.nan
        contours_i_same[badpix]=np.nan
        
        a = fig.add_subplot(2,3,5)
        try:
            plt.tight_layout()
        except ValueError:
            print("all NaN")
            print("==========================================================================================")
        
        #If the differences between the     
        if ((maximum - minimum) < .2):
            maximum = median + 0.1
            minimum = median - 0.1
        

        #plots the metallicity map   
        imgplot = plt.imshow(logOH12, cmap = "viridis", extent = shapemap, vmin = minimum, vmax = maximum, zorder = 1)

        plt.title("Metallicity Map")
        
        #tries to plot the i band image contours 
        try:
            csss=plt.gca().contour(contours_i_same, 8, colors = 'k', alpha = 0.6, extent = shapemap, origin = 'upper', zorder = 3)
            #plt.contour(logOH12, 20, colors='k')
        except ValueError:
            print("Value error! Skipping the log0H12 contour plotting....")
            print("==========================================================================================")

        
        #plots the one effective radius contour
        css = plt.gca().contour(r_Re*2,[2],extent=shapemap, colors='red', origin = 'upper', alpha = .6, zorder = 2, z = 1, edgecolors = "black")


        #inverts y axis to match the other bits
        plt.gca().invert_yaxis()
        
        plt.xlabel('Arcseconds')
        plt.ylabel('Arcseconds')
        cb = plt.colorbar(shrink = .7)
        cb.set_label('12+log(O/H)', rotation = 270, labelpad = 25)
        #plt.xlim, plt.ylim
       

  
        ##Saves the final image. Can change the dpi (increase for higher quality but larger images)
        print("Saving...")
        ##plt.show()
        plt.savefig('/home/celeste/Documents/astro_research/manga_images/final_images/TERABYTE/MPL7/logcube_' + plateifu +"_v5.2.png", bbox_inches = 'tight', dpi = 84)

        #plt.show()
        plt.close('all')
        print("Done with this one.")
        print("--------------------------------------------------")
print(failed_logcube)
print(failed_maps)
print(failed_other)
