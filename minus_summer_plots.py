from astropy.io import fits
import numpy as np
from astropy.table import Table, Column
import matplotlib.image as img
import matplotlib.pyplot as plt
import astropy.table as t
import math
import matplotlib


'''
badpix30 = (stel - gas <= np.asarray(30)) and (stel - gas > np.asarray(40))
badpix40 = (stel - gas <= 40) and (stel - gas > 50)
badpix50 = (stel - gas <= 50)
badpix_all = (stel - gas >= 30)
'''

def big_loop(stel, gas, stel_err, gas_err, mass, name):
    minus = np.absolute(stel - gas)

    badpix30 = ((minus <= 30) | (minus > 40))
    badpix40 = ((minus <= 40) | (minus > 50))
    badpix50 = (minus <=50)
    badpix_all = (minus >= 30)

    stel30 = np.array(stel)
    stel40 = np.array(stel)
    stel50 = np.array(stel)
    gas30 = np.array(gas)
    gas40 = np.array(gas)
    gas50 = np.array(gas)

    stel30[badpix30] = np.nan
    stel40[badpix40] = np.nan
    stel50[badpix50] = np.nan
    gas30[badpix30] = np.nan
    gas40[badpix40] = np.nan
    gas50[badpix50] = np.nan


    stel[badpix_all] = np.nan
    gas[badpix_all] = np.nan


    fig = plt.figure(figsize=(10,9))

    ax1 = fig.add_subplot(111)

    size = 80
    
    sub = np.absolute(stel - gas)
    sub30 = np.absolute(stel30 - gas30)
    sub40 = np.absolute(stel40 - gas40)
    sub50 = np.absolute(stel50 - gas50)
    
    stel_err = stel_err/3
    gas_err = gas_err/3
    
    ##Error prop. for subtraction
    
    sub1 = np.square(stel)*np.square(stel_err)
    sub2 = np.square(gas)*np.square(gas_err)
    sub3 = 2*(stel)*(gas)*(stel_err)*(gas_err)
    
    sub_err = np.sqrt(sub1 + sub2 - sub3)

    ax1.scatter(mass, sub, edgecolors = "black", color = "gray", label = "Difference b/t\n" + 'Gas PA and Stellar ' + "\nPA is $\leq$ 30$\degree$", s = size)
    ax1.scatter(mass, sub30, edgecolors = "black", color = "yellow", label = "30$\degree \leq$ x $<$ 40$\degree$", s = size)
    ax1.scatter(mass, sub40, edgecolors = "black", color = "orange", label = "40$\degree \leq$ x $<$ 50$\degree$", s = size)
    ax1.scatter(mass, sub50, edgecolors = "black", color = "red", label = "x $>$ 50$\degree$", s = size)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''
    ax1.errorbar(stel, gas, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 40, fmt = None)
    ax1.errorbar(stel30, gas30, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 4, fmt=None)
    ax1.errorbar(stel40, gas40, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 4, fmt=None)
    ax1.errorbar(stel50, gas50, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 3, fmt=None)
    '''

    size = 10
    '''
    ax1.errorbar(mass, sub, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt = None)
    ax1.errorbar(mass, sub30, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(mass, sub40, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(mass, sub50, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    '''
    
    plt.title("Gas Position Angle - Stellar Position Angle vs Galaxy Mass")
    plt.ylabel("Gas PA - Stellar PA")
    plt.xlabel('Log Galaxy Mass')
    plt.ylim(-1.75, 180)
    plt.xlim(8.5, 11.5)
    
    #plt.show()

    plt.savefig('/home/celeste/Documents/astro_research/summer_2018/' + str(name) + '_pa_nocaps.png')
    plt.close('all')
    
def big_loop2(stel, gas, stel_err, gas_err, mass, name):
    minus = np.absolute(stel - gas)

    badpix30 = ((minus <= 30) | (minus > 40))
    badpix40 = ((minus <= 40) | (minus > 50))
    badpix50 = (minus <=50)
    badpix_all = (minus >= 30)

    stel30 = np.array(stel)
    stel40 = np.array(stel)
    stel50 = np.array(stel)
    gas30 = np.array(gas)
    gas40 = np.array(gas)
    gas50 = np.array(gas)

    stel30[badpix30] = np.nan
    stel40[badpix40] = np.nan
    stel50[badpix50] = np.nan
    gas30[badpix30] = np.nan
    gas40[badpix40] = np.nan
    gas50[badpix50] = np.nan


    stel[badpix_all] = np.nan
    gas[badpix_all] = np.nan


    fig = plt.figure(figsize=(10,9))

    ax1 = fig.add_subplot(111)

    size = 80
    
    sub = np.absolute(stel - gas)
    sub30 = np.absolute(stel30 - gas30)
    sub40 = np.absolute(stel40 - gas40)
    sub50 = np.absolute(stel50 - gas50)
    
    stel_err = stel_err/3
    gas_err = gas_err/3
    
    ##Error prop. for subtraction
    
    sub1 = np.square(stel)*np.square(stel_err)
    sub2 = np.square(gas)*np.square(gas_err)
    sub3 = 2*(stel)*(gas)*(stel_err)*(gas_err)
    
    sub_err = np.sqrt(sub1 + sub2 - sub3)

    ax1.scatter(mass, sub, edgecolors = "black", color = "gray", label = "Difference b/t\n" + 'DRPall PA and Gas ' + "\nPA is $\leq$ 30$\degree$", s = size)
    ax1.scatter(mass, sub30, edgecolors = "black", color = "yellow", label = "30$\degree \leq$ x $<$ 40$\degree$", s = size)
    ax1.scatter(mass, sub40, edgecolors = "black", color = "orange", label = "40$\degree \leq$ x $<$ 50$\degree$", s = size)
    ax1.scatter(mass, sub50, edgecolors = "black", color = "red", label = "x $>$ 50$\degree$", s = size)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''
    ax1.errorbar(stel, gas, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 40, fmt = None)
    ax1.errorbar(stel30, gas30, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 4, fmt=None)
    ax1.errorbar(stel40, gas40, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 4, fmt=None)
    ax1.errorbar(stel50, gas50, xerr=stel_err/3, yerr=gas_err/3, capsize = 7, color = "black", errorevery = 3, fmt=None)
    '''

    size = 10
    '''
    ax1.errorbar(mass, sub, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt = None)
    ax1.errorbar(mass, sub30, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(mass, sub40, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(mass, sub50, yerr=sub_err, xerr=0, capsize = 0, color = "black", fmt=None)
    '''
    
    plt.title("DRPall Position Angle - Gas Position Angle vs Galaxy Mass")
    plt.ylabel("DRPall PA - Gas PA")
    plt.xlabel('Log Galaxy Mass')
    plt.ylim(-1.75, 180)
    plt.xlim(8.5, 11.5)
    
    #plt.show()

    plt.savefig('/home/celeste/Documents/astro_research/summer_2018/' + str(name) + '_pa_nocaps.png')
    plt.close('all')
    
    
#start the plot making    
slopes = fits.open('/home/celeste/Documents/astro_research/summer_2018/pa_datav4.fits')

stel = slopes[1].data['STEL_PA']
stel_err = slopes[1].data['STEL_PA_ERR']
gas = slopes[1].data['GAS_PA']
gas_err = slopes[1].data['GAS_PA_ERR']
pa = slopes[1].data['PA']
iden = slopes[1].data['GALAXY_ID']

stel = np.asarray(stel)
stel_err = np.asarray(stel_err)
gas = np.asarray(gas)
gas_err = np.asarray(gas_err)
pa = np.asarray(pa)
iden = np.asarray(iden)

mass = []


for x in range(0, len(iden)):
    drpall = t.Table.read('/home/celeste/Documents/astro_research/drpall-v2_3_1.fits')
    obj = drpall[drpall['plateifu']==str(iden[x])][0]
    mass_num = math.log10(obj['nsa_elpetro_mass'])-np.log10(.49)
    mass.append(mass_num)
    print(str(iden[x]) + ' and done\n-------------------------')

print(mass)


big_loop(stel, gas, stel_err, gas_err, mass, 'stel-gas_vs_mass')

#big_loop2(pa, gas, 0, gas_err, mass, 'drpallpa-gas_vs_mass')
