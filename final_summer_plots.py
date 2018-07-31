from astropy.io import fits
import numpy as np
from astropy.table import Table, Column
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib


'''
badpix30 = (stel - gas <= np.asarray(30)) and (stel - gas > np.asarray(40))
badpix40 = (stel - gas <= 40) and (stel - gas > 50)
badpix50 = (stel - gas <= 50)
badpix_all = (stel - gas >= 30)
'''

def big_loop(stel, gas, stel_err, gas_err, name, xname, yname):
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

    ax1.scatter(stel, gas, edgecolors = "black", color = "gray", label = "Difference b/t\n" + str(name) + "\nPA is $\leq$ 30$\degree$", s = size)
    ax1.scatter(stel30, gas30, edgecolors = "black", color = "yellow", label = "30$\degree \leq$ x $<$ 40$\degree$", s = size)
    ax1.scatter(stel40, gas40, edgecolors = "black", color = "orange", label = "40$\degree \leq$ x $<$ 50$\degree$", s = size)
    ax1.scatter(stel50, gas50, edgecolors = "black", color = "red", label = "x $>$ 50$\degree$", s = size)

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
    ax1.errorbar(stel, gas, xerr=stel_err/3, yerr=gas_err/3, capsize = 0, color = "black", fmt = None)
    ax1.errorbar(stel30, gas30, xerr=stel_err/3, yerr=gas_err/3, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(stel40, gas40, xerr=stel_err/3, yerr=gas_err/3, capsize = 0, color = "black", fmt=None)
    ax1.errorbar(stel50, gas50, xerr=stel_err/3, yerr=gas_err/3, capsize = 0, color = "black", fmt=None)
    plt.title("Gas Position Angle vs Stellar Position Angle")
    plt.xlabel(str(xname) + " PA")
    plt.ylabel(str(yname) + ' PA')

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

big_loop(stel, gas, stel_err, gas_err,'stel_vs_gas', 'Stellar', 'Gas')
big_loop(pa, stel, 0, stel_err, 'drpall_vs_stel', 'DRPall', 'Stellar')
big_loop(gas, pa, gas_err, 0, 'gas_vs_drpall', 'Gas', 'DRPall')
