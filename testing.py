from astropy.io import fits
import numpy as np
from astropy.table import Table, Column

#hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-7443-1901-MAPS-HYB10-GAU-MILESHC.fits.gz')

slopes = fits.open('/home/celeste/Documents/astro_research/summer_2018/pa_datav4.fits')
print(slopes.info())

stel = slopes[1].data['STEL_PA']
gas = slopes[1].data['GAS_PA']
pa = slopes[1].data['PA']
iden = slopes[1].data['GALAXY_ID']

#print(slopes[1].header)


new_iden20 = []
new_iden30 = []
new_iden40 = []
new_iden50 = []

for x in range(0, stel.size):
    if gas[x]-stel[x]>30:
        #print(iden[x])
        new_iden30.append(iden[x])
    if gas[x]-stel[x]>20:
        #print(iden[x])
        new_iden20.append(iden[x])
    if gas[x]-stel[x]>40:
        #print(iden[x])
        new_iden40.append(iden[x])
    if gas[x]-stel[x]>50:
        print(iden[x])
        
        new_iden50.append(iden[x])
        
new_iden20 = np.array(new_iden20)
new_iden30 = np.array(new_iden30)
new_iden40 = np.array(new_iden40)
new_iden50 = np.array(new_iden50)

print(new_iden20.size)
print(new_iden30.size)
print(new_iden40.size)
print(new_iden50.size)
asdf
t = Table()

#t['OFF_20'] = Column(new_iden20, description = 'galaxy IDs where the gas PA and stel PA difference is greater than 20' )
#t['OFF_30'] = Column(new_iden30, description = 'galaxy IDs where the gas PA and stel PA difference is greater than 30' )
#t['OFF_40'] = Column(new_iden40, description = 'galaxy IDs where the gas PA and stel PA difference is greater than 40' )
t['OFF_50'] = Column(new_iden50, description = 'galaxy IDs where the gas PA and stel PA difference is greater than 50' )

t.write('/home/celeste/Documents/astro_research/summer_2018/pa_off50v1.fits')
