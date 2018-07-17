from astropy.io import fits

hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-7443-1901-MAPS-HYB10-GAU-MILESHC.fits.gz')

slopes = fits.open('/home/celeste/Documents/astro_research/summer_2018/pa_datav2.fits')
print(slopes.info())

names = slopes[1].data['STEL_PA']
expec = slopes[1].data['GAS_PA']
calc = slopes[1].data['PA']
iden = slopes[1].data['GALAXY_ID']

print(slopes[1].header)
print(names.size)
print(expec.size)
print(calc.size)
print(iden.size)
