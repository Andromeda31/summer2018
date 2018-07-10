from astropy.io import fits

hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-7443-1901-MAPS-HYB10-GAU-MILESHC.fits.gz')

slopes = fits.open('/home/celeste/Documents/astro_research/summer_2018/slopesv2.fits')
print(slopes.info())

names = slopes[1].data['Y_INT_CALC']
expec = slopes[1].data['Y_INT_EXPEC']
calc = slopes[1].data['CALCULATED_ERR']

print(slopes[1].header)
print(names.size)
print(expec.size)
print(calc.size)
