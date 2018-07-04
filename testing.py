from astropy.io import fits

hdulist = fits.open('/media/celeste/Hypatia/MPL7/HYB/allmaps/manga-7443-1901-MAPS-HYB10-GAU-MILESHC.fits.gz')
print(hdulist.info())

slopes = fits.open('/home/celeste/Documents/astro_research/summer_2018/slopes.fits')
print(slopes.info())

names = slopes['PRIMARY'].data
names.info()
