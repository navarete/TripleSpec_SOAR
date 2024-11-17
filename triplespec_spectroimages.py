from astropy.io import fits
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation,SkyCoord
import astropy.units as u
from astropy.time import Time
import os

def tspec_order_section(image, order):
    """
    Extract the section of the TripleSpec data for a given order.

    This function returns the specified spectral order section from a TripleSpec image.
    The valid range for orders is from 3 to 7, each corresponding to different spectral bands (K, H, J, and Y).
    The section is extracted based on predefined pixel ranges, and adjustments are made to ensure that traces
    from adjacent orders are properly excluded.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D array representing the image from which the order section will be extracted.
    
    order : int
        The spectral order to extract, where the value should be between 3 and 7. Each order corresponds to a specific band:
        - Order 3: K band
        - Order 4: H band
        - Order 5: J band
        - Order 6: Y band
        - Order 7: Y band (blue)

    Returns
    -------
    image_section : numpy.ndarray
        The extracted section of the image corresponding to the specified order. Pixel values outside the region
        of interest for adjacent orders are set to `np.nan`.

    Raises
    ------
    ValueError
        If the order is outside the valid range of 3 to 7, a ValueError is raised.

    Notes
    -----
    The section intervals and adjustments for each order are specific to the TripleSpec spectrograph configuration.
    The function applies cuts to exclude traces from adjacent orders, ensuring that the correct spectral data is extracted.

    Example
    -------
    >>> import numpy as np
    >>> image = np.random.random((2048, 2048))  # Example spectrograph image
    >>> order_section = tspec_order_section(image, order=5)
    >>> print(order_section.shape)
    """
    # make sure order is within the expected range
    if order < 3 or order > 7:
        raise ValueError("<ERROR> Order n={} is outside the limits n=[3-7].".format(order))
    
    # Set the interval along the slit for each order
    if order == 3:
        x_order = [874,980] # K band
    elif order == 4:
        x_order = [704,815] # H band
    elif order == 5:
        x_order = [529,685] # J band
    elif order == 6:
        x_order = [319,555] # Y band
    else:
        x_order = [245,421] # Y band (blue)

    # Apply the cut, also clip the n=7 data to remove the empty region to the blue
    if order != 7:
        image_section = image[:,x_order[0]:x_order[1]].copy()
    else:
        image_section = image[1020:,x_order[0]:x_order[1]].copy()

    # Exclude traces from adjacent orders to ensure we're properly fitting the actual order
    if order == 5:
        image_section[1389:,0:25] = np.nan # from n=6
    if order == 6:
        image_section[:380,210:]  = np.nan # from n=5
        image_section[1000:,:100] = np.nan # from n=7
    if order == 7:
        image_section[:201,154:]  = np.nan # from n=6
    
    # Return the section for the corresponding order
    return image_section

def tspec_grid_spatial(spatial_solution, flat_header):
    """
    Generate a uniform spatial grid for the TripleSpec data.

    This function computes a uniform spatial grid based on the spectral data input and the plate scale
    from the image's header. It identifies the spatial range of the data and generates a new grid
    to be used for interpolation.

    Parameters
    ----------
    spatial_solution : numpy.ndarray
        A 2D array representing the spectral data as a function of the spatial position.

    flat_header : dict-like object
        Header information from the FITS file.
        It must include the 'PLTSCALE' keyword, representing the plate scale in arcseconds per pixel.

    Returns
    -------
    spatial_grid : numpy.ndarray
        A 1D array representing the uniform spatial grid on which the spectrum can be interpolated.

    Notes
    -----
    The function calculates the minimum and maximum spatial range where finite data points exist
    in each wavelength slice. It then creates a uniform grid using the plate scale (PLTSCALE) to
    ensure that the data is consistently spaced in the spatial direction.

    Example
    -------
    >>> import numpy as np
    >>> spatial_solution = np.random.random((100, 50))  # Example spectral data
    >>> flat_header = {'PLTSCALE': 0.2}  # Plate scale in arcseconds per pixel
    >>> spatial_grid = tspec_grid_spatial(spatial_solution, flat_header)
    >>> print(spatial_grid.shape)
    """    
    
    # Initialize spatial range
    spatial_range = None
    
    # Slice the data as a function of wavelength
    for dwave in range(0,spatial_solution.shape[0]):
        idx_finite = np.isfinite(spatial_solution[dwave,:])
        n_finite   = np.sum(idx_finite)
        if n_finite > 0:
            sl = spatial_solution[dwave,idx_finite]
            spatial_range_dw = [min(sl),max(sl)]
            if spatial_range is None:
                spatial_range = spatial_range_dw
            else:
                if min(spatial_range_dw) < min(spatial_range):
                    spatial_range[0] = min(spatial_range_dw)
                if max(spatial_range_dw) > max(spatial_range):
                    spatial_range[1] = max(spatial_range_dw)
            
    # Create a uniform array on the spatial axis
    plate_scale = flat_header['PLTSCALE']
    spatial_range = [ min(spatial_range), max(spatial_range) ]
    n_spaxels = round( ( spatial_range[1] - spatial_range[0] ) / plate_scale ) + 1

    # create new spatial grid to interpolate the spectrum
    spatial_grid =  np.arange(n_spaxels) * plate_scale + spatial_range[0]
    
    return spatial_grid


def tspec_grid_spatial_all_orders(spatial_solution, flat_header):
    """
    Generate a uniform spatial grid for the TripleSpec data.

    This function computes a uniform spatial grid based on the combined spatial ranges of all orders 
    (from n=3 to n=7) in the TripleSpec spectrograph data. It extracts the spatial extent for each 
    order and generates a unified spatial grid using the plate scale from the image's header.

    Parameters
    ----------
    spatial_solution : numpy.ndarray
        A 2D array representing the spectrograph image data where each row corresponds to a wavelength
        and each column corresponds to a spatial position.

    flat_header : dict-like object
        Header information from the FITS file or a similar data structure that contains metadata
        about the observations. It must include the 'PLTSCALE' keyword, representing the plate scale
        in arcseconds per pixel.

    Returns
    -------
    spatial_grid : numpy.ndarray
        A 1D array representing the uniform spatial grid on which the spectrum can be interpolated,
        covering the entire spatial range of all orders.

    Notes
    -----
    The function iterates through each spectral order (from n=3 to n=7) to determine the minimum and
    maximum spatial ranges where finite data points exist. It then creates a unified spatial grid
    using the plate scale (PLTSCALE) to ensure consistent spacing in the spatial direction.

    Example
    -------
    >>> import numpy as np
    >>> spatial_solution = np.random.random((1024, 512))  # Example spectrograph image
    >>> flat_header = {'PLTSCALE': 0.2}  # Plate scale in arcseconds per pixel
    >>> spatial_grid = tspec_grid_spatial_all_orders(spatial_solution, flat_header)
    >>> print(spatial_grid.shape)
    """
    
    # Initialize lists to store minimum and maximum spatial ranges
    min_spatial_value = []
    max_spatial_value = []
    
    # Loop through each spectral order from 3 to 7
    for order in range(3,8):
        
        spatial_solution_per_order =  tspec_order_section(spatial_solution, order)

        # Initialize spatial range for the current order
        spatial_range_per_order = None
        
        # Slice the data as a function of wavelength
        for dwave in range(0,spatial_solution_per_order.shape[0]):
            idx_finite = np.isfinite(spatial_solution_per_order[dwave,:])
            n_finite   = np.sum(idx_finite)
            if n_finite > 0:
                sl = spatial_solution_per_order[dwave,idx_finite]
                spatial_range_dw = [min(sl),max(sl)]
                if spatial_range_per_order is None:
                    spatial_range_per_order = spatial_range_dw
                else:
                    if min(spatial_range_dw) < min(spatial_range_per_order):
                        spatial_range_per_order[0] = min(spatial_range_dw)
                    if max(spatial_range_dw) > max(spatial_range_per_order):
                        spatial_range_per_order[1] = max(spatial_range_dw)
          
        min_spatial_value.append(spatial_range_per_order[0])
        max_spatial_value.append(spatial_range_per_order[1])

    # Create a uniform array on the spatial axis
    plate_scale = flat_header['PLTSCALE']
    spatial_range = [ min(min_spatial_value), max(max_spatial_value) ]
    n_spaxels = round( ( spatial_range[1] - spatial_range[0] ) / plate_scale ) + 1

    # Create new spatial grid to interpolate the spectrum
    spatial_grid =  np.arange(n_spaxels) * spat_disp + spat_range[0]
    
    return spatial_grid

def tspec_flux_calibration_order(flux_solution, order, new_wavelength_grid, saturation_limit=30000):
    """
    Perform flux calibration for a specific spectral order of TripleSpec data.

    This function takes a flux calibration solution and applies it to a given spectral order, 
    interpolating the calibration to a specified wavelength grid. It also identifies the saturation 
    limit for the flux values.

    Parameters
    ----------
    flux_solution : numpy.ndarray
        A 3D array containing flux calibration solutions. The first dimension corresponds to 
        the orders (3 to 7), the second dimension holds the wavelength and flux values, 
        and the third dimension is the corresponding data.

    order : int
        The spectral order (between 3 and 7) for which the flux calibration will be applied.

    new_wavelength_grid : numpy.ndarray
        A 1D array representing the new wavelength grid onto which the flux calibration will be interpolated.

    saturation_limit : float, optional
        A threshold value for identifying saturation in the flux data. Default is 30000 DN/s. 
        The minimum saturated flux value will be calculated based on this limit and the flux calibration.

    Returns
    -------
    fintp : numpy.ndarray
        A 1D array containing the interpolated flux calibration values corresponding to the new 
        wavelength grid (new_wavelength_grid).

    saturation_limit_flux : float
        The minimum saturated flux value, calculated from the flux calibration data.

    Notes
    -----
    The function first identifies valid (finite) entries in both the wavelength and flux arrays. 
    It then calculates the minimum saturated flux by multiplying the saturation limit by the finite 
    flux values. The flux calibration spectrum is interpolated onto the provided wavelength grid.

    Example
    -------
    >>> flux_solution = np.random.random((5, 2, 100))  # Example flux solution
    >>> order = 4  # Spectral order
    >>> new_wavelength_grid = np.linspace(1.0, 2.0, 50)  # New wavelength grid
    >>> fintp, saturation_limit_flux = tspec_flux_calibration_order(flux_solution, order, new_wavelength_grid)
    >>> print(fintp.shape, sat_flux)
    """
    
    # Retrieve the wavelength and flux information from the specified order
    fcal_wave = flux_solution[order-3,0,:]
    fcal_flux = flux_solution[order-3,1,:]
    
    # Identify NaN values in the wavelength and flux arrays
    idxfw = np.isfinite(fcal_wave) 
    idxff = np.isfinite(fcal_flux)
    idxf  = idxfw * idxff

    # Calculate the saturation limit in flux units
    saturation_limit_flux = np.nanmin(saturation_limit * fcal_flux[idxf])
    
    # Interpolate the flux-calibration spectra into the 'new_wavelength_grid' grid 
    ffx = interpolate.interp1d(fcal_wave[idxf], fcal_flux[idxf], fill_value='extrapolate', kind='linear')
    flux_interpolated = ffx(new_wavelength_grid)
    
    return flux_interpolated, saturation_limit_flux

def tspec_interp_spatial_grid(image_section, spatial_solution, new_spatial_grid):    
    """
    Interpolate the original TripleSpec 2d image from a given order into a new spatial grid.

    This function takes a 2d image data and the corresponding spatial solution, 
    interpolating the image data onto a new (linear) spatial grid defined by `new_spatial_grid`. 

    Parameters
    ----------
    image_section : numpy.ndarray
        A 2D array representing image data to be interpolated. 

    spatial_solution : numpy.ndarray
        A 2D array of the same shape as `image_section` containing spatial solution for the image. 
        It should represent the spatial coordinates corresponding to the image data.

    new_spatial_grid : numpy.ndarray
        A 1D array of new spatial coordinates onto which the image section will be interpolated.

    Returns
    -------
    image_interp_spatial : numpy.ndarray
        A 2D array of the interpolated image data, with shape (n_sg, n_wavelengths), 
        where `n_sg` is the length of the input spatial grid `new_spatial_grid`.

    Notes
    -----
    The function iterates through each wavelength slice of the image data, performing linear interpolation 
    of the finite spatial values. The resulting interpolated values are then organized into a new array. 
    Finally, the spatial axis of the interpolated image is flipped to ensure proper orientation.

    Example
    -------
    >>> image_section = np.random.random((100, 50))  # Example image data
    >>> spatial_solution = np.random.random((100, 50))  # Corresponding spatial values
    >>> new_spatial_grid = np.linspace(0, 1, 40)  # New spatial grid
    >>> interp_image = tspec_interp_spatial_grid(image_section, spatial_solution, new_spatial_grid)
    >>> print(interp_image.shape)
    (40, 100)
    """
    
    # Initialize the output array
    image_interp_spatial  = np.ndarray((new_spatial_grid.shape[0],image_section.shape[0]))
    
    # Slice the data as a function of wavelength
    for dwave in range(0, image_section.shape[0]):
        idx_finite = np.isfinite(spatial_solution[dwave,:])
        n_finite   = np.sum(idx_finite)       # Identify finite values
        if n_finite > 0:
            sl = spatial_solution[dwave,idx_finite]   # Finite spatial values (x vals)
            fx = interpolate.interp1d(sl, image_section[dwave,idx_finite], kind='linear', fill_value='extrapolate')
            image_interp_spatial[:,dwave] = fx(new_spatial_grid)  # Interpolated values onto new spatial grid
            
    # Flip the spatial axis to make it right
    image_interp_spatial = np.flip(image_interp_spatial,axis=0)            
            
    return image_interp_spatial

def tspec_grid_spectral(wave_solution, spec_disp):
    """
    Create a uniform spectral grid based on the input wavelength solution and specified dispersion.

    This function generates a linear wavelength grid for the spectrum of a given order, accounting for  
    the input wavelength values and a desired spectral dispersion.

    Parameters
    ----------
    wave_solution : numpy.ndarray
        A 1D array containing the original wavelength values. 
        Values should be in the same units (e.g., nanometers, micrometers).

    spec_disp : float
        The spectral dispersion value, indicating the pixel-to-wavelength conversion.

    Returns
    -------
    spectral_grid : numpy.ndarray
        A 1D array representing the newly created uniform wavelength grid. 
        The length of the grid is determined based on the dispersion and the range of the input wavelengths.

    Raises
    ------
    ValueError
        If the order is outside the valid range of 3 to 7, a ValueError is raised.

    Notes
    -----
    The function first excludes any zero values from the input wavelength array. If there are no 
    non-zero values, an error message is printed. The function also checks the maximum wavelength 
    value to determine the appropriate number of pixels for the output grid, which is fixed at 
    2040 pixels for certain orders (3 to 6). If the maximum wavelength is below a specified threshold, 
    the number of pixels is calculated based on the range of the input wavelengths and the spectral dispersion.

    Example
    -------
    >>> wave_solution = np.array([0.5, 0.7, 0.9, 1.1, 1.5])  # Example wavelengths
    >>> spec_disp = 0.001  # Example spectral dispersion
    >>> spectral_grid = tspec_grid_spectral(wave_solution, spec_disp)
    >>> print(spectral_grid)
    [0.5     0.501   0.502   ...]
    """
    
    # Create uniform array on the dispersion axis
    idx_nonzero = ( wave_solution > 0 )  # Exclude zero values.
    n_nonzero   = np.sum(idx_nonzero)
    
    if (n_nonzero == 0):
        raise ValueError("<ERROR> All zero values.")
    
    # For orders 3 to 6, the dispersion axis size should be fixed on 2040 pixels
    if np.nanmax(wave_solution) > 1.07:
        n_pixels = 2040 # this works for n=3-6 orders
    else:
        n_pixels = round( ( np.nanmax(wave_solution) - np.nanmin(wave_solution) ) / spec_disp )

    # Create new wavelength grid to interpolate the spectrum
    spectral_grid = np.nanmin(wave_solution) + np.arange(n_pixels) * spec_disp

    return spectral_grid


def get_heliocentric_velocity(header):
    """
    Calculate the heliocentric velocity correction for astronomical observations.

    This function computes the heliocentric velocity correction using information from the FITS file header.
    It calculates the correction based on the target's sky coordinates and the observation date, assuming
    that the observations were made at the SOAR telescope. The location of the telescope can be modified if necessary.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header of the FITS file, which should contain the following keywords:
        - 'RA': Right Ascension of the target in hour angle.
        - 'DEC': Declination of the target in degrees.
        - 'DATE-OBS': Observation date and time in a format recognized by `astropy.time.Time`.

    Returns
    -------
    helio_vel : astropy.units.Quantity
        The heliocentric velocity correction in km/s.

    Notes
    -----
    The default location is set to the SOAR telescope. If using another telescope, modify the location coordinates accordingly.
    Based on the example provided in https://docs.astropy.org/en/stable/coordinates/velocities.html

    Example
    -------
    >>> from astropy.io import fits
    >>> from astropy.time import Time
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> import astropy.units as u
    >>> header = fits.getheader('observation.fits')
    >>> heliocentric_velocity = get_heliocentric_velocity(header)
    >>> print(f'Heliocentric velocity correction: {heliocentric_velocity}')
    """
    
    # define SOAR location - modify if using another felescope
    # coordinates from https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc0490.pdf
    soar = EarthLocation.from_geodetic(lat=-30.238*u.deg, lon=-70.734*u.deg, height=2738*u.m)
    # get target's coordinates
    target_coordinates = SkyCoord(header['RA'], header['DEC'], unit=(u.hourangle, u.deg), frame='icrs')
    # calculates the heliocentric velocity based on the time of obsevations and location of the telescope
    helio_vel = target_coordinates.radial_velocity_correction('heliocentric', obstime=Time(header['DATE-OBS']), location=soar)
    # stores the velocity in km/s units
    helio_vel = helio_vel.to(u.km/u.s)
    
    # Return the heliocentric velocity in km/s units
    return helio_vel

def tspec_heliocentric_correction(sci_header, spectral_grid):
    """
    Correct the wavelength grid for heliocentric velocity.

    This function applies the heliocentric velocity correction to the given spectral grid 
    based on the observational header information. It returns the corrected wavelength grid 
    and the heliocentric velocity in km/s units.

    Parameters
    ----------
    sci_header : astropy.io.fits.Header
        The header information from a FITS file, containing the necessary observational 
        data to calculate the heliocentric velocity (e.g., RA, DEC, and DATE-OBS).

    spectral_grid : numpy.ndarray
        A 1D array representing the original wavelength grid of the spectrum. 
        The wavelengths should be in consistent units (e.g., angstroms, nanometers).

    Returns
    -------
    corrected_spectral_grid : numpy.ndarray
        A 1D array representing the corrected wavelength grid after applying the heliocentric 
        velocity adjustment.

    vhelio : float
        The heliocentric velocity in kilometers per second (km/s) used for the correction.

    Example
    -------
    >>> from astropy.io import fits
    >>> sci_header = fits.getheader('spectrum.fits')  # Example FITS file header
    >>> spectral_grid = np.array([4000, 5000, 6000])  # Example wavelength grid in angstroms
    >>> corrected_grid, vhelio = tspec_heliocentric_correction(sci_header, spectral_grid)
    >>> print(corrected_grid, vhelio)
    [4001.23 5001.23 6001.23] 12.34
    """
    # Calculate heliocentric velocity
    helio_velocity = get_heliocentric_velocity(sci_header)

    # Apply correction to the spectral grid
    corrected_spectral_grid = spectral_grid * ( 1. + helio_velocity.value / 2.99792E5 )
    
    return corrected_spectral_grid, helio_velocity.value


def tspec_interp_spectral_grid(image_spat_interp, interp_wave_spat, new_spectral_grid):
    """
    Interpolate the spatially corrected TripleSpec spectro image onto a new linear spectral grid.

    This function takes an image that has been interpolated in the spatial direction and 
    interpolates it onto a specified new wavelength grid.

    Parameters
    ----------
    image_spat_interp : numpy.ndarray
        A 2D array where each row represents a spatial slice of the image 
        that has been corrected for spatial variations. The shape should be 
        (n_slits, n_wavelengths).

    interp_wave_spat : numpy.ndarray
        A 2D array of the same shape as `image_spat_interp` that contains 
        the corresponding wavelengths for each spatial slice. The shape should be 
        (n_slits, n_wavelengths).

    new_spectral_grid : numpy.ndarray
        A 1D array representing the new spectral grid onto which the 
        spatially corrected image will be interpolated. This array should 
        be in consistent units with `interp_wave_spat`.

    Returns
    -------
    interp_image_spec : numpy.ndarray
        A 2D array where each row corresponds to a spatial slice of the image 
        interpolated onto the new spectral grid `new_spectral_grid`. The shape will be 
        (n_slits, n_new_wavelengths), where `n_new_wavelengths` is the length 
        of the input spectral grid `new_spectral_grid`.

    Example
    -------
    >>> image_spat_interp = np.array([[1, 2, 3], [4, 5, 6]])  # Example spatially corrected image
    >>> interp_wave_spat = np.array([[4000, 5000, 6000], [4000, 5000, 6000]])  # Wavelengths for each slice
    >>> new_spectral_grid = np.array([4100, 4200, 4300])  # New spectral grid
    >>> result = tspec_interp_spectral_grid(image_spat_interp, interp_wave_spat, new_spectral_grid)
    >>> print(result)
    [[1.5 2.5 3.5]
     [4.5 5.5 6.5]]
    """
    interp_image_spec = np.ndarray((image_spat_interp.shape[0], new_spectral_grid.shape[0]))

    # Slice the data as a function of the spatial dimension
    for d_slit in range(0, image_spat_interp.shape[0]):
        wave_dlist = interp_wave_spat[d_slit,:]
        
        fwi = interpolate.interp1d(wave_dlist, image_spat_interp[d_slit,:], kind='linear', fill_value='extrapolate')
        
        # now interpolate the row and add it to the final image
        interp_image_spec[d_slit,:] = fwi(new_spectral_grid)

    return interp_image_spec

def tspec_create_new_header(sci_header, new_spectral_grid, new_spatial_grid):
    """
    Create a new FITS header for interpolated data based on an existing TripleSpec header.

    This function creates a new FITS header by copying selected keywords from the original 
    TripleSpec science header and adds new keywords that describe the interpolated data's 
    spectral (`new_spectral_grid`) and spatial (`new_spatial_grid`) grid.

    Parameters
    ----------
    sci_header : astropy.io.fits.Header
        The original FITS header from the science data, containing metadata for the observation.

    new_spectral_grid : numpy.ndarray
        A 1D array representing the new spectral grid for the interpolated data. The first and 
        second elements of this array are used to set the wavelength axis information.

    new_spatial_grid : numpy.ndarray
        A 1D array representing the new spatial grid for the interpolated data. The first and 
        second elements of this array are used to set the spatial axis information.

    Returns
    -------
    header_out : astropy.io.fits.Header
        A new FITS header with copied metadata from the original header and additional information 
        describing the new spectral and spatial axes.

    Notes
    -----
    - The function assumes that `DATE-OBS` is in ISO format when converting it to Modified Julian Date (MJD).
    - The keyword `EXPTIME0` is set to the original exposure time from the science header, and a new 
      `EXPTIME` value is set to 1.0, indicating flux is per second.

    Example
    -------
    >>> from astropy.io import fits
    >>> from astropy.time import Time
    >>> sci_header = fits.Header({'DATE-OBS': '2022-10-01T00:00:00', 'EXPTIME': 1200.0})
    >>> new_spectral_grid = [1.1, 1.2]  # Example spectral grid
    >>> new_spatial_grid = [10.0, 10.5]  # Example spatial grid
    >>> new_header = tspec_create_new_header(sci_header, new_spectral_grid, new_spatial_grid)
    >>> print(new_header)
    """
        
    # Copy existing keywords to the new header
    kws=['OBJECT', 'PROPID', 'NIGHTID', 'OBSID', 'FILENAME', 'OBSMODE', 'OBSERVER', 'INSTRUM',
         'OBSTYPE', 'DATE-OBS', 'FSAMPLE', 'COADDS', 'RON', 'GAIN', 'OBSERVAT', 'TELESCOP', 
         'MJD', 'UT', 'DATE', 'RA', 'DEC', 'HA', 'TELAZ', 'TELEL', 'SIDEREAL', 'AIRMASS', 'IPA', 'PARALL']
    header_out = fits.Header()
    for kw in kws:
        if kw in sci_header:
            header_out[kw] = (sci_header[kw], sci_header.comments[kw])

    # Add new axis information
    header_out['CRVAL1'] = new_spectral_grid[0]
    header_out['CRPIX1'] = 1
    header_out['CDELT1'] = new_spectral_grid[1]-new_spectral_grid[0]
    header_out['CRVAL2'] = new_spatial_grid[0]
    header_out['CRPIX2'] = 1
    header_out['CDELT2'] = new_spatial_grid[1]-new_spatial_grid[0]
    header_out['EXPTIME'] = ( 1.0, 'Flux is per second')
    header_out['EXPTIME0'] = ( sci_header['EXPTIME'], sci_header.comments['EXPTIME'])
    # Convert DATE-OBS to MJD
    date_obs = sci_header['DATE-OBS']
    time_obj = Time(date_obs, format='isot', scale='utc')  # Assumes ISO format for DATE-OBS
    mjd = time_obj.mjd
    header_out['MJD-OBS'] = ( mjd, 'MJD shutter open' )

    return header_out
 
def tspec_confidence_interval_for_plotting(spec_image, quantile=0.95):
    """
    Computes the corresponding pixel values for a given confidence interval for visualization purposes.

    This function calculates the lower and upper bounds of a specified confidence interval 
    for the pixel values in a 2D spectral image. The interval can be used to plot the data 
    while excluding outliers.

    Parameters
    ----------
    spec_image : numpy.ndarray
        A 2D array representing the spectral image, where each element corresponds to a pixel value.
    
    quantile : float, optional
        The desired quantile for the confidence interval, by default 0.95. This should be a value 
        between 0 and 1, representing the fraction of the data to include within the interval.

    Returns
    -------
    range_conf : list of float
        A list containing the lower and upper bounds of the confidence interval.

    Notes
    -----
    - The function sorts all the pixel values in the spectral image and then uses the quantile value 
      to determine the range of values that fall within the specified confidence interval.
    - The default quantile of 0.95 corresponds to a 95% confidence interval.

    Example
    -------
    >>> import numpy as np
    >>> spec_image = np.random.normal(loc=0, scale=1, size=(100, 100))
    >>> range_conf = tspec_confidence_interval_for_plotting(spec_image, quantile=0.95)
    >>> print("Confidence interval:", range_conf)
    """    
    
    # Calculate the indices for the lower and upper bounds of the confidence interval
    n_elements = spec_image.shape[0] * spec_image.shape[1]
    lowindex = int(((1.0 - quantile) / 2) * n_elements)
    highindex = n_elements - lowindex - 1
    # Sort the values in the image data
    sortvalues = np.sort(spec_image, axis=None)
    # Extract the confidence range based on the calculated indices
    confidence_interval = [sortvalues[lowindex], sortvalues[highindex]]
    
    return confidence_interval


def tspec_plot_txt_order(order):
    """
    Generate a title text string for a given spectral order.

    This function returns a descriptive string for plotting purposes, indicating 
    the wavelength band and the spectral order number associated with the input.

    Parameters
    ----------
    order : int
        The spectral order number, expected to be between 3 and 7, inclusive.

    Returns
    -------
    title_order : str
        A string indicating the wavelength band and spectral order, formatted 
        for use in plot titles.

    Raises
    ------
    ValueError
        If the input `order` is not in the range of 3 to 7.

    Example
    -------
    >>> tspec_plot_txt_order(3)
    '(K-band, n=3)'

    >>> tspec_plot_txt_order(5)
    '(J-band, n=5)'
    """    

    if order == 3:
        title_order = '(K-band, n=3)'
    elif order == 4:
        title_order = '(H-band, n=4)'
    elif order == 5:
        title_order = '(J-band, n=5)'
    elif order == 6:
        title_order = '(Y-band, n=6)'
    elif order == 7:
        title_order = '(Y-band, n=7)'
    else:
        raise ValueError("Invalid order. Expected order between 3 and 7.")

    return title_order

######################################

def tspec_extract2d(science_frame, flat_solution, arc_solution, outfolder, 
                    sky_frame=None, flux_solution=None, 
                    vhelio_correction=False, reverse_slit=False,
                    sky_ratio=1.0, plot_ci=0.95, plot_aspectratio=2.5, plot_dpi=300):
    """
    Convert the cross-dispersed TripleSpec data into linearized spectro-images for each spectral order.
    
    Parameters
    ----------

    science_frame : str
        The full path of the on-source Science frame.
        
    flat_solution : str
        The full path of the flat-field solution frame
        (output from Spextool's calibration procedure).
        
    arc_solution : str
        The full path of the arc solution frame contaning both wavelength and spatial dispersion solution
        (output from Spextool's calibration procedure).
    
    outfolder : str
        The full path of the outfolder directory. Must exist otherwise the code will fail.
    
    sky_frame : str (default=None)
        The full path of the off-source Science frame.
    
    flux_solution : str (default=None)
        The full path of the flux-calibration solution frame.
        (output from Spextool's telluric correction procedure).
    
    vhelio_correction : bool (default=False)
        Compute and apply heliocentric velocity correction.
        
    reverse_slit : bool (default=False)
        If True, invert the spatial dispersion of the slit.
        
    sky_ratio : float (default=1.0)
        Multiplicative factor for the sky frame to improve sky subtraction correction.
    
    plot_ci : float (default=0.95)
        Confidence interval to remove outliers before exhibiting the spectral images.
    
    plot_aspectratio : float (default=2.5)
        Sets the aspect ratio for plotting the resulting spectral images.
    
    plot_dpi : int (defalt=300)
        DPI of the resulting PDF image showing the spectral images. 

    """
    # check if input files exist:
    if not os.path.isfile(science_frame):
        raise ValueError("  <ERROR> 'science_frame' does not exist. check input file.")
    if not os.path.isfile(flat_solution):
        raise ValueError("  <ERROR> 'flat_solution' does not exist. check input file.")
    if not os.path.isfile(arc_solution):
        raise ValueError("  <ERROR> 'arc_solution' does not exist. check input file.")
    if sky_frame is not None:
        if not os.path.isfile(sky_frame):
            raise ValueError("  <ERROR> 'sky_frame' does not exist. check input file.")
    if flux_solution is not None:
        if not os.path.isfile(flux_solution):
            raise ValueError("  <ERROR> 'flux_solution' does not exist. check input file.")
    
    # create outfolder if not exists
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)        
    
    # define saturation limit (in DN) for TripleSpec
    saturation_limit = 30000.00
    
    # read arc_solution file and get the WCS information
    arc_header    = fits.open(arc_solution)[0].header
    wave_solution = fits.open(arc_solution)[1].data
    spat_solution = fits.open(arc_solution)[2].data
    
    # check if the 'arc_solution' contains the required keywords
    if (arc_header['DISPO03'] <= 0):
        raise ValueError("  <ERROR> 'arc_solution' does not have the required keywords. check input file.")
        
    # read flat_solution file and get the WCS information
    flat_header = fits.open(flat_solution)[0].header
    flat_image  = fits.open(flat_solution)[1].data
    
    # check if the 'flat_solution' contains the required keywords
    if (flat_header['PLTSCALE'] <= 0):
        raise ValueError("  <ERROR> 'flat_solution' does not have the required keywords. check input file.")

    # now reads the sci image
    hdu_sci    = fits.open(science_frame)[0]
    image_sci  = hdu_sci.data
    sci_header = hdu_sci.header
    
    sci_exptime = sci_header['EXPTIME']
    
    if sky_frame is not None:
        hdu_sky    = fits.open(sky_frame)[0]
        image_sky  = hdu_sky.data
        sky_header = hdu_sky.header
        
        sky_exptime = sky_header['EXPTIME']
        
        if sky_exptime != sci_exptime:
             raise ValueError('Sci and Sky frame have different exposure times {} and {}'.format(sci_exptime,sky_exptime))
        
    if flux_solution is not None:
        flux = fits.open(flux_solution)[0]
        flux_image  = flux.data
        flux_header = flux.header
     
    # the dispersion for each order is given in the header of the arc solution
    dispersion = [ arc_header['DISPO03'], arc_header['DISPO04'], arc_header['DISPO05'], arc_header['DISPO06'], arc_header['DISPO07'] ]

    # flat correct the input AB images
    image_sci /= flat_image
    
    if sky_frame is not None:
        image_sky /= flat_image
        # make A-B
        image_sci = image_sci - ( image_sky * sky_ratio )
    
    # loop into orders 3 to 7
    for order in range(3, 8):
        
        cut_sci  = tspec_order_section(image_sci, order)
        cut_wave = tspec_order_section(wave_solution, order)
        cut_spat = tspec_order_section(spat_solution, order)

        # get the spatial grid to interpolate the data
        sy = tspec_grid_spatial(cut_spat, flat_header)
        
        # set the spectral dispersion for the order
        spec_disp = dispersion[order-3]
        
        # get the spectral grid to interpolate the data
        sw = tspec_grid_spectral(cut_wave, spec_disp)
        
        # now interpolate the data on the y-direction using the linear 'sy' scale
        interp_sci  = tspec_interp_spatial_grid(cut_sci,  cut_spat, sy)
        interp_wave = tspec_interp_spatial_grid(cut_wave, cut_spat, sy)
        
        # now interpolate the science frame into the new wavelength grid 'sw'
        img_out = tspec_interp_spectral_grid(interp_sci, interp_wave, sw) 
        
        # interpolate the flux calibration spectrum into the new spectral grid 'sw'
        if flux_solution is not None:
            flux_calibrate, sat_limit = tspec_flux_calibration_order(flux_image, order, sw, saturation_limit)
            img_out *= flux_calibrate
        else:
            flux_calibrate = None
            # in case of no flux correction, the saturation limit will be in DN/s units
            sat_limit = saturation_limit
        
        # truncate pixels above saturation limit threshold
        idx_sat = (img_out >= sat_limit)
        n_sat   = np.sum(idx_sat)
        if n_sat > 0:
            img_out[idx_sat] = sat_limit
        
        # heliocentric correction
        if vhelio_correction:
            sw_orig = sw
            sw, vhelio = tspec_heliocentric_correction(sci_header, sw)

            if order == 3:
                # I'm checking the heliocentric velocity correction around Br-Gamma line
                wave_limits_vhelio = (2.164,2.168)
                spec_vhelio = np.sum(img_out[5:11,:],axis=0)
                idx = ( sw >= wave_limits_vhelio[0] ) * ( sw <= wave_limits_vhelio[1] )
                ylim_vhelio = [np.nanmin(spec_vhelio[idx]),np.nanmax(spec_vhelio[idx])]
                ylim_vhelio[1] *= 1.1
                
                plt.figure(figsize=(4*plot_aspectratio,4))
                plt.plot(sw_orig,spec_vhelio, label='Original')

                plt.plot(sw, spec_vhelio, label='Vhelio={:.2f}km/s'.format(vhelio))
                plt.title("Heliocentric velocity correction")

                plt.axvline(2.16612,color='black',ls='--',label='Br-gamma')
                plt.xlim(wave_limits_vhelio)
                plt.ylim(ylim_vhelio)
                plt.xlabel("Wavelength (micron)")
                plt.ylabel("Flux")
                plt.legend()
                plt.show()

        # in case the user wants to reverse the slit direction
        if reverse_slit is True:
            img_out = np.flip(img_out,axis=0)
        
        # now divide the output by the exposure time
        img_out  /= sci_exptime
        sat_limit /= sci_exptime
        
        # create new header and output file name
        header_out = tspec_create_new_header(sci_header, sw, sy)
        
        # add specific keywords on the output header
        root_on = science_frame.split('/')[-1]
        if sky_frame is not None:
            root_off = sky_frame.split('/')[-1]
            header_out['TS2D_MOD'] = ( 'ON-OFF', 'ON-OF or ON' )
            header_out['TS2D_ON']  = ( root_on,  'ON image'    )
            header_out['TS2D_OFF'] = ( root_off, 'OFF image'   )
        else:
            header_out['TS2D_MOD'] = ( 'ON',     'ON-OF or ON' )
            header_out['TS2D_ON']  = ( root_on,  'ON image'    )
            
        if vhelio_correction:
            header_out['VHELIO']   = ( vhelio, 'Heliocentric velocity correction (in km/s)' )
        
        flag_fluxcor = True if flux_solution else False
        header_out['TS2D_FLC']   = ( flag_fluxcor, 'True if flux calibration was applied' )
        header_out['YUNITS']     = ( flux_header['YUNITS'], flux_header.comments['YUNITS'] ) if flux_solution else ( 'DN/sec', 'Flux units (per DN/sec)' )
        header_out['TS2D_SAT']   = ( sat_limit, 'Saturation limit (in YUNITS units)' )
        header_out['TS2D_NST']   = ( n_sat,   'Number of saturated pixels' )    

        # create output names based on the ON-Source file
        output = outfolder + root_on.split('.fits')[0] + '_2d_{}'.format(order)
        
        # Create a PrimaryHDU with your numpy array as data and the new header
        hdu_out = fits.PrimaryHDU(data=img_out, header=header_out)
        hdu_out.writeto(output + '.fits', overwrite=True)

        # now plot
        range_conf = tspec_confidence_interval_for_plotting(img_out, quantile=plot_ci)
        
        # now plot
        plt.figure(figsize=(4*plot_aspectratio,4))
        img=plt.imshow(img_out, origin='lower', aspect='auto', vmin=range_conf[0], vmax=range_conf[1], extent=[sw.min(), sw.max(), sy.min(), sy.max()])
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Slit direction (")')
        plt.colorbar(img,label=header_out['YUNITS'], pad=0.025 / plot_aspectratio, aspect=15)
        plt.title(sci_header['OBJECT'] + ' ' + tspec_plot_txt_order(order))
        plt.savefig(output+'.pdf',format='pdf',dpi=plot_dpi, bbox_inches = 'tight')
        plt.tight_layout()
        plt.show()
        