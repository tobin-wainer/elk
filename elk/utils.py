import numpy as np


def flux_to_mag(flux):
    # TODO: These numbers deserve some explanation
    m1 = 10
    f1 = 15000
    mag = 2.5 * np.log10(f1 / flux) + m1
    return mag


def flux_err_to_mag_err(flux, flux_err):
    # TODO: Same as above
    d_mag_d_flux = -2.5 / (flux * np.log(10))
    m_err_squared = abs(d_mag_d_flux)**2 * flux_err**2
    return np.sqrt(m_err_squared)
