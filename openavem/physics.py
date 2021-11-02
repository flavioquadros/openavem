"""
Physical constants and general purpose math|physics functions
"""

##
# Imports
##

# Import system modules

# Import additional modules
import numpy as np
from geographiclib.geodesic import Geodesic

# Import modules from the project


##
# Parameters
##


##
# Constants
##

FT_TO_METER = 0.3048            # [ft] -> [m]
KT_TO_MPS = 1.852/3.6           # [kt] -> [m/s]
NM_TO_METER = 1852              # [nm] -> [m]
FPM_TO_MPS = FT_TO_METER / 60   # [ft/min] -> [m/s]

PSI_TO_PA = 6894.76             # [psi] -> [Pa]
BAR_TO_PSI = 14.5038            # [bar] -> [psi]

FUEL_LHV = 43.1e6               # [J/kg]

LTO_CEILING = 3000 * FT_TO_METER

R_EARTH = 6371e3  # radius of the Earth in [m]

AIR_CP = 1.005     # specific heat capacity at constant pressure [kJ/(kg.K)]
AIR_GAMMA = 1.4    # specific heat ratio [-]
AIR_R = 287.05287  # gas constant [m2/(K.s2)]

# ICAO ISA constants at mean surface level
MSL_T = 288.15   # temperature [K]
MSL_P = 101325   # pressure [Pa]
MSL_RHO = 1.225  # specific mass [kg/m3]
MSL_A = 340.294  # speed of sound [m/s]

ISA_T_GRAD = -0.0065    # ISA tropospheric temperature gradient [K/m]
ISA_TROPOPAUSE = 11000  # ISA tropopause geopotential altitude [m]
ISA_T_TROPOPAUSE = MSL_T + ISA_T_GRAD * ISA_TROPOPAUSE

G0 = 9.80665  # standard gravity [m/s2]

##
# Global variables
##

geod = Geodesic.WGS84  # define the WGS84 ellipsoid
geod_sphere = Geodesic(R_EARTH, 0)  # define the WGS84 ellipsoid
   
##
# Functions
##
   
def eng_notation(x, unit='', precision=3):
    """
    Format a number into engineering notation

    Parameters
    ----------
    x : float
        Number to be formatted.
    unit : str, optional
        Unit symbol. The default is ''.
    precision : int, optional
        Number of significant digits. E.g. pi with precision 3 is 3.14.
        The default is 3.

    Returns
    -------
    s : str
        Representation of the number in engineering notation.

    """
    x = float(x)
    unit = str(unit)
    precision = max(1, int(precision))
    
    if x == 0:
        return f'{0:.{precision-1}f} {unit}'
    
    if x < 0:
        sign = '-'
        x = -x
    else:
        sign = ''
    
    exp = int(np.floor(np.floor(np.log10(x))/3)*3)
    exp = np.clip(exp, -24, 24)
    si_prefixes = {-24: 'y',
                   -21: 'z',
                   -18: 'a',
                   -15: 'f',
                   -12: 'p',
                   -9:  'n',
                   -6:  'μ',
                   -3:  'm',
                   0:   '',
                   3:   'k',
                   6:   'M',
                   9:   'G',
                   12:  'T',
                   15:  'P',
                   18:  'E',
                   21:  'Z',
                   24:  'Y'}
    prefix = si_prefixes[exp]
    significand = x / 10.0**exp
    sig_len = len(str(int(significand)))
    ndecimals = max(0, int(precision - sig_len))
    s = f'{sign}{significand:.{ndecimals}f} {prefix}{unit}'
    
    return s


def isa_trop(Hp, dT=0, dp=0):
    """
    Temperature, pressure, density, speed of sound at given altitude (ICAO ISA)

    Parameters
    ----------
    Hp : float
        Geopotential altitude in meters.
    dT : float, optional
        Temperature difference at MSL in K. The default is 0.
    dp : float, optional
        Pressure difference at MSL in Pa. The default is 0.

    Returns
    -------
    T : float
        Temperature in K.
    p : float
        Pressure in Pa.
    rho : float
        Density in kg/m3.
    a : float
        Speed of sound in m/s.

    """
    if Hp < ISA_TROPOPAUSE:
        T = MSL_T + dT + ISA_T_GRAD * Hp
        p = MSL_P * ((T - dT) / MSL_T)**(-G0 / (ISA_T_GRAD * AIR_R))
    else:
        T = ISA_T_TROPOPAUSE + dT
        p = (
            MSL_P * ((T - dT) / MSL_T)**(-G0 / (ISA_T_GRAD * AIR_R))
            * np.exp(-G0 / (AIR_R * ISA_T_TROPOPAUSE) * (Hp - ISA_TROPOPAUSE))
        )
    rho = p / (AIR_R * T)
    a = np.sqrt(AIR_GAMMA * AIR_R * T)
    
    return T, p, rho, a


def spherical_dist_rad(phi1, phi2, lambda1, lambda2, R=6367000):
    """
    Great circle distance in meters from coordinates in radians

    Parameters
    ----------
    phi1 : float
        Starting point latitude in radians north.
    phi2 : float
        End point latitude in radians north.
    lambda1 : float
        Starting point longitude in radians east.
    lambda2 : float
        End point longitude in radians east.
    R : float, optional
        Radius of the Earth in meters. The default is 6367000.

    Returns
    -------
    d : float
        Great circle distance in meters.

    """
    d = R * 2 * np.arcsin(np.sqrt(
        np.square(np.sin((phi2 - phi1) / 2))
        + np.cos(phi1) * np.cos(phi2) * np.square(np.sin((lambda2 - lambda1)
                                                         / 2))))
    
    return d


def distance(lat1, lon1, lat2, lon2, earth='sphere'):
    """
    Return forward azimuth and geodesic distance between two points

    Parameters
    ----------
    lat1 : float
        Starting point latitude in degress north.
    lon1 : float
        Starting point longitude in degress east.
    lat2 : float
        End point latitude in degress north.
    lon2 : float
        End point longitude in degress east.
    earth : str, optional
        Earth model to use. Possible values are 'sphere' and 'WSG84'.
        The default is 'sphere'.

    Raises
    ------
    NotImplementedError
        If earth is not recognized.

    Returns
    -------
    fwd_azimuth : float
        Forward azimuth in degrees clockwise from north [0, +360).
    dist : float
        Geodesic distance in meters.

    """
    if earth == 'sphere':
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        fwd_azimuth = np.arctan2(
            np.sin(dlon), np.cos(lat1)*np.tan(lat2)-np.sin(lat1)*np.cos(dlon)
        )
        # Convert range from [-pi,pi] to [0,360]
        fwd_azimuth = np.degrees(fwd_azimuth)
        fwd_azimuth = (fwd_azimuth + 360) % 360
        
        a = (np.sin(dlat/2.0)**2
             + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        dist = R_EARTH * c
    elif earth == 'WGS84':
        geo_dict = geod.Inverse(lat1, lon1, lat2, lon2)
        fwd_azimuth = geo_dict['azi1']
        dist = geo_dict['s12']
    else:
        raise NotImplementedError(
            'Selected earth model not implemented'
        )
    
    return fwd_azimuth, dist


def spherical_surfarea_rad(phi1, phi2, lambda1, lambda2, R=6367000):
    """
    Surface area in meters squared from coordinates in radians

    Parameters
    ----------
    phi1 : float
        Starting point latitude in radians north.
    phi2 : float
        End point latitude in radians north.
    lambda1 : float
        Starting point longitude in radians east.
    lambda2 : float
        End point longitude in radians east.
    R : float, optional
        Radius of the Earth in meters. The default is 6367000.

    Returns
    -------
    S : float
        Area in square meters.

    """
    dlambda = (lambda2 - lambda1) % (2 * np.pi)
    dlambda = np.where(dlambda==0, 2*np.pi, dlambda)
    
    dcosphi = np.cos(phi2) - np.cos(phi1)
    dcosphi = np.where(dcosphi==0, 2, dcosphi)
    
    S = R*R * dlambda * dcosphi
    return S


def direct_geodesic_prob(lat, lon, azi, dist, earth='sphere'):
    """
    Return new (lat, lon, azi) after travelling dist at azimuth azi

    Parameters
    ----------
    lat : float
         Starting latitude in degress north.
    lon : float
        Starting longitude in degress east.
    azi : float
        Starting azimuth in degrees clockwise from north (-180, +180].
    dist : float
        Geodesic distance to travel in meters.
    earth : str, optional
        Earth model to use. Possible values are 'sphere' and 'WSG84'.
        The default is 'sphere'.

    Raises
    ------
    NotImplementedError
        If earth is not recognized.

    Returns
    -------
    lat2 : float
        End point latitude in degrees north.
    lon2 : float
        End point longitude in degrees east.
    azi2 : float
        End point azimuth in degrees clockwise from north (-180, +180].

    """
    if earth == 'sphere':
        lat = np.radians(lat)
        lon = np.radians(lon)
        azi = np.radians(azi)
        
        sigma_12 = dist / R_EARTH
        lat2 = np.arcsin(np.sin(lat) * np.cos(sigma_12)
                         + np.cos(lat) * np.sin(sigma_12) * np.cos(azi))
        dlon = np.arctan(np.sin(sigma_12) * np.sin(azi)
                         / (np.cos(lat) * np.cos(sigma_12)
                            - np.sin(lat) * np.sin(sigma_12) * np.cos(azi)))
        lon2 = lon + dlon
        azi2 = np.arctan2(np.sin(azi),
                          (np.cos(sigma_12) * np.cos(azi)
                           - np.tan(lat) * np.sin(sigma_12)))
        
        lat2 = lat2 * 180 / np.pi
        lon2 = lon2 * 180 / np.pi
        azi2 = azi2 * 180 / np.pi
    elif earth == 'WGS84':
        geo_dict = geod.Direct(lat, lon, azi, dist)
        lat2 = geo_dict['lat2']
        lon2 = geo_dict['lon2']
        azi2 = geo_dict['azi2']
    else:
        raise NotImplementedError(
            'Only spherical Earth model is supported currently (ellps=None)'
        )
    
    return lat2, lon2, azi2
    
