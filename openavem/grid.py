"""
Gridding functionalities
"""

##
# Imports
##

# Import system modules
import os

# Import additional modules
import numpy as np
import xarray as xr

# Import modules from the project
from openavem.physics import eng_notation, FT_TO_METER, geod_sphere


###
# Contants
##

# Vertical grid edges in meters
# (http://wiki.seas.harvard.edu/geos-chem/index.php/
#   GEOS-Chem_vertical_grids#47-layer_reduced_vertical_grid)
VEDGES_LTO = [-6, 1000, 80581]

GC72_ALT_I = [-6, 123, 254, 387, 521, 657, 795, 934, 1075, 1218, 1363, 1510,
              1659, 1860, 2118, 2382, 2654, 2932, 3219, 3665, 4132, 4623, 5142,
              5692, 6277, 6905, 7582, 8320, 9409, 10504, 11578, 12633, 13674,
              14706, 15731, 16753, 17773, 18807, 19855, 20920, 22004, 23108,
              24240, 25402, 26596, 27824, 29085, 30382, 31716, 33101, 34539,
              36030, 37574, 39173, 40825, 42529, 44286, 46092, 47946, 49844,
              51788, 53773, 55794, 57846, 59924, 62021, 64130, 66245, 68392,
              70657, 73180, 76357, 80581]
GC72_ALT_M = [58, 189, 320, 454, 589, 726, 864, 1004, 1146, 1290, 1436, 1584,
              1759, 1988, 2249, 2517, 2792, 3074, 3439, 3896, 4375, 4879, 5413,
              5980, 6585, 7237, 7943, 8846, 9936, 11021, 12086, 13134, 14170,
              15198, 16222, 17243, 18269, 19309, 20364, 21438, 22531, 23648,
              24794, 25971, 27180, 28423, 29701, 31015, 32372, 33782, 35244,
              36759, 38328, 39951, 41627, 43355, 45134, 46962, 48835, 50754,
              52716, 54717, 56752, 58816, 60902, 63004, 65115, 67243, 69440,
              71812, 74594, 78146]
GC47_ALT_I = [-6, 123, 254, 387, 521, 657, 795, 934, 1075, 1218, 1363, 1510,
              1659, 1860, 2118, 2382, 2654, 2932, 3219, 3665, 4132, 4623, 5142,
              5692, 6277, 6905, 7582, 8320, 9409, 10504, 11578, 12633, 13674,
              14706, 15731, 16753, 17773, 19855, 22004, 24240, 26596, 31716,
              37574, 44286, 51788, 59924, 68392, 80581]
GC47_ALT_M = [58, 189, 320, 454, 589, 726, 864, 1004, 1146, 1290, 1436, 1584,
              1759, 1988, 2249, 2517, 2792, 3074, 3439, 3896, 4375, 4879, 5413,
              5980, 6585, 7237, 7943, 8846, 9936, 11021, 12086, 13134, 14170,
              15198, 16222, 17243, 18727, 20836, 23020, 25307, 28654, 34024,
              40166, 47135, 54834, 63053, 72180]

TO_GC_SPC = {'fuelburn': 'FUELBURN',
             'nox': 'NO2',
             'co': 'CO',
             'hc': 'HC',
             'nvpm': 'nvPM',
             'nvpmnum': 'nvPM_N'}

GMAO_025x03125_LAT = np.arange(-90, 90+0.25, 0.25)
GMAO_025x03125_LON = np.arange(-180, 180, 0.3125)

GMAO_05x0625_LAT = np.arange(-90, 90+0.5, 0.5)
GMAO_05x0625_LAT_EDGES = np.concatenate([[-90],
                                         np.arange(-89.75, 90, 0.5),
                                         [90]])
GMAO_05x0625_LON = np.arange(-180, 180, 0.625)
GMAO_05x0625_LON_EDGES = np.concatenate([[-180-0.625/2],
                                         np.arange(-180+0.625/2, 180, 0.625)])

GMAO_2x25_LAT = np.concatenate(([-89.5], np.arange(-88, 88+2, 2), [89.5]))
GMAO_2x25_LON = np.arange(-180, 180, 2.5)

GMAO_4x5_LAT = np.concatenate(([-89], np.arange(-86, 86+4, 4), [89]))
GMAO_4x5_LON = np.arange(-180, 180, 5)

GENERIC_1x1_LAT = np.arange(-89.5, 89.5, 1)
GENERIC_1x1_LON = np.arange(-179.5, 179.5+1, 1)

# Transition altitude to pressure levels [m]
ALT_TRANSITION = 10000 * FT_TO_METER


###
# Classes
##


###
# Functions
##

def print_ds_totals(ds):
    """
    Print emission totals from a grid

    Parameters
    ----------
    ds : xr.Dataset
        Grid containing the emissions.

    Returns
    -------
    None.

    """
    sums = ds.sum()
    if sums['FUELBURN'] == 0:
        # Nothing to print if there are no emissions
        print('No fuel was burned, which might not be as good as it sounds...')
        return
    
    col_widths = [12, 10, 14]
    print(f'{"Species":{col_widths[0]}}'
          + f'{"Sum  ":>{col_widths[1]}}'
          + f'{"EI    ":>{col_widths[2]}}')
    for v in ds.data_vars:
        if v == 'nvPM_N':
            str_sum = f'{sums[v].values:.2e}'
            ei = sums[v] / sums['FUELBURN']
            str_ei = f'{ei.values:.2e} /kg'
        else:
            str_sum = eng_notation(sums[v] * 1e3, 'g')
            ei = sums[v] / sums['FUELBURN'] * 1e3
            str_ei = eng_notation(ei, 'g/kg')
        print(f'{v:{col_widths[0]}}'
              + f'{str_sum:>{col_widths[1]}}'
              + f'{str_ei:>{col_widths[2]}}')


def save_nc4(ds, fname):
    if not fname.lower().endswith('.nc4'):
        fname = fname + '.nc4'
    
    n = len(ds.dims) - 2
    chunksizes = n * [1] + [ds.lat.size, ds.lon.size]
    encoding = dict(zlib=True, shuffle=True, complevel=1,
                    chunksizes=chunksizes)
    
    if isinstance(ds, xr.Dataset):
        enc = {v:encoding for v in list(ds.data_vars.keys())}
        ds.to_netcdf(fname, encoding=enc)
    elif isinstance(ds, xr.DataArray):
        ds.to_netcdf(fname, encoding={ds.name: encoding})
    else:
        raise TypeError('ds must be xr.Dataset or xr.DataArray.'
                        + f'{type(ds)} was given.')


def grid_from_simcfg(simcfg):
    vert = simcfg.grid_vertical
    if vert == 'uniform':
        ds_grid = make_3dgrid()
    elif vert.startswith('GC'):
        if simcfg.grid_vertical.endswith('r'):
            ds_grid = make_geoschem_grid(lvls=int(vert[2:-1]),
                                         reduced_vert_grid=True)
        else:
            ds_grid = make_geoschem_grid(lvls=int(vert[2:]))
    elif simcfg.grid_vertical == 'none':
        ds_grid = make_geoschem_grid(lvls=0)

    if simcfg.save_cfg_in_nc4:
        ds_grid.attrs['simcfg'] = simcfg.dumps()
    
    return ds_grid
    

def make_3dgrid(alt_top=None,
                lat=None,
                lon=None,
                add_vars=True,
                fill_value=0.0,
                emiss_bytime=False,
                emiss_byarea=False,
                attrs={'Conventions': 'COARDS', 'Format': 'NetCDF-4'}):
    """
    Create xr.Dataset with cartesian lat x lon x lev coordinates

    Parameters
    ----------
    alt_top : array_like, optional
        Altitude of the top edge of grid boxes in meters. If None, a grid with
        midpoints at 0, 500, ..., 44500 ft is created. The default is None.
    lat : array_like, optional
        Data values of the lat coordinate. If None, the GMAO 0.5 x 0.625 grid
        lat values will be used. The default is None.
    lon : array_like, optional
        Data values of the lon coordinate. If None, the GMAO 0.5 x 0.625 grid
        lon values will be used. The default is None.
    add_vars : bool, optional
        Whether to create data_vars in the dataset. The default is True.
    fill_value : float, optional
        Value to fill created data variables. Ignored if add_vars is False.
        The default is 0.0.
    emiss_bytime : bool, optional
        If True, the units attribute of data vars contains '/s'. The default
        is False.
    emiss_byarea : TYPE, optional
        If True, the units attribute of data vars contains '/m2'. The default
        is False.
    attrs : dict, optional
        Attributes to be added to the dataset. The default is
        {'Conventions': 'COARDS', 'Format': 'NetCDF-4'}.

    Raises
    ------
    ValueError
        If alt_top is not strictly increasing.

    Returns
    -------
    ds : xr.Dataset
        Dataset with coordinates defining a grid.
    
    """
    # First, get a 2D grid
    ds = make_geoschem_grid(lvls=0,
                            lat=lat,
                            lon=lon,
                            add_vars=add_vars,
                            fill_value=fill_value,
                            emiss_bytime=emiss_bytime,
                            emiss_byarea=emiss_byarea,
                            attrs=attrs)
    
    # Create vertical coordinates
    if alt_top is None:
        h_step = 500 * FT_TO_METER
        h_max = 44500 * FT_TO_METER
        alt_mid = np.arange(0, h_max + h_step, h_step)
        alt_top = (alt_mid[:-1] + alt_mid[1:]) / 2
        alt_top = np.append(alt_top, alt_top[-1] + h_step)
    else:
        if not isinstance(alt_top, np.ndarray):
            alt_top = np.array(alt_top)
        if np.any(np.diff(alt_top) <= 0):
            raise ValueError('alt_top should be strictly increasing.')
        alt_mid = np.concatenate(([min(0, (3*alt_top[0] - alt_top[1]) / 2)],
                                  (alt_top[1:] + alt_top[:-1])/2))
    
    dv_lev = xr.Variable('lev',
                         np.arange(1, len(alt_top) + 1),
                         attrs={'long_name': 'Vertical level',
                                'units': 'level',
                                'positive': 'up',
                                'axis': 'Z'})
    
    # Add vertical dimension to dataset
    ds = ds.expand_dims(lev=dv_lev).copy(deep=True)
    ds = ds.transpose('lev', 'lat', 'lon')
    ds = ds.assign_coords(alt_mid=('lev', alt_mid),
                          alt_top=('lev', alt_top))
    ds.alt_mid.attrs = {
        'long_name': ('Altitude of grid box midpoint'),
        'units': 'm',
        'positive': 'up',
        'axis': 'Z'}
    ds.alt_top.attrs = {
        'long_name': ('Altitude of grid box top edge'),
        'units': 'm',
        'positive': 'up',
        'axis': 'Z'}
    
    return ds


def make_geoschem_grid(lvls=0,
                       max_alt=None,
                       lat=None,
                       lon=None,
                       add_vars=True,
                       fill_value=0.0,
                       reduced_vert_grid=False,
                       emiss_bytime=False,
                       emiss_byarea=False,
                       attrs={'Conventions': 'COARDS', 'Format': 'NetCDF-4'}):
    """
    Create xr.Dataset with coordinates for GEOS-Chem classic

    Parameters
    ----------
    lvls : int, optional
        Number of vertical levels. Must be at most 72, or 42 if
        reduced_vert_grid is True. No lev coordinate is created if both lvl and
        max_alt are 0. The default is 0.
    max_alt : int, optional
        If lvls is 0, the grid will have the least amount of vertical levels
        necessary to include the altitude represented in meters by max_alt. If
        None, max_alt is ignored. The default is None.
    lat : array_like, optional
        Data values of the lat coordinate. If None, the GMAO 0.5 x 0.625 grid
        lat values will be used. The default is None.
    lon : array_like, optional
        Data values of the lon coordinate. If None, the GMAO 0.5 x 0.625 grid
        lon values will be used. The default is None.
    add_vars : bool, optional
        Whether to create data_vars in the dataset. The default is True.
    fill_value : float, optional
        Value to fill created data variables. Ignored if add_vars is False.
        The default is 0.0.
    reduced_vert_grid : bool, optional
        If True, use GEOS-Chem reduced 47-layer vertical grid. If False, use
        the 72-layer grid. The default is False.
    emiss_bytime : bool, optional
        If True, the units attribute of data vars contains '/s'. The default
        is False.
    emiss_byarea : TYPE, optional
        If True, the units attribute of data vars contains '/m2'. The default
        is False.
    attrs : dict, optional
        Attributes to be added to the dataset. The default is
        {'Conventions': 'COARDS', 'Format': 'NetCDF-4'}.

    Raises
    ------
    ValueError
        If lvls is not in the range [0, 72|47] or if lat or lon are not
        strictly increasing.

    Returns
    -------
    ds : xr.Dataset
        Dataset with coordinates defining a grid.
    
    """
    lvls = int(lvls)
    if reduced_vert_grid:
        maxlvls = 47
    else:
        maxlvls = 72
    if lvls < 0 or lvls > maxlvls:
        raise ValueError(
            f'lvls must be >= 0 and <= {maxlvls}. {lvls} was passed.')
    
    if lat is None:
        lat   = GMAO_05x0625_LAT
        lat_e = GMAO_05x0625_LAT_EDGES
    else:
        if not isinstance(lat, np.ndarray):
            lat = np.array(lat)
        diff = np.diff(lat)
        if np.any(diff <= 0):
            raise ValueError('lat values must be strictly increasing')
        dlat = np.median(diff)
        first = max(-90, lat[0] - dlat / 2)
        last = min(90, lat[-1] + dlat / 2)
        mid = (lat[1:] + lat[:-1]) / 2
        lat_e = np.concatenate([[first], mid, [last]])
        
    if lon is None:
        lon   = GMAO_05x0625_LON
        lon_e = GMAO_05x0625_LON_EDGES
    else:
        if not isinstance(lon, np.ndarray):
            lon = np.array(lon)
        diff = np.diff(lon)
        if np.any(diff <= 0):
            raise ValueError('lon values must be strictly increasing')
        dlon = np.median(diff)
        first = lon[0] - dlon / 2
        last = lon[-1] + dlon / 2
        mid = (lon[1:] + lon[:-1]) / 2
        lon_e = np.concatenate([[first], mid, [last]])
    
    # Convert [kg] -> [kg/s] | [kg/s] -> [kg/(m2.s)]?
    unitsattr = 'kg' + emiss_byarea * '/m2' + emiss_bytime * '/s'
    
    dv_lat = xr.Variable('lat', lat, attrs={'long_name': 'Latitude',
                                        'units': 'degrees_north',
                                        'axis': 'Y'})
    dv_lon = xr.Variable('lon', lon, attrs={'long_name': 'Longitude',
                                            'units': 'degrees_east',
                                            'axis': 'X'})
    
    if max_alt is not None and lvls == 0:
        if reduced_vert_grid:
            lvls = np.searchsorted(GC47_ALT_I, max_alt)
        else:
            lvls = np.searchsorted(GC72_ALT_I, max_alt)
        lvls = max(1, lvls)
    
    if lvls > 0:
        dv_lev = xr.Variable('lev',
                             np.arange(1, lvls + 1),
                             attrs={'long_name': 'GEOS-Chem levels',
                                    'units': 'level',
                                    'positive': 'up',
                                    'axis': 'Z'})
        ds = xr.Dataset(coords={'lev': dv_lev, 'lat': dv_lat, 'lon': dv_lon},
                        attrs=attrs)
        if reduced_vert_grid:
            ds = ds.assign_coords(alt_mid=('lev', GC47_ALT_M[:lvls]))
            ds = ds.assign_coords(alt_top=('lev', GC47_ALT_I[1:lvls+1]))
        else:
            ds = ds.assign_coords(alt_mid=('lev', GC72_ALT_M[:lvls]))
            ds = ds.assign_coords(alt_top=('lev', GC72_ALT_I[1:lvls+1]))
        ds.alt_mid.attrs = {
            'long_name': ('Altitude of grid box midpoint'),
            'units': 'm',
            'positive': 'up',
            'axis': 'Z'}
        ds.alt_top.attrs = {
            'long_name': ('Altitude of grid box top edge for a column at '
                          + 'atmospheric sea level'),
            'units': 'm',
            'positive': 'up',
            'axis': 'Z'}
        da_empty = xr.DataArray(fill_value,
                                coords=ds.coords,
                                attrs={'units': unitsattr})
        da_empty = da_empty.transpose('lev', 'lat', 'lon')
    else:
        ds = xr.Dataset(coords={'lat': dv_lat, 'lon': dv_lon}, attrs=attrs)
        da_empty = xr.DataArray(fill_value,
                                coords=ds.coords,
                                attrs={'units': unitsattr})
        da_empty = da_empty.transpose('lat', 'lon')
    
    # Add coordinates of lat and lon edges
    ds = ds.assign_coords(lat_south=('lat', lat_e[:-1]),
                          lat_north=('lat', lat_e[1:]),
                          lon_west=('lon', lon_e[:-1]),
                          lon_east=('lon', lon_e[1:]))
    ds.lat_south.attrs = {
        'long_name': ('Latitude of grid box south edge'),
        'units': 'degrees_north',
        'axis': 'Y'}
    ds.lat_north.attrs = {
        'long_name': ('Latitude of grid box north edge'),
        'units': 'degrees_north',
        'axis': 'Y'}
    ds.lon_west.attrs = {
        'long_name': ('Longitude of grid box west edge'),
        'units': 'degrees_east',
        'axis': 'X'}
    ds.lon_east.attrs = {
        'long_name': ('Longitude of grid box east edge'),
        'units': 'degrees_east',
        'axis': 'X'}
    
    if add_vars:
        ds = ds.assign({'FUELBURN': da_empty.copy(),
                        'NO2': da_empty.copy(),
                        'HC': da_empty.copy(),
                        'CO': da_empty.copy(),
                        'nvPM': da_empty.copy(),
                        'nvPM_N': da_empty.copy(),
                        })
        ds['FUELBURN'].attrs['long_name'] = 'Aviation fuel burned'
        ds['NO2'].attrs['long_name'] = 'Aviation emitted NOx (NO2 base)'
        ds['HC'].attrs['long_name'] = 'Aviation emitted HC'
        ds['CO'].attrs['long_name'] = 'Aviation emitted CO'
        ds['nvPM'].attrs['long_name'] = ('Aviation emitted non-volatile '
                                         + 'particulate matter (mass)')
        ds['nvPM_N'].attrs['long_name'] = ('Aviation emitted non-volatile '
                                           + 'particulate matter (number)')
    
    return ds


def grid_nearest_neighbor(segs, ds, multiplier=1,
                          species=['fuelburn', 'nox', 'co', 'hc', 'nvpm',
                                   'nvpmnum'],
                          reset=False, inplace=True, pos='start'):
    """
    Allocate emission segments into cartesian grid by nearest neighbor

    Parameters
    ----------
    segs : list of openavem.EmissionSegment
        Emission segments to be added to the grid.
    ds : xr.Dataset
        Dataset containing the grid.
    multiplier : float
        Factor that multiplies all emissions. Used for adding multiple flights
        with the same type and origin/destination pair.
        The default is 1.
    species : list of str, optional
        Names of the emission fields in the segments that are to be allocated.
        The default is ['fuelburn', 'nox', 'co', 'hc', 'nvpm', 'nvpmnum'].
    reset : bool, optional
        If True, previous emissions are discarded. If False, emissions are
        added. The default is True.
    pos : string, optional
        In which part of the segment emissions are concentrated. Possible
        values are '(s)tart', '(e)nd', '(m)id'. The default is 'start'.

    Raises
    ------
    Exception
        If pos[0] is different than 's', 'e' and 'm'.

    Returns
    -------
    ds : xr.Dataset
        Dataset with emissions added.
        
    Notes
    -----
    Segments that fall outside the grid will result in an IndexError.
    
    """
    # Do nothing if there are no emission segments
    if len(segs) == 0:
        if reset:
            ds = xr.zeros_like(ds)
        return ds
    
    if pos[0] not in ['s', 'e', 'm']:
        raise Exception('Invallid position argument (!= "s", "e", "m")')
    
    lats = ds['lat'].values
    lons = ds['lon'].values
    if 'lev' in ds.dims:
        alt_tops = ds['alt_top'].values
        shape = (len(alt_tops), len(lats), len(lons))
    else:
        shape = (len(lats), len(lons))
    newemissions = {
        spc: np.zeros(shape) for spc in species
    }
    # Double lon vector so longitude can roll over
    lons = np.concatenate([lons, lons + 360])
    
    # _v suffix for vector
    if pos[0] == 's':
        lat_v = [s.lat_s for s in segs]
        lon_v = [s.lon_s for s in segs]
        h_v   = [s.h_s for s in segs]
    elif pos[0] == 'e':
        lat_v = [s.lat_e for s in segs]
        lon_v = [s.lon_e for s in segs]
        h_v   = [s.h_e for s in segs]
    elif pos[0] == 'm':
        lat_v = [(s.lat_s + s.lat_e) / 2 for s in segs]
        lon_v = np.array([(s.lon_s + s.lon_e) / 2 for s in segs])
        roll = [abs(s.lon_s - s.lon_e) > 180 for s in segs]
        lon_v = np.where(roll, lon_v + 180, lon_v)
        h_v   = [(s.h_s + s.h_e) / 2 for s in segs]
    
    lat_high_i_v = lats.searchsorted(lat_v)
    lat_high_i_v = lat_high_i_v.clip(max=len(lats)-1)
    lat_low_i_v = (lat_high_i_v - 1).clip(min=0)
    lat_high_v = lats[lat_high_i_v]
    lat_low_v = lats[lat_low_i_v]
    go_low_v = lat_v - lat_low_v < lat_high_v - lat_v
    target_lat_v = np.where(go_low_v, lat_low_i_v, lat_high_i_v)
    
    lon_high_i_v = lons.searchsorted(lon_v)
    lon_high_i_v = lon_high_i_v.clip(max=len(lons)-1)
    lon_low_i_v = (lon_high_i_v - 1).clip(min=0)
    lon_high_v = lons[lon_high_i_v]
    lon_low_v = lons[lon_low_i_v]
    go_low_v = lon_v - lon_low_v < lon_high_v - lon_v
    target_lon_v = np.where(go_low_v, lon_low_i_v, lon_high_i_v)
    target_lon_v = np.mod(target_lon_v, len(lons) / 2)
    
    if 'lev' in ds.dims:
        target_h_v = alt_tops.searchsorted(h_v)
        target_h_v = target_h_v.clip(max=len(alt_tops)-1)
        
        for s, lev_i, lat_i, lon_i in zip(segs, target_h_v, target_lat_v,
                                          target_lon_v):
            lon_i = int(lon_i)
            for spc in species:
                newemissions[spc][lev_i, lat_i, lon_i] += s.__dict__[spc]
    else:
        for s, lat_i, lon_i in zip(segs, target_lat_v, target_lon_v):
            lon_i = int(lon_i)
            for spc in species:
                newemissions[spc][lat_i, lon_i] += s.__dict__[spc]
    
    for spc in species:
        newemissions[spc] *= multiplier
        gc_spc = TO_GC_SPC[spc]
        if inplace:
            if reset:
                ds[gc_spc].values = newemissions[spc]
            else:
                ds[gc_spc].values += newemissions[spc]
        else:
            ds = ds.copy(deep=True)
            if reset:
                ds[gc_spc].values = newemissions[spc]
            else:
                ds[gc_spc].values += newemissions[spc]
    
    return ds


def grid_supersampled(segs, ds, multiplier=1,
                      step_hor=1e3, step_vert=500.1*FT_TO_METER,
                      species=['fuelburn', 'nox', 'co', 'hc', 'nvpm', 'nvpmnum'],
                      reset=False):
    """Allocate emission into cartesian grid by super sampling segments
    
    Parameters
    ----------
    segs : list of openavem.EmissionSegment
        List of emission segments to allocate into the grid.
    ds : xr.Dataset
        Contains a cartesian (lat x lon) grid with lat and lon being in
        ascending order.
    multiplier : float
        Factor that multiplies all emissions. Used for adding multiple flights
        with the same aircraft type and origin/destination pair.
        The default is 1.
    step_hor : float, optional
        Length of maximum horizontal step in meters. The default is 10e3.
    step_vert : float, optional
        Length of maximum vertical step in meters.
        The default is 500.1*FT_TO_METER.
    species : list of str, optional
        Names of the emission fields in the segments that are to be allocated.
        The default is ['fuelburn', 'nox', 'co', 'hc', 'nvpm', 'nvpmnum'].
    reset : bool, optional
        If True, previous emissions are discarded. If False, emissions are
        added. The default is True.
    
    Raises
    ------
    Exception
        If any emission segment is missing necessary data (lat, lon, h, t,
        emissions).

    Returns
    -------
    ds : xr.Dataset
        Grid with emissions added
    
    """
    three_d = 'lev' in ds.dims
    
    # Do nothing if there are no emission segments
    if len(segs) == 0:
        if reset:
            ds = xr.zeros_like(ds)
        return ds
    
    # Temporary matrix to hold new emissions to be added
    #  (with double width so longitude rolls over)
    shape = ds[TO_GC_SPC[species[0]]].shape
    if three_d:
        len_lev, len_lat, len_lon = shape
        newemissions = {
            spc: np.zeros((len_lev, len_lat, 2*len_lon)) for spc in species
        }
        alt_top = ds.alt_top.values
    else:
        len_lat, len_lon = shape
        newemissions = {
            spc: np.zeros((len_lat, 2*len_lon)) for spc in species
        }
    lat_edges = ds.lat_south.values
    lon_edges = ds.lon_west.values
    lon_edges = np.concatenate([lon_edges, lon_edges+360])
    
    # Check if segments have all necessary data
    bad_data = np.isnan([[s.lat_s, s.lon_s, s.h_s,
                          s.lat_e, s.lon_e, s.h_e,
                          s.t_s, s.t_e]
                         + [s.__dict__[spc] for spc in species]
                         for s in segs]).any(axis=1)
    if bad_data.any():
        i = bad_data.argmax()
        print('np.nan found in the following emission segment:')
        print(segs[i].__dict__)
        raise Exception('Emission segment(s) with bad data')
    
    # Process emissions from each segment
    for seg in segs:
        if seg.lat_s == seg.lat_e and seg.lon_s == seg.lon_e:
            if not three_d or seg.h_s == seg.h_e:
                # The segment is actually a point
                visited_lat = np.array([seg.lat_s])
                visited_lon = np.array([seg.lon_s])
                visited_h = np.array([seg.h_s])
                weights = np.array([1.0])
            else:
                # Going straight up/down
                if seg.h_s < seg.h_e:
                    step = step_vert
                else:
                    step = -step_vert
                visited_h = np.arange(seg.h_s, seg.h_e, step)
                visited_h = np.append(visited_h, [seg.h_e])
                n_subs = len(visited_h) - 1
                visited_lat = np.full(n_subs + 1, seg.lat_s)
                visited_lon = np.full(n_subs + 1, seg.lon_s)
                if n_subs == 1:
                    # Simple case of no extra sampling
                    weights = np.array([0.5, 0.5])
                else:
                    tot_dist = seg.h_e - seg.h_s
                    e_frac = (seg.h_e - visited_h[-2]) / tot_dist
                    weights = np.concatenate([[step/2/tot_dist],
                                              np.full(n_subs-2, step/tot_dist),
                                              [step/2/tot_dist + e_frac/2],
                                              [e_frac/2]])
        else:        
            # Divide segment into "subsegments"
            line = geod_sphere.InverseLine(seg.lat_s, seg.lon_s,
                                           seg.lat_e, seg.lon_e)
            tot_dist = line.s13
            sub_lengths = np.arange(step_hor, tot_dist, step_hor)
            vert_steps = np.arange(
                step_vert, abs(seg.h_e - seg.h_s), step_vert)
            if len(vert_steps) > len(sub_lengths):
                # Steps are determined by vertical distance
                sub_lengths = vert_steps / abs(seg.h_e - seg.h_s) * tot_dist
                step = sub_lengths[0]
            else:
                # Steps are determined by horizontal distance
                step = step_hor
            n_subs = len(sub_lengths) + 1
            
            # Find position of each sample (subsegment endpoints)
            visited_lat = [seg.lat_s, seg.lat_e]
            visited_lon = [seg.lon_s, seg.lon_e]
            visited_h = [seg.h_s, seg.h_e]
            if n_subs > 1:
                for s12 in sub_lengths:
                    p = line.Position(s12)
                    visited_lat.append(p['lat2'])
                    visited_lon.append(p['lon2'])
                visited_h = np.concatenate(
                    [visited_h,
                     (sub_lengths / tot_dist) * (seg.h_e - seg.h_s) + seg.h_s])
                e_frac = (tot_dist - sub_lengths[-1]) / tot_dist
                weights = np.concatenate([[step/2/tot_dist],
                                          [e_frac/2],
                                          np.full(n_subs-2, step/tot_dist),
                                          [step/2/tot_dist + e_frac/2]])
            else:
                # Simple case of no extra sampling
                weights = np.array([0.5, 0.5])
            
        # Make sure we don't create or destroy emissions
        try:
            assert np.isclose(weights.sum(), 1)
        except:
            print(seg.__dict__)
            print(weights)
            print(weights.sum())
            raise
            
        # Find where the positions land on the grid
        lat_idx = lat_edges.searchsorted(visited_lat, 'right') - 1
        lon_idx = (lon_edges.searchsorted(visited_lon) - 1) % len_lon
        lowest_lat_idx = lat_idx.min()
        lowest_lon_idx = lon_idx.min()
        highest_lat_idx = lat_idx.max()
        highest_lon_idx = lon_idx.max()
        
        if three_d:
            lev_idx = alt_top.searchsorted(visited_h).clip(max=len(ds.lev)-1)
            lowest_lev_idx = lev_idx.min()
            highest_lev_idx = lev_idx.max()
        
            # Prepare 3D matrix with weights
            weight_grid = np.zeros([highest_lev_idx - lowest_lev_idx + 1,
                                    highest_lat_idx - lowest_lat_idx + 1,
                                    highest_lon_idx - lowest_lon_idx + 1])
            for i in range(len(weights)):
                weight_grid[lev_idx[i] - lowest_lev_idx,
                            lat_idx[i] - lowest_lat_idx,
                            lon_idx[i] - lowest_lon_idx] += weights[i]
            
            # Add 3D emissions for this segment
            for spc in species:
                newemissions[spc][lowest_lev_idx:highest_lev_idx+1,
                                  lowest_lat_idx:highest_lat_idx+1,
                                  lowest_lon_idx:highest_lon_idx+1] += (
                                      weight_grid * seg.__dict__[spc]
                                  )
        else:
            # Prepare 2D matrix with weights
            weight_grid = np.zeros([highest_lat_idx - lowest_lat_idx + 1,
                                    highest_lon_idx - lowest_lon_idx + 1])
            for i in range(len(weights)):
                weight_grid[lat_idx[i] - lowest_lat_idx,
                            lon_idx[i] - lowest_lon_idx] += weights[i]
        
            # Add 2D emissions for this segment
            for spc in species:
                newemissions[spc][lowest_lat_idx:highest_lat_idx+1,
                                  lowest_lon_idx:highest_lon_idx+1] += (
                                      weight_grid * seg.__dict__[spc]
                                  )

    # Now add newemissions to main grid
    for spc in species:
        # Roll temporary grid back into a [-180:+180] longitude range
        if three_d:
            newemissions[spc] = (newemissions[spc][:,:,:len_lon]
                                 + newemissions[spc][:,:,len_lon:])
        else:
            newemissions[spc] = (newemissions[spc][:,:len_lon]
                                 + newemissions[spc][:,len_lon:])
        newemissions[spc] *= multiplier
        gc_spc = TO_GC_SPC[spc]
        if reset:
            ds[gc_spc].values = newemissions[spc]
        else:
            ds[gc_spc].values += newemissions[spc]
    
    return ds


def load_splits(simcfg):
    """
    Load emission splits

    Parameters
    ----------
    simcfg : SimConfig
        Configuration of simulation options.

    Returns
    -------
    subds : dict of xr.Dataset
        Gridded emissions calculated for each fragment ("sub-task") of the job
        (split by aircraft type or by country of departure).
    loaded_fpaths : list of str
        Paths to .nc4 files that were loaded.

    """
    subds = {}
    basename_length = len(simcfg.output_name) + 1
    fnames = os.listdir(simcfg.splitdir)
    loaded_fpaths = []
    for f in fnames:
        if f.startswith(simcfg.output_name + '_') and f.endswith('.nc4'):
            fpath = os.path.join(simcfg.splitdir, f)
            loaded_fpaths.append(fpath)
            sub = f[basename_length:-4]
            subds[sub] = xr.load_dataset(fpath)
    
    return subds, loaded_fpaths


def merge_splits(ds, subds, simcfg, verbose=False):
    """
    Merge emission splits into a single Dataset across a new dimension

    Parameters
    ----------
    ds : xr.Dataset
        Gridded emissions, summed across all aircraft types and countries.
    subds : dict of xr.Dataset
        Gridded emissions calculated for each fragment ("sub-task") of the job
        (split by aircraft type or by country of departure).
    simcfg : SimConfig
        Configuration of simulation options.
    verbose : bool
        If True, print splits as they are processed.

    Returns
    -------
    ds : xr.Dataset
        Gridded emissions with new dimension.

    """
    dim = simcfg.split_job
    
    # Add overall emissions
    ds = ds.expand_dims(dim)
    ds = ds.assign_coords({dim: (dim, ['*'])})
    dss = [ds]
    ds.close()
    
    for sub in subds:
        if verbose:
            print(sub, end=', ', flush=True)
        # Add split emissions
        subds[sub] = subds[sub].expand_dims(dim)
        subds[sub] = subds[sub].assign_coords({dim: (dim, [sub])})
        dss.append(subds[sub])
        subds[sub].close()
    ds = xr.concat(dss, dim)
    
    return ds


##
# Main routine
##

if __name__ == '__main__':
    pass
