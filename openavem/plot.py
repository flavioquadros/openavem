"""
Plotting functionalities
"""

##
# Imports
##

# Import system modules
import copy

# Import additional modules
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import colorcet as cc


###
# Contants
##

MINLOGVALUE = 1e-9
DEFAULT_NLEVELS = 16
DEFAULT_CRANGE = 1e2

# Regional definitions matching GEOS-Chem
AREAS = {'G': [-180, 180, -90, 90],
         'EU': [-30, 50, 30, 70],
         'NA': [-140, -40, 10, 70],
         'AS': [60, 150, -11, 55]}
CLONS = {'G':0, 'EU':0, 'NA':-90, 'AS':105}
R_XLIM = {'EU': (-2750426.116179934, 4584043.526966555),
          'NA': (-4964018.842429203, 5026069.077959571),
          'AS': (-4510023.924036825, 4572663.145203998)}
R_YLIM = {'EU': (3643853.564079695, 7774469.607891487),
          'NA': (1234041.3450299501, 7774469.607891487),
          'AS': (-1356888.7854123458, 6386579.968095268)}


###
# Classes
##


###
# Functions
##

def refine(a, res):
    # if not a:
    #     return a
    refined = np.array([a.pop()])
    while a:
        refined = np.append(refined, np.linspace(refined[-1], a.pop(), res))
    return refined
    

def plot_basemap(prj='Mollweide', rgn='G', grid=False, grid_size=[0.625, 0.5],
                ax=None, fsize=[12, 6], tight_layout={}, nl=False, clon=None):
    """Plot a blank cartopy map with coastlines"""
    if ax is None:
        fig = plt.figure(figsize=fsize, tight_layout=tight_layout)
    else:
        fig = ax.figure
    area = AREAS[rgn]
    
    # Central longitude for Mollweide projection is set to 0 if the region
    # contains the 0 meridian, and it is set to the middle longitude otherwise
    if clon is None:
        if area[0] <= 0 and area[1] >= 0:
            clon = 0
        else:
            clon = (area[1]+area[0])/2
    
    # Plot base map in the appropriate projection
    if prj == 'Mollweide':
        if ax == None:
            ax = plt.axes(projection=ccrs.Mollweide(central_longitude=clon))
        if rgn != 'G':
            ax.set_xlim(R_XLIM[rgn])
            ax.set_ylim(R_YLIM[rgn])
    else:
        if ax == None:
            ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_xlim([area[0],area[1]])
        ax.set_ylim([area[2],area[3]])
        if grid:
            ax.set_xticks((np.arange(area[0],
                                     area[1]+0.1,
                                     grid_size[0],
                                     dtype=np.float32)).tolist())
            ax.set_yticks((np.arange(area[2],
                                     area[3]+0.1,
                                     grid_size[1],
                                     dtype=np.float32)).tolist())
    
    # Add coast lines, with higher resolution for regional maps
    if rgn == 'G':
        ax.coastlines(resolution='110m');
    else:
        ax.coastlines(resolution='10m');

    # Add griid lines
    if grid:
        xlocs = (np.arange(area[0],
                           area[1]+grid_size[0]+0.1,
                           grid_size[0],
                           dtype=np.float32)
                 - grid_size[0]/2).tolist()
        ylocs = (np.arange(area[2],
                           area[3]+grid_size[1]+0.1,
                           grid_size[1])
                 - grid_size[1]/2).tolist()
        ax.gridlines(linewidth=0.5, linestyle='-', xlocs=xlocs, ylocs=ylocs)
    
    # Add country borders for regional maps
    if rgn != 'G':
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.25,
                       edgecolor=[0,0,0,0.25])
        
    # Add the Netherlands in red for EU
    if rgn == 'EU' and nl:
        # Get outline of the Netherlands
        countries = shapereader.natural_earth(resolution='10m',
                                              category='cultural',
                                              name='admin_0_countries')
        for country in shapereader.Reader(countries).records():
            if country.attributes['SU_A3'] == 'NLD':
                netherlands = country.geometry
                break
        else:
            raise ValueError('Unable to find the NLD boundary.')

        ax.add_geometries([netherlands], ccrs.PlateCarree(), edgecolor='red',
                          facecolor='none')
    
    return fig, ax


def plot_colormesh(dss, vmax=None, vmin=None, linthresh=None, scale='lin',
                  drawColorbar=True, tickfmt=None, cmap='Reds', alpha=1.0,
                  rgn='G', prj='Mollweide', grid=False, grid_size=[5, 4],
                  fig=None, ax=None, fsize=[12, 6]):
    """Colormesh plot of a DataArray (dss) in a global map"""
    # Get base map
    if ax is None:
        if fig is None:
            fig, ax = plot_basemap(prj=prj, rgn=rgn, grid=grid,
                                   grid_size=grid_size, fsize=fsize)
        else:
            ax = fig.gca()
    else:
        fig = ax.figure
    plt.sca(ax)
    
    if vmax is None:
        vmax = dss.max()
    
    if vmin is None:
        vmin = dss.min()
    
    # For Mollweide projection, we have to remove the northern-most latitude
    # to not crash, for some reason
    if prj == 'Mollweide':
        dss = dss.sel(lat=slice(-89, 89))
    
    # Plot colormesh
    if scale == 'log':
        if vmin <= 0:
            vmin = max(MINLOGVALUE, dss.min())
        mesh = dss.plot(norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax),
                        cmap=cmap, alpha=alpha, transform=ccrs.PlateCarree(),
                        add_colorbar=drawColorbar)
    elif scale == 'symLog':
        absmax = vmax;
        if not linthresh or linthresh <= 0:
            linthresh = max(abs(dss).quantile(0.1), absmax/DEFAULT_CRANGE)
        mesh = dss.plot(norm=mplcolors.SymLogNorm(linthresh=linthresh,
                                                  linscale=0.5, vmin=-absmax,
                                                  vmax=absmax, base=np.e),
                        cmap=cmap, alpha=alpha, transform=ccrs.PlateCarree(),
                        add_colorbar=drawColorbar)
    else:
        mesh = dss.plot(vmax=vmax, vmin=vmin, cmap=cmap, alpha=alpha,
                        transform=ccrs.PlateCarree(),
                        add_colorbar=drawColorbar)
    if drawColorbar and tickfmt is not None:
        mesh.colorbar.formatter = ticker.FormatStrFormatter(tickfmt)
        mesh.colorbar.update_ticks()
    cbar = mesh.colorbar
    
    if 'long_name' in dss.attrs and 'time' in dss.coords:
        ax.set_title(
            dss.long_name + ' @ '
            + np.datetime_as_string(dss['time'].values,'D')
        )
    
    return fig, ax, cbar


def plot_fuelburn(ds, vmax=None, vmin=None):
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    
    if isinstance(ds, xr.DataArray):
        da = ds
    elif isinstance(ds, xr.Dataset):
        da = ds['FUELBURN']
    else:
        raise TypeError('Input must be xr.Dataset or xr.DataArray')

    if 'lev' in da.dims:
        da = da.sum('lev')
    
    if vmax is None:
        vmax = float(da.max())

    if vmin is None:
        vmin = vmax / 1e4
    
    cmap = copy.copy(cc.cm.CET_L17)
    cmap.set_bad([1, 1, 1, 0])
    fig, ax, cb = plot_colormesh(da,
                                 vmax=vmax,
                                 vmin=vmin,
                                 scale='log',
                                 cmap=cmap)
    
    return fig, ax, cb


def plot_wind(ds, lev=29, vmax=80, coarsen=1):
    """
    Quiver plot of wind at a given vertical level

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing wind speed (WS) in m/s and wind direction (WDIR) in
        degrees clockwise from north, or containing wind in the latitude (V) and
        longitude (U) directions in m/s.
    lev : int, optional
        Label of the vertical level to be selected. The default is 29.
    vmax : float, optional
        Maximum value for the plot. The default is 80.
    coarsen : int, optional
        Number of cells to average to coarsen the grid. If 1, the grid is not
        coarsened. The default is 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure created.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axis with the plot.
    cbar : matplotlib.colorbar.Colorbar
        Colorbar for the quiver plot.

    """
    ds = ds.sel(lev=29)
    
    if 'U' not in ds.data_vars:
        ds['U'] = ds.WS * np.sin(ds.WDIR * np.pi / 180)
    if 'V' not in ds.data_vars:
        ds['V'] = ds.WS * np.cos(ds.WDIR * np.pi / 180)
    
    if coarsen > 1:
        nlat = len(ds.lat)
        nlon = len(ds.lon)
        ds = ds.isel(lat=slice(0, nlat-nlat%coarsen),
                     lon=slice(0, nlon-nlon%coarsen))
        ds = ds.coarsen(lat=coarsen, lon=coarsen).mean()
        ds['WS'] = np.sqrt(ds.U**2 + ds.V**2)
        ds['WDIR'] = np.arctan2(ds.U, ds.V) * 180 / np.pi
    
    X = ds.lon.values
    Y = ds.lat.values
    U = ds.U.values
    V = ds.V.values
    C = ds.WS
    
    fig, ax = plot_basemap()
    
    norm = mplcolors.Normalize(vmin=0, vmax=vmax, clip=False)
    cmap = mplcolors.ListedColormap(plt.cm.viridis(np.linspace(0.25, 1, 10)),
                                    "name")
    
    Q = ax.quiver(X, Y, U, V, C, transform=ccrs.PlateCarree(), pivot='mid',
                  norm=norm, cmap=cmap, scale=vmax*24)
    cbar = fig.colorbar(Q, ticks=np.linspace(0, vmax,
                                             num=min(11, int(vmax/10+1))),
                        format='%.1f', label='Wind speed [m/s]')
    
    return fig, ax, cbar


##
# Main routine
##

