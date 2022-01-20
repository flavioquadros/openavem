"""
Functions to read input data
"""

##
# Imports
##

# Import system modules
import os
import warnings

# Import additional modules
import numpy as np
import pandas as pd
import xarray as xr

# Import modules from the project
from .core import Airport, Engine, APU
from . import physics
from . import dir_openavem


###
# Contants
##

FILEPATH_AC_CRUISE_FL = os.path.join(
    dir_openavem, r'aircraft/cruise_fl.csv')
FILEPATH_AC_ICAO_DESCRIPTION = os.path.join(
    dir_openavem, r'aircraft/ac_icao_description.csv')
FILEPATH_AC_LOADFACTORS = os.path.join(
    dir_openavem, r'aircraft/load_factors.csv')
FILEPATH_AC_MILITARY = os.path.join(
    dir_openavem, r'aircraft/military.csv')
FILEPATH_AC_NATSGRP = os.path.join(
    dir_openavem, r'aircraft/natsgrp.csv')
FILEPATH_AC_REPLACEMENTS = os.path.join(
    dir_openavem, r'aircraft/typecode_replacements.csv')
FILEPATH_ICAO_ACTYPES = os.path.join(
    dir_openavem, r'aircraft/AircraftTypes_31Dec2020.json')

FILEPATH_ISO_CC = os.path.join(
    dir_openavem, r'airports/iso_country_codes.csv')
FILEPATH_OPENFLIGHTS_APTS = os.path.join(
    dir_openavem, r'airports/airports.dat')
FILEPATH_OURAIRPORTS_APTS = os.path.join(
    dir_openavem, r'airports/OurAirports/airports.csv')
FILEPATH_OURAIRPORTS_RUNWAYS = os.path.join(
    dir_openavem, r'airports/OurAirports/runways.csv')
FILEPATH_TIM_STETTLER = os.path.join(
    dir_openavem, r'airports/tim_stettler.csv')
FILEPATH_MOVEMENTS = os.path.join(
    dir_openavem, r'airports/yearly_movements.csv')

FILEPATH_ENG_PROPERTIES = os.path.join(
    dir_openavem, r'engines/eng_properties.csv')
FILEPATH_ENG_ALLOCATIONS = os.path.join(
    dir_openavem, r'engines/eng_allocations.csv')
FILEPATH_APU_PROPERTIES = os.path.join(
    dir_openavem, r'engines/apu_properties.csv')

# Correction factors for EEDB fuel flow to account for installation effects
# of air bleed
BFFM2_FUEL_CORRECTION = np.array([1.100, 1.020, 1.013, 1.010])


###
# Parameters and global variables
##


###
# Classes
##


###
# Functions
##

def load_all_engines(filepath=FILEPATH_ENG_PROPERTIES):
    """
    Load all engines from filepath and return a dict to the created objects

    Parameters
    ----------
    filepath : str, optional
        Path to .csv file containing engine data.
        The default is FILEPATH_ENG_PROPERTIES.

    Returns
    -------
    engines : dict
        Dictionary of loaded engines with unique identifiers (UID) as keys.

    """
    if filepath is None:
        filepath = FILEPATH_ENG_PROPERTIES
    
    # Load everything
    engine_data = pd.read_csv(filepath,
                              index_col='uid',
                              dtype={'uid': str,
                                     'manufacturer': str,
                                     'combustor_description': str,
                                     'name': str,
                                     'wikipedia_link': str})
    engine_data.fillna({'combustor_description': '', 'wikipedia_link': ''},
                       inplace=True)
    
    # Create Engine objects and put them in a dict
    engines = {uid: Engine(uid, engine_data.loc[uid].to_dict())
               for uid in engine_data.index}
    
    return engines
    

def load_eng_allocations(filepath=FILEPATH_ENG_ALLOCATIONS):
    if filepath is None:
        filepath = FILEPATH_ENG_ALLOCATIONS
    return pd.read_csv(filepath, index_col='typecode')
    

def load_apus(filepath=FILEPATH_APU_PROPERTIES):
    if filepath is None:
        filepath = FILEPATH_APU_PROPERTIES
    apu_data = pd.read_csv(filepath, index_col='name', dtype={'name': str})
    apus = {n: APU(n, apu_data.loc[n].to_dict()) for n in apu_data.index}
    
    return apus


def allocate_eng(acs, simcfg, engines={}, apus={}, eng_allocations=None):
    """
    Assign engines and APU to aircraft

    Parameters
    ----------
    acs : dict of bada.Aircraft, list of bada.Aircraft, or bada.Aircraft
        Aircraft that will be assigned engines and APU.
    simcfg : SimConfig
        Configuration of simulation options.
    engines : dict of Engine, optional
        Dict of str to Engine objects that can be assigned to aircraft. If an
        empty dict is provided, it will be populated with load_all_engines().
        The default is {}.
    apus : dict of APU, optional
        Dict of str to APU objects that can be assigned to aircraft. If an
        empty dict is provided, it will be populated with load_apus().
        The default is {}.
    eng_allocations : pd.DataFrame or None, optional
        Engine and APU assignments. If None, load_eng_allocations() will be
        called to obtain it. The default is None.

    Raises
    ------
    TypeError
        If acs is not the expected type.

    Returns
    -------
    engines : dict of Engine
        Engines as given or as loaded (if engines={} was passed).
    apus : TYPE
        APUs as given or as loaded (if apus={} was passed).
    eng_allocations : TYPE
        Allocations as given or as loaded (if eng_allocations=None was passed).

    """
    # Convert acs to dict if needed
    if not hasattr(acs, '__iter__'):
        acs = {acs.typecode: acs}
    elif isinstance(acs, list):
        acs = {ac.typecode: ac for ac in acs}
    elif not isinstance(acs, dict):
        raise TypeError('acs must be a dict, list or single Aircraft object.'
                        + f' {type(acs)} was passed.')
    
    # Load engines if not given
    if len(engines) == 0:
        engines = load_all_engines(simcfg.eng_properties_path)
    
    # Load APUs if not given
    if len(apus) == 0:
        apus = load_apus(simcfg.apu_properties_path)
    
    # Load allocations if not given
    if eng_allocations is None:
        eng_allocations = load_eng_allocations(simcfg.eng_allocations_path)
    
    # Go over allocations
    for typecode, alloc in eng_allocations.iterrows():
        if typecode in acs:
            ac = acs[typecode]
            # Allocate engine
            if alloc['uid'] in engines:
                ac.eng = engines[alloc['uid']]
            else:
                warnings.warn(f'engine allocation failed "{typecode}": '
                              + f'"{alloc["uid"]}"')
            # Allocate APU
            if alloc['apu'] == 'none':
                ac.hasapu = False
                ac.apu = None
            elif alloc['apu'] == 'unknown/unavailable':
                ac.hasapu = True
            elif alloc['apu'] in apus:
                ac.hasapu = True
                ac.apu = apus[alloc['apu']]
            elif simcfg.apu_warn_missing:
                warnings.warn(f'APU allocation failed "{typecode}": '
                              + f'"{alloc["apu"]}"')
            ac.apu_acrp = alloc['apu_acrp']
            ac.apu_icao_adv = alloc['apu_icao_adv']
            ac.apu_icao_simple = alloc['apu_icao_simple']
    
    # Substitute missing engines
    if simcfg.use_replacement_eng:
        for ac in acs.values():
            if (not hasattr(ac, 'eng')
                and ac.mdl_typecode in acs
                and hasattr(acs[ac.mdl_typecode], 'eng')):
                ac.eng = acs[ac.mdl_typecode].eng
    
    # Substitute missing APUs
    if simcfg.use_replacement_apu:
        for ac in acs.values():
            if (not hasattr(ac, 'apu')
                and ac.mdl_typecode in acs
                and hasattr(acs[ac.mdl_typecode], 'apu')):
                # Aircraft has no specific APU model but substitute has
                ac.apu = acs[ac.mdl_typecode].apu
                ac.apu_acrp = acs[ac.mdl_typecode].apu_acrp
                ac.apu_icao_adv = acs[ac.mdl_typecode].apu_icao_adv
                ac.apu_icao_simple = acs[ac.mdl_typecode].apu_icao_simple
            elif (not hasattr(ac, 'apu_acrp')
                  and ac.mdl_typecode in acs
                  and hasattr(acs[ac.mdl_typecode], 'apu_acrp')):
                # Aircraft has no generic APU type, so substitute anyway
                ac.apu_acrp = acs[ac.mdl_typecode].apu_acrp
                ac.apu_icao_adv = acs[ac.mdl_typecode].apu_icao_adv
                ac.apu_icao_simple = acs[ac.mdl_typecode].apu_icao_simple
    
    return engines, apus, eng_allocations


def load_icao_actypes(filepath=FILEPATH_ICAO_ACTYPES):
    """
    Load ICAO aircraft type definitions from a .json into a pd.Dataframe

    Parameters
    ----------
    filepath : str, optional
        Path to the JSON file. The default is FILEPATH_ICAO_ACTYPES.

    Returns
    -------
    actypes : pd.Dataframe
        Data read, with special type designators added.
    
    References
    ----------
    ICAO DOC 8643 - Aircraft Type Designators.

    """
    actypes = pd.read_json(filepath)
    # Add special designators manually
    special = pd.DataFrame([
            ['Aircraft type not (yet) assigned a designator', 'ZZZZ'],
            ['Airship', 'SHIP'],
            ['Balloon', 'BALL'],
            ['Glider', 'GLID'],
            ['Microlight aircraft', 'ULAC'],
            ['Microlight autogyro', 'GYRO'],
            ['Microlight helicopter', 'UHEL'],
            ['Powered parachute/Paraplane', 'PARA'],
            ['Sailplane', 'GLID'],
            ['Ultralight aircraft', 'ULAC'],
            ['Ultralight autogyro', 'GYRO'],
            ['Ultralight helicopter', 'UHEL']],
        columns = ['ModelFullName', 'Designator']
    )
    actypes = pd.concat([actypes, special], ignore_index=True)
    
    # Add non-official special designators manually
    special = pd.DataFrame([
            ['Drone', 'DRON'],
            ['Aircraft under test', 'TEST']],
        columns = ['ModelFullName', 'Designator']
    )
    actypes = pd.concat([actypes, special], ignore_index=True)
    
    return actypes


def load_ac_icao_description(filepath=FILEPATH_AC_ICAO_DESCRIPTION):
    return pd.read_csv(filepath, header=0, index_col='Designator',
                       squeeze=True)

def load_ac_military(filepath=FILEPATH_AC_MILITARY):
    return pd.read_csv(filepath, header=None, squeeze=True)


def load_ac_replacements(filepath=FILEPATH_AC_REPLACEMENTS):
    return pd.read_csv(filepath, index_col=0, header=0, squeeze=True)
    

def load_ac_natsgrp(filepath=FILEPATH_AC_NATSGRP):
    return pd.read_csv(filepath, index_col='typecode')


def load_load_factors(filepath=FILEPATH_AC_LOADFACTORS):
    return pd.read_csv(filepath, index_col=['year', 'month'],
                       na_values={'month': ['*']})


def determine_payload_m_frac(simcfg):
    """
    Sets simcfg.payload_m_frac based on its value and simcfg.yyyymm

    Parameters
    ----------
    simcfg : SimConfig
        Configuration of simulation options.

    Raises
    ------
    Exception
        If payload_m_frac is None and a factor is not found for the month given
        by yyyymm.

    Returns
    -------
    m_frac : float
        Payload mass factor set in simcfg.

    """
    if simcfg.payload_m_frac is None:
        if simcfg.yyyymm is not None:
            year = int(simcfg.yyyymm[:4])
            month = int(simcfg.yyyymm[4:6])
            loadfactors = load_load_factors()
            if (year, month) in loadfactors.index:
                m_frac = float(loadfactors.loc[(year, month)])
            elif (year, np.nan) in loadfactors.index:
                m_frac = float(loadfactors.loc[(year, np.nan)])
            else:
                raise Exception('Load factor not found for month'
                                + f' {simcfg.yyyymm}')
        else:
            raise Exception("payload_m_frac and yyyymm are None, "
                            + "couldn't determine payload mass fraction")
    else:
        m_frac = simcfg.payload_m_frac
        
    if m_frac < 0 or m_frac > 100:
        raise ValueError('payload_m_frac needs to be between 0% and 100%'
                         + f', but {m_frac} was given.')
    elif m_frac > 1.0:
        m_frac /= 100.0
    simcfg.payload_m_frac = m_frac
    
    return m_frac


def load_cruise_fl(filepath=FILEPATH_AC_CRUISE_FL):
    return pd.read_csv(filepath, index_col='typecode')
   

def load_apts_from_simcfg(simcfg, iata_keys=False, fill_h=True, runways=True, movements=True, as_df=False):
    """
    Load airport data into dict of core.Airport according to simcfg

    Parameters
    ----------
    simcfg : SimConfig
        Configuration of simulation options.
    iata_keys : bool, optional
        If True, use IATA codes as keys. If False, use ICAO codes.
        The default is False.
    fill_h : bool, optional
        If True, fill missing altitude values with 0. The default is True.
    runways : bool, optional
        If True, call load_runway_lengths to assign runway max length and
        number. The default is True.
    movements : bool, optional
        If True, call load_yearly_movements to assign number of yearly aircraft
        movements. The default is True.
    as_df : bool, optional
        If True, return a pd.DataFrame instead of a dict. The default is False.
    
    Returns
    -------
    apts : dict or pd.DataFrame
        Dictionary of IATA or ICAO codes to core.Airport objects containing the
        data read.

    """
    apts = load_airports(source=simcfg.apt_source,
                         iata_keys=iata_keys,
                         closed=simcfg.apt_closed,
                         heliport=simcfg.apt_heliport,
                         runways=runways,
                         movements=movements,
                         tim_stettler=(simcfg.ltocycle=='Stettler'),
                         as_df=as_df)
    return apts


def load_airports(openflights_path=FILEPATH_OPENFLIGHTS_APTS,
                  ourairports_path=FILEPATH_OURAIRPORTS_APTS,
                  iso_cc_path=FILEPATH_ISO_CC,
                  source='all', iata_keys=False, closed=False, heliport=False,
                  fill_h=True, runways=True, movements=True,
                  tim_stettler=False, as_df=False):
    """
    Load airport data into dict of core.Airport

    Parameters
    ----------
    openflights_path : str, optional
        Path to csv file with OpenFlights airport database.
        The default is FILEPATH_OPENFLIGHTS_APTS.
    ourairports_path : str, optional
        Path to csv file with OurAirports airport database.
        The default is FILEPATH_OURAIRPORTS_APTS.
    source : str, optional
        Airport databases to use. Possible values are 'openflights',
        'ourairports', and 'all'. If 'all', use both OpenFlights and
        OurAirports, with preference for the first. The default is 'all'.
    iata_keys : bool, optional
        If True, use IATA codes as keys. If False, use ICAO codes.
        The default is False.
    closed : bool, optional
        If True, also load airports marked as "closed". The default is False.
    heliport : bool, optional
        If True, also load airports marked as "heliport". The default is False.
    fill_h : bool, optional
        If True, fill missing altitude values with 0. The default is True.
    runways : bool, optional
        If True, call load_runway_lengths to assign runway max length and
        number. The default is True.
    movements : bool, optional
        If True, call load_yearly_movements to assign number of yearly aircraft
        movements. The default is True.
    tim_stettler : bool, optional
        If True, call load_runway_lengths, load_yearly_movements and
        load_tim_stettler to assign time-in-mode values. The default is False.
    as_df : bool, optional
        If True, return a pd.DataFrame instead of a dict. The default is False.

    Raises
    ------
    Exception
        If database has duplicated ICAO codes, or if it has duplicated IATA
        codes and iata_keys is True.

    Returns
    -------
    apts : dict or pd.DataFrame
        Dictionary of IATA or ICAO codes to core.Airport objects containing the
        data read.

    """
    source = source.lower()
    if source not in ['openflights', 'ourairports', 'all']:
        raise ValueError(f'source "{source}" not recognized')
    
    iso_cc = pd.read_csv(iso_cc_path,
                         index_col='iso_country',
                         usecols=['iso_country', 'country', 'domestic_iso'],
                         keep_default_na=False)
    
    apt_types = ['airport', 'small_airport', 'medium_airport', 'large_airport',
                 'seaplane_base']
    if closed:
        apt_types += ['closed', 'closed_later']
    if heliport:
        apt_types.append('heliport')
    
    if source == 'openflights' or source == 'all':
        try:
            # If Open Flights database has headers
            df = pd.read_csv(
                openflights_path,
                usecols=['icao', 'iata', 'type', 'name', 'lat', 'lon', 'alt',
                         'country', 'tzoffset', 'dst', 'tz'],
                index_col='icao',
                na_values={'icao': r'\N', 'iata': r'\N'},
                dtype={'icao': str, 'iata': str}
            )
        except ValueError:
            # If Open Flights database doen't have headers
            df = pd.read_csv(
                openflights_path,
                names=['id', 'name', 'city', 'country', 'iata', 'icao', 'lat',
                       'lon', 'alt', 'tzoffset', 'dst', 'tz', 'type',
                       'source'],
                index_col='icao',
                na_values={'icao': r'\N', 'iata': r'\N'},
                dtype={'icao': str, 'iata': str}
            )
            df = df[['iata', 'type', 'name', 'lat', 'lon', 'alt', 'country',
                     'tzoffset', 'dst', 'tz']]
        df = df[df.index.notna()]
        df = df[df.type.isin(apt_types)]
        if iata_keys:
            df = df.dropna(subset=['iata'])
            if not df.iata.is_unique:
                print('OpenFlights database contains duplicated IATA code(s):')
                print(df[df.iata.duplicated()].iata.unique())
                warnings.warn('Duplicated IATA in OpenFlights')
        if not df.index.is_unique:
            print('OpenFlights database contains duplicated ICAO code(s):')
            print(df[df.index.duplicated()].index.unique())
            raise Exception('Duplicated ICAO in OpenFlights')
        df['alt'] *= physics.FT_TO_METER
        df = df.assign(domestic_iso=df['country'].map(
            iso_cc.set_index('country')['domestic_iso']))
    
    if source == 'ourairports' or source == 'all':
        df2 = pd.read_csv(
            ourairports_path,
            usecols=['icao', 'iata', 'type', 'name', 'lat', 'lon', 'alt',
                     'iso_country', 'tz'],
            index_col='icao',
            header=0,
            na_values={'iata': '0'},
            dtype={'icao': str, 'iata': str}
        )
        df2 = df2[df2.type.isin(apt_types)]
        if iata_keys:
            df2 = df2.dropna(subset=['iata'])
            if not df2.iata.is_unique:
                print('OurAirports database contains duplicated IATA code(s):')
                print(df2[df2.iata.duplicated()].iata.unique())
                warnings.warn('Duplicated IATA in OurAirports')
        if not df2.index.is_unique:
            print('OurAirports database contains duplicated ICAO code(s):')
            print(df2[df2.index.duplicated()].index.unique())
            raise Exception('Duplicated ICAO in OurAirports')
        df2['alt'] *= physics.FT_TO_METER
        df2['tzoffset'] = None
        df2['dst'] = None
        df2 = df2.assign(domestic_iso=df2['iso_country'].map(
            iso_cc[~iso_cc.index.duplicated()]['domestic_iso']))
    
    if source == 'ourairports':
        df = df2
    elif source == 'all':
        df = df.combine_first(df2)
    
    if fill_h:
        df.fillna({'alt': 0}, inplace=True)
    
    df = df.set_index(df.index.str.upper())
    df['iata'] = df['iata'].str.upper()
    
    if as_df:
        if iata_keys:
            apts = df.reset_index().set_index('iata')
        else:
            apts = df
        return apts
    
    if iata_keys:
        apts = {iata: Airport(icao, iata, lat, lon, h, tz, dst, cc)
                for icao, iata, lat, lon, h, tz, dst, cc
                in zip(df.index, df.iata, df.lat, df.lon, df.alt,
                       df.tz, df.dst, df.domestic_iso)}
    else:
        if iata_keys:
            apts = {iata: Airport(icao, iata, lat, lon, h, tz, dst, cc)
                    for icao, iata, lat, lon, h, tz, dst, cc
                    in zip(df.index, df.iata, df.lat, df.lon, df.alt,
                           df.tz, df.dst, df.domestic_iso)}
        else:
            apts = {icao: Airport(icao, iata, lat, lon, h, tz, dst, cc)
                    for icao, iata, lat, lon, h, tz, dst, cc
                    in zip(df.index, df.iata, df.lat, df.lon, df.alt,
                           df.tz, df.dst, df.domestic_iso)}
    if runways or tim_stettler:
        load_runway_lengths(apts)
    
    if movements or tim_stettler:
        load_yearly_movements(apts)
    
    if tim_stettler:
        load_tim_stettler(apts)
    
    return apts


def load_runway_lengths(apts, filepath=FILEPATH_OURAIRPORTS_RUNWAYS):
    """
    Load runway quantity and max length from OurAirport

    Parameters
    ----------
    apts : list of core.Airport, or dict of core.Airport, or pd.DataFrame
        Airports for which runway data will be loaded.
    filepath : str, optional
        Path to a .csv file with the runway data.
        The default is FILEPATH_OURAIRPORTS_RUNWAYS.
    
    Raises
    ------
    TypeError
        If apts is not the expected type.

    Returns
    -------
    None.

    """
    runways = pd.read_csv(filepath, index_col='id')
    
    if isinstance(apts, list) and isinstance(apts[0], Airport):
        icao_list = [a.icao for a in apts]
    elif (isinstance(apts, dict)
          and isinstance(next(iter(apts.values())), Airport)):
        icao_list = [a.icao for a in apts.values()]
    elif isinstance(apts, pd.DataFrame):
        if 'icao' in apts.columns:
            icao_list = list(apts.icao.unique())
        else:
            icao_list = list(apts.index.unique())
    else:
        raise TypeError('apts should be a (list or dict) of Airport')
    
    runways = runways[(runways.airport_ident.isin(icao_list))
                      & (runways.closed==0)]
    runways = runways.groupby('airport_ident')
    lengths = runways.length_ft.max() * physics.FT_TO_METER
    numbers = runways.size()
    
    if isinstance(apts, pd.DataFrame):
        if 'icao' in apts.columns:
            apts['n_runways'] = apts.icao.map(numbers)
            apts['max_runway_length'] = apts.icao.map(lengths)
        else:
            apts['n_runways'] = numbers
            apts['max_runway_length'] = lengths
    else:
        for apt in apts:
            if isinstance(apt, str):
                apt = apts[apt]
            icao = apt.icao
            
            if icao in numbers.index:
                apt.n_runways = numbers.loc[icao]
                apt.max_runway_length = lengths.loc[icao]


def load_yearly_movements(apts, filepath=FILEPATH_MOVEMENTS):
    """
    Load number of yearly aircraft movements for given airports

    Parameters
    ----------
    apts : list of core.Airport, or dict of core.Airport, or pd.DataFrame
        Airports for which movement data will be loaded.
    filepath : str, optional
        Path to a .csv file with the yearly movement data.
        The default is FILEPATH_MOVEMENTS.

    Returns
    -------
    None.

    """
    try:
        movements = pd.read_csv(filepath, index_col='icao', squeeze=True)
    except FileNotFoundError:
        warnings.warn('could not find number of airport movements at '
                      + f'"{filepath}"')
        return
    
    if isinstance(apts, pd.DataFrame):
        if 'icao' in apts.columns:
            apts['yearly_movements'] = apts.icao.map(movements)
        else:
            apts['yearly_movements'] = movements
    else:
        for apt in apts:
            if isinstance(apt, str):
                apt = apts[apt]
            icao = apt.icao
            
            if icao in movements.index:
                apt.yearly_movements = movements.loc[icao]


def load_tim_stettler(apts, filepath=FILEPATH_TIM_STETTLER):
    """
    Load LTO times-in-mode into a list of airports

    Parameters
    ----------
    apts : list of core.Airport, or dict of core.Airport, or pd.DataFrame
        Airports for which TIM will be assigned.
    filepath : str, optional
        Path to a .csv file with the TIMs.
        The default is FILEPATH_TIM_STETTLER.

    Raises
    ------
    TypeError
        If apts is not the expected type.
        
    Returns
    -------
    None.
    
    References
    ----------
    Stettler, M.E.J., S. Eastham, and S.R.H. Barrett. “Air Quality and Public
    Health Impacts of UK Airports. Part I: Emissions.” Atmospheric Environment
    45, no. 31 (October 2011): 5415–24.
    https://doi.org/10.1016/j.atmosenv.2011.07.012.

    """
    if isinstance(apts, pd.DataFrame):
        # TIMs cannot be added to a DataFrame, so do nothing
        return
    
    DEFAULT_APT = 'LTN'
    tims = pd.read_csv(filepath, index_col=['iata', 'natsgrp'])  # times [s]
    
    if DEFAULT_APT in tims.index:
        default_tims = tims.loc[DEFAULT_APT]
    else:
        default_iata = tims.index[0][0]
        print("Warning (load_tim_stettler): couldn't find TIMs for"
              + f" '{DEFAULT_APT}' in '{filepath}', using {default_iata}"
              + " instead")
        default_tims = tims.loc[default_iata]
    
    for apt in apts:
        if isinstance(apt, str):
            apt = apts[apt]
        elif not isinstance(apt, Airport):
            raise TypeError('apts should be a (list or dict) of Airport')
        iata = apt.iata
        if iata in tims.index:
            apt.tim_stettler = tims.loc[iata]
        else:
            apt.tim_stettler = default_tims.copy(deep=True)
            if (hasattr(apt, 'max_runway_length')
                and not np.isnan(apt.max_runway_length)):
                # Stettler suggests taxi time of 1 s per 10 m of max runway
                a = apt.max_runway_length / 10
                apt.tim_stettler = apt.tim_stettler.assign(taxiin=a)
                apt.tim_stettler = apt.tim_stettler.assign(taxiout=a)
            if (hasattr(apt, 'yearly_movements')
                and hasattr(apt, 'n_runways')):
                c = apt.yearly_movements / apt.n_runways
                b = 0.002 * (c - 100000)
                b = min(600, max(0, b))
                apt.tim_stettler = apt.tim_stettler.assign(hold=b)


def load_met(simcfg, forceload=False):
    """
    Load wind data based on simcfg

    Parameters
    ----------
    simcfg : SimConfig
        Configuration of simulation options.
    forceload : bool, optional
        If True, load met even if simcfg.wind is "none" or
        simcfg.fly_nonlto is False. The default is False.

    Raises
    ------
    Exception
        If simcfg is missing the `wind` or `wind_filepath` attributes.

    Returns
    -------
    met : xr.Dataset or None
        Dataset containing wind data. Is None if simcfg.forcecload and either
        simcfg.wind is "none" or simcfg.fly_nonlto is False.

    """
    if not hasattr(simcfg, 'wind'):
        raise Exception('"wind" configuration missing')
    if not forceload and (simcfg.wind == 'none' or not simcfg.fly_nonlto):
        met = None
    else:
        if not hasattr(simcfg, 'wind_filepath'):
            raise Exception('"wind_filepath" configuration missing')
        met = xr.open_dataset(simcfg.wind_filepath)
    
    return met

