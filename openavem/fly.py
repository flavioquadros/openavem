"""
Calculates emissions for a flight
"""

##
# Imports
##

# Import system modules
import time
import os
import sys
import multiprocessing
import warnings
import logging

# Import additional modules
import numpy as np
import pandas as pd
from scipy import interpolate

# Import from the project
from .core import NothingToProcess, ModelDataMissing
from .core import EmissionSegment, SimConfig, Phase, rng, APU, Airport
from .core import list_if_few
from . import physics
from . import bada
from . import grid
from . import plot
from . import dir_openavem
from .datasources import (load_apts_from_simcfg,
                          load_yearly_movements,
                          load_tim_stettler,
                          load_ac_icao_description,
                          load_ac_replacements,
                          load_ac_military,
                          determine_payload_m_frac,
                          allocate_eng,
                          load_met)

###
# Contants
##

DIR_FLIGHTLISTS = os.path.join(dir_openavem, 'flightlists/')

DEPARTURE = 2
ARRIVAL   = 1

LTO_CEILING = 3000 * physics.FT_TO_METER
DES_EST_GRADE = 1000 / 3 * physics.FT_TO_METER / physics.NM_TO_METER  # [-]

MIN_FLIGHTDIST = 2000  # [m]

EST_RESERVEFUEL_PCT = 0.05  # [-]

EST_LONGHAUL_DIVERSION  = 200 * physics.NM_TO_METER
EST_SHORTHAUL_DIVERSION = 100 * physics.NM_TO_METER

EST_LONGHAUL_LOWALTHOLD  =  30  # [min]
EST_SHORTHAUL_LOWALTHOLD =  40  # [min]

PHASE_IDLE = 0
PHASE_APP  = 1
PHASE_CO   = 2
PHASE_TO   = 3

LTO_ICAO_TAXI    = 0
LTO_ICAO_TAXIIN  = 0
LTO_ICAO_TAXIOUT = 4
LTO_ICAO_APP     = 1
LTO_ICAO_CO      = 2
LTO_ICAO_TO      = 3
TIM_ICAO = [7.0*60, 4.0*60, 2.2*60, 0.7*60, 19.0*60]
LTO_ICAO_THRUST = [0.07, 0.30, 0.85, 1.00]
LTO_FOCA_THRUST = [0.12, 0.45, 0.85, 1.00]


###
# Classes
##


###
# Functions
##

def clip_thrust_to_icao(f):
    """Clip thrust to range of ICAO LTO cycle"""
    return min(max(f, LTO_ICAO_THRUST[0]), LTO_ICAO_THRUST[-1])


def nvpm_ei_lto(eng, simcfg):
    """
    Get nvPM mass and number ICAO LTO emission indices according to simcfg

    Parameters
    ----------
    eng : Engine
        Engine.
    simcfg : SimConfig
        Configuration of simulation options.

    Raises
    ------
    ValueError
        If simcfg.nvpm_lto or simcfg.nvpmnum_lto are not recognized.

    Returns
    -------
    nvpm_ei : np.array
        nvPM mass emission indices in kg/kg_fuel at ICAO LTO thrusts.
    nvpmnum_ei : np.array
        nvPM number emission indices in 1/kg_fuel at ICAO LTO thrusts.

    """
    # Mass emission indices
    methods = simcfg.nvpm_lto
    if isinstance(methods, str):
        # Handle a string being provided instead of a list of str
        methods = [methods]
    
    for method in methods:
        if method == 'measured':
            if np.isnan(eng.nvpm_lto).any():
                # Try next selected method
                continue
            else:
                nvpm_ei = eng.nvpm_lto
                break
        elif method == 'FOA4':
            if not hasattr(eng, 'foa4') or eng.foa4['zero']:
                # Try next selected method
                continue
            else:
                nvpm_ei = eng.foa4['ei_lto']
                break
        elif method == 'SCOPE11':
            if not hasattr(eng, 'scope11') or eng.scope11['zero']:
                # Try next selected method
                continue
            else:
                nvpm_ei = eng.scope11['ei_lto']
                break
        elif method == 'FOA3':
            if not hasattr(eng, 'foa3') or eng.foa3['zero']:
                # Try next selected method
                continue
            else:
                nvpm_ei = eng.foa3['ei_lto']
                break
        elif method == 'FOX':
            if not hasattr(eng, 'fox') or eng.fox['zero']:
                # Try next selected method
                continue
            else:
                nvpm_ei = eng.fox['EI_ref']
                break
        elif method == 'constantEI':
            nvpm_ei = np.array(4*[simcfg.nvpm_lto_EI])
            break
        elif method == 'none':
            nvpm_ei = np.array([0, 0, 0, 0])
            break
        else:
            raise ValueError(f'simcfg.nvpm_lto "{method}" not recognized')
    else:
        raise Exception('No method for LTO nvPM EI calculation succeded')
    # [mg/kg] -> [kg/kg]
    nvpm_ei = nvpm_ei / 1e6
        
    # Number emission indices
    methods = simcfg.nvpmnum_lto
    if isinstance(methods, str):
        # Handle a string being provided instead of a list of str
        methods = [methods]
    
    for method in methods:
        if method == 'measured':
            if np.isnan(eng.nvpmnum_lto).any():
                # Try next selected method
                continue
            else:
                nvpmnum_ei = eng.nvpmnum_lto
                break
        elif method == 'FOA4':
            if not hasattr(eng, 'foa4') or eng.foa4['zero']:
                # Try next selected method
                continue
            else:
                nvpmnum_ei = eng.foa4['num_ei_lto']
                break
        elif method == 'SCOPE11':
            if not hasattr(eng, 'scope11') or eng.scope11['zero']:
                # Try next selected method
                continue
            else:
                nvpmnum_ei = eng.scope11['num_ei_lto']
                break
        elif method == 'constantEI':
            nvpmnum_ei = np.array(4*[simcfg.nvpmnum_lto_EI * 1e14])
            break
        elif method == 'none':
            nvpmnum_ei = np.array([0, 0, 0, 0])
            break
        else:
            raise ValueError(f'simcfg.nvpm_lto "{method}" not recognized')
    else:
        raise Exception('No method for LTO nvPM_num EI calculation succeded')
    
    return nvpm_ei, nvpmnum_ei


def lto_stettler(ac, apt, departure, simcfg, t0=0):
    """
    Calculate LTO emissions using Stettler et al (2011) method

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    apt : Airport
        Airport of departure or arrival.
    departure : bool
        If True, calculate departure emissions. If False, calculate arrival
        emissions.
    simcfg : SimConfig
        Configuration of simulation options.
    t0 : float, optional
        Initial time in seconds. The default is 0.

    Returns
    -------
    emissions : EmissionSegment
        Calculated emissions.
    
    Notes
    -----
    Initial climb, climb out, and approach phases are spread vertically, but
        are still concentrated at the airport's nominal (lat, lon).
    AEIC random variable scheme:
        approach:  uniform(21%-30%)
        taxiin: uniform(4%-7%)
        landing, hold, taxiout = taxi
        reverse: 30% (90% of flights with natsgrp <= 6)
        takeoff: uniform(75%-99%) with a 10% chance of 100% if natsgrp <= 6
        takeoff: 100% if natsgrp >= 7
        initialclimb = takeoff
        climbout = min(takeoff, 85%)
        taxiacc_arr = triangular(7%-17%, mode=10%)
        taxiacc_dep = taxiacc_arr
        
    References
    ----------
    Aviation Emissions Inventory Code v2.1. Nicholas Simone, Marc Stettler.
        https://lae.mit.edu/codes/ accessed 06/Sep/2019.
    
    Simone, Nicholas W., Marc E.J. Stettler, and Steven R.H. Barrett. “Rapid
        Estimation of Global Civil Aviation Emissions with Uncertainty
        Quantification.” Transportation Research Part D: Transport and
        Environment 25 (December 2013): 33–41.
        https://doi.org/10.1016/j.trd.2013.07.001.
    
    Stettler, M.E.J., S. Eastham, and S.R.H. Barrett. “Air Quality and Public
        Health Impacts of UK Airports. Part I: Emissions.” Atmospheric
        Environment 45, no. 31 (October 2011): 5415–24.
        https://doi.org/10.1016/j.atmosenv.2011.07.012.

    Watterson, John, Charles Walker, and Brian Eggleston. “Revision to the
        Method of Estimating Emissions from Aircraft in the UK Greenhouse Gas
        Inventory.” Netcen, July 2004.

    """
    DEFAULT_THRUST = {
        'approach':     0.25,
        'landing':      0.055,
        'taxiin':       0.055,
        'taxiout':      0.055,
        'hold':         0.055,
        'takeoff':      0.90,
        'initialclimb': 0.90,
        'climbout':     0.85,
        'taxiacc_dep':  0.10,
        'taxiacc_arr':  0.10,
    }
    DEFAULT_REVERSE = False
    eng = ac.eng
    
    thrusts = DEFAULT_THRUST
    neng = ac.opf['aircraft']['n_engs']
    
    tims = apt.tim_stettler.loc[ac.natsgrp]
    # Taxi acceleration TIMs modeled like in AEIC
    if tims['hold'] > 0:
        tims = tims.append(pd.Series([20, 10], ['taxiacc_dep', 'taxiacc_arr']))
    else:
        tims = tims.append(pd.Series([10, 10], ['taxiacc_dep', 'taxiacc_arr']))
    
    if departure:
        if ac.natsgrp > 6:
            # Default thrusts are different for small aircraft
            # (AEIC assumes no de-rated takeoff)
            thrusts['takeoff'] = 1.00
            thrusts['initialclimb'] = 1.00
    else:
        # Reverse thurst modeled like AEIC
        if simcfg.reverse_rand_occurence:
            apply_reverse = rng.random() < 0.1
        else:
            apply_reverse = DEFAULT_REVERSE
        
        if apply_reverse:
            tims['landing'] = max(0, tims['landing'] - tims['reverse'])
            thrusts['reverse'] = tims['reverse_setting']
        else:
            tims['reverse'] = 0
    
    fflow = {
        m: np.interp(thrusts[m],
                     [0] + LTO_ICAO_THRUST,
                     np.concatenate(([0], eng.bffm2['wf'])))
        for m in thrusts
    }
    
    nvpm_ei, nvpmnum_ei = nvpm_ei_lto(eng, simcfg)
    nvpm_interp_pchip = interpolate.pchip(LTO_ICAO_THRUST, nvpm_ei)
    nvpm_interp_lin = interpolate.interp1d(
        LTO_ICAO_THRUST, nvpm_ei, bounds_error=False,
        fill_value=(LTO_ICAO_THRUST[0], LTO_ICAO_THRUST[-1]))
    nvpmnum_interp = interpolate.interp1d(LTO_ICAO_THRUST, nvpmnum_ei)
    
    lat = apt.lat
    lon = apt.lon
    h   = apt.h
    t   = t0
    
    # Atmospheric conditions taken as ISA at altitude h
    T, p, rho, a = physics.isa_trop(h)
    theta_amb = T / physics.MSL_T
    delta_amb = p / physics.MSL_P
    
    # List to hold all emission segments created here
    emissions = []
    
    if departure:
        # Emissions at origin
        # Ground level phases
        for mode, phasenum in zip(['taxiout', 'taxiacc_dep',
                                   'hold', 'takeoff'],
                                  [Phase.TAXIOUT, Phase.TAXIACC_DEP,
                                   Phase.HOLD, Phase.TAKEOFF]):
            dt = tims[mode]
            if dt == 0:
                continue
            seg = EmissionSegment(lat, lon, h, lat, lon, h)
            seg.t_s = t
            t += dt
            seg.t_e = t
            seg.tas = 0
            seg.phasenum = phasenum
            # adjusted_f: fuel flow at particular atmospheric conditions [kg/s]
            adjusted_f = fflow[mode] * delta_amb / theta_amb**3.8
            seg.fuelburn = dt * adjusted_f * neng
            clipped_thrust = clip_thrust_to_icao(thrusts[mode])
            if simcfg.nvpm_interp == 'PCHIP':
                seg.nvpm = seg.fuelburn * nvpm_interp_pchip(clipped_thrust)
            if simcfg.nvpm_interp != 'PCHIP' or seg.nvpm <= 0:
                seg.nvpm = seg.fuelburn * nvpm_interp_lin(clipped_thrust)
            seg.nvpmnum = seg.fuelburn * nvpmnum_interp(clipped_thrust)
            emissions.append(seg)
        
        # Climb
        # Note that in BADA: H_max_to = 400 ft and H_max_ic = 2000 ft
        #  while in Watterson, initialclimb = 0-450 m and climbout = 450-1000 m
        mode = 'initialclimb'
        seg = EmissionSegment(lat, lon, h, lat, lon, h + 450)
        dt = tims[mode]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = ac.opf['aero']['Vstall_ic']
        seg.phasenum = Phase.INITIAL_CLIMB
        M = seg.tas / (physics.AIR_R * physics.AIR_GAMMA * T)**0.5
        adjusted_f = (fflow[mode]
                      * delta_amb / theta_amb**3.8 / np.exp(0.2 * M**2))
        seg.fuelburn = dt * adjusted_f * neng
        clipped_thrust = clip_thrust_to_icao(thrusts[mode])
        if simcfg.nvpm_interp == 'PCHIP':
            seg.nvpm = seg.fuelburn * nvpm_interp_pchip(clipped_thrust)
        if simcfg.nvpm_interp != 'PCHIP' or seg.nvpm <= 0:
            seg.nvpm = seg.fuelburn * nvpm_interp_lin(clipped_thrust)
        seg.nvpmnum = seg.fuelburn * nvpmnum_interp(clipped_thrust)
        emissions.append(seg)
        
        mode = 'climbout'
        seg = EmissionSegment(lat, lon, h + 450, lat, lon, h + LTO_CEILING)
        dt = tims[mode]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = ac.opf['aero']['Vstall_ic']
        seg.phasenum = Phase.CLIMBOUT
        M = seg.tas / (physics.AIR_R * physics.AIR_GAMMA * T)**0.5
        adjusted_f = (fflow[mode]
                      * delta_amb / theta_amb**3.8 / np.exp(0.2 * M**2))
        seg.fuelburn = dt * adjusted_f * neng
        clipped_thrust = clip_thrust_to_icao(thrusts[mode])
        if simcfg.nvpm_interp == 'PCHIP':
            seg.nvpm = seg.fuelburn * nvpm_interp_pchip(clipped_thrust)
        if simcfg.nvpm_interp != 'PCHIP' or seg.nvpm <= 0:
            seg.nvpm = seg.fuelburn * nvpm_interp_lin(clipped_thrust)
        seg.nvpmnum = seg.fuelburn * nvpmnum_interp(clipped_thrust)
        emissions.append(seg)
        
    else:
        # Emissions at destination
        # Approach
        mode = 'approach'
        seg = EmissionSegment(lat, lon, h + LTO_CEILING, lat, lon, h)
        dt = tims[mode]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = ac.opf['aero']['Vstall_app']
        seg.phasenum = Phase.APPROACH
        M = seg.tas / (physics.AIR_R * physics.AIR_GAMMA * T)**0.5
        adjusted_f = (fflow[mode]
                      * delta_amb / theta_amb**3.8 / np.exp(0.2 * M**2))
        seg.fuelburn = dt * adjusted_f * neng
        clipped_thrust = clip_thrust_to_icao(thrusts[mode])
        if simcfg.nvpm_interp == 'PCHIP':
            seg.nvpm = seg.fuelburn * nvpm_interp_pchip(clipped_thrust)
        if simcfg.nvpm_interp != 'PCHIP' or seg.nvpm <= 0:
            seg.nvpm = seg.fuelburn * nvpm_interp_lin(clipped_thrust)
        seg.nvpmnum = seg.fuelburn * nvpmnum_interp(clipped_thrust)
        emissions.append(seg)
        
        # Ground level phases
        for mode, phasenum in zip(['landing', 'reverse',
                                   'taxiacc_arr', 'taxiin'],
                                  [Phase.LANDING, Phase.REVERSE,
                                   Phase.TAXIACC_ARR, Phase.TAXIIN]):
            dt = tims[mode]
            if dt == 0:
                continue
            seg = EmissionSegment(lat, lon, h, lat, lon, h)
            seg.t_s = t
            t += dt
            seg.t_e = t
            seg.tas = 0
            seg.phasenum = phasenum
            adjusted_f = fflow[mode] * delta_amb / theta_amb**3.8
            seg.fuelburn = dt * adjusted_f * neng
            clipped_thrust = clip_thrust_to_icao(thrusts[mode])
            if simcfg.nvpm_interp == 'PCHIP':
                seg.nvpm = seg.fuelburn * nvpm_interp_pchip(clipped_thrust)
            if simcfg.nvpm_interp != 'PCHIP' or seg.nvpm <= 0:
                seg.nvpm = seg.fuelburn * nvpm_interp_lin(clipped_thrust)
            seg.nvpmnum = seg.fuelburn * nvpmnum_interp(clipped_thrust)
            emissions.append(seg)
    
    bffm2(emissions, ac)
    
    return emissions
        
    
def lto_icao(ac, apt, departure, simcfg, t0=0):
    """
    Calculate LTO emissions using the ICAO cycle

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    apt : Airport
        Airport of departure or arrival.
    departure : bool
        If True, calculate departure emissions. If False, calculate arrival
        emissions.
    simcfg : SimConfig
        Configuration of simulation options.
    t0 : float, optional
        Initial time in seconds. The default is 0.

    Returns
    -------
    emissions : list of EmissionSegment
        Calculated emissions.

    """
    eng = ac.eng
    neng = ac.opf['aircraft']['n_engs']
    
    lat = apt.lat
    lon = apt.lon
    h   = apt.h
    t   = t0
    
    # List to hold all emission segments created here
    emissions = []
    
    nvpm_ei, nvpmnum_ei = nvpm_ei_lto(eng, simcfg)
    
    if departure:
        # Emissions at origin
        # Taxi out / ground idle
        seg = EmissionSegment(lat, lon, h, lat, lon, h)
        dt = TIM_ICAO[LTO_ICAO_TAXIOUT]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = 0
        seg.phasenum = Phase.TAXIOUT
        seg.fuelburn = dt * eng.fuel_idle * neng
        seg.nox = seg.fuelburn * eng.nox_idle / 1000
        seg.co = seg.fuelburn * eng.co_idle / 1000
        seg.hc = seg.fuelburn * eng.hc_idle / 1000
        seg.nvpm = seg.fuelburn * nvpm_ei[LTO_ICAO_TAXI]
        seg.nvpmnum = seg.fuelburn * nvpmnum_ei[LTO_ICAO_TAXI]
        emissions.append(seg)
        
        # Takeoff
        seg = EmissionSegment(lat, lon, h, lat, lon, h)
        dt = TIM_ICAO[LTO_ICAO_TO]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = 0
        seg.phasenum = Phase.TAKEOFF
        seg.fuelburn = dt * eng.fuel_to * neng
        seg.nox = seg.fuelburn * eng.nox_to / 1000
        seg.co = seg.fuelburn * eng.co_to / 1000
        seg.hc = seg.fuelburn * eng.hc_to / 1000
        seg.nvpm = seg.fuelburn * nvpm_ei[LTO_ICAO_TO]
        seg.nvpmnum = seg.fuelburn * nvpmnum_ei[LTO_ICAO_TO]
        emissions.append(seg)
        
        # Climb-out
        seg = EmissionSegment(lat, lon, h, lat, lon, h + LTO_CEILING)
        dt = TIM_ICAO[LTO_ICAO_CO]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = ac.opf['aero']['Vstall_ic']
        seg.phasenum = Phase.CLIMBOUT
        seg.fuelburn = dt * eng.fuel_co * neng
        seg.nox = seg.fuelburn * eng.nox_co / 1000
        seg.co = seg.fuelburn * eng.co_co / 1000
        seg.hc = seg.fuelburn * eng.hc_co / 1000
        seg.nvpm = seg.fuelburn * nvpm_ei[LTO_ICAO_CO]
        seg.nvpmnum = seg.fuelburn * nvpmnum_ei[LTO_ICAO_CO]
        emissions.append(seg)
    
    else:
        # Emissions at destination
        # Approach
        seg = EmissionSegment(lat, lon, h + LTO_CEILING, lat, lon, h)
        dt = TIM_ICAO[LTO_ICAO_APP]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = ac.opf['aero']['Vstall_app']
        seg.phasenum = Phase.APPROACH
        seg.fuelburn = dt * eng.fuel_app * neng
        seg.nox = seg.fuelburn * eng.nox_app / 1000
        seg.co = seg.fuelburn * eng.co_app / 1000
        seg.hc = seg.fuelburn * eng.hc_app / 1000
        seg.nvpm = seg.fuelburn * nvpm_ei[LTO_ICAO_APP]
        seg.nvpmnum = seg.fuelburn * nvpmnum_ei[LTO_ICAO_APP]
        emissions.append(seg)
        
        # Taxi in / ground idle
        seg = EmissionSegment(lat, lon, h, lat, lon, h)
        dt = TIM_ICAO[LTO_ICAO_TAXIIN]
        seg.t_s = t
        t += dt
        seg.t_e = t
        seg.tas = 0
        seg.phasenum = Phase.TAXIIN
        seg.fuelburn = dt * eng.fuel_idle * neng
        seg.nox = seg.fuelburn * eng.nox_idle / 1000
        seg.co = seg.fuelburn * eng.co_idle / 1000
        seg.hc = seg.fuelburn * eng.hc_idle / 1000
        seg.nvpm = seg.fuelburn * nvpm_ei[LTO_ICAO_TAXI]
        seg.nvpmnum = seg.fuelburn * nvpmnum_ei[LTO_ICAO_TAXI]
        emissions.append(seg)
    
    return emissions
    
    
def get_apu(method, ac, apus):
    """
    Get apu for given aircraft and EI method

    Parameters
    ----------
    method : str
        Method used to select emission indices. Possible values are
        "ICAOsimple", "ICAOadv", "ACRP", "specific", "none".
    ac : mada.Aircraft
        Modeled aircraft.
    apus : dict of str to APU
        Dictionary of name strings to APU objects.

    Raises
    ------
    KeyError
        If the required APU is not in apus.
    ValueError
        If the method is not recognized.

    Returns
    -------
    apu : APU, None, or nan
        APU selected. None if the aircraft has no APU or method is "none".
        NaN if the aircraft has no APU listed (information missing).

    """
    if method == 'specific':
        if hasattr(ac, 'apu'):
            apu = ac.apu
        else:
            apu = np.nan
    else:
        if method == 'ICAOsimple':
            aputype = ac.apu_icao_simple
            apukey = f'icao_simple_{aputype}'
        elif method == 'ICAOadv':
            aputype = ac.apu_icao_adv
            apukey = f'icao_adv_{aputype}'
        elif method == 'ACRP':
            aputype = ac.apu_acrp
            apukey = f'acrp_{aputype}'
        elif method == 'none':
            aputype = 'none'
        else:
            raise ValueError(f'APU EI method "{method}" not recognized')
        
        if aputype == 'none':
            apu = None
        else:
            try:
                apu = apus[apukey]
            except KeyError:
                raise KeyError(f'Could not find "{apukey}" in the apus '
                               + 'provided')
    
    return apu


def apu_run(ac, apt, departure, simcfg, apus, t0=0):
    """
    Calculate APU emissions using the "ICAO-Simple" method

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    apt : Airport
        Airport of departure or arrival.
    departure : bool
        If True, calculate departure emissions. If False, calculate arrival
        emissions.
    simcfg : SimConfig
        Configuration of simulation options.
    apus : dict of str to APU
        Dictionary containing APU objected indexed with the keys
        "icao_simple_business", "icao_simple_short" and "icao_simple_long"
    t0 : float, optional
        Initial or final (see "before" arg) time in seconds. The default is 0.

    Returns
    -------
    emissions : list of EmissionSegment
        Calculated emissions.

    """
    # List to hold all emission segments created here
    emissions = []
    
    tim = simcfg.apu_tim
        
    if tim == 'none' or (ac.apu_icao_simple == 'business'
                         and not simcfg.apu_businessjet):
        # No APU for you
        return emissions
    
    lat = apt.lat
    lon = apt.lon
    h   = apt.h
    
    if tim == 'ICAOadv':
        # Multiple-mode model
        # Time-in-mode
        if departure:
            modes = ['nl', 'ecs', 'mes']
            phasenums = {'nl':    Phase.APU_STARTUP,
                         'ecs':   Phase.APU_GATEOUT,
                         'mes':   Phase.APU_MES,
                         'cycle': Phase.APU_DEP}
            if ac.opf['aircraft']['n_engs'] > 2:
                tims = {'nl':  3.0 * 60,
                        'ecs': 5.3 * 60,
                        'mes': 140}
            else:
                tims = {'nl':  3.0 * 60,
                        'ecs': 3.6 * 60,
                        'mes': 35}
        else:
            modes = ['ecs']
            phasenums = {'ecs':   Phase.APU_GATEIN,
                         'cycle': Phase.APU_ARR}
            tims = {'ecs': 15.0 * 60}
        tims['cycle'] = sum(tims.values())
        
        # Choose EI for fuel, NOx, CO, HC
        gasmethods = simcfg.apu_eigas
        if isinstance(gasmethods, str):
            gasmethods = [gasmethods]
        for gasmethod in gasmethods:
            apu = get_apu(gasmethod, ac, apus)
            if apu is None:
                # No APU, no emissions
                return emissions
            elif isinstance(apu, APU):
                break
        else:
            raise Exception('No method for APU gas EI calculation succeded')
        
        # If no mode specific EIs are present, we'll revert to single-mode
        missing_ei =  np.isnan([apu.fuel_nl, apu.fuel_ecs, apu.fuel_mes,
                                apu.nox_nl, apu.nox_ecs, apu.nox_mes,
                                apu.co_nl, apu.co_ecs, apu.co_mes,
                                apu.hc_nl, apu.hc_ecs, apu.hc_mes]).any()
        if missing_ei:
            gasmodes = ['cycle']
        else:
            gasmodes = modes
        
        # Create emission segments
        segs = {k: EmissionSegment(lat, lon, h, lat, lon, h) for k in gasmodes}
        
        # Calculate gas emissions
        for k in gasmodes:
            segs[k].fuelburn = tims[k] * apu.fuel[k]
            segs[k].nox      = segs[k].fuelburn * apu.nox[k] / 1e3
            segs[k].co       = segs[k].fuelburn * apu.co[k] / 1e3
            segs[k].hc       = segs[k].fuelburn * apu.hc[k] / 1e3
            segs[k].phasenum = phasenums[k]
            # Initiate nvPM as 0 and recalculate later
            segs[k].nvpm    = 0
            segs[k].nvpmnum = 0
        
        # Choose EI for nvPM
        pmmethods = simcfg.apu_eipm
        if isinstance(pmmethods, str):
            pmmethods = [pmmethods]
        for pmmethod in pmmethods:
            apu_pm = get_apu(pmmethod, ac, apus)
            if apu_pm is None:
                # gasmethod found an APU, but now we find that there is none
                warnings.warn('conflicting information about whether aircraft '
                              + f'"{ac.typecode}" has an APU')
                break
            elif isinstance(apu_pm, APU):
                hasnvpm = (not np.isnan([apu_pm.fuel_cycle,
                                         apu_pm.nvpm_cycle]).any()
                           or not np.isnan([apu_pm.nvpm_nl, apu_pm.nvpm_ecs,
                                            apu_pm.nvpm_mes]).any())
                haspm10 = (simcfg.apu_pm10_as_nvpm
                           and (not np.isnan([apu_pm.fuel_cycle,
                                              apu_pm.pm10_cycle]).any()
                                or not np.isnan([apu_pm.pm10_nl,
                                                 apu_pm.pm10_ecs,
                                                 apu_pm.pm10_mes]).any()))
                if hasnvpm or haspm10:
                    break
        else:
            raise Exception('No method for APU nvPM EI calculation succeded')
        
        if apu_pm is None:
            for k in segs:
                segs[k].nvpm    = 0
                segs[k].nvpmnum = 0
        else:
             # Without mode specific EIs, revert to single-mode just for nvPM
            missing_nvpm = np.isnan([apu_pm.nvpm_nl, apu_pm.nvpm_ecs,
                                     apu_pm.nvpm_mes]).any()
            missing_pm10 = np.isnan([apu_pm.pm10_nl, apu_pm.pm10_ecs,
                                     apu_pm.pm10_mes]).any()
            missing_ei =  (missing_nvpm
                           and (missing_pm10 or not simcfg.apu_pm10_as_nvpm))
            if missing_ei:
                pmmodes = ['cycle']
            else:
                pmmodes = modes
            
            for k in pmmodes:
                if k not in segs:
                    segs[k] = EmissionSegment(lat, lon, h, lat, lon, h)
                    segs[k].fuelburn = 0
                    segs[k].nox      = 0
                    segs[k].co       = 0
                    segs[k].hc       = 0
                    segs[k].phasenum = phasenums[k]
                if np.isnan(apu_pm.nvpm[k]):
                    segs[k].nvpm = (
                        tims[k] * apu_pm.fuel[k] * apu_pm.pm10[k] / 1e3)
                else:
                    segs[k].nvpm = (
                        tims[k] * apu_pm.fuel[k] * apu_pm.nvpm[k] / 1e3)
                    if not np.isnan(apu_pm.nvpmnum[k]):
                        segs[k].nvpmnum = (tims[k] * apu_pm.fuel[k]
                                           * apu_pm.nvpmnum[k])
            
        if departure:
            if 'ecs' in segs:
                segs['mes'].t_e = t0
                t = t0 - tims['mes']
                segs['mes'].t_s = t
                segs['ecs'].t_e = t
                t -= tims['ecs']
                segs['ecs'].t_s = t
                segs['nl'].t_e = t
                t -= tims['nl']
                segs['nl'].t_s = t0
                emissions += [segs['nl'], segs['ecs'], segs['mes']]
            if 'cycle' in segs:
                segs['cycle'].t_s = t0 - tims['cycle']
                segs['cycle'].t_e = t0
                emissions.append(segs['cycle'])
        else:
            if 'cycle' in segs:
                segs['cycle'].t_s = t0
                segs['cycle'].t_e = t0 + tims['cycle']
                emissions.append(segs['cycle'])
            if 'ecs' in segs:
                segs['ecs'].t_s = t0
                segs['ecs'].t_e = t0 + tims['ecs']
                emissions.append(segs['ecs'])
    
    else:
        # Single-mode model
        seg = EmissionSegment(lat, lon, h, lat, lon, h)
        if tim == 'ICAOsimple':
            # Assume 50:50 split between arrival and departure
            if ac.apu_icao_simple == 'longhaul':
                t = 75 * 60 / 2
            else:
                t = 45 * 60 / 2
        elif tim == 'ATAmid':
            # Assume 50:50 split between arrival and departure
            if hasattr(apt, 'gnd_power') and apt.gnd_power:
                t = (0.23 + 0.26) / 2 * 3600 / 2
            else:
                if ac.apu_icao_adv.startswith('large'):
                    t = (1.0 + 1.5) / 2 * 3600 / 2
                else:
                    t = 0.87 * 3600 / 2
        elif tim == 'AEIC':
            if departure:
                t = 1804
            else:
                t = 1050
        elif tim == 'AEDT':
            if hasattr(apt, 'gnd_power') and apt.gnd_power:
                t = 3.5 * 60
            else:
                t = 13 * 60
        else:
            raise ValueError(f'apu_tim "{tim}" not recognized')
        
        # Choose EI for fuel, NOx, CO, HC
        gasmethods = simcfg.apu_eigas
        if isinstance(gasmethods, str):
            # Handle a string being provided instead of a list of str
            gasmethods = [gasmethods]
        
        for gasmethod in gasmethods:
            apu = get_apu(gasmethod, ac, apus)
            if apu is None:
                # No APU, no emissions
                return emissions
            elif (isinstance(apu, APU)
                  and not np.isnan([apu.fuel_cycle, apu.nox_cycle,
                                    apu.co_cycle, apu.hc_cycle]).any()):
                break
        else:
            raise Exception('No method for APU gas EI calculation succeded')
        
        seg.fuelburn = t * apu.fuel_cycle
        seg.nox      = seg.fuelburn * apu.nox_cycle / 1e3
        seg.co       = seg.fuelburn * apu.co_cycle / 1e3
        seg.hc       = seg.fuelburn * apu.hc_cycle / 1e3
        
        # Choose EI for nvPM
        pmmethods = simcfg.apu_eipm
        if isinstance(pmmethods, str):
            # Handle a string being provided instead of a list of str
            pmmethods = [pmmethods]
                
        for pmmethod in pmmethods:
            apu = get_apu(pmmethod, ac, apus)
            if apu is None:
                break
            elif (isinstance(apu, APU)
                  and (simcfg.apu_pm10_as_nvpm and not np.isnan(apu.pm10_cycle)
                       or not np.isnan(apu.nvpm_cycle))):
                break
        else:
            raise Exception('No method for APU nvPM EI calculation succeded')
        
        if apu is None:
            # gasmethod found an APU, but not we find that there is none
            warnings.warn('conflicting information about whether aircraft '
                          + f'"{ac.typecode}" has an APU')
            seg.nvpm     = 0
            seg.nvpmnum  = 0
        else:
            # PM mass
            if np.isnan(apu.nvpm_cycle):
                seg.nvpm = seg.fuelburn * apu.pm10_cycle / 1e3
            else:
                seg.nvpm = seg.fuelburn * apu.nvpm_cycle / 1e3
            # PM number
            if np.isnan(apu.nvpmnum_cycle):
                seg.nvpmnum = 0
            else:
                seg.nvpmnum = seg.fuelburn * apu.nvpmnum_cycle
    
        if departure:
            seg.phasenum = Phase.APU_DEP
            seg.t_s = t0 - t
            seg.t_e = t0
        else:
            seg.phasenum = Phase.APU_ARR
            seg.t_s = t0
            seg.t_e = t0 + t
    
        emissions = [seg]
    
    return emissions


def lto(ac, apt, departure, simcfg, apus, t0=0):
    """
    Calculate LTO emissions

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    apt : Airport
        Airport of departure or arrival.
    departure : bool
        If True, calculate departure emissions. If False, calculate arrival
        emissions.
    simcfg : SimConfig
        Configuration of simulation options.
    apus : dict of APU
        Dictionary of APU objects with str as keys
    t0 : float, optional
        Initial time in seconds. The default is 0.

    Raises
    ------
    NotImplementedError
        If simcfg.ltocycle is not in ['ICAO', 'Stettler', 'none'].

    Returns
    -------
    emissions : list of EmissionSegment
        Calculated emissions.

    """
    # APU emissions
    apu_emissions = apu_run(ac, apt, departure, simcfg, apus, t0=t0)
    
    # Main engine emissions
    if simcfg.ltocycle == 'ICAO':
        main_emissions = lto_icao(ac, apt, departure, simcfg, t0)
    elif simcfg.ltocycle == 'Stettler':
        main_emissions = lto_stettler(ac, apt, departure, simcfg, t0)
    elif simcfg.ltocycle == 'none':
        main_emissions = []
    else:
        raise ValueError('fly.lto: method not recognized')
    
    # Concatenate segments
    if departure:
        emissions = apu_emissions + main_emissions
    else:
        emissions = main_emissions + apu_emissions
    
    # Altitude coordinate transformation
    if simcfg.transform_alt:
        for e in emissions:
            e.h_s -= apt.h
            e.h_e -= apt.h
    
    return emissions


def short_cruise_alt(d, eng_type):
    """
    Give default cruise altitude for short flights

    Parameters
    ----------
    d : float
        Flight length in meters.
    eng_type : str
        Engine type. Possible values are 'Jet', 'Turboprop', and 'Piston'.

    Returns
    -------
    h : float or None
        Return default altitude if flight is shorter than 200 NM. If flight is
        too long or eng_type is not recognized, None is returned.
    
    References
    ----------
    Kim B, Fleming G, Balasubramanian S, Malwitz A, Lee J, Ruggiero J,
        Waitz I, Klima K, Stouffer V, Long D, Kostiuk P, Locke M, Holsclaw C,
        Morales A, McQueen E and Gillete W 2005 System for assessing Aviation’s
        Global Emissions (SAGE), Version 1.5, Technical Manual (Federal
        Aviation Administration)
    
    """
    d /= physics.NM_TO_METER
    if eng_type == 'Jet':
        if d < 100:
            fl = 170
        
        elif d < 150:
            fl = 230
        
        elif d < 200:
            fl = 250
        
        else:
            fl = None
    
    elif eng_type == 'Turboprop' or eng_type == 'Piston':
        if d < 100:
            fl = 85
        
        elif d < 200:
            fl = 150
            
        else:
            fl = None
        
    if fl is None:
        h = None
    else:
        h = fl * 100 * physics.FT_TO_METER
    
    return h


def non_lto(ac, apt_origin, apt_destination, simcfg, t0=0, met=None):
    """
    Calculate emissions for the non-LTO portion of a flight using BADA

    Parameters
    ----------
    ac : bada.Aircraft
        Aircraft to fly.
    apt_origin : pd.Series
        Origin airport.
    apt_destination : pd.Series
        Destination airport.
    simcfg : SimConfig
        Configuration of simulation options.
    t0 : float
        Initial time in seconds.
    met : xr.Dataset or None, optional
        Meteorological conditions. Can be None if simulation is configured to
        not use them. The default is None.

    Returns
    -------
    flight_emissions : list of EmissionSegment
        Emissions calculated.
        
    References
    ----------
    Aviation Emissions Inventory Code v2.1. Nicholas Simone, Marc Stettler.
        https://lae.mit.edu/codes/ accessed 06/Sep/2019.

    """
    H_TOLERANCE = 0.1  # tolerance to conclude climb and descent [m]
    
    # Initial position
    lat = apt_origin.lat
    lon = apt_origin.lon
    azi, flightdist = physics.distance(lat, lon, apt_destination.lat,
                                       apt_destination.lon)
    if flightdist < MIN_FLIGHTDIST:
        msg = ('Non-LTO not implemented for short flights.'
               + f' Flight distance of {flightdist:.0f} m,'
               + f' between {apt_origin.icao}'
               + f' and {apt_destination.icao},'
               + f' < limit of {MIN_FLIGHTDIST} m')
        raise NotImplementedError(msg)
    
    # non-LTO starts and ends at 3 kft above the airports
    h_cl_s = apt_origin.h + LTO_CEILING
    h_des_e = apt_destination.h + LTO_CEILING
    
    # --- Climb ---
    
    # Determine altitude of end of climb
    eng_type = ac.opf['aircraft']['eng_type']
    if simcfg.cruise_fl == 'bada-sage':
        h_cl_e = (ac.opf['envelope']['hMO']-7000)*physics.FT_TO_METER
        cap = short_cruise_alt(flightdist, eng_type)
        if cap is not None:
            h_cl_e = min(h_cl_e, cap)
    elif simcfg.cruise_fl == 'openavem':
        if ac.h_cr is not None:
            h_cl_e = ac.h_cr
        else:
            h_cl_e = (ac.opf['envelope']['hMO']-7000)*physics.FT_TO_METER
        cap = short_cruise_alt(flightdist, eng_type)
        if cap is not None:
            h_cl_e = min(h_cl_e, cap)
    else:
        # cruise_fl = 'bada'
        h_cl_e = (ac.opf['envelope']['hMO']-7000)*physics.FT_TO_METER
    
    h_cl_e = max(h_cl_s, h_cl_e)
    h_cl_e = min(h_cl_e, ac.opf['envelope']['hMO']*physics.FT_TO_METER)
    
    if h_cl_e < h_cl_s:
        msg = ('Trying to climb to a lower altitude '
               + f'({h_cl_s:.0f}->{h_cl_e:.0f}) '
               + f'({ac.typecode} at {apt_origin.icao}/{apt_origin.iata})')
        warnings.warn(msg)
        # Lets assume we will cruise at h_MO then
        h_cl_s = h_cl_e
    
    # Get coordinates of met data
    if met is None:
        met_lats = None
        met_lons = None
        met_cruise = None
    else:
        met_lats = met.lat.values
        met_lons = met.lon.values
        met_lons = np.append(met_lons, met_lons[0]+360)
        # Get slice of met data at cruise level
        ilev = met.h_edge.values.searchsorted(h_cl_e) - 1
        ilev = max(ilev, 0)
        met_cruise = met.isel(lev=ilev)
        
    
    h_step = simcfg.hstep_cl
    
    # Estimate starting mass
    # Method used by AEIC
    m = ac.opf['mass']['max']
    emissions = bada.cruise_fuel(ac, h_cl_e, apt_origin.lat, apt_origin.lon,
                                 apt_destination.lat, apt_destination.lon,
                                 flightdist, m, 0, simcfg,
                                 met_cruise, met_lats, met_lons)
    # +5% of reserve fuel
    est_fuel = (1 + EST_RESERVEFUEL_PCT) * emissions.fuelburn
    
    # Fuel for diversion and hold
    est_time = emissions.t_e
    low_alt_f = ac.ptf['table'].dropna(subset=['flo_cr']).iloc[0].flo_cr
    if est_time > 60 * 180:
        # Long haul
        est_fuel += emissions.fuelburn * EST_LONGHAUL_DIVERSION / flightdist
        est_fuel += low_alt_f * EST_LONGHAUL_LOWALTHOLD
    else:
        # Short haul
        est_fuel += emissions.fuelburn * EST_SHORTHAUL_DIVERSION / flightdist
        est_fuel += low_alt_f * EST_SHORTHAUL_LOWALTHOLD
    
    # Estimated payload mass
    est_pyld = simcfg.payload_m_frac * ac.opf['mass']['pyld']
    
    m_start = ac.opf['mass']['min'] + est_pyld + est_fuel
    m_start = min(m_start, ac.opf['mass']['max'])
    if simcfg.debug_m_est:
        print(f'm_start = {m_start:,.0f} kg    est_pyld = {est_pyld:,.0f} kg'
              + f'    est_fuel = {est_fuel:,.0f}')
    
    # Total distance flown until the current point of the flight [m]
    totdist = 0
    dist_to_dest = flightdist
    t = t0
    h = h_cl_s
    h_next = h + h_step
    m = m_start
    flight_emissions = []
    while h < h_cl_e - H_TOLERANCE:
        # Abort climb early if it's already time to descend
        est_des_dist = (h - h_des_e) / DES_EST_GRADE
        if h > h_des_e and dist_to_dest <= est_des_dist:
            break
        
        # Simulate this climb step
        emissions = bada.climb_fuel(ac, h, h_next, lat, lon, azi, m, t, simcfg,
                                    met, met_lats, met_lons)
        emissions.dist_flown_s = totdist
        flight_emissions.append(emissions)
        if simcfg.debug_climb_step:
            print(
                f'{h:6.0f} m -> {h_next:6.0f} m\t'
                + f'fuel: {emissions.fuelburn:.0f} kg'
            )
        
        m  -= emissions.fuelburn
        lat = emissions.lat_e
        lon = emissions.lon_e
        azi, dist_to_dest = physics.distance(lat, lon, apt_destination.lat,
                                             apt_destination.lon)
        totdist += emissions.dist
        t   = emissions.t_e
        h   = h_next
        h_next += h_step
        h_next = min(h_next, h_cl_e)
    else:
        # This fixes h in case it's off h_cl_e by less than H_TOLERANCE
        # note that this doesn't execute if climb ended prematurely
        h = h_cl_e
    
    if simcfg.verify_flightfuel > 0:
        total_fuelburn = (
            np.array([seg.fuelburn for seg in flight_emissions]).sum()
        )
        if simcfg.debug_climb_step:
            print(f'Total climb fuel burn: {total_fuelburn:.0f} kg\n')
        last_total_fuelburn = total_fuelburn
    
    # Cruise ---
    
    # Estimate cruise length
    est_des_dist = (h - h_des_e) / DES_EST_GRADE
    dist_to_go = dist_to_dest - est_des_dist
    
    s_step = simcfg.sstep_cr
    cruise_line = physics.geod_sphere.InverseLine(
        lat, lon, apt_destination.lat, apt_destination.lon
    )
    
    # s = distance from the start of cruise to the current point
    last_s = 0
    s = min(s_step, dist_to_go)
    while last_s < dist_to_go:
        p = cruise_line.Position(s)
        global temp
        temp = p
        lat_next = p['lat2']
        lon_next = p['lon2']
        dist = p['s12'] - last_s
        emissions = bada.cruise_fuel(ac, h, lat, lon, lat_next, lon_next, dist,
                                     m, t, simcfg, met_cruise, met_lats,
                                     met_lons)
        emissions.azi_e = p['azi2']
        emissions.dist_flown_s = totdist + last_s
        emissions.dist_to_go_s = dist_to_go - last_s
        flight_emissions.append(emissions)
        if simcfg.debug_cruise_step:
            print(f'{last_s/1000:6.0f} m -> {s/1000:6.0f} m\t'
                  + f'fuel: {emissions.fuelburn:.0f} kg')
        
        m  -= emissions.fuelburn
        last_s = s
        lat = lat_next
        lon = lon_next
        t   = emissions.t_e
        s += s_step
        s = min(s, dist_to_go)
    
    if simcfg.verify_flightfuel > 0:
        total_fuelburn = (
            np.array([seg.fuelburn for seg in flight_emissions]).sum()
        )
        if simcfg.debug_cruise_step:
            print('Total cruise fuel burn: '
                  + f'{total_fuelburn-last_total_fuelburn:.0f} kg\n')
        last_total_fuelburn = total_fuelburn
    
    # Descent ---
    
    h_des_s = h
    h_des_e = min(h_des_s, apt_destination.h + LTO_CEILING)
    h_step = simcfg.hstep_des
    h_next = h - h_step
    while h > h_des_e + H_TOLERANCE:
        azi, dist_to_go = physics.distance(lat, lon, apt_destination.lat,
                                           apt_destination.lon)
        emissions = bada.descent_fuel(ac, h, h_next, lat, lon, azi, m, t,
                                      simcfg, met, met_lats, met_lons)
        emissions.dist_to_go_s = dist_to_go
        flight_emissions.append(emissions)
        if simcfg.debug_descent_step:
            print(f'{h:6.0f} m -> {h_next:6.0f} m\t'
                  + f'fuel: {emissions.fuelburn:.1f} kg')
        
        m  -= emissions.fuelburn
        lat = emissions.lat_e
        lon = emissions.lon_e
        azi = emissions.azi_e
        t   = emissions.t_e
        h   = h_next
        h_next -= h_step
        h_next = max(h_next, h_des_e)
    h = h_des_e
    
    if simcfg.verify_flightfuel > 0:
        total_fuelburn = (
            np.array([seg.fuelburn for seg in flight_emissions]).sum()
        )
        if simcfg.debug_descent_step:
            print('Total descent fuel burn: '
                  + f'{total_fuelburn-last_total_fuelburn:.0f} kg\n')
        if total_fuelburn > (simcfg.verify_flightfuel
                             * (ac.opf['mass']['max']
                                - ac.opf['mass']['min'])):
            msg = (f'fuel burn ({total_fuelburn:,.1f} kg) > '
                   + f'{simcfg.verify_flightfuel:.2f} * (m_max - m_min)'
                   + f' [{simcfg.verify_flightfuel:.2f} * '
                   + f"{ac.opf['mass']['max'] - ac.opf['mass']['min']:,.1f}"
                   + f' kg] for {ac.typecode} flying {apt_origin.icao} -> '
                   + f'{apt_destination.icao}')
            warnings.warn(msg)
        
    if hasattr(ac, 'eng'):
        eng = ac.eng
        if hasattr(eng, 'bffm2'):
            # Calculate NOx, CO, HC, BC emissions for all segments
            flight_emissions = bffm2(flight_emissions, ac)
        
        # Calculate nvPM emissions for all segments
        if eng.type == 'P':
            methods = simcfg.nvpm_cruise_piston
            if isinstance(methods, str):
                # Handle a string being provided instead of a list of str
                methods = [methods]
            for method in methods:
                if method == 'AEDT-FOA4':
                    if (np.isnan(eng.nvpm_lto).any()
                        and (not hasattr(eng, 'foa4') or eng.foa4['zero'])):
                        continue
                    else:
                        nvpm_piston(flight_emissions, ac)
                        break
                elif method == 'constantEI':
                    for seg in flight_emissions:
                        seg.nvpm = seg.fuelburn * simcfg.nvpm_cruise_EI / 1e6
                        seg.nvpmnum = (seg.fuelburn
                                       * simcfg.nvpmnum_cruise_EI * 1e14)
                    break
                elif method == 'none':
                    for seg in flight_emissions:
                        seg.nvpm = 0
                        seg.nvpmnum = 0
                    break
                else:
                    raise ValueError(f'simcfg.nvpm_cruise_piston "{method}" '
                                     + 'not recognized')
            else:
                raise Exception(
                    'No method for LTO nvPM EI calculation succeded')
        else:
            methods = simcfg.nvpm_cruise
            if isinstance(methods, str):
                # Handle a string being provided instead of a list of str
                methods = [methods]
            
            for method in methods:
                if method == 'AEDT-FOA4':
                    if (np.isnan(eng.nvpm_lto).any()
                        and (not hasattr(eng, 'foa4') or eng.foa4['zero'])):
                        continue
                    else:
                        nvpm_cruise_aedt(flight_emissions, ac)
                        break
                elif method == 'FOX':
                    if not hasattr(eng, 'fox') or eng.fox['zero']:
                        continue
                    else:
                        nvpm_fox(flight_emissions, ac)
                        for seg in flight_emissions:
                            seg.nvpmnum = 0
                        break
                elif method == 'constantEI':
                    for seg in flight_emissions:
                        seg.nvpm = seg.fuelburn * simcfg.nvpm_cruise_EI / 1e6
                        seg.nvpmnum = (seg.fuelburn
                                       * simcfg.nvpmnum_cruise_EI * 1e14)
                    break
                elif method == 'none':
                    for seg in flight_emissions:
                        seg.nvpm = 0
                        seg.nvpmnum = 0
                    break
                else:
                    raise ValueError(f'simcfg.nvpm_cruise "{method}" '
                                     + 'not recognized')
            else:
                raise Exception(
                    'No method for LTO nvPM EI calculation succeded')
        
    if simcfg.lat_inefficiency == 'AEIC':
        lat_ineff_aeic(flight_emissions, flightdist, simcfg.lat_ineff_cr_max)
    elif simcfg.lat_inefficiency == 'FEAT':
        lat_ineff_feat(flight_emissions, flightdist)
    elif simcfg.lat_inefficiency != 'none':
        raise ValueError("Couldn't recognize simcfg.lat_inefficiency "
                         + f'"{simcfg.lat_inefficiency}"')        
    
    if simcfg.transform_alt:
        for seg in flight_emissions:
            if seg.phasenum == Phase.CLIMB:
                if seg.h_s < grid.ALT_TRANSITION:
                    seg.h_s = (
                        LTO_CEILING + ((seg.h_s - h_cl_s)
                                       / (grid.ALT_TRANSITION - h_cl_s)
                                       * (grid.ALT_TRANSITION - LTO_CEILING)))
                if seg.h_e < grid.ALT_TRANSITION:
                    seg.h_e = (
                        LTO_CEILING + ((seg.h_e - h_cl_s)
                                       / (grid.ALT_TRANSITION - h_cl_s)
                                       * (grid.ALT_TRANSITION - LTO_CEILING)))
            elif seg.phasenum == Phase.DESCENT:
                if seg.h_s < grid.ALT_TRANSITION:
                    seg.h_s = (
                        LTO_CEILING + ((seg.h_s - h_des_e)
                                       / (grid.ALT_TRANSITION - h_des_e)
                                       * (grid.ALT_TRANSITION - LTO_CEILING)))
                if seg.h_e < grid.ALT_TRANSITION:
                    seg.h_e = (
                        LTO_CEILING + ((seg.h_e - h_des_e)
                                       / (grid.ALT_TRANSITION - h_des_e)
                                       * (grid.ALT_TRANSITION - LTO_CEILING)))
    
    return flight_emissions


def nvpm_fox(segs, ac, Ts=physics.MSL_T, ps=physics.MSL_P, max_EI=1000):
    """
    Add nvpm mass emissions to segments with the FOX method

    Parameters
    ----------
    segs : list of EmissionSegment
        Segments with fuel burn.
    ac : bada.Aircraft
        Modeled aircraft.
    Ts : float, optional
        Ambient (static) temperature at compressure inlet in K.
        The default is physics.MSL_T.
    ps : float, optional
        Ambient absolute (static) pressure at compressure inlet in Pa.
        The default is physics.MSL_P.
    max_EI : float, optional
        Maximum allowable EI in mg/kg. The calculated EI will be clipped by
        this limit. The default is 1000.

    Returns
    -------
    segs : list of EmissionSegment
        Segments with NOx, CO, and HC emissions added.
    
    References
    ----------
    Stettler M E J, Boies A M, Petzold A and Barrett S R H 2013 Global Civil
        Aviation Black Carbon Emissions Environ. Sci. Technol. 47 10397–404
    
    Aviation Emissions Inventory Code v2.1. Nicholas Simone, Marc Stettler.
        https://lae.mit.edu/codes/ accessed 06/Sep/2019.

    """
    eng = ac.eng
    if eng.type not in ['TF', 'MTF']:
        for seg in segs:
            seg.nvpm = 0
        return segs
    neng = ac.opf['aircraft']['n_engs']
    
    # Overall pressure ratio [-]
    pr = eng.pr
    
    # eta_p: polytropic efficiency [-]
    eta_p = 0.9
    # gamma: specific heat ratio of the air [-]
    gamma = physics.AIR_GAMMA
    
    # Use takeoff as the reference point
    p3ref = eng.fox['p3ref'][-1]
    Tflref = eng.fox['Tflref'][-1]
    AFRref = eng.fox['AFRref'][-1]
    C_BCref = eng.fox['C_BCref'][-1]
    
    for seg in segs:
        # f: fuel mass flow rate [kg/s]
        dt = seg.t_e - seg.t_s
        f = seg.fuelburn / dt / neng
        # mf_rel: percentage of fuel mass flow rate to its maximum value [-]
        mf_rel = f / eng.fuel_to
        
        # Ambient atmospheric conditions
        Ts, ps, rho, a = physics.isa_trop(seg.h_s)
        # M: Mach number [-]
        M = seg.tas / a
        # T2: temperature at compressor inlet [K]
        T2 = Ts * (1 + (gamma - 1) / 2 * M**2)
        # p2: pressure at compressor inlet [K]
        p2 = ps * (1 + (gamma - 1) / 2 * M**2)**(gamma / (gamma - 1))
        
        # p3: absolute pressure at the combustor [Pa]
        p3 = ((pr - 1) * mf_rel + 1) * physics.MSL_P
        # T3: temperature at the combustor [K]
        T3 = T2 * (p3/p2)**((gamma - 1)/ (gamma * eta_p))
        
        # Tfl: flame temperature [K]
        Tfl = 0.9 * T3 + 2120
        
        # AFR: air to fuel ratio [-]
        # AEIC used +0.0078, but Stettler et al 2013 write 0.008
        AFR = 1 / (0.0121 * mf_rel + 0.008)
        
        # C_BC: black carbon concentration [mg/m3]
        C_BC = (C_BCref
                * (AFRref / AFR)**2.5
                * (p3 / p3ref)**1.35
                * np.exp(-20000 / Tfl + 20000 / Tflref))
        C_BC = np.clip(C_BC, 0, max_EI)
        
        # Q: volumetric flow rate [m3/kg]
        Q = 0.776 * AFR + 0.877
        
        # EI: emission index [g/kg]
        ei = C_BC * Q / 1e3
        
        seg.nvpm = seg.fuelburn * ei / 1e3
    
    return segs


def nvpm_piston(segs, ac, foca_power_settings=False):
    if ac.eng.type != 'P':
        raise Exception('Expected a piston engine')
    
    # Default engine thrust setting
    def_F = {Phase.TAXI:          0.12,
             Phase.TAXIIN:        0.12,
             Phase.TAXIACC_DEP:   0.12,
             Phase.TAKEOFF:       1.00,
             Phase.INITIAL_CLIMB: 1.00,
             Phase.CLIMBOUT:      0.85,
             Phase.CLIMB:         1.00,
             Phase.CRUISE:        0.65,
             Phase.DESCENT:       0.16,
             Phase.APPROACH:      0.45,
             Phase.LANDING:       0.12,
             Phase.REVERSE:       1.00,
             Phase.TAXIOUT:       0.12,
             Phase.TAXIACC_ARR:   0.12,
             Phase.TAXI:          0.12}
    eng = ac.eng
    if np.isnan(eng.nvpm_lto).any():
        pchip_interp = interpolate.pchip(LTO_FOCA_THRUST, eng.foa4['ei_lto'])
        lin_interp = interpolate.interp1d(LTO_FOCA_THRUST, eng.foa4['ei_lto'])
        lin_numinterp = interpolate.interp1d(LTO_FOCA_THRUST,
                                             eng.foa4['num_ei_lto'])
    else:
        pchip_interp = interpolate.pchip(LTO_FOCA_THRUST, eng.nvpm_lto)
        lin_interp = interpolate.interp1d(LTO_FOCA_THRUST, eng.nvpm_lto)
        lin_numinterp = interpolate.interp1d(LTO_FOCA_THRUST, eng.nvpmnum_lto)
    
    for seg in segs:
        # Engine thrust setting
        if foca_power_settings:
            F = def_F[seg.phasenum]
        else:
            F = seg.fuelburn / (seg.t_e - seg.t_s) / eng.fuel_to
            F = np.clip(F, 0.12, 1.00)
        
        # EIs at calculated power level at ground conditions
        ei_m = pchip_interp(F)
        if ei_m <= 0:
            ei_m = lin_interp(F)
        ei_num = lin_numinterp(F)
        
        seg.nvpm = ei_m * seg.fuelburn / 1e6
        seg.nvpmnum = ei_num * seg.fuelburn
            
    return segs
    
    
def nvpm_cruise_aedt(segs, ac, aedt_power_settings=False, t_flame='Zahavi'):
    """
    Add nvPM mass and number emissions to cruise segments with AEDT-FOA4 method

    Parameters
    ----------
    segs : list of EmissionSegment
        Segments with fuel burn.
    ac : bada.Aircraft
        Modeled aircraft.
    aedt_power_settings : bool, optional
        If True, set thrust level to AEDT default values according to flight
        phase. If false, thrust level is proportional to fuel burn.
        The default is False.
    t_flame : str, optional
        Method used to estimate flame temperature. Possible values are
        'Zahavi', 'Najjar', 'Lefebvre', and 'linear'. The default is 'Zahavi'.

    Returns
    -------
    segs : list of EmissionSegment
        Segments with NOx, CO, and HC emissions added.
    
    References
    ----------
    U.S. Department of Transportation 2021 Aviation Environmental Design Tool
        (AEDT), Technical Manual, Version 3d

    Peck J, Oluwole O O, Wong H-W and Miake-Lye R C 2013 An algorithm to
        estimate aircraft cruise black carbon emissions for use in developing a
        cruise emissions inventory Journal of the Air & Waste Management
        Association 63 367–75
    
    """
    # Default engine thrust setting
    def_F = {Phase.TAXI:          0.07,
             Phase.TAXIIN:        0.07,
             Phase.TAXIACC_DEP:   0.10,
             Phase.TAKEOFF:       1.00,
             Phase.INITIAL_CLIMB: 1.00,
             Phase.CLIMBOUT:      0.85,
             Phase.CLIMB:         1.00,
             Phase.CRUISE:        0.95,
             Phase.DESCENT:       0.16,
             Phase.APPROACH:      0.30,
             Phase.LANDING:       0.07,
             Phase.REVERSE:       1.00,
             Phase.TAXIOUT:       0.07,
             Phase.TAXIACC_ARR:   0.10,
             Phase.TAXI:          0.07}
    eng = ac.eng
    if np.isnan(eng.nvpm_lto).any():
        pchip_interp = interpolate.pchip(LTO_ICAO_THRUST, eng.foa4['ei_lto'])
        lin_interp = interpolate.interp1d(LTO_ICAO_THRUST, eng.foa4['ei_lto'])
        lin_numinterp = interpolate.interp1d(LTO_ICAO_THRUST,
                                             eng.foa4['num_ei_lto'])
    else:
        pchip_interp = interpolate.pchip(LTO_ICAO_THRUST, eng.nvpm_lto)
        lin_interp = interpolate.interp1d(LTO_ICAO_THRUST, eng.nvpm_lto)
        lin_numinterp = interpolate.interp1d(LTO_ICAO_THRUST, eng.nvpmnum_lto)
    
    for seg in segs:
        # Atmospheric conditions
        h = seg.h_s
        T, p, rho, a = physics.isa_trop(h)
        
        # Mach number
        M = seg.tas / a
        
        # Total temperature and pressure
        gamma = physics.AIR_GAMMA
        Tt = T * (1 + (gamma - 1) / 2 * M**2)
        pt = p * ((1 + (gamma - 1) / 2 * M**2)**(gamma / (gamma - 1)))
        
        # Engine thrust setting
        if aedt_power_settings:
            F = def_F[seg.phasenum]
        else:
            F = seg.fuelburn / (seg.t_e - seg.t_s) / eng.fuel_to
            F = np.clip(F, 0.07, 1.15)
        
        # Compressor efficiency
        if F < 0.1:
            eta_comp = 0.65
        elif F < 0.3:
            eta_comp = 0.70
        else:
            eta_comp = 0.85
        
        # T and p at the combustor (cruise)
        p3 = pt * (1 + F * (eng.pr - 1))
        T3 = Tt * (1 + ((p3 / pt)**((gamma - 1) / gamma) - 1) / eta_comp)
        
        # T and p at the combustor (ground)
        Tgnd = physics.MSL_T
        pgnd = physics.MSL_P
        T3ref = T3
        p3ref = pgnd * ((1 + eta_comp * (T3ref / Tgnd - 1))
                        **(gamma / (gamma - 1)))

        # Engine power at ground level
        Fref = (p3ref / pgnd - 1) / (eng.pr - 1)
        
        # EIs at calculated power level at ground conditions
        Fref_clipped = clip_thrust_to_icao(Fref)
        ei_ref = pchip_interp(Fref_clipped)
        if ei_ref <= 0:
            ei_ref = lin_interp(Fref_clipped)
        ei_num = lin_numinterp(Fref_clipped)
        
        # Flame temperatures
        if t_flame == 'Zahavi':
            Tfl = 2281 * (p3**0.009375 + 0.000178 * (p3**0.055) * (T3 - 298))
            Tflref = 2281 * (p3ref**0.009375 + 0.000178 * (p3ref**0.055)
                             * (T3ref - 298))
        elif t_flame == 'Najjar':
            Tfl = 1067.93 * T3**0.13306
            Tflref = 1067.93 * T3ref**0.13306
        elif t_flame == 'Lefebvre':
            Tfl = 0.6 * T3 + 1800
            Tflref = 0.6 * T3ref + 1800
        else:
            # == 'linear'
            Tfl = 0.9 * T3 + 2120
            Tflref = 0.9 * T3ref + 2120

        # Change of stoichiometric ratio is disconsidered
        phi = 1.0
        phi_ref = 1.0
        
        ei_m = (ei_ref
                * (phi / phi_ref)**2.5
                * (p3 / p3ref)**1.35
                * np.exp(-20000/Tfl + 20000/Tflref))
        
        seg.nvpm = ei_m * seg.fuelburn / 1e6
        seg.nvpmnum = ei_num * seg.fuelburn
            
    return segs
        

def bffm2(segs, ac, clip_wf_idle=True):
    """
    Add NOx, CO, HC emissions to segments using Boeing's Fuel Flow Method 2

    Parameters
    ----------
    segs : list of EmissionSegment
        Segments with fuel burn.
    ac : bada.Aircraft
        Modeled aircraft.
    clip_wf_idle : bool, optional
        If True, limit emission index interpolations to a minimum fuel flow of
        equal to the certification idle point (7% thrust). If False,
        interpolation can go to zero fuel flow. The default is True.

    Returns
    -------
    segs : list of EmissionSegment
        Segments with NOx, CO, and HC emissions added.
    
    References
    ----------
    Baughcum, Steven L, Terrance G Tritz, Stephen C Henderson, and David C
        Pickett. “Scheduled Civil Aircraft Emission Inventories for 1992:
        Database Development and Analysis.” Langley Research Center, April
        1996.

    """
    # Relative humidity (constant at 60% for now) for NOx altitude correction
    PHI = 0.6
    
    # Number of engines and emission parameters
    neng = ac.opf['aircraft']['n_engs']
    params = ac.eng.bffm2
    for seg in segs:
        # Fuel flow rate [kg/s]
        dt = seg.t_e - seg.t_s
        Wf = seg.fuelburn / dt / neng
        if dt < 0.001:
            warnings.warn(f'dt = {dt} at {seg.__dict__}')
        
        if clip_wf_idle:
            # Adopt Wf_Idle as a minimum
            Wf = max(Wf, ac.eng.fuel_idle)
        
        # Conversion to equivalent standard (T, p)
        h = seg.h_s
        # Atmospheric conditions taken as ISA at altitude h
        T, p, rho, a = physics.isa_trop(h)
        theta_amb = T / physics.MSL_T
        delta_amb = p / physics.MSL_P
        M = seg.tas / a
        
        # Fuel flow factor [kg/s]
        Wff = Wf / delta_amb * theta_amb**3.8 * np.exp(0.2 * M**2)
        
        # Calculate log and clip to range between (0 or Idle) and TO
        if clip_wf_idle:
            log_wff = np.clip(np.log10(Wff), params['logwf'][PHASE_IDLE],
                              params['logwf'][PHASE_TO])
        else:
            log_wff = np.clip(np.log10(Wff), None, params['logwf'][PHASE_TO])
        
        # NOx
        if not params['zero_nox']:
            # zero_nox being False means we have the required parameters
            # Calculate (R)eference EI [g/kg]
            rei_nox = np.power(10, (
                params['nox_fit'][1] + log_wff * params['nox_fit'][0]
            ))
            
            # Convert REI to EI at flying conditions
            beta = (7.90298 * (1 - 373.15 / T) + 3.00571
                    + 5.02808 * np.log10(373.15 / T)
                    + 1.3816e-7 * (1 - 10**(11.344 * (1 - T / 373.15)))
                    + 8.1328e-3 * (10**(3.49149 * (1 - 373.15 / T)) - 1)
                )
            # Pv = saturation vapor pressure [Pa]
            p_v = 0.014504 * 10**beta
            # omega = specific humidity [-]
            omega = 0.62198 * PHI * p_v / (p / physics.PSI_TO_PA - PHI * p_v)
            H = -19 * (omega - 0.0063)
            # Emissions index [g/kg]
            ei_nox = (
                rei_nox * np.exp(H) / theta_amb**(3.3/2) * delta_amb**(1.02/2)
            )
            
            # Calculate and save emissions [kg]
            seg.nox = ei_nox * seg.fuelburn / 1000
        else:
            seg.nox = 0
        
        # CO
        if not params['zero_co']:
            # zero_co being False means we have the required parameters
            # Calculate (R)eference EI [g/kg]
            if log_wff < params['co_break']:
                rei_co = np.power(10, (
                    params['co_fitl'][1]
                    + (log_wff - params['logwf'][PHASE_IDLE])
                    * params['co_fitl'][0]
                ))
            else:
                rei_co = np.power(10, params['co_fith'])
            
            # Convert REI to EI at flying conditions [g/kg]
            ei_co = rei_co * theta_amb**3.3 / delta_amb**1.02
            
            # Calculate and save emissions [kg]
            seg.co = ei_co * seg.fuelburn / 1000
        else:
            seg.co = 0
        
        # HC
        if not params['zero_hc']:
            # zero_hc being False means we have the required parameters
            # Calculate (R)eference EI [g/kg]
            if log_wff < params['hc_break']:
                rei_hc = np.power(10, (
                    params['hc_fitl'][1]
                    + (log_wff - params['logwf'][PHASE_IDLE])
                    * params['hc_fitl'][0]
                ))
            else:
                rei_hc = np.power(10, params['hc_fith'])
            
            # Convert REI to EI at flying conditions [g/kg]
            ei_hc = rei_hc * theta_amb**3.3 / delta_amb**1.02
            
            # Calculate and save emissions [kg]
            seg.hc = ei_hc * seg.fuelburn / 1000
        else:
            seg.hc = 0
            
    return segs


def lat_ineff_isineu(s):
    return s.lat_s >= 36 and s.lat_s <= 72 and s.lon_s >= -13 and s.lon_s <= 45


def lat_ineff_aeic(segs, d, max_cr_mult=2.0):
    """
    Apply lateral inefficiency to a flight based on AEIC's method

    Parameters
    ----------
    segs : list of EmissionSegment
        Segments on which to apply inefficiency.
    d : float
        Total geodesic length of the flight in meters.
    max_cr_mult: float
        Maximum multiplier that can be applied to cruise emission. This is
        needed since the multiplier goes to infinity as cruise distance goes
        to zero, creating a scenario approaching 0 * inf.

    Returns
    -------
    None.
    
    References
    ----------
    Aviation Emissions Inventory Code v2.1. Nicholas Simone, Marc Stettler.
        https://lae.mit.edu/codes/ accessed 06/Sep/2019.
    
    """
    # Lateral inneficiency constants
    LATINEFF_RANGE_DEP = 50 * physics.NM_TO_METER
    LATINEFF_RANGE_ARR = 50 * physics.NM_TO_METER
    LATINEFF_DEP_US = 1.156
    LATINEFF_DEP_EU = 1.180
    LATINEFF_CR_A_US = 1.029
    LATINEFF_CR_A_EU = 1.020
    LATINEFF_CR_B_US = 22 * physics.NM_TO_METER
    LATINEFF_CR_B_EU = 12 * physics.NM_TO_METER
    LATINEFF_ARR_US = 1.554
    LATINEFF_ARR_EU = 1.530

    # Departure scale factor
    if lat_ineff_isineu(segs[0]):
        mult_dep = LATINEFF_DEP_EU
    else:
        mult_dep = LATINEFF_DEP_US
    
    # Cruise scale factors
    d -= LATINEFF_RANGE_DEP + LATINEFF_RANGE_ARR
    if d > 0:
        mult_cr = (LATINEFF_CR_A_US * d + LATINEFF_CR_B_US) / d
        mult_cr_eu = (LATINEFF_CR_A_EU * d + LATINEFF_CR_B_EU) / d
        mult_cr = min(mult_cr, max_cr_mult)
        mult_cr_eu = min(mult_cr_eu, max_cr_mult)
    else:
        mult_cr = 1
        mult_cr_eu = 1
    
    # Arrival scale factor
    if lat_ineff_isineu(segs[-1]):
        mult_arr = LATINEFF_ARR_EU
    else:
        mult_arr = LATINEFF_ARR_US
    
    # Apply factors to each segment based on flight phase and location
    for seg in segs:
        if seg.phasenum == Phase.CLIMB:
            if seg.dist_flown_s <= LATINEFF_RANGE_DEP:
                multiplier = mult_dep
            else:
                if lat_ineff_isineu(seg):
                    multiplier = mult_cr_eu
                else:
                    multiplier = mult_cr
        elif seg.phasenum == Phase.CRUISE:
            if seg.dist_flown_s <= LATINEFF_RANGE_DEP:
                multiplier = mult_dep
            elif seg.dist_to_go_s <= LATINEFF_RANGE_ARR:
                multiplier = mult_arr
            else:
                if lat_ineff_isineu(seg):
                    multiplier = mult_cr_eu
                else:
                    multiplier = mult_cr
        elif seg.phasenum == Phase.DESCENT:
            if seg.dist_to_go_s <= LATINEFF_RANGE_ARR:
                multiplier = mult_arr
            else:
                if lat_ineff_isineu(seg):
                    multiplier = mult_cr_eu
                else:
                    multiplier = mult_cr
        else:
            multiplier = 1
        
        for spc in ['fuelburn', 'nox', 'co', 'hc', 'nvpm', 'nvpmnum']:
            if hasattr(seg, spc):
                seg.__dict__[spc] *= multiplier
    
    return


def lat_ineff_feat(segs, d):
    """
    Apply lateral inefficiency to a flight based on FEAT's method

    Parameters
    ----------
    segs : list of EmissionSegment
        Segments on which to apply inefficiency.
    d : float
        Total geodesic length of the flight in meters.
    max_mult: float
        Maximum multiplier that can be applied. This is needed since the
        multiplier goes to infinity as distance goes to zero, creating a
        scenario approaching 0 * inf.

    Returns
    -------
    None.
    
    References
    ----------
    Seymour K, Held M, Georges G and Boulouchos K 2020 Fuel Estimation in Air
    Transportation: Modeling global fuel consumption for commercial aviation
    Transportation Research Part D: Transport and Environment 88 102528

    """
    if d <= 0:
        return
    
    LATINEFF_A = 1.0387
    LATINEFF_B = 40.5491 * physics.NM_TO_METER
    MIN_D = 100e3
    MAX_MULT = (LATINEFF_A * MIN_D + LATINEFF_B) / MIN_D
    
    # Scale factor
    mult = (LATINEFF_A * d + LATINEFF_B) / d
    mult = min(mult, MAX_MULT)
    
    # Apply factors to each segment
    for seg in segs:
        for spc in ['fuelburn', 'nox', 'co', 'hc', 'nvpm', 'nvpmnum']:
            if hasattr(seg, spc):
                seg.__dict__[spc] *= mult
    
    return


def fly_list_ongrid(fcounts, airports, thisbada, simcfg, apus, met=None,
                    ds_grid=None):
    """
    Calculate emissions from list of flights and add them to a grid

    Parameters
    ----------
    fcounts : pd.Series
        Number of times each route is flown, with an index of the form
        (ac, org, des) where
            ac is a str of the aircraft's ICAO typecode
            org is a str of the origin's ICAO or IATA code
            des is a str of the destination's ICAO or IATA code
    airports : dict
        Dictionary of ICAO or IATA codes (matching fcounts.org and fcounts.des)
        to Airport objects.
    thisbada : bada.Bada
        Bada instance.
    simcfg : SimConfig
        Configuration of simulation options.
    apus : dict of str to APU
        Dictionary containing APU objected indexed with the keys
        "icao_simple_business", "icao_simple_short" and "icao_simple_long"
    met : xr.Dataset or None, optional
        Meteorological conditions. Can be None if simulation is configured to
        not use them. The default is None.
    ds_grid : ds.DataFrame, optional
        Dataframe containing the grid. If None, a new grid is created with
        default parameters. The default is None.
    
    Returns
    -------
    ds_grid : pd.DataFrame
        Grid with calculated emissions added.

    """
    if ds_grid is None:
        ds_grid = grid.grid_from_simcfg(simcfg)
    
    if not simcfg.grid_everyflight:
        # Hold all emission segments before gridding
        all_emissions = []
    
    fstats = []
    for flight, n in zip(fcounts.index, fcounts):
        typecode = flight[0]
        origin = flight[1]
        destination = flight[2]
        if simcfg.debug_flist:
            print(f'{typecode} from {origin} to {destination}...', end=' ')
        ac = thisbada.acs[typecode]
        if origin not in airports:
            if simcfg.ignore_missing_apts:
                warnings.warn(f'Missing airport: "{origin}"')
                continue
            else:
                raise ModelDataMissing(f'Missing airport: "{origin}"')
        apt_origin = airports[origin]
        if destination not in airports:
            if simcfg.ignore_missing_apts:
                warnings.warn(f'Missing airport "{destination}"')
                continue
            else:
                raise ModelDataMissing(f'Missing airport: "{destination}"')
        apt_destination = airports[destination]
        
        if simcfg.remove_same_od:
            _, flightdist = physics.distance(apt_origin.lat, apt_origin.lon,
                                               apt_destination.lat,
                                               apt_destination.lon)
            if flightdist < MIN_FLIGHTDIST:
                msg = (f'Flight distance of {flightdist:.0f} m,'
                       + f' between {apt_origin.icao}'
                       + f' and {apt_destination.icao},'
                       + f' < limit of {MIN_FLIGHTDIST} m')
                warnings.warn(msg)
                continue
        
        dep_emissions = lto(ac, apt_origin, departure=True, simcfg=simcfg,
                            apus=apus, t0=0)
        if len(dep_emissions) > 0:
            t = dep_emissions[-1].t_e
        else:
            t = 0
        
        if simcfg.fly_nonlto:
            nonlto_emissions = non_lto(ac, apt_origin, apt_destination, simcfg,
                                       t, met)
            t = nonlto_emissions[-1].t_e
        else:
            nonlto_emissions = []
        
        arr_emissions = lto(ac, apt_destination, departure=False,
                            simcfg=simcfg, apus=apus, t0=t)
        emissions = dep_emissions + nonlto_emissions + arr_emissions
        
        if simcfg.debug_flist:
            fuel_dep = np.array([s.fuelburn for s in dep_emissions]).sum()
            fuel_nonlto = np.array([s.fuelburn
                                    for s in nonlto_emissions]).sum()
            fuel_arr = np.array([s.fuelburn for s in arr_emissions]).sum()
            print(f'fuel: {fuel_dep:7,.0f} / '
                  + f'{fuel_nonlto:7,.0f} / {fuel_arr:7,.0f} kg\tn = {n:.0f}')
        
        if simcfg.return_flight_durations:
            t = emissions[-1].t_e
            totfuel = np.array([s.fuelburn for s in emissions]).sum()
            fstats.append({'typecode': typecode,
                           'origin': origin,
                           'destination': destination,
                           'duration': t,
                           'fuel': totfuel,
                           'n': n})
        
        if len(emissions) > 0:
            if simcfg.grid_everyflight:
                # Grid emissions after every flight
                if simcfg.grid_method == 'supersampling':
                    ds_grid = grid.grid_supersampled(emissions, ds_grid,
                                                     multiplier=n)
                elif simcfg.grid_method == 'nearestneighbor':
                    ds_grid = grid.grid_nearest_neighbor(emissions, ds_grid,
                                                         multiplier=n)
                else:
                    raise ValueError(f'simcfg.grid_method "{simcfg.grid_method}" '
                                     + 'not recognized')
            else:
                # Keep emission segments in memory and hold off gridding
                # Multiplier needs to be applied now however
                for seg in emissions:
                    for spc in ['fuelburn', 'nox', 'co', 'hc', 'nvpm',
                                'nvpmnum']:
                        if hasattr(seg, spc):
                            seg.__dict__[spc] *= n
                all_emissions += emissions
    
    if not simcfg.grid_everyflight and len(fcounts) > 0 and len(emissions) > 0:
        # Grid all emissions for the list of flights
        if simcfg.grid_method == 'supersampling':
            ds_grid = grid.grid_supersampled(all_emissions, ds_grid,
                                             multiplier=1)
        elif simcfg.grid_method == 'nearestneighbor':
            ds_grid = grid.grid_nearest_neighbor(all_emissions, ds_grid,
                                                 multiplier=1)
        else:
            raise ValueError(f'simcfg.grid_method "{simcfg.grid_method}" '
                             + 'not recognized')
    
    if simcfg.return_flight_durations:
        return ds_grid, fstats
    
    return ds_grid


def testflight(typecode='B77W', apt_origin='GRU', apt_destination='AMS',
               iata_keys=True, simcfg=None,
               figname=None, plotfuel=True, cfg_params={},
               verbose=1):
    """
    Calculate emissions from a single flight

    Parameters
    ----------
    typecode : str, optional
        Aircraft ICAO typecode. The default is 'B77W'.
    apt_origin_IATA : str, optional
        IATA code of the origin airport. The default is 'GRU'.
    apt_destination_IATA : str, optional
        IATA code of the destination airport. The default is 'AMS'.
    simcfg : SimConfig or None, optional
        Configuration of simulation options. If None, create a new object with
        default configuration and parameters from cfg_params.
        The default is None.
    figname : str or None, optional
        File name to save a plot of calculated fuel burn. If None, the figure
        is not saved. The default is None.
    plotfuel : bool, optional
        If True, plot calculated fuel burn. The default is True.
    cfg_params : dict, optional
        Parameters to initialize simcfg, if none is provided. This argument is
        ignored if simcfg is passed. The default is {}.

    Returns
    -------
    ds : xr.Dataset
        Gridded emissions calculated.
    emissions : list of EmissionSegment
        Calculated emissions.

    """
    if simcfg is None:
        if 'ignore_missing_engs' not in cfg_params:
            cfg_params['ignore_missing_engs'] = False
        simcfg = SimConfig(**cfg_params)
    
    # Load payload mass fraction from weight load factor if needed
    determine_payload_m_frac(simcfg)
    
    if verbose >= 1:
        print(f'{"Loading airport data...":33}', end='', flush=True)
    if not isinstance(apt_origin, (str, Airport)):
        raise TypeError('apt_origin must be str or Airport')
    if not isinstance(apt_destination, (str, Airport)):
        raise TypeError('apt_destination must be str or Airport')
    if isinstance(apt_origin, str) or isinstance(apt_destination, str):
        apts = load_apts_from_simcfg(simcfg, iata_keys=iata_keys)
    if isinstance(apt_origin, str):
        apt_origin = apts[apt_origin]
    if isinstance(apt_destination, str):
        apt_destination = apts[apt_destination]
    if simcfg.ltocycle == 'Stettler':
        load_yearly_movements([apt_origin, apt_destination])
        load_tim_stettler([apt_origin, apt_destination])
    if verbose >= 1:
        print(' done')
    
    if verbose >= 1:
        print(f'{"Loading BADA...":33}', end='', flush=True)
    activebada = bada.Bada(load_all_acs=False, acs_to_load=[typecode])
    ac = activebada.acs[typecode]
    if ac.mdl_typecode != typecode:
        activebada.load_ac(ac.mdl_typecode)
    acs = activebada.acs
    if verbose >= 1:
        print(' done')
    
    if verbose >= 1:
        print(f'{"Assigning engines to aircraft...":33}', end='', flush=True)
    engines, apus, _ = allocate_eng(acs, simcfg)
    if verbose >= 1:
        print(' done')
    
    if verbose >= 1:
        print(f'{"Loading meteorological data...":33}', end='', flush=True)
    met = load_met(simcfg)
    if verbose >= 1:
        print(' done')
    
    # Departure LTO
    dep_emissions = lto(ac, apt_origin, departure=True, simcfg=simcfg,
                        apus=apus, t0=0)
    if len(dep_emissions) > 0:
        t = dep_emissions[-1].t_e
    else:
        t = 0
    
    # Non-LTO
    nonlto_emissions = []
    if simcfg.fly_nonlto:
        nonlto_emissions = non_lto(ac, apt_origin, apt_destination, simcfg,
                                   t, met)
        t = nonlto_emissions[-1].t_e
    
    # Arrival LTO
    arr_emissions = lto(ac, apt_destination, departure=False, simcfg=simcfg,
                        apus=apus, t0=t)
    
    emissions = dep_emissions + nonlto_emissions + arr_emissions

    ds = grid.grid_from_simcfg(simcfg)
    if simcfg.grid_method == 'supersampling':
        ds = grid.grid_supersampled(emissions, ds)
    elif simcfg.grid_method == 'nearestneighbor':
        ds = grid.grid_nearest_neighbor(emissions, ds)
    else:
        raise ValueError(f'simcfg.grid_method "{simcfg.grid_method}" '
                         + 'not recognized')
    
    if verbose >= 1:
        print('\n\t-- Total emissions --')
        grid.print_ds_totals(ds)
        print()
    
    if plotfuel:
        if verbose >= 1:
            print(f'{"Plotting fuel burn...":33}', end='')
        da = ds.FUELBURN
        da = da.sel(lat=slice(-89, 89))
        if 'lev' in da.coords:
            da = da.sum('lev')
        
        fig, ax = plot.plot_basemap(grid=False, grid_size=[0.625, 0.5],
                                    fsize=[12, 6])
        vmax = da.max()
        vmin = vmax/1e3
        plot.plot_colormesh(da, scale='log', vmax=vmax, vmin=vmin,
                            cmap=plot.cc.cm.CET_L17, ax=ax)
        ax.set_global()
        if verbose >= 1:
            print(' done')
        
        if figname is not None:
            if verbose >= 1:
                print(f'Saving plot as "{figname}"...', end='')
            if not os.path.isdir(simcfg.outputdir):
                os.mkdir(simcfg.outputdir)
            fig.savefig(simcfg.outputdir + figname, dpi=300,
                        bbox_inches='tight')
            if verbose >= 1:
                print(' done')
    
    return ds, emissions


def test_ac(typecode, simcfg=None, maxroutes=None, load_subflist=False,
            figname=None, plotfuel=True):
    """
    Calculate emissions from one aircraft type in a list of flights

    Parameters
    ----------
    typecode : str
        Aircraft ICAO typecode.
    simcfg : SimConfig or None, optional
        Configuration of simulation options. If None, create a new object with
        default configuration. The default is None.
    maxroutes : int or None, optional
        Limit the number of routes to fly. If None, all the routes in the
        flight list are run. The default is None.
    load_subflist : bool, optional
        If True, load a (subset) flight list from "subflist.csv.bz2". If False,
        a larger flight list is loaded. The default is True.
    figname : str or None, optional
        File name to save a plot of calculated fuel burn. If None, the figure
        is not saved. The default is None.
    plotfuel : bool, optional
        If True, plot calculated fuel burn. The default is True.

    Returns
    -------
    ds : xr.Dataset
        Gridded emissions calculated.

    """
    if simcfg is None:
        simcfg = SimConfig(debug_flist=True,
                           ignore_missing_engs=False)
    
    if maxroutes is None:
        print(f'Testing aircraft "{typecode}"')
    else:
        print(f'Testing aircraft "{typecode}" for up to {maxroutes} routes')
    
    # Load payload mass fraction from weight load factor if needed
    determine_payload_m_frac(simcfg)
    
    print(f'{"Loading airport data...":33}', end='', flush=True)
    apts = load_apts_from_simcfg(simcfg)
    print(' done')
    
    print(f'{"Loading BADA...":33}', end='', flush=True)
    activebada = bada.Bada(load_all_acs=False, upsample_ptf=1)
    activebada.load_ac(typecode, upsample_ptf=1)
    print(' done')
    
    print(f'{"Assigning engines to aircraft...":33}', end='', flush=True)
    ac = activebada.acs[typecode]
    engines, apus, _ = allocate_eng(ac, simcfg)
    print(' done')
    
    print(f'{"Loading meteorological data...":33}', end='', flush=True)
    met = load_met(simcfg)
    print(' done')
    
    print(f'{"Reading flight list...":33}', end='', flush=True)
    if load_subflist:
        subflist = pd.read_csv(DIR_FLIGHTLISTS+'subflist.csv.bz2',
                               index_col=['typecode', 'origin_aeic',
                                          'destination_aeic'],
                               dtype={'nFlights': int},
                               squeeze=True)
    else:
        flist = pd.read_csv(
            DIR_FLIGHTLISTS
            + 'flightlist_20190901_20190930_v0-7_processed.csv.bz2',
            usecols=['callsign', 'typecode', 'origin', 'destination',
                      'firstseen', 'origin_aeic', 'destination_aeic']
        )
        flist = flist.dropna(subset=['typecode',
                                     'origin_aeic',
                                     'destination_aeic'])
        flist = flist[flist.origin_aeic != flist.destination_aeic]
        subflist = flist[flist.typecode==typecode]
        subflist = subflist.groupby(
            ['typecode', 'origin_aeic', 'destination_aeic']
        )
        subflist = subflist.size().rename('nflights')
        subflist.to_csv(DIR_FLIGHTLISTS+'subflist.csv.bz2')
    subflist = subflist[:maxroutes]
    print(' done')
    
    print(f'Flying {len(subflist)} routes')
    time_s = time.time()
    ds = fly_list_ongrid(subflist, apts, activebada, simcfg, apus, met)
    time_spent = time.time() - time_s
    print(f'Time to run flights = {time_spent:.0f} s')
    print(f'{time_spent*1000/len(subflist):.0f} ms/flight\n')
    
    print(ds.sum())
    
    if plotfuel:
        print(f'{"Plotting fuel burn...":33}', end='')
        da = ds.FUELBURN
        da = da.sel(lat=slice(-89, 89))
        fig, ax = plot.plot_basemap(grid=False, grid_size=[0.625, 0.5],
                                    fsize=[12, 6])
        vmax = da.max()
        vmin = vmax/1e3
        plot.plot_colormesh(da, scale='log', vmax=vmax, vmin=vmin,
                            cmap=plot.cc.cm.CET_L17, ax=ax)
        ax.set_global()
        print(' done')
        
        if figname is not None:
            print(f'Saving plot as "{figname}"...', end='')
            if not os.path.isdir(simcfg.outputdir):
                os.mkdir(simcfg.outputdir)
            fig.savefig(simcfg.outputdir + figname, dpi=300,
                        bbox_inches='tight')
            print(' done')
    
    return ds


def run_fcounts(fcounts_path, iata_keys=False, simcfg=None, maxroutes=None,
                figname=None, plotfuel=True):
    """
    Calculate emissions from all flights in a list

    Parameters
    ----------
    fcounts_path : str
        Path to CSV file containing the flight counts.
    iata_keys : bool, optional
        Airport codes in the list of flights: IATA if True, ICAO if False.
        The default is False.
    simcfg : SimConfig or None, optional
        Configuration of simulation options. If None, create a new object with
        default configuration. The default is None.
    maxroutes : int or None, optional
        Limit the number of routes to fly. If None, all the routes in the
        flight list are run. The default is None.
    figname : str or None, optional
        File name to save a plot of calculated fuel burn. If None, the figure
        is not saved. The default is None.
    plotfuel : bool, optional
        If True, plot calculated fuel burn. The default is True.

    Raises
    ------
    ValueError
        If unknown or unsupported aircraft are requested, or if any simcfg
        attribute value is not recognized.
    NothingToProcess
        If, after applying filters according to simcfg, there are no flights
        left to fly.
    ModelDataMissing
        If aircraft, engine, APU, or airport data needed is not found.

    Returns
    -------
    subds : dict of xr.Dataset
        Gridded emissions calculated for each fragment ("sub-task") of the job
        (split by aircraft type or by country of departure).
    overall_ds : xr.Dataset
        Gridded emissions, summed across all aircraft types and countries.
    all_fstats : pd.DataFrame
        Returned only if simcfg.return_flight_durations is True.
        Flight duration in seconds, fuel burn in kg, and number of flights for
        each (typecode-origin-destination) triple.

    """
    # Get settings for the run
    if simcfg is None:
        simcfg = SimConfig()
    
    # Start log
    fmt = ('[%(asctime)s] %(levelname)s:%(process)d:%(module)s'
           + ':%(funcName)s:%(message)s')
    # Previous handlers are cleared in case they weren't shutdown properly
    root = logging.getLogger()
    root.handlers.clear()
    if simcfg.logfile is None:
        logging.basicConfig(level=simcfg.loglvl,
                            format=fmt)
    else:
        logging.basicConfig(filename=simcfg.logfile, filemode='w',
                            level=simcfg.loglvl,
                            format=fmt)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(simcfg.loglvl_stream)
        formatter = logging.Formatter('%(levelname)s:%(process)d:%(module)s'
                                      + ':%(funcName)s:%(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.captureWarnings(True)
    logging.info(simcfg.__dict__)
    
    # Load flight counts
    print(f'Loading flight counts from "{fcounts_path}"...', flush=True)
    if not os.path.isfile(fcounts_path):
        fcounts_path = DIR_FLIGHTLISTS + fcounts_path
    fcounts = pd.read_csv(fcounts_path,
                          index_col=['typecode', 'origin', 'destination'],
                          dtype={'nFlights': int},
                          squeeze=True)
    fcounts = fcounts[:maxroutes]
    
    # Apply typecode replacements
    if simcfg.ac_replacements:
        ac_replacements = load_ac_replacements()
        fcounts = fcounts.rename(index=ac_replacements, level='typecode')
    
    # Ignore military typecodes
    if simcfg.ignore_military:
        military = load_ac_military()
        print(f'- Ignoring {len(military)} military aircraft type(s)')
        fcounts = fcounts[
            ~fcounts.index.isin(military, level='typecode')
        ]
    
    # Apply typecode allowlist or blocklist
    if len(simcfg.acs_allow) > 0:
        print(f'- Flights restricted to the {len(simcfg.acs_allow)} aircraft '
              + 'in simcfg.acs_allow', end='')
        if len(simcfg.acs_allow) <= 10:
            print(f' ({", ".join(simcfg.acs_allow)})')
        else:
            print()
        fcounts = fcounts[
            fcounts.index.isin(simcfg.acs_allow, level='typecode')
        ]
    elif len(simcfg.acs_block) > 0:
        dropped = fcounts[fcounts.index.isin(simcfg.acs_block,
                                             level='typecode')]
        ndropped = int(dropped.sum())
        print(f'- Ignoring {ndropped:,.0f} flight(s) from'
              + f' {len(dropped):,.0f} aircraft-route(s) involving '
              + f'{len(simcfg.acs_block)} aircraft in simcfg.acs_block')
        fcounts = fcounts[
            ~fcounts.index.isin(simcfg.acs_block, level='typecode')
        ]
    
    # Remove non-airplanes
    if simcfg.filter_nonairplanes:
        # Check if type codes are in ICAO DOC 8643
        acs_to_fly = pd.Series(fcounts.index.unique(level='typecode'))
        ac_icao_description = load_ac_icao_description()
        unidentified_acs = acs_to_fly[
            ~acs_to_fly.isin(ac_icao_description.index)
        ].values
        if len(unidentified_acs) > 0:
            print('- The following aircraft type code(s) were not recognized:')
            print(unidentified_acs)
            if simcfg.ignore_unknown_actypes:
                missing_fcounts = fcounts[fcounts.index.isin(unidentified_acs,
                                                             level='typecode')]
                ndropped = int(missing_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(missing_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(unidentified_acs)} type code(s)')
                acs_to_fly = acs_to_fly[
                    acs_to_fly.isin(ac_icao_description.index)
                ]
                fcounts = fcounts[fcounts.index.isin(acs_to_fly,
                                                     level='typecode')]
            else:
                raise ValueError('Pre-check failed: unknown aircraft type(s)')
        
        # Check if aircraft are airplanes
        # (Landplane, Amphibious, Seaplane, or Tiltrotor)
        airplanes_icao = ac_icao_description.index[
            (ac_icao_description.str[0].isin(['L', 'A', 'S', 'T']))]
        non_airplanes = (
            acs_to_fly[~acs_to_fly.isin(airplanes_icao)].values
        )
        if len(non_airplanes) > 0:
            print('- The following non-airplanes were found:')
            print(non_airplanes)
            if simcfg.ignore_nonairplanes:
                dropped_fcounts = fcounts[
                    fcounts.index.isin(non_airplanes, level='typecode')
                ]
                ndropped = int(dropped_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(dropped_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(non_airplanes)} non-airplane(s)')
                acs_to_fly = acs_to_fly[acs_to_fly.isin(airplanes_icao)]
                fcounts = fcounts[fcounts.index.isin(acs_to_fly,
                                                     level='typecode')]
            else:
                raise ValueError('Pre-check failed: unsupported aircraft')
    
    # Remove flights with origin == destination
    if simcfg.remove_same_od:
        same_od = fcounts[fcounts.index.get_level_values('origin')
                          == fcounts.index.get_level_values('destination')]
        fcounts = fcounts[fcounts.index.get_level_values('origin')
                          != fcounts.index.get_level_values('destination')]
        if len(same_od) > 0:
            print(f'- Ignoring {same_od.sum():,.0f} flight(s) '
                  + f'involving {len(same_od):,.0f} aircraft-route(s) '
                  + 'with the same origin and destination')
    
    # Load payload mass fraction (from weight load factor if needed)
    determine_payload_m_frac(simcfg)
    
    print('Done loading flight counts\n', flush=True)
    
    if len(fcounts) == 0:
        raise NothingToProcess('Zero flights to run.')
    
    # Load airports
    print('Loading airport data...', flush=True)
    apts = load_apts_from_simcfg(simcfg, iata_keys=iata_keys)
    apts_df = load_apts_from_simcfg(simcfg, iata_keys=iata_keys, as_df=True)
    apts_df = apts_df[~apts_df.index.duplicated(keep='first')]
    if simcfg.precheck_apts:
        apts_to_fly = pd.Series(np.unique(
            np.concatenate([fcounts.index.unique(level='origin'),
                            fcounts.index.unique(level='destination')])
        ))
        missing_apts = apts_to_fly[~apts_to_fly.isin(apts)].values
        if len(missing_apts) > 0:
            print('- Data for the following airport code(s) are missing:')
            print(missing_apts)
            if simcfg.ignore_missing_apts:
                dropped_fcounts = fcounts[
                    (fcounts.index.isin(missing_apts, level='origin'))
                    | (fcounts.index.isin(missing_apts, level='destination'))
                ]
                ndropped = int(dropped_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(dropped_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(missing_apts)} missing airport(s)')
                fcounts = fcounts[
                    (fcounts.index.isin(apts, level='origin'))
                    & (fcounts.index.isin(apts, level='destination'))
                ]
            else:
                msg = 'Pre-check failed: missing airport data'
                raise ModelDataMissing(msg)
    
    # Apply country allowlists or blocklists to origin
    if len(simcfg.apt_cco_allow) > 0:
        print(f'- Flights restricted to the {len(simcfg.apt_cco_allow)}'
              + ' departure country code(s) in simcfg.apt_cco_allow'
              + f' ({list_if_few(simcfg.apt_cco_allow, 10)})')
        o_country = (fcounts
                     .index.get_level_values('origin')
                     .map(apts_df.domestic_iso))
        fcounts = fcounts[o_country.isin(simcfg.apt_cco_allow)]
    elif len(simcfg.apt_cco_block) > 0:
        o_country = (fcounts
                     .index.get_level_values('origin')
                     .map(apts_df.domestic_iso))
        dropped = fcounts[o_country.isin(simcfg.apt_cco_block)]
        ndropped = int(dropped.sum())
        print(f'- Ignoring {ndropped:,.0f} flight(s) from'
              + f' {len(dropped):,.0f} aircraft-route(s) involving '
              + f'{len(simcfg.apt_cco_block)} departure country code(s) in '
              + f'simcfg.acs_block ({list_if_few(simcfg.apt_cco_block, 10)})')
        fcounts = fcounts[~o_country.isin(simcfg.apt_cco_block)]
    
    # Apply country allowlists or blocklists to destination
    if len(simcfg.apt_ccd_allow) > 0:
        print(f'- Flights restricted to the {len(simcfg.apt_ccd_allow)}'
              + ' arrival country code(s) in simcfg.apt_ccd_allow'
              + f' ({list_if_few(simcfg.apt_ccd_allow, 10)})')
        d_country = (fcounts
                     .index.get_level_values('destination')
                     .map(apts_df.domestic_iso))
        fcounts = fcounts[d_country.isin(simcfg.apt_ccd_allow)]
    elif len(simcfg.apt_ccd_block) > 0:
        d_country = (fcounts
                     .index.get_level_values('destination')
                     .map(apts_df.domestic_iso))
        dropped = fcounts[d_country.isin(simcfg.apt_ccd_block)]
        ndropped = int(dropped.sum())
        print(f'- Ignoring {ndropped:,.0f} flights from'
              + f' {len(dropped):,.0f} aircraft-route(s) involving '
              + f'{len(simcfg.apt_ccd_block)} arrival country code(s) in '
              + f'simcfg.acs_block ({list_if_few(simcfg.apt_ccd_block, 10)})')
        fcounts = fcounts[~d_country.isin(simcfg.apt_ccd_block)]
    
    # Restrict to domestic or international flights only
    if simcfg.fly_international == 'no' or simcfg.fly_international == 'only':
        # Flights to/from antarctica are always treated as domestic
        ANTARCTICA_CC = 'AQ'
        o_country = (fcounts
                     .index.get_level_values('origin')
                     .map(apts_df.domestic_iso))
        d_country = (fcounts
                     .index.get_level_values('destination')
                     .map(apts_df.domestic_iso))
        domestic = ((o_country == d_country)
                    | (o_country == ANTARCTICA_CC)
                    | (d_country == ANTARCTICA_CC))
        if simcfg.fly_international == 'no':
            dropped_fcounts = fcounts[~domestic]
            ndropped = int(dropped_fcounts.sum())
            print(f'- Ignoring {ndropped:,.0f} international flight(s) from'
                  + f' {len(dropped_fcounts):,.0f} aircraft-route(s)')
            fcounts = fcounts[domestic]
        else:
            dropped_fcounts = fcounts[domestic]
            ndropped = int(dropped_fcounts.sum())
            print(f'- Ignoring {ndropped:,.0f} domestic flight(s) from'
                  + f' {len(dropped_fcounts):,.0f} aircraft-route(s)')
            fcounts = fcounts[~domestic]
    
    print('Done loading airport data\n', flush=True)
    
    if len(fcounts) == 0:
        raise NothingToProcess('Zero flights to run.')
    
    # Load aircraft data
    print('Loading aircraft data...', flush=True)
    acs_to_fly = pd.Series(fcounts.index.unique(level='typecode'))
    if simcfg.precheck_acs:
        activebada = bada.Bada(load_all_acs=False)
        unsupported_acs = (
            acs_to_fly[~acs_to_fly.isin(activebada.synonym.index)].values)
        acs_to_fly = acs_to_fly[~acs_to_fly.isin(unsupported_acs)]
        if len(unsupported_acs) > 0:
            print('- The following aircraft was/were not present in BADA:')
            print(unsupported_acs)
            if simcfg.ignore_unsupported_acs:
                dropped_fcounts = fcounts[fcounts.index.isin(unsupported_acs,
                                                             level='typecode')]
                ndropped = int(dropped_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(dropped_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(unsupported_acs)} missing aircraft '
                      + 'type(s)')
                fcounts = fcounts[fcounts.index.isin(acs_to_fly,
                                                     level='typecode')]
            else:
                msg = 'Pre-check failed: missing aircraft data'
                raise ModelDataMissing(msg)
        ac_models_used = (
            activebada.synonym.loc[acs_to_fly]['modeled_as'].values)
        acs_to_load = np.unique(np.concatenate((acs_to_fly, ac_models_used)))
        activebada.load_acs(acs_to_load,
                            load_custom_h_cr=simcfg.cruise_fl=='openavem')
    else:
        activebada = bada.Bada(load_all_acs=False, acs_to_load=acs_to_fly,
                               load_custom_h_cr=simcfg.cruise_fl=='openavem')
    acs = activebada.acs
    print('Done loading BADA\n', flush=True)
    
    # Assign engines to aircraft
    print('Assigning engines to aircraft...', flush=True)
    engines, apus, _ = allocate_eng(acs, simcfg)
    # Check for missing engines and APU
    if simcfg.precheck_engs:
        haseng = pd.Series([hasattr(acs[ac], 'eng') for ac in acs],
                           index=acs.keys())
        haseng = haseng[haseng].index
        missing_eng = acs_to_fly[~acs_to_fly.isin(haseng)].values
        if len(missing_eng) > 0:
            print('- The following aircraft are missing engine data:')
            print(missing_eng)
            if simcfg.ignore_missing_engs:
                dropped_fcounts = fcounts[fcounts.index.isin(missing_eng,
                                                             level='typecode')]
                ndropped = int(dropped_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(dropped_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(missing_eng)} aircraft type(s)')
                fcounts = fcounts[fcounts.index.isin(haseng, level='typecode')]
            else:
                raise ModelDataMissing('Pre-check failed: missing engine data')
        hasapu = pd.Series(
            [hasattr(acs[ac], 'apu_acrp') for ac in acs], index=acs.keys())
        hasapu = hasapu[hasapu].index
        missing_apu = acs_to_fly[~acs_to_fly.isin(hasapu)].values
        if len(missing_apu) > 0:
            print('- The following aircraft are missing APU data:')
            print(missing_apu)
            if simcfg.ignore_missing_apu:
                dropped_fcounts = fcounts[fcounts.index.isin(missing_apu,
                                                             level='typecode')]
                ndropped = int(dropped_fcounts.sum())
                print(f'- Ignoring {ndropped:,.0f} flight(s) from'
                      + f' {len(dropped_fcounts):,.0f} aircraft-route(s)'
                      + f' involving {len(missing_apu)} aircraft type(s)')
                fcounts = fcounts[fcounts.index.isin(hasapu, level='typecode')]
            else:
                raise ModelDataMissing('Pre-check failed: missing APU data')
    print('Done assigning engines to aircraft\n', flush=True)
    
    if len(fcounts) == 0:
        raise NothingToProcess('Zero flights to run.')
    
    if not simcfg.multiprocessing or simcfg.dryrun:
        print(f'{"Loading meteorological data...":33}', end='', flush=True)
        met = load_met(simcfg)
        print(' done\n', flush=True)
    
    acs_to_fly = pd.Series([i[0] for i in fcounts.index]).unique()
    print(f'Flying {len(fcounts):,.0f} aircraft-route(s)'
          + f' ({fcounts.sum():,.0f} flight(s),'
          + f' {len(acs_to_fly)} aircraft type(s))\n', flush=True)
    
    if ((simcfg.save_emissions or plotfuel)
        and not os.path.isdir(simcfg.outputdir)):
        os.mkdir(simcfg.outputdir)
    
    if simcfg.save_emissions and simcfg.save_splits:
        if simcfg.split_job == 'cc_o':
            splitdir =  os.path.join(simcfg.outputdir, 'by_cc_o')
        elif simcfg.split_job == 'cc_d':
            splitdir =  os.path.join(simcfg.outputdir, 'by_cc_d')
        else:
            splitdir =  os.path.join(simcfg.outputdir, 'by_ac')
        if not os.path.isdir(splitdir):
            os.mkdir(splitdir)
        simcfg.splitdir = splitdir
    
    if simcfg.multiprocessing:
        # Multi-thread
        time_s = time.time()
        
        # Distribute workload to processes
        nthreads = simcfg.nthreads
        if simcfg.split_job == 'cc_o':
            ccs = (fcounts
                   .index.get_level_values('origin')
                   .map(apts_df.domestic_iso))
            n_persub = pd.value_counts(ccs)
        elif simcfg.split_job == 'cc_d':
            ccs = (fcounts
                   .index.get_level_values('destination')
                   .map(apts_df.domestic_iso))
            n_persub = pd.value_counts(ccs)
        elif simcfg.split_job == 'ac':
            n_persub = fcounts.groupby(level='typecode').count()
            ccs = None
        else:
            raise ValueError('simcfg.split_job not recognized'
                             + f' ({simcfg.split_job})')
        n_persub = n_persub.sort_values(ascending=False)
        n_by_thread = np.zeros(nthreads)
        tasks = [[] for _ in range(nthreads)]
        i_process = 0
        for sub, n in n_persub.items():
            tasks[i_process].append(sub)
            n_by_thread[i_process] += n
            i_process = n_by_thread.argmin()
        # Trim nthreads if there are workers with no task
        threads_used = (n_by_thread == 0).argmax()
        if threads_used > 0:
            nthreads = threads_used
            tasks = tasks[:threads_used]
            n_by_thread = n_by_thread[:threads_used]
        print(f'Routes per process: {n_by_thread}\n', flush=True)
        
        if simcfg.dryrun:
            print('Stopping here since simcfg.dry_run is True', flush=True)
            return None, None
        
        with multiprocessing.Pool(nthreads) as pool:
            try:
                if simcfg.split_job in ['cc_o', 'cc_d']:
                    result_objects = [
                        pool.apply_async(process_subs,
                                         (fcounts[ccs.isin(task)],
                                          apts,
                                          activebada,
                                          simcfg,
                                          apus,
                                          ccs[ccs.isin(task)]),
                                         error_callback=_worker_error_callback)
                        for task in tasks
                    ]
                else:
                    result_objects = [
                        pool.apply_async(process_subs,
                                         (fcounts.loc[task],
                                          apts,
                                          activebada,
                                          simcfg,
                                          apus),
                                         error_callback=_worker_error_callback)
                        for task in tasks
                    ]
                # Do work
                results = []
                for r_obj in result_objects:
                    try:
                        results.append(r_obj.get())
                    except Exception as err:
                        logging.error('A worker has failed!')
                        logging.error(err, exc_info=True)
                        if not simcfg.continue_on_error:
                            print('Trying to terminate all processes...',
                                  flush=True)
                            pool.terminate()
                            raise
            except:
                pool.close()
                pool.join()
                # End log and merge logs from child processes
                logging.shutdown()
                with open(simcfg.logfile, 'a') as mainlog:
                    # Try to merge logs of workers that returned
                    for r in results:
                        if r['log_fname'] is not None:
                            try:
                                with open(r['log_fname'], 'r') as childlog:
                                    mainlog.write(childlog.read())
                                os.remove(r['log_fname'])
                            except FileNotFoundError:
                                pass
                    # Try to merge logs of workers that were terminated
                    for i in range(1, 1+nthreads):
                        log_fname = (os.path.splitext(simcfg.logfile)[0]
                                     + f'_{i}.log')
                        try:
                            with open(log_fname, 'r') as childlog:
                                mainlog.write(childlog.read())
                            os.remove(log_fname)
                        except FileNotFoundError:
                            pass
                raise
        
        subds = {}
        overall_ds = grid.grid_from_simcfg(simcfg)
        # Keep ds.attrs to re set them later
        attrs = overall_ds.attrs
        all_fstats = []
        for r in results:
            if simcfg.return_splits:
                subds.update(r['ds'])
                for sub in r['ds']:
                    overall_ds += r['ds'][sub]
            else:
                overall_ds += r['ds']
            if simcfg.return_flight_durations:
                all_fstats.extend(r['fstats'])
        overall_ds.attrs = attrs
        all_fstats = pd.DataFrame(data={
            'typecode': [s['typecode'] for s in all_fstats],
            'origin': [s['origin'] for s in all_fstats],
            'destination': [s['destination'] for s in all_fstats],
            'duration': [s['duration'] for s in all_fstats],
            'fuel': [s['fuel'] for s in all_fstats],
            'n': [s['n'] for s in all_fstats]})
        
        if not np.all([r['success'] for r in results]):
            print('\nErrors seem to have occured along the way! The log file '
                  + f'({simcfg.logfile}) might have more information.\n')
        
        time_spent = time.time() - time_s
        print('---------------------')
        print(f'Total time = {time_spent:6,.0f} s')
        print('---------------------')
        
        # End log and merge logs from child processes
        logging.shutdown()
        with open(simcfg.logfile, 'a') as mainlog:
            for r in results:
                if r['log_fname'] is not None:
                    try:
                        with open(r['log_fname'], 'r') as childlog:
                            mainlog.write(childlog.read())
                        os.remove(r['log_fname'])
                    except FileNotFoundError:
                        pass
        
    else:
        # Single-thread
        time_s = time.time()
        subds = {}
        overall_ds = grid.grid_from_simcfg(simcfg)
        all_fstats = []
        success = True
        
        if simcfg.split_job == 'cc_o':
            # Split work by origin country code
            o_country = (fcounts
                         .index.get_level_values('origin')
                         .map(apts_df.domestic_iso))
            subs = pd.unique(o_country)
        elif simcfg.split_job == 'cc_d':
            # Split work by origin country code
            d_country = (fcounts
                         .index.get_level_values('destination')
                         .map(apts_df.domestic_iso))
            subs = pd.unique(d_country)
        elif simcfg.split_job == 'ac':
            # Split work by aircraft type
            subs = acs_to_fly
        else:
            raise ValueError('simcfg.split_job not recognized'
                             + f' ({simcfg.split_job})')
        
        if simcfg.dryrun:
            print('Stopping here since simcfg.dry_run is True', flush=True)
            return None, None
        
        for sub in subs:
            try:
                if simcfg.split_job == 'cc_o':
                    sub_flist = fcounts[o_country==sub]
                    savename = simcfg.output_name + '_' + sub.lower()
                    print(f'Flying from {sub}: {len(sub_flist):,.0f} routes',
                          flush=True)
                elif simcfg.split_job == 'cc_d':
                    sub_flist = fcounts[d_country==sub]
                    savename = simcfg.output_name + '_' + sub.lower()
                    print(f'Flying to {sub}: {len(sub_flist):,.0f} routes',
                          flush=True)
                else:
                    sub_flist = fcounts.xs(sub, level='typecode',
                                           drop_level=False)
                    savename = simcfg.output_name + '_' + sub
                    print(f'Flying {sub}: {len(sub_flist):,.0f} routes',
                          flush=True)
                
                time_sub_s = time.time()
                if simcfg.return_flight_durations:
                    ds, fstats = fly_list_ongrid(sub_flist, apts, activebada,
                                                 simcfg, apus, met)
                    all_fstats.extend(fstats)
                else:
                    ds = fly_list_ongrid(sub_flist, apts, activebada,
                                         simcfg, apus, met)
                time_sub_spent = time.time() - time_sub_s
                print(f'Time to run flights = {time_sub_spent:.0f} s,',
                      end='\t')
                print(f'{time_sub_spent*1000/len(sub_flist):.0f} ms/flight')
            except Exception as err:
                success = False
                logging.error(f'{sub} failed!')
                logging.error(err, exc_info=True)
                print(f'{sub} failed!')
                if not simcfg.continue_on_error:
                    raise
            else:
                if simcfg.save_emissions and simcfg.save_splits:
                    fpath = os.path.join(simcfg.splitdir, savename)
                    grid.save_nc4(ds, fpath)
                if simcfg.return_splits:
                    subds[sub] = ds
                # Keep ds attributes
                attrs = overall_ds.attrs
                overall_ds += ds
                overall_ds.attrs = attrs
        
        all_fstats = pd.DataFrame(data={
            'typecode': [s['typecode'] for s in all_fstats],
            'origin': [s['origin'] for s in all_fstats],
            'destination': [s['destination'] for s in all_fstats],
            'duration': [s['duration'] for s in all_fstats],
            'fuel': [s['fuel'] for s in all_fstats],
            'n': [s['n'] for s in all_fstats]})
        
        if not success:
            print('\nErrors seem to have occured along the way! The log file '
                  + f'({simcfg.logfile}) might have more information.\n')
        
        time_spent = time.time() - time_s
        print('---------------------')
        print(f'Total time = {time_spent:6,.0f} s')
        print('---------------------')
        
        logging.shutdown()

    grid.print_ds_totals(overall_ds)
    if simcfg.save_emissions:
        print('\nSaving overall emissions...', end='')
        if simcfg.save_splits and simcfg.merge_saved_splits:
            print(' Concatenating splits... ', end='')
            if len(subds) == 0:
                subds, split_fpaths = grid.load_splits(simcfg)
            overall_ds = grid.merge_splits(overall_ds, subds, simcfg,
                                           verbose=True)
        fpath = os.path.join(simcfg.outputdir, simcfg.output_name) + '.nc4'
        grid.save_nc4(overall_ds, fpath)
        if simcfg.save_splits and simcfg.merge_saved_splits:
            for fpath in split_fpaths:
                try:
                    os.remove(fpath)
                except PermissionError as err:
                    msg = 'Failed to delete split emissions. ' + err.__str__()
                    warnings.warn(msg)
            try:
                os.rmdir(simcfg.splitdir)
            except FileNotFoundError:
                pass
        print('done')
    
    if plotfuel:
        print(f'\n{"Plotting fuel burn...":33}', end='')
        da = overall_ds.FUELBURN
        if simcfg.split_job in da.dims:
            da = da.sel({simcfg.split_job: '*'})
        da = da.sel(lat=slice(-89, 89))
        if 'lev' in da.dims:
            da = da.sum('lev')
        vmax = da.max()
        vmin = vmax/1e3
        if vmin <= 0:
            print(" Couldn't do it: no fuel burn!")
        else:
            fig, ax = plot.plot_basemap(grid=False, grid_size=[0.625, 0.5],
                                        fsize=[12, 6])
            plot.plot_colormesh(da, scale='log', vmax=vmax, vmin=vmin,
                                cmap=plot.cc.cm.CET_L17, ax=ax)
            ax.set_global()
            print(' done')
            
            if figname is not None:
                print(f'Saving plot as "{figname}"...', end='')
                if not os.path.isdir(simcfg.outputdir):
                    os.mkdir(simcfg.outputdir)
                fig.savefig(simcfg.outputdir + figname, dpi=300,
                            bbox_inches='tight')
                print(' done')
    
    if simcfg.return_flight_durations:
        return subds, overall_ds, all_fstats
    
    return subds, overall_ds


def process_subs(fcounts, apts, activebada, simcfg, apus, ccs=None):
    """
    Process subtasks assigned to a worker in a multiprocess run

    Parameters
    ----------
    fcounts : pd.Series
        Number of times each route is flown, with an index of the form
        (ac, org, des) where
            ac is a str of the aircraft's ICAO typecode
            org is a str of the origin's ICAO or IATA code
            des is a str of the destination's ICAO or IATA code
    apts : dict of str to Airport
        Dictionary of IATA or ICAO codes to Airport objects.
    activebada : bada.Bada
        Bada instance.
    simcfg : SimConfig
        Configuration of simulation options.
    apus : dict of str to APU
        Dictionary of name strings to APU objects.
    ccs : pd.Index or None, optional
        Country codes of origin or destination if simcfg.split_job is 'cc_o' or
        'cc_d', or None if split_job is 'ac'. The default is None.

    Raises
    ------
    err
        Any exception during task execution is caught here and re-raised if
        simcfg.continue_on_error is False.

    Returns
    -------
    result : dict
        Results from this worker, where
            ds is a xr.Dataset with gridded emissions
            fstats is a dict with duration and fuel burn per flight
            wnum is the number of this worker
            sucess is a bool that is True if no exceptions were raised
            log_fname is the name of the log file

    """
    worker_number = multiprocessing.current_process().name.split('-')[-1]
    if simcfg.return_splits:
        ds_worker = {}
    else:
        ds_worker = grid.grid_from_simcfg(simcfg)
    all_fstats = []
    tot_nroutes = len(fcounts)
    
    # Flag to signal errors
    success = True
    
    if tot_nroutes == 0:
        result = dict(ds=ds_worker,
                      fstats=all_fstats,
                      wnum=worker_number,
                      success=success,
                      log_fname=None)
        return result
    
    fmt = ('[%(asctime)s] %(levelname)s:%(processName)s:%(module)s'
           + ':%(funcName)s:%(message)s')
    if simcfg.logfile is None:
        logging.basicConfig(level=simcfg.loglvl,
                            format=fmt, force=True)
    else:
        log_fname = (os.path.splitext(simcfg.logfile)[0]
                     + f'_{worker_number}.log')
        logging.basicConfig(filename=log_fname, filemode='w',
                            level=simcfg.loglvl,
                            format=fmt, force=True)
        root = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(simcfg.loglvl_stream)
        formatter = logging.Formatter('%(levelname)s:%(processName)s:'
                                      + '%(module)s:%(funcName)s:%(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.info(f'pid #{os.getpid()} has started working on'
                + f' {tot_nroutes} routes')
    
    time_s_all = time.time()
    print(f'[{worker_number}] Loading meteorological data...', flush=True)
    met = load_met(simcfg)
    print(f'[{worker_number}] Meteorological data loaded', flush=True)
    if simcfg.split_job in ['cc_o', 'cc_d']:
        subs = ccs.unique()
    else:
        subs = fcounts.index.get_level_values('typecode').unique()
    for sub in subs:
        try:
            if simcfg.split_job == 'cc_o':
                sub_flist = fcounts[ccs==sub]
                savename = simcfg.output_name + '_' + sub.lower()
                prep = 'from '
            elif simcfg.split_job == 'cc_d':
                sub_flist = fcounts[ccs==sub]
                savename = simcfg.output_name + '_' + sub.lower()
                prep = 'to '
            else:
                sub_flist = fcounts.xs(sub, level='typecode', drop_level=False)
                savename = simcfg.output_name + '_' + sub
                prep = ''
            nroutes = len(sub_flist)
            print(f'[{worker_number}] Flying {prep}{sub}: {nroutes:,.0f}'
                  + ' routes', flush=True)
            time_s = time.time()
            if simcfg.return_flight_durations:
                subds, fstats = fly_list_ongrid(sub_flist, apts,
                                                activebada, simcfg, apus, met)
                all_fstats.extend(fstats)
            else:
                subds = fly_list_ongrid(sub_flist, apts,
                                        activebada, simcfg, apus, met)
            if simcfg.save_emissions and simcfg.save_splits:
                fpath = os.path.join(simcfg.splitdir, savename)
                grid.save_nc4(subds, fpath)
            if simcfg.return_splits:
                ds_worker[sub] = subds
            else:
                ds_worker += subds
            time_spent = time.time() - time_s
            print(f'[{worker_number}] Time to run {sub:4} flights = '
                  + f'{time_spent:4.0f} s,\t'
                  + f'{time_spent*1000/nroutes:.0f} ms/flight', flush=True)
        except Exception as err:
            success = False
            logger.error(f'{sub} failed!')
            logger.error(err, exc_info=True)
            print(f'[{worker_number}] {sub} failed!', flush=True)
            if not simcfg.continue_on_error:
                raise
    time_spent_all = time.time() - time_s_all
    
    result = dict(ds=ds_worker,
                  fstats=all_fstats,
                  wnum=worker_number,
                  success=success,
                  log_fname=log_fname)
    
    print(f'Worker [{worker_number}] finished {tot_nroutes:7,.0f} '
              + f'routes, {time_spent_all:5,.0f} s, '
              + f'{time_spent_all*1000/tot_nroutes:3.0f} ms/flight',
              flush=True)
    
    return result


def _worker_error_callback(e):
    # [TODO] manage to stop all workers properly
    pass
        
