"""
Core
"""

##
# Imports
##

# Import system modules
import logging
from enum import Enum

# Import additional modules
import numpy as np

# Import modules from the project
from . import physics


###
# Contants
##

FILEPATH_WIND = './met/wind_dummy.nc4'

DIR_FLIGHTLISTS = r'./flightlists/'
DIR_OUTPUT = r'./output/'

# ICAO LTO cycle engine thrust setting [fraction of full rated thrust]
# values are for idle, approach, climb-out, takeoff
LTO_ICAO_THRUST = np.array([0.07, 0.30, 0.85, 1.0])
LTO_ICAO_IDLE = 0
LTO_ICAO_APP = 1
LTO_ICAO_CO = 2
LTO_ICAO_TO = 3

# Correction factors for EEDB fuel flow to account for installation effects
# of air bleed
BFFM2_FUEL_CORRECTION = np.array([1.100, 1.020, 1.013, 1.010])

###
# Parameters and global variables
##

rng = np.random.default_rng()


###
# Classes
##

class OpenAVEMError(Exception):
    pass


class NothingToProcess(OpenAVEMError):
    pass


class ModelDataMissing(OpenAVEMError):
    pass


class SimConfig:
    """
    Configuration of a model run. See description on docs/configuration.md
    """
    DEFAULT_OPTIONS = {
        'save_cfg_in_nc4':          True,
        'save_emissions':           True,
        'filter_nonairplanes':      True,
        'outputdir':                DIR_OUTPUT,
        'splitdir':                 DIR_OUTPUT,
        'output_name':              'run_output',
        'logfile':                  './logs/log.log',
        'loglvl':                   logging.INFO,
        'loglvl_stream':            logging.WARNING,
        'continue_on_error':        True,
        'yyyymm':                   None,
        'dryrun':                   False,
        'precheck_apts':            True,
        'precheck_acs':             True,
        'precheck_engs':            True,
        'ignore_missing_apts':      False,
        'ignore_unknown_actypes':   True,
        'ignore_nonairplanes':      True,
        'ignore_unsupported_acs':   True,
        'ignore_military':          True,
        'ignore_missing_engs':      False,
        'ignore_missing_apu':       False,
        'apu_warn_missing':         False,
        'apt_source':               'all',
        'apt_closed':               False,
        'apt_heliport':             False,
        'remove_same_od':           True,
        'ac_replacements':          True,
        'use_replacement_eng':      True,
        'use_replacement_apu':      True,
        'fly_nonlto':               True,
        'fly_international':        'yes',
        'transform_alt':            True,
        'eng_properties_path':      None,
        'eng_allocations_path':     None,
        'apu_properties_path':      None,
        'apu_tim':                  'ICAOadv',
        'apu_eigas':                ['specific', 'ICAOadv'],
        'apu_eipm':                 ['specific', 'ICAOadv'],
        'apu_pm10_as_nvpm':         False,
        'apu_businessjet':          True,
        'ltocycle':                 'Stettler',
        'reverse_rand_occurence':   False,
        'nvpm_lto':                 ['measured', 'FOA4', 'none'],
        'nvpm_lto_EI':              30,
        'nvpmnum_lto':              ['measured', 'FOA4', 'none'],
        'nvpmnum_lto_EI':           0,
        'nvpm_interp':              'PCHIP',
        'nvpm_cruise':              ['AEDT-FOA4', 'constantEI', 'none'],
        'nvpm_cruise_EI':           30,
        'nvpmnum_cruise_EI':        0,
        'nvpm_cruise_piston':       ['AEDT-FOA4', 'none'],
        'lat_inefficiency':         'FEAT',
        'lat_ineff_cr_max':         2.0,
        'cruise_fl':                'openavem',
        'hstep_cl':                 1000 * physics.FT_TO_METER,
        'hstep_des':                1000 * physics.FT_TO_METER,
        'sstep_cr':                 50 * physics.NM_TO_METER,
        'payload_m_frac':           0.700,
        'wind':                     'none',
        'wind_filepath':            FILEPATH_WIND,
        'grid_vertical':            'GC33',
        'grid_everyflight':         False,
        'grid_method':              'supersampling',
        'debug_flist':              False,
        'debug_m_est':              False,
        'debug_climb_step':         False,
        'debug_cruise_step':        False,
        'debug_descent_step':       False,
        'verify_flightfuel':        0,
        'return_flight_durations':  False,
        'split_job':                'ac',
        'return_splits':            False,
        'save_splits':              True,
        'merge_saved_splits':       True,
        'acs_allow':                [],
        'acs_block':                [],
        'apt_cco_allow':            [],
        'apt_cco_block':            [],
        'apt_ccd_allow':            [],
        'apt_ccd_block':            [],
        'bada_clip_mlo':            True,
        'bada_clip_mhi':            False,
        'multiprocessing':          False,
        'nthreads':                 15
    }
    
    def __init__(self, **kwargs):
        # Initialize default configuration
        for key in SimConfig.DEFAULT_OPTIONS:
            if key in kwargs:
                self.__dict__[key] = kwargs[key]
            else:
                self.__dict__[key] = SimConfig.DEFAULT_OPTIONS[key]
    
    def dumps(self):
        from json import dumps
        return dumps(self.__dict__)


class Phase(Enum):
    """
    Enumerator for flight phases
    
    All non-LTO phases should be < Phase.LTO
    All LTO phases should be >= Phase.LTO
    """
    # Non-LTO
    NONLTO        = 0
    CLIMB         = 1
    CRUISE        = 2
    DESCENT       = 3
    # LTO, generic
    LTO           = 64
    GROUND        = 65
    TAXI          = 66
    APU_GROUND    = 67
    # LTO, departure
    APU_DEP       = 72
    APU_STARTUP   = 73
    APU_GATEOUT   = 74
    APU_MES       = 76
    TAXIOUT       = 80
    TAXIACC_DEP   = 81
    HOLD          = 82
    TAKEOFF       = 83
    INITIAL_CLIMB = 84
    CLIMBOUT      = 85
    # LTO, arrival
    APPROACH      = 96
    LANDING       = 97
    REVERSE       = 98
    TAXIACC_ARR   = 99
    TAXIIN        = 100
    APU_ARR       = 104
    APU_GATEIN    = 105
    
    def islto(self):
        if isinstance(self, Phase):
            k = self.value
        else:
            k = int(k)
        return k >= Phase.LTO.value
    
    def isarrival(self):
        if isinstance(self, Phase):
            k = self.value
        else:
            k = int(k)
        return k >= Phase.APPROACH.value
    
    def isdeparture(self):
        if isinstance(self, Phase):
            k = self.value
        else:
            k = int(k)
        return k >= Phase.APU_DEP.value and k < Phase.APPROACH.value
    
    def isapu(self):
        if isinstance(self, Phase):
            k = self.value
        else:
            k = int(k)
        return (k >= Phase.APU_DEP.value and k <= Phase.APU_MES.values
                or k >= Phase.APU_ARR.value)


class EmissionSegment:
    """(lat, lon, h, t) segment containing mass of emissions"""
    def __init__(self, lat_s, lon_s, h_s, lat_e, lon_e, h_e):
        self.lat_s = lat_s
        self.lon_s = lon_s
        self.h_s   = h_s
        
        self.lat_e = lat_e
        self.lon_e = lon_e
        self.h_e   = h_e
        
    def __len__(self):
        """3-D segment length in meters, rounded down"""
        _, dist = physics.distance(self.lat_s, self.lon_s,
                                   self.lat_e, self.lat_e)
        
        return int(np.sqrt(dist**2 + (self.h_e - self.h_s)**2))
    
    def start_coord(self):
        """Return coordinates of starting point"""
        return self.lat_s, self.lon_s

    def end_coord(self):
        """Return coordinates of end point"""
        return self.lat_e, self.lon_e
    
    def reverse(self, inplace=False):
        """Swap ends of segment, deleting azimuth"""
        if inplace:
            temp = self.lat_e
            self.lat_e = self.lat_s
            self.lat_s = temp
            
            temp = self.lon_e
            self.lon_e = self.lon_s
            self.lon_s = temp
            
            temp = self.h_e
            self.h_e = self.h_s
            self.h_s = temp
            
            if hasattr(self, 't_e'):
                temp = self.t_e
                if hasattr(self, 't_s'):
                    self.t_e = self.t_s
                else:
                    del self.t_e
                self.t_s = temp
            elif hasattr(self, 't_s'):
                self.t_e = self.t_s
            
            if hasattr(self, 'azi_s'):
                del self.azi_s
            if hasattr(self, 'azi_e'):
                del self.azi_e
        
            return self
        
        else:
            newseg = EmissionSegment(self.lat_e, self.lon_e, self.h_e,
                                     self.lat_s, self.lon_s, self.h_s)
            if hasattr(self, 't_e'):
                newseg.t_s = self.t_e
            if hasattr(self, 't_s'):
                newseg.t_e = self.t_s
            
            return newseg
        

class Engine():
    """Aircraft main engine"""
    def __init__(self, uid=None, properties={}, estimate_missing_sn=True):
        """
        Parameters
        ----------
        uid : str or None, optional
            Unique identifier to assign to engine. This overides a value passed
            in the properties argument if present. If None, no UID is assigned.
            The default is None.
        properties : dict, optional
            Properties to assign to engine. Fuel flow rates in kg/s, EIs in
            g/kg, thrust in kN. The default is {}.
        estimate_missing_sn : bool, optional
            If True, missing mode specific smoke number values are estimated
            from the maximum SN value according to the procedure on ICAO's
            Airport Air Quality Manual. The default is True.

        Returns
        -------
        None.
        
        References
        ----------
        International Civil Aviation Organization 2020 Airport Air Quality
            Manual (International Civil Aviation Organization)

        """
        if len(properties) == 0:
            if uid is not None:
                self.uid = str(uid)
            return
        
        default_nan = [
            'nox_idle', 'nox_app', 'nox_co', 'nox_to',
            'co_idle', 'co_app', 'co_co', 'co_to',
            'hc_idle', 'hc_app', 'hc_co', 'hc_to',
            'sn_idle', 'sn_app', 'sn_co', 'sn_to', 'sn_max',
            'fuel_idle', 'fuel_app', 'fuel_co', 'fuel_to',
            'nvpm_idle', 'nvpm_app', 'nvpm_co', 'nvpm_to',
            'nvpmnum_idle', 'nvpmnum_app', 'nvpmnum_co', 'nvpmnum_to'
        ]
        fields = ['uid', 'manufacturer', 'name', 'combustor_description',
                  'type', 'bp_ratio', 'pr', 'rated_thrust',
                  'wikipedia_link'] + default_nan
        
        for f in default_nan:
            self.__dict__[f] = np.nan
            
        for field in fields:
            if field in properties:
                self.__dict__[field] = properties[field]
        
        if uid is not None:
                self.uid = str(uid)
        
        self.nox_lto = np.array([self.nox_idle, self.nox_app,
                                 self.nox_co, self.nox_to])
        
        self.co_lto = np.array([self.co_idle, self.co_app,
                                self.co_co, self.co_to])
        
        self.hc_lto = np.array([self.hc_idle, self.hc_app,
                                self.hc_co, self.hc_to])
        
        self.sn_lto = np.array([self.sn_idle, self.sn_app,
                                self.sn_co, self.sn_to])
        if np.isnan(self.sn_max):
            self.sn_max = self.sn_lto.max()
        if (np.isnan(self.sn_lto).any() and estimate_missing_sn
            and not np.isnan(self.sn_max)):
            if self.manufacturer == 'Aviadvigatel':
                sf = np.array([0.3, 0.8, 1.0, 1.0])
            elif self.name.startswith('CF34'):
                sf = np.array([0.3, 0.3, 0.4, 1.0])
            elif self.manufacturer == 'Textron Lycoming':
                sf = np.array([0.3, 0.6, 1.0, 1.0])
            elif (self.manufacturer == 'CFM International'
                  and self.combustor_description in ['DAC', 'DAC I', 'DAC II',
                                                     'DAC-II']):
                sf = np.array([1.0, 0.3, 0.3, 0.3])
            else:
                sf = np.array([0.3, 0.3, 0.9, 1.0])
            self.sn_lto = np.nan_to_num(self.sn_lto, nan=self.sn_max*sf)
            self.sn_idle = self.sn_lto[0]
            self.sn_app = self.sn_lto[1]
            self.sn_co = self.sn_lto[2]
            self.sn_to = self.sn_lto[3]
        
        self.fuel_lto = np.array([self.fuel_idle, self.fuel_app,
                                  self.fuel_co, self.fuel_to])
        
        self.nvpm_lto = np.array([self.nvpm_idle, self.nvpm_app,
                                    self.nvpm_co, self.nvpm_to])
        
        self.nvpmnum_lto = np.array([self.nvpmnum_idle, self.nvpmnum_app,
                                    self.nvpmnum_co, self.nvpmnum_to])
        
        self.fcorr_lto = self.fuel_lto * BFFM2_FUEL_CORRECTION
        self.fcorr_idle = self.fcorr_lto[LTO_ICAO_IDLE]
        self.fcorr_app  = self.fcorr_lto[LTO_ICAO_APP]
        self.fcorr_co   = self.fcorr_lto[LTO_ICAO_CO]
        self.fcorr_to   = self.fcorr_lto[LTO_ICAO_TO]
        
        self.prepare_bffm2()
        self.prepare_scope11()
        self.prepare_foa4()
        self.prepare_foa3()
        self.prepare_fox()
    
    def prepare_bffm2(self, debug=False):
        """
        Calculate regression coeficients for the Boeing Fuel FLow Method 2

        Parameters
        ----------
        debug : bool, optional
            If True, print eventual issues found in the application of BFFM2.
            The default is False.
        
        Raises
        ------
        Exception
            If fuel flow rate is missing or is <= 0.

        Returns
        -------
        None.

        Notes
        -----
        Workarounds for issues listed in Table 7 of SAGE 1.5 technical manual
            are marked by comments.
        
        References
        ----------
        Baughcum S L, Tritz T G, Henderson S C and Pickett D C 1996 Scheduled
            Civil Aircraft Emission Inventories for 1992: Database Development
            and Analysis (Langley Research Center)
        
        Kim B, Fleming G, Balasubramanian S, Malwitz A, Lee J, Ruggiero J,
            Waitz I, Klima K, Stouffer V, Long D, Kostiuk P, Locke M, Holsclaw
            C, Morales A, McQueen E and Gillete W 2005 System for assessing
            Aviation’s Global Emissions (SAGE), Version 1.5, Technical Manual
            (Federal Aviation Administration)
        
        """
        # Parameters will be stored in a dictionary in eng
        self.bffm2 = {}
        
        if np.isnan(self.fuel_lto).any():
            raise Exception('fuel flow value missing')
        if (self.fuel_lto <= 0).any():
            raise Exception('fuel flow <= 0')
        # Correct fuel flow for installation effects
        wf = self.fuel_lto * BFFM2_FUEL_CORRECTION
        logwf = np.log10(wf)
        self.bffm2['wf'] = wf
        self.bffm2['logwf'] = logwf
        
        # List of issues found
        issues = []
        
        # Minimum value to clip EIs [g/kg]
        MIN_EI = 1e-6
        
        # NOx
        if np.isnan(self.nox_lto).all() or (self.nox_lto == 0).all():
            # Issue 7
            issues.append('(NOx) issue #7: all EIs missing')
            self.bffm2['zero_nox'] = True
        else:
            self.bffm2['zero_nox'] = False
            if (self.nox_lto == 0).any():
                # Issue 6
                issues.append('(NOx) issue #6: some EIs equal to 0')
            lognox = np.log10(np.where(self.nox_lto == 0,
                                       MIN_EI, self.nox_lto))
            self.bffm2['nox_fit'] = np.polyfit(logwf, lognox, 1)
        
        # CO
        if np.isnan(self.co_lto).all() or (self.co_lto == 0).all():
            # Issue 7
            issues.append('(CO) issue #7: all EIs missing')
            self.bffm2['zero_co'] = True
        else:
            self.bffm2['zero_co'] = False
            if (self.co_lto == 0).any():
                # Issue 6
                issues.append('(CO) issue #6: some EIs equal to 0')
            logco = np.log10(np.where(self.co_lto == 0,
                                      MIN_EI, self.co_lto))
            if logwf[LTO_ICAO_APP] == logwf[LTO_ICAO_IDLE]:
                slope = 0
            else:
                slope = (
                    (logco[LTO_ICAO_APP] - logco[LTO_ICAO_IDLE])
                    / (logwf[LTO_ICAO_APP] - logwf[LTO_ICAO_IDLE])
                )
            co_fitl = np.array([slope, logco[LTO_ICAO_IDLE]])
            co_fith = (logco[LTO_ICAO_CO] + logco[LTO_ICAO_TO]) / 2
            
            # Work around some potential issues
            if slope == 0:
                if logco[LTO_ICAO_APP] < logco[LTO_ICAO_CO]:
                    # Consider this like issue 2, and use only the lower Wf
                    # horizontal line
                    issues.append('(CO) issue #2: intersection at EI higher'
                                  + ' than EI_App')
                    co_fith = logco[LTO_ICAO_APP]
                    co_break = logwf[LTO_ICAO_APP]
                else:
                    # Consider this like issue 3
                    issues.append('(CO) issue #3: EI_App > EI_Idle')
                    co_fitl = np.array([0, co_fith])
                    co_break = logwf[LTO_ICAO_IDLE]
            else:
                co_break = (
                    logwf[LTO_ICAO_IDLE]
                    + (co_fith - logco[LTO_ICAO_IDLE]) / slope
                )
                if co_break > logwf[LTO_ICAO_CO]:
                    # Issue 1
                    issues.append('(CO) issue #1: intersection at Wff higher'
                                  + ' than Wff_CO')
                    co_break = logwf[LTO_ICAO_CO]
                elif co_break < logwf[LTO_ICAO_APP] and slope < 0:
                    # Issue 2
                    issues.append('(CO) issue #2: intersection at EI higher'
                                  + ' than EI_App')
                    co_fith = logco[LTO_ICAO_APP]
                    co_break = logwf[LTO_ICAO_APP]
                elif slope >= 0:
                    # Issue 3
                    issues.append('(CO) issue #3: EI_App > EI_Idle')
                    co_fitl = np.array([0, co_fith])
                    co_break = logwf[LTO_ICAO_IDLE]
            self.bffm2['co_fitl'] = co_fitl
            self.bffm2['co_fith'] = co_fith
            self.bffm2['co_break'] = co_break
        
        # HC
        if np.isnan(self.hc_lto).all() or (self.hc_lto == 0).all():
            # Issue 7
            issues.append('(HC) issue #7: all EIs missing')
            self.bffm2['zero_hc'] = True
        else:
            self.bffm2['zero_hc'] = False
            if (self.hc_lto == 0).any():
                # Issue 6
                issues.append('(HC) issue #6: some EIs equal to 0')
            loghc = np.log10(np.where(self.hc_lto == 0,
                                      MIN_EI, self.hc_lto))
            slope = (
                (loghc[LTO_ICAO_APP] - loghc[LTO_ICAO_IDLE])
                / (logwf[LTO_ICAO_APP] - logwf[LTO_ICAO_IDLE])
            )
            hc_fitl = np.array([slope, loghc[LTO_ICAO_IDLE]])
            hc_fith = (loghc[LTO_ICAO_CO] + loghc[LTO_ICAO_TO]) / 2
            
            # Work around some potential issues
            if slope == 0:
                if loghc[LTO_ICAO_APP] < loghc[LTO_ICAO_CO]:
                    # Consider this like issue 2, and use only the lower Wf
                    # horizontal line
                    issues.append('(HC) issue #2: intersection at EI higher'
                                  + ' than EI_App')
                    hc_fith = loghc[LTO_ICAO_APP]
                    hc_break = logwf[LTO_ICAO_APP]
                else:
                    # Consider this like issue 3
                    issues.append('(CO) issue #3: EI_App > EI_Idle')
                    hc_fitl = np.array([0, hc_fith])
                    hc_break = logwf[LTO_ICAO_IDLE]
            else:
                hc_break = (
                    logwf[LTO_ICAO_IDLE]
                    + (hc_fith - loghc[LTO_ICAO_IDLE]) / slope
                )
                if hc_break > logwf[LTO_ICAO_CO]:
                    # Issue 1
                    issues.append('(HC) issue #1: intersection at Wff higher'
                                  + ' than Wff_CO')
                    hc_break = logwf[LTO_ICAO_CO]
                elif hc_break < logwf[LTO_ICAO_APP] and slope < 0:
                    # Issue 2
                    issues.append('(HC) issue #2: intersection at EI higher'
                                  + ' than EI_App')
                    hc_fith = loghc[LTO_ICAO_APP]
                    hc_break = logwf[LTO_ICAO_APP]
                elif slope >= 0:
                    # Issue 3
                    issues.append('(CO) issue #3: EI_App > EI_Idle')
                    hc_fitl = np.array([0, hc_fith])
                    hc_break = logwf[LTO_ICAO_IDLE]
            self.bffm2['hc_fitl'] = hc_fitl
            self.bffm2['hc_fith'] = hc_fith
            self.bffm2['hc_break'] = hc_break

        if debug:
            if hasattr(self, 'uid'):
                name = self.uid
            else:
                name = '[no-uid]'
            for issue in issues:
                print(f'BFFM2 of "{name}": {issue}')

    def prepare_scope11(self):
        """
        Assign EI(nvPM) in mg/kg for LTO points from SN using SCOPE11 method

        Returns
        -------
        None.
        
        References
        ----------
        Wayson, Roger L, Gregg G Fleming, and Ralph Iovinelli. “Methodology to
            Estimate Particulate Matter Emissions from Certified Commercial
            Aircraft Engines.” Journal of the Air & Waste Management
            Association 59, no. 1 (January 2009): 91–100.
            https://doi.org/10.3155/1047-3289.59.1.91.
        
        Agarwal, Akshat, Raymond L. Speth, Thibaud M. Fritz, S. Daniel Jacob,
            Theo Rindlisbacher, Ralph Iovinelli, Bethan Owen, Richard C.
            Miake-Lye, Jayant S. Sabnis, and Steven R. H. Barrett. “SCOPE11
            Method for Estimating Aircraft Black Carbon Mass and Particle
            Number Emissions.” Environmental Science & Technology 53, no. 3
            (February 5, 2019): 1364–73.
            https://doi.org/10.1021/acs.est.8b04060.
        
        """
        # Parameters will be stored in this dictionary
        self.scope11 = {}
        
        # Flag if there are no smoke numbers
        if self.sn_lto.max() <= 0 or np.isnan(self.sn_lto).all():
            self.scope11['zero'] = True
            return
        else:
            self.scope11['zero'] = False
        
        # Constants from the SN-C_BCi regression
        k1 = 648.4   # [μg/m³]
        k2 = 0.0766
        k3 = -1.098
        k4 = -3.064
        
        # Constants from the C_BCi-C_BCe regression
        a1 = 3.219
        a2 = 312.5  # [μg/m³]
        a3 = 42.6   # [μg/m³]
        
        # Volumetric flow rate calculation follows Wayson et al. 2009
        # A hydrogen content of 13.8% is assumed, following Agarwal et al.
        #  2019, leading to:
        #     Q = 0.776*AFR + 0.767 (for non-mixed exhaust)
        #     Q = 0.776*AFR*(1+BP_ratio) + 0.767 (for mixed exhaust)
        afr = np.array([106, 83, 51, 45])
        
        if self.type == 'MTF':
            beta = self.bp_ratio
        else:
            beta = 0
        
        sn = self.sn_lto
        # C_BCi: BC concentration at measuring instrument [μg/m³]
        C_BCi = k1 * np.exp(k2 * sn) / (1 + np.exp(k3 * (sn + k4)))
        # Q: volumetric flow rate [m³/kg-fuel]
        Q = 0.776 * afr * (1 + beta) + 0.767
        # k_slm: system loss correction [-]
        k_slm = np.log((a1 * C_BCi * (1 + beta) + a2)
                       / (C_BCi * (1 + beta) + a3))
        k_slm = k_slm.clip(min=1)
        # C_BCe: BC concentration at engine exit plane [μg/m³]
        C_BCe = C_BCi * k_slm
        # EI_m: mass emission index [mg/kg-fuel]
        EI_m = C_BCe * Q / 1e3
        EI_m = np.where(sn==0, 0, EI_m)
        self.scope11['ei_lto'] = EI_m
        
        # Number of particles
        
        # Note that some constants (rho_a and LCV) have slightly different
        # values than those in physics.py (which uses the values considered by
        # BADA). I kept the constants used by Agarwal et al. for consistency
        # with their regression
        
        # Ambient air density [kg/m³]
        rho_a = 1.2
        # Total temperature [K] and pressure [Pa] at gas turbine inlet
        T_t2 = physics.MSL_T
        p_t2 = physics.MSL_P
        # Polytropic efficiency [-]
        eta_p = 0.9
        # Lower calorific value of fuel [MJ/kg]
        LCV = 43.2
        # Heat capacity at constant pressure of exhaust [kJ/(kg.K)]
        cp_e = 1.250
        
        # Total temperature at combustor inlet [K]
        T_t3 = T_t2 * self.pr**((physics.AIR_GAMMA - 1) / (physics.AIR_GAMMA
                                                         * eta_p))
 
        # Total temperature [K], pressure [Pa], and density [kg/m³]
        # at turbine inlet
        p_t4 = p_t2 * (1 + (self.pr - 1) * LTO_ICAO_THRUST)
        T_t4 = (afr * physics.AIR_CP * T_t3 + LCV) / (cp_e * (1 + afr))
        rho_t4 = p_t4 / (physics.AIR_R * T_t4)
        
        # BC concentration at the combustor exit [μg/m³]
        C_BCc = C_BCe * (1 + beta) * rho_t4 / rho_a
        
        # Constants from the GMD-C_BCc regression
        a = 5.08   # [nm]
        b = 0.185  # [-]
        
        # Geometric mean diameter [nm]
        GMD = a * C_BCc**b
        
        # Effective density of soot [kg/m³]
        rho_soot = 1000
        # Geometric standard deviation [-]
        sigma = 1.8
        # EI_N: number emission index [particles/kg-fuel]
        EI_N = 1e21 * 6 * EI_m / (np.pi * rho_soot * GMD**3
                                  * np.exp(4.5*(np.log(sigma))**2))
        self.scope11['num_ei_lto'] = EI_N
        
        # [TODO] SCOPE11 for cruise not yet implemented
        self.scope11['cruise'] = False
    
    def prepare_foa4(self):
        """
        Assign EI(nvPM) in mg/kg for LTO points from SN using FOA4.0 method

        Returns
        -------
        None.
        
        References
        ----------
        International Civil Aviation Organization 2020 Airport Air Quality
            Manual (International Civil Aviation Organization)
        
        Notes
        -----
        The only difference between SCOPE11 and FOA4.0 is that FOA uses
            standard values for particle GMD instead of calculating engine
            specific values.
        
        """
        # Parameters will be stored in this dictionary
        self.foa4 = {}
        
        # Flag if there are no smoke numbers
        if self.sn_lto.max() <= 0 or np.isnan(self.sn_lto).all():
            self.foa4['zero'] = True
            return
        else:
            self.foa4['zero'] = False
        
        # Constants from the SN-C_BCi regression
        k1 = 648.4   # [μg/m³]
        k2 = 0.0766
        k3 = -1.098
        k4 = -3.064
        
        # Constants from the C_BCi-C_BCe regression
        a1 = 3.219
        a2 = 312.5  # [μg/m³]
        a3 = 42.6   # [μg/m³]
        
        # Volumetric flow rate calculation follows Wayson et al. 2009
        # A hydrogen content of 13.8% is assumed, following Agarwal et al.
        #  2019, leading to:
        #     Q = 0.776*AFR + 0.767 (for non-mixed exhaust)
        #     Q = 0.776*AFR*(1+BP_ratio) + 0.767 (for mixed exhaust)
        afr = np.array([106, 83, 51, 45])
        
        if self.type == 'MTF':
            beta = self.bp_ratio
        else:
            beta = 0
        
        sn = self.sn_lto
        # C_BCi: BC concentration at measuring instrument [μg/m³]
        C_BCi = k1 * np.exp(k2 * sn) / (1 + np.exp(k3 * (sn + k4)))
        # Q: volumetric flow rate [m³/kg-fuel]
        # The Airport AQ Manual has the first constant as 0.777, but the
        #  original source (Wayson et al. 2009) has it 0.776
        Q = 0.776 * afr * (1 + beta) + 0.767
        # k_slm: system loss correction [-]
        k_slm = np.log((a1 * C_BCi * (1 + beta) + a2)
                       / (C_BCi * (1 + beta) + a3))
        k_slm = k_slm.clip(min=1)
        # C_BCe: BC concentration at engine exit plane [μg/m³]
        C_BCe = C_BCi * k_slm
        # EI_m: mass emission index [mg/kg-fuel]
        EI_m = C_BCe * Q / 1e3
        EI_m = np.where(sn==0, 0, EI_m)
        self.foa4['ei_lto'] = EI_m
        
        # Number of particles
        
        # Geometric mean diameter [nm]
        GMD = np.array([20, 20, 40, 40])
        
        # Effective density of soot [kg/m³]
        rho_soot = 1000
        # Geometric standard deviation [-]
        sigma = 1.8
        # EI_N: number emission index [particles/kg-fuel]
        EI_N = 1e21 * 6 * EI_m / (np.pi * rho_soot * GMD**3
                                  * np.exp(4.5*(np.log(sigma))**2))
        self.foa4['num_ei_lto'] = EI_N
        
        # [TODO] FOA4 for cruise not yet implemented
        self.foa4['cruise'] = False
    
    def prepare_foa3(self):
        """
        Assign EI(nvPM) in mg/kg for LTO points from SN using FOA3 method

        Returns
        -------
        None.
        
        References
        ----------
        Wayson, Roger L, Gregg G Fleming, and Ralph Iovinelli. “Methodology to
            Estimate Particulate Matter Emissions from Certified Commercial
            Aircraft Engines.” Journal of the Air & Waste Management
            Association 59, no. 1 (January 2009): 91–100.
            https://doi.org/10.3155/1047-3289.59.1.91.
        
        """
        # Parameters will be stored in this dictionary
        self.foa3 = {}
        
        # Flag if there are no smoke numbers
        if self.sn_lto.max() <= 0 or np.isnan(self.sn_lto).all():
            self.foa3['zero'] = True
            return
        else:
            self.foa3['zero'] = False
        
        # Volumetric flow rate calculation follows Wayson et al. 2009
        afr = np.array([106, 83, 51, 45])
        
        if self.type == 'MTF':
            beta = self.bp_ratio
        else:
            beta = 0
        
        sn = self.sn_lto
        # CI: concentration index [mg/m³]
        CI_lowSN = 0.0694 * sn**1.24
        CI_highSN = 0.0297 * sn**2 - 1.802 * sn + 31.94
        CI = np.where(sn <= 30, CI_lowSN, CI_highSN)
        # Q: volumetric flow rate [m³/kg-fuel]
        Q = 0.776 * afr * (1 + beta) + 0.767
        # EI: mass emission index [mg/kg-fuel]
        EI = CI * Q
        EI = np.where(sn == 0, 0, EI)
        self.foa3['ei_lto'] = EI
    
    def prepare_fox(self):
        """
        Assign EI(nvPM) in mg/kg for LTO points from SN using FOX method

        Returns
        -------
        None.

        References
        ----------
        Stettler M E J, Boies A M, Petzold A and Barrett S R H 2013 Global
        Civil Aviation Black Carbon Emissions Environ. Sci. Technol. 47
        10397–404
        
        """
        # Parameters will be stored in this dictionary
        self.fox = {}
        
        # Flag if the method is not applicable
        if (self.type not in ['TF', 'MTF', 'TP']
            or np.isnan(self.pr)
            or np.isnan(self.fuel_lto).any()):
            self.fox['zero'] = True
            return
        else:
            self.fox['zero'] = False
        
        # maxEI: max limit of EI [mg/kg]
        max_EI = 1000
        # eta_p: polytropic efficiency [-]
        eta_p = 0.9
        # gamma: specific heat ratio of the air [-]
        gamma = physics.AIR_GAMMA
        # A_form: formation constant [mg.s/(kg.m3)]
        A_form = 356
        E_form = -6390
        # A_ox: oxidation constant [mg.s/(kg.m3)]
        A_ox = 608
        E_ox = -19778
            
        # f: fuel mass flow rate [kg/s]
        f = self.fuel_lto
        # mf_rel: percentage of fuel mass flow rate to its maximum value [-]
        mf_rel = f / f.max()
        
        # Temperature [K] and pressure [Pa] at compressor inlet
        #  at static reference conditions
        T2 = physics.MSL_T
        p2 = physics.MSL_P
        
        # p3: absolute pressure at the combustor [Pa]
        p3 = ((self.pr - 1) * mf_rel + 1) * physics.MSL_P
        # T3: temperature at the combustor [K]
        T3 = T2 * (p3/p2)**((gamma - 1)/ (gamma * eta_p))
        # Tflref: flame temperature [K]
        Tfl = 0.9 * T3 + 2120
        
        # AFRref: air to fuel ratio [-]
        # AEIC used +0.0078, but Stettler et al 2013 write 0.008
        AFR = 1 / (0.0121 * mf_rel + 0.008)
        
        # C_BCref: black carbon concentration [mg/m3]
        C_BC = (f * (A_form * np.exp(E_form / Tfl)
                     - A_ox * AFR * np.exp(E_ox / Tfl)))
        C_BC = np.clip(C_BC, 0, max_EI)
        
        # Qref: volumetric flow rate [m3/kg]
        Q = 0.776 * AFR + 0.877
        
        # EI: emission index [mg/kg]
        EI = C_BC * Q
                
        self.fox['p3ref'] = p3
        self.fox['Tflref'] = Tfl
        self.fox['AFRref'] = AFR
        self.fox['C_BCref'] = C_BC
        self.fox['EI_ref'] = EI

        
class APU():
    """Auxiliary Power Unit"""
    
    MODE_CYCLE = 0
    MODE_NL    = 1
    MODE_ECS   = 2
    MODE_MES   = 3
    
    def __init__(self, name, properties={}):
        """
        Parameters
        ----------
        name : str
            Name of the APU.
        properties : TYPE, optional
            Properties to assign to engine. Fuel flow rates in kg/s, EIs in
            g/kg, thrust in kN. The default is {}.

        Returns
        -------
        None.

        """
        default_nan = [
            'fuel_cycle', 'fuel_nl', 'fuel_ecs', 'fuel_mes',
            'nox_cycle', 'nox_nl', 'nox_ecs', 'nox_mes',
            'co_cycle', 'co_nl', 'co_ecs', 'co_mes',
            'hc_cycle', 'hc_nl', 'hc_ecs', 'hc_mes',
            'pm10_cycle', 'pm10_nl', 'pm10_ecs', 'pm10_mes',
            'nvpm_cycle', 'nvpm_nl', 'nvpm_ecs', 'nvpm_mes',
            'nvpmnum_cycle', 'nvpmnum_nl', 'nvpmnum_ecs', 'nvpmnum_mes'
        ]
        fields = ['name'] + default_nan
        
        for f in default_nan:
            self.__dict__[f] = np.nan
        
        for field in fields:
            if field in properties:
                self.__dict__[field] = properties[field]
        
        modes = ['cycle', 'nl', 'ecs', 'mes']
        for spc in ['fuel', 'nox', 'co', 'hc', 'pm10', 'nvpm', 'nvpmnum']:
            self.__dict__[spc] = {k: properties[f'{spc}_{k}'] for k in modes}


class Airport():
    def __init__(self, icao, iata, lat, lon, h, timezone=None, dst=None,
                 cc=None):
        self.icao = icao
        self.iata = iata
        self.lat  = lat
        self.lon  = lon
        self.h    = h
        
        self.timezone = timezone
        self.dst      = dst
        
        self.cc = cc


def list_if_few(arr, nmax=10):
    """
    Return string joining list members if the length is <= nmax

    Parameters
    ----------
    arr : list of str
        List of items to join in one string.
    nmax : int, optional
        If len(arr) is greater than nmax, an empty string is returned.
        The default is 10.

    Returns
    -------
    s : str
        Items of arr joined if len(arr) <= nmax, or an empty string otherwise.

    """
    if len(arr) <= nmax:
        s = ', '.join(arr)
    else:
        s = ''
    
    return s

