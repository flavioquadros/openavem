"""
Implements the BADA aircraft performance model to calculate emissions

References
----------
User Manual for the Base of Aircraft Data (BADA) Revision 3.15,
    EEC Technical/Scientific Report No. 19/03/18-45, EUROCONTROL, May 2019

"""

##
# Imports
##

# Import system modules
import os
import warnings

# Import additional modules
import pandas as pd
import numpy as np

# Import from the project
from . import physics
from .core import EmissionSegment, Phase
from openavem.datasources import load_ac_natsgrp, load_cruise_fl
from . import dir_openavem

##
# Constants
##

LTO_CEILING = 3000 * physics.FT_TO_METER

ENG_OVERALL_EFF = 0.15

ENG_TYPE_SHORTER = {'Jet': 'jet',
                    'Turboprop': 'turbo',
                    'Piston': 'piston'}

# Tolerance when comparing aircraft mass to its acceptable limits [kg]
MASS_TOLERANCE = 0.1

PHASE_BADA_STR = {Phase.CLIMB:          'cl',
                   Phase.CRUISE:        'cr',
                   Phase.DESCENT:       'des',
                   Phase.TAKEOFF:       'to',
                   Phase.INITIAL_CLIMB: 'ic',
                   Phase.APPROACH:      'app',
                   Phase.LANDING:       'lnd',
                   Phase.GROUND:        'gnd',
                   Phase.HOLD:          'hold'}


##
# Parameters and global variables
##

bada_dir = os.path.join(dir_openavem, 'BADA/')


##
# Classes
##

class Bada:
    """Hold all BADA parameters and data"""
    def __init__(self, data_dir=bada_dir, load_all_acs=True, acs_to_load=[],
                 upsample_ptf=1, load_custom_h_cr=True):
        self.data_dir = data_dir
        self.gpf      = read_gpf(data_dir+'BADA.GPF')
        self.synonym  = read_synonym(data_dir+'SYNONYM.NEW')
        
        self.acs = {}
        if load_all_acs:
            self.load_acs(self.synonym.index.to_list(), upsample_ptf,
                          load_custom_h_cr)
        elif len(acs_to_load) > 0:
            ac_models_used = self.synonym.loc[acs_to_load]['modeled_as'].values
            acs_to_load = np.unique(np.concatenate((acs_to_load, 
                                                    ac_models_used)))
            self.load_acs(acs_to_load, upsample_ptf, load_custom_h_cr)
    
    def load_acs(self, acs_to_load, upsample_ptf=1, load_custom_h_cr=True):
        if len(acs_to_load) == 0:
            msg = 'Tried to load aircraft, but an empty list was given'
            warnings.warn(msg)
        else:
            loaded_any_ac = False
            for typecode in acs_to_load:
                if typecode not in self.synonym.index:
                    warnings.warn(f"Aircraft '{typecode}' not found in BADA's "
                                  + "synonym file")
                else:
                    modeled_as = self.synonym.loc[typecode]['modeled_as']
                    try:
                        self.load_ac(typecode, self.data_dir, modeled_as,
                                     upsample_ptf, load_natsgrp=False)
                        loaded_any_ac = True
                    except FileNotFoundError:
                        warnings.warn(
                            f"Couldn't load files for aircraft '{typecode}'")
            if not loaded_any_ac:
                raise Exception(
                    f"Couldn't load any aircraft data at '{self.data_dir}'")
            self.load_natsgrp()
            if load_custom_h_cr:
                self.load_h_cr()
    
    def load_ac(self, typecode, data_dir=None, modeled_as=None, upsample_ptf=0,
                load_natsgrp=True, load_h_cr=True):
        """Load aircraft data and add to Bada.acs[typecode]"""
        if data_dir is None:
            data_dir = self.data_dir
        self.acs[typecode] = Aircraft(typecode, data_dir, modeled_as, self,
                                      upsample_ptf)
        if load_natsgrp:
            self.load_natsgrp(acs=[typecode])
        if load_h_cr:
            self.load_h_cr(acs=[typecode])
    
    def load_natsgrp(self, filepath=None, acs=[]):
        if filepath is None:
            nats = load_ac_natsgrp()
        else:
            nats = load_ac_natsgrp(filepath)
        if len(acs) == 0:
            acs = list(self.acs.keys())
        for ac in acs:
            if ac in nats.index:
                self.acs[ac].natsgrp = nats.loc[ac][0]

    def load_h_cr(self, filepath=None, acs=[]):
        if len(acs) == 0:
            acs = list(self.acs.keys())
            if len(acs) == 0:
                return
        if filepath is None:
            cruise_h = load_cruise_fl() * 100 * physics.FT_TO_METER
        else:
            cruise_h = load_cruise_fl(filepath) * 100 * physics.FT_TO_METER
        for ac in self.acs:
            modeled_as = self.acs[ac].mdl_typecode
            if ac in cruise_h.index:
                self.acs[ac].h_cr = cruise_h.loc[ac].values[0]
            elif modeled_as in cruise_h.index:
                self.acs[ac].h_cr = cruise_h.loc[modeled_as].values[0]
            else:
                self.acs[ac].h_cr = None


class Aircraft:
    """Hold modeled data for an aircraft type"""
    def __init__(self, typecode, data_dir, bada_typecode=None, bada=None,
                 upsample_ptf=0):
        self.typecode = typecode
        self.mdl_typecode = bada_typecode
        self.parentbada = bada
        
        if bada_typecode is None:
            bada_typecode = typecode
        
        filename = (bada_typecode + '_____')[:6]
        self.opf = read_opf(data_dir + filename + '.OPF')
        self.apf = read_apf(data_dir + filename + '.APF')
        self.ptf = read_ptf(data_dir + filename + '.PTF')
        self.ptd = read_ptd(data_dir + filename + '.PTD')
        
        if upsample_ptf > 0:
            fls = (
                np.append(np.arange(self.ptf['table'].index[0],
                              self.ptf['table'].index[-1],
                              upsample_ptf),
                          self.ptf['table'].index[-1])
            )
            self.ptf['table'] = self.ptf['table'].reindex(fls)
            self.ptf['table'] = self.ptf['table'].interpolate()
            del fls
        
        # Separate engine type attribute for convinience
        self.eng_type = ENG_TYPE_SHORTER[self.opf['aircraft']['eng_type']]
        
        # Complement OPF data with minimum speeds
        if self.parentbada is not None:
            self.opf['envelope']['Vmin_cr'] = (
                self.parentbada.gpf['C_v_min', self.eng_type, 'cr']
                * self.opf['aero']['Vstall_cr']
            )
            self.opf['envelope']['Vmin_app'] =  (
                self.parentbada.gpf['C_v_min', self.eng_type, 'app']
                * self.opf['aero']['Vstall_app']
            )


##
# Functions
##
   
def fixed_spacing_floats(s, fmt, nan_value=np.nan):
    """
    Return floats contained in given positions of string s

    Parameters
    ----------
    s : str
        String to parse.
    fmt : list of int
        Alternating gaps and lengths, starting with a gap.
    nan_value : ?, optional
        Value to be returned for failed casts to float. The default is np.nan.

    Raises
    ------
    ValueError
        If len(fmt) < 2.

    Returns
    -------
    values : list of float
        Extracted values.
    
    """
    if len(fmt) < 2:
        raise ValueError('fmt must have length >= 2')
    
    values = []
    pos = 0
    for i in range(int(len(fmt)/2)):
        pos += fmt[i*2]
        length = fmt[i*2+1]
        value = s[pos:pos+length]
        try:
            value = float(value)
        except ValueError:
            value = nan_value
        values.append(value)
        pos += length
    
    return values


def fixed_spacing_strings(s, fmt):
    """
    Return stripped substrings contained in given positions of string s

    Parameters
    ----------
    s : str
        String to parse.
    fmt : list of int
        Alternating gaps and lengths, starting with a gap.

    Raises
    ------
    ValueError
        If len(fmt) < 2.

    Returns
    -------
    values : list of str
        Extracted values.
    
    """
    if len(fmt) < 2:
        raise ValueError('fmt must have length >= 2')
    
    values = []
    pos = 0
    for i in range(int(len(fmt)/2)):
        pos += fmt[i*2]
        length = fmt[i*2+1]
        value = s[pos:pos+length]
        value = value.strip()
        values.append(value)
        pos += length
    
    return values


def read_synonym(filepath=None):
    """
    Read list of supported aircraft from the SYNONYM.NEW file

    Parameters
    ----------
    filepath : str, optional
        If none, path is set to bada_dir + 'SYNONYM.NEW'. The default is None.

    Returns
    -------
    synonym : pd.DataFrame
        Table of synonyms.
    
    """
    if filepath is None:
        filepath = bada_dir + 'SYNONYM.NEW'
    
    SUPPORTED = '-'
    # EQUIVALENT = '*'
    ICAO_TRUE = 'Y'
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        rows = []
        for line in lines[18:]:
            if line[:2] == 'CD':
                # data line
                data = fixed_spacing_strings(line, [3, 1, 1, 4, 3, 18, 1, 25,
                                                    1, 6, 2, 1])
                direct = data[0] == SUPPORTED
                modeled_as = data[4].rstrip('_')
                icao = data[5] == ICAO_TRUE
                rows.append([data[1], direct, data[2], data[3], data[4], icao,
                             modeled_as])
        synonym = pd.DataFrame(data=rows, columns=['typecode',
                                                   'direct',
                                                   'manufacturer',
                                                   'name',
                                                   'file',
                                                   'icao',
                                                   'modeled_as'])
        synonym = synonym.set_index('typecode')
    
    return synonym


def read_gpf(filepath=None):
    """
    Read global aircraft parameters from the Global Parameter File

    Parameters
    ----------
    filepath : str, optional
        If none, path is set to bada_dir + 'BADA.GPF'. The default is None.

    Returns
    -------
    gpf : dict
        Extracted data.
    
    """
    if filepath is None:
        filepath = bada_dir + 'BADA.GPF'
    gpf = {}
    with open(filepath, 'r') as f:
        for line in f:
            linetype = line[:2]
            if linetype == 'FI':
                # end-of-file line
                break
            if linetype == 'CD':
                # data line
                data = line.split()
                parameter = data[1]
                for flight in data[2].split(','):
                    for engine in data[3].split(','):
                        for phase in data[4].split(','):
                            gpf[(parameter, engine, phase)] = float(data[5])
    return gpf


def read_opf(filepath):
    """
    Read an Operations Performance File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    opf : dict
        Extracted data.
    
    """
    opf = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Aircraft block
        opf['aircraft'] = {}
        line = lines[13]
        i = 2+3
        opf['aircraft']['badaname'] = line[i:i+6]
        i += 6+9
        opf['aircraft']['n_engs'] = int(line[i:i+1])
        i += 1+12
        opf['aircraft']['eng_type'] = line[i:i+9].rstrip()
        i += 9+17
        opf['aircraft']['wakecat'] = line[i:i+1]
        
        # Mass block
        opf['mass'] = {}
        line = lines[18].split()
        opf['mass']['ref']  = float(line[1]) * 1000
        opf['mass']['min']  = float(line[2]) * 1000
        opf['mass']['max']  = float(line[3]) * 1000
        opf['mass']['pyld'] = float(line[4]) * 1000
        opf['mass']['Gw']   = float(line[5])
        
        # Flight envelope block
        opf['envelope'] = {}
        line = lines[21].split()
        opf['envelope']['VMO']  = float(line[1])
        opf['envelope']['MMO']  = float(line[2])
        opf['envelope']['hMO']  = float(line[3])
        opf['envelope']['hmax'] = float(line[4])
        opf['envelope']['Gt']   = float(line[5])
        
        # Aerodynamics block
        opf['aero'] = {}
        line = lines[25].split()
        opf['aero']['S']         = float(line[2])
        opf['aero']['Clbo(M=0)'] = float(line[3])
        opf['aero']['k']         = float(line[4])
        opf['aero']['CM16']      = float(line[5])
        
        phase_translation = {'CR': 'cr', 'IC': 'ic', 'TO': 'to', 'AP': 'app',
                             'LD': 'lnd'}
        for j in range(28, 33):
            line = lines[j]
            phase = phase_translation[line[5:7]]
            opf['aero'][f'Vstall_{phase}'] = float(line[20:30])
            opf['aero'][f'CD0_{phase}']    = float(line[33:43])
            opf['aero'][f'CD2_{phase}']    = float(line[46:56])
        
        line = lines[38]
        opf['aero']['CD0LDG'] = float(line[33:43])
        
        # Engine thrust block
        opf['thrust'] = {}
        line = lines[44].split()
        opf['thrust']['CTC1'] = float(line[1])
        opf['thrust']['CTC2'] = float(line[2])
        opf['thrust']['CTC3'] = float(line[3])
        opf['thrust']['CTC4'] = float(line[4])
        opf['thrust']['CTC5'] = float(line[5])
        
        line = lines[46].split()
        opf['thrust']['CTdeslow']  = float(line[1])
        opf['thrust']['CTdeshigh'] = float(line[2])
        opf['thrust']['Hpdes']     = float(line[3])
        opf['thrust']['CTdesapp']  = float(line[4])
        opf['thrust']['CTdesld']   = float(line[5])
        
        line = lines[48].split()
        opf['thrust']['Vdesref'] = float(line[1])
        opf['thrust']['Mdesref'] = float(line[2])
        
        # Fuel consumption block
        opf['fuel'] = {}
        line = lines[51].split()
        opf['fuel']['Cf1'] = float(line[1])
        opf['fuel']['Cf2'] = float(line[2])
        
        line = lines[53].split()
        opf['fuel']['Cf3'] = float(line[1])
        opf['fuel']['Cf4'] = float(line[2])
        
        line = lines[55].split()
        opf['fuel']['Cfcr'] = float(line[1])
        
        # Ground movement block
        opf['ground'] = {}
        line = lines[58].split()
        opf['ground']['TOL']    = float(line[1])
        opf['ground']['LDL']    = float(line[2])
        opf['ground']['span']   = float(line[3])
        opf['ground']['length'] = float(line[4])
        
    return opf


def read_apf(filepath):
    """
    Read an Airlines Procedures File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    apf : dict
        Extracted data.

    """
    apf = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # File Identification Block
        line = lines[12].split()
        try:
            apf['LO_min'] = float(line[2]) * 1000
        except ValueError:
            apf['LO_min'] = np.nan
        
        try:
            apf['AV_min'] = float(line[7]) * 1000
        except ValueError:
            apf['AV_min'] = apf['LO_min']
        
        try:
            apf['HI_min'] = float(line[7]) * 1000
        except ValueError:
            apf['HI_min'] = apf['AV_min']
        
        try:
            apf['HI_max'] = float(line[14]) * 1000
        except:
            apf['HI_max'] = np.nan
        
        try:
            apf['AV_max'] = float(line[9]) * 1000
        except ValueError:
            apf['AV_max'] = apf['HI_max']
            
        try:
            apf['LO_max'] = float(line[4]) * 1000
        except ValueError:
            apf['LO_max'] = apf['AV_max']
            
        # Procedures Specification Block
        # Read only default procedures
        for config, j in zip(['default_LO', 'default_AV', 'default_HI'],
                             [20, 21, 22]):
            line = lines[j]
            procedures = {}
            
            procedures['Vcl1'] = float(line[27:30])
            procedures['Vcl2'] = float(line[31:34])
            procedures['Mcl'] = float(line[35:37])
            
            procedures['Vcr1'] = float(line[47:50])
            procedures['Vcr2'] = float(line[51:54])
            procedures['Mcr'] = float(line[55:57])
            
            procedures['Mdes'] = float(line[59:61])
            procedures['Vdes2'] = float(line[62:65])
            procedures['Vdes1'] = float(line[66:69])
            
            apf[config] = procedures
        
    return apf


def read_ptf(filepath):
    """
    Read a Performance Table File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    ptf : dict
        Extracted data.

    """
    ptf = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Header
        line = lines[7]
        ptf['Vcl1'] = float(line[11:14])
        ptf['Vcl2'] = float(line[15:18])
        ptf['Mcl'] = float(line[23:27])
        ptf['mlo'] = float(line[41:])
        
        line = lines[8]
        ptf['Vcr1'] = float(line[11:14])
        ptf['Vcr2'] = float(line[15:18])
        ptf['Mcr'] = float(line[23:27])
        ptf['mnom'] = float(line[41:54])
        ptf['hMO'] = float(line[71:])
        
        line = lines[9]
        ptf['Vdes1'] = float(line[11:14])
        ptf['Vdes2'] = float(line[15:18])
        ptf['Mdes'] = float(line[23:27])
        ptf['mhi'] = float(line[41:])
        
        # Table
        table = []
        irow = 0
        line = lines[16]
        while line[0] != '=':
            values = fixed_spacing_floats(line, [0, 3, 4, 3, 3, 5, 1, 5, 1, 5,
                                                 5, 3, 3, 5, 1, 5, 1, 5, 3, 5,
                                                 5, 3, 2, 5, 2, 5])
            table.append(values)
            irow += 1
            line = lines[16+irow*2]
        ptf['table'] = pd.DataFrame(
            data=table,
            columns=['FL', 'Vcr', 'flo_cr', 'fnom_cr', 'fhi_cr',
                     'Vcl', 'ROCDlo_cl', 'ROCDnom_cl', 'ROCDhi_cl', 'fnom_cl',
                     'Vdes', 'ROCDnom_des', 'fnom_des']
        ).set_index('FL')
    
    return ptf


def read_ptd(filepath):
    """
    Read a Performance Table Data (file) for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    ptd : dict
        Extracted data.

    """
    ptd = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Low mass climb
        iline = 8
        table = []
        line = lines[iline]
        while True:
            values = fixed_spacing_floats(line, [0, 6, 1, 3, 1, 6, 1, 7, 1, 7,
                                                 1, 8, 1, 8, 1, 7, 1, 6, 1, 9,
                                                 1, 9, 1, 7, 1, 7, 1, 7, 1, 8,
                                                 1, 7])
            if np.isnan(values[0]):
                break
            table.append(values)
            iline += 1
            line = lines[iline]
        ptd['cl_lo'] = pd.DataFrame(
            data=table,
            columns=['FL', 'T', 'p', 'rho', 'a', 'TAS', 'CAS', 'M', 'mass',
                     'Thrust', 'Drag', 'Fuel', 'ESF', 'ROC', 'TDC', 'PWC']
        ).set_index('FL')
        
        # Nominal mass climb
        iline += 6
        table = []
        line = lines[iline]
        while True:
            values = fixed_spacing_floats(line, [0, 6, 1, 3, 1, 6, 1, 7, 1, 7,
                                                 1, 8, 1, 8, 1, 7, 1, 6, 1, 9,
                                                 1, 9, 1, 7, 1, 7, 1, 7, 1, 8,
                                                 1, 7])
            if np.isnan(values[0]):
                break
            table.append(values)
            iline += 1
            line = lines[iline]
        ptd['cl_nom'] = pd.DataFrame(
            data=table,
            columns=['FL', 'T', 'p', 'rho', 'a', 'TAS', 'CAS', 'M', 'mass',
                     'Thrust', 'Drag', 'Fuel', 'ESF', 'ROC', 'TDC', 'PWC']
        ).set_index('FL')
        
        # High mass climb
        iline += 6
        table = []
        line = lines[iline]
        while True:
            values = fixed_spacing_floats(line, [0, 6, 1, 3, 1, 6, 1, 7, 1, 7,
                                                 1, 8, 1, 8, 1, 7, 1, 6, 1, 9,
                                                 1, 9, 1, 7, 1, 7, 1, 7, 1, 8,
                                                 1, 7])
            if np.isnan(values[0]):
                break
            table.append(values)
            iline += 1
            line = lines[iline]
        ptd['cl_hi'] = pd.DataFrame(
            data=table,
            columns=['FL', 'T', 'p', 'rho', 'a', 'TAS', 'CAS', 'M', 'mass',
                     'Thrust', 'Drag', 'Fuel', 'ESF', 'ROC', 'TDC', 'PWC']
        ).set_index('FL')
        
        # Nominal mass descent
        iline += 5
        table = []
        line = lines[iline]
        while True:
            values = fixed_spacing_floats(line, [0, 6, 1, 3, 1, 6, 1, 7, 1, 7,
                                                 1, 8, 1, 8, 1, 7, 1, 6, 1, 9,
                                                 1, 9, 1, 7, 1, 7, 1, 7, 1, 8,
                                                 1, 8])
            if np.isnan(values[0]):
                break
            table.append(values)
            iline += 1
            line = lines[iline]
        ptd['des_nom'] = pd.DataFrame(
            data=table,
            columns=['FL', 'T', 'p', 'rho', 'a', 'TAS', 'CAS', 'M', 'mass',
                     'Thrust', 'Drag', 'Fuel', 'ESF', 'ROC', 'TDC', 'gammaTAS']
        ).set_index('FL')
        
    return ptd


def get_min_speed(ac, phase, H, V):
    """
    Return minimum CAS for given flight phase, H_AGL, V_CAS

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    phase : int
        Number representing flight phase according to Phase.
    H : float
        Geopotential altitude above ground level in meters.
    V : float
        Calibrated airspeed in meters per second.

    Raises
    ------
    Exception
        If phase is not recognized or not implemented.

    Returns
    -------
    minCAS : float
        Minimum CAS for given condition.

    """
    gpf = ac.parentbada.gpf
    opf = ac.opf
    
    if phase is Phase.CLIMB:
        if H < gpf['H_max_to', ac.eng_type, 'to']:
            Vstall = opf['aero']['Vstall_to']
        elif H <= gpf['H_max_ic', ac.eng_type, 'ic']:
            Vstall = opf['aero']['Vstall_ic']
        else:
            Vstall = opf['aero']['Vstall_cr']
   
    elif phase == Phase.CRUISE:
        Vstall = opf['aero']['Vstall_cr']
    
    elif phase == Phase.DESCENT:
        if H > gpf['H_max_app', ac.eng_type, 'app']:
            Vstall = opf['aero']['Vstall_cr']
        elif V >= opf['envelope']['Vmin_cr'] + 10:
            Vstall = opf['aero']['Vstall_cr']
        elif H > gpf['H_max_ld', ac.eng_type, 'lnd']:
            Vstall = opf['aero']['Vstall_app']
        elif V >= opf['envelope']['Vmin_app'] + 10:
            Vstall = opf['aero']['Vstall_app']
        else:
            Vstall = opf['aero']['Vstall_lnd']
    
    else:
        raise Exception(f'phase "{phase}" not recognized or not implemented')
    
    if phase == Phase.TAKEOFF:
        C_v_min = gpf['C_v_min_to', ac.eng_type, 'to']
    else:
        C_v_min = gpf['C_v_min', ac.eng_type, PHASE_BADA_STR[phase]]

    minCAS = C_v_min * Vstall
    
    return minCAS


def scheduled_CAS(ac, phase, fl, mass):
    """
    Return calibrated airspeed according to BADA's modeled speed schedule

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    phase : int
        Number representing flight phase according to Phase.
    fl : float
        Flight level.
    mass : str
        Aircraft mass condition. Possible values are 'LO', 'AV', and 'HI' for
        low, average, and high, respectively.

    Raises
    ------
    Exception
        If aircraft eng_type, mass condition, or flight phase are not
        recognized.

    Returns
    -------
    Vcas : float
        Scheduled calibrated airspeed.

    Notes
    -----
    This function does not apply above the Mach transition altitude!
    
    """
    eng_type = ac.opf['aircraft']['eng_type']
    if eng_type not in ['Jet', 'Turboprop', 'Piston']:
        raise Exception(
            f'eng_type "{eng_type}" != "Jet", "Turboprop", "Piston"')
    
    if mass not in ['LO', 'AV', 'HI']:
        raise Exception(f'mass condition "{mass}" != "LO", "AV", "HI"')
    
    if phase not in [Phase.CLIMB, Phase.CRUISE, Phase.DESCENT]:
        raise Exception(f'phase "{phase}" not recognized')
    
    apf = ac.apf
    opf = ac.opf
    gpf = ac.parentbada.gpf
    
    if phase is Phase.CLIMB:
        if eng_type == 'Jet':
            if fl < 100:
                Vcas = min(apf[f'default_{mass}']['Vcl1'], 250)
                if fl < 60:
                    Vstall_to = opf['aero']['Vstall_to']
                    C_v_min = gpf['C_v_min', 'jet', 'cl']
                    if fl < 15:
                        Vd_cl = gpf['V_cl_1', 'jet', 'cl']
                    elif fl < 30:
                        Vd_cl = gpf['V_cl_2', 'jet', 'cl']
                    elif fl < 40:
                        Vd_cl = gpf['V_cl_3', 'jet', 'cl']
                    elif fl < 50:
                        Vd_cl = gpf['V_cl_4', 'jet', 'cl']
                    else:
                        Vd_cl = gpf['V_cl_5', 'jet', 'cl']
                    # Ensure that CAS is not higher at lower altitudes
                    Vcas = min(Vcas, C_v_min * Vstall_to + Vd_cl)
            else:
                Vcas = apf[f'default_{mass}']['Vcl2']
        else:
            # (Turboprop or Piston)
            if fl < 100:
                Vcas = min(apf[f'default_{mass}']['Vcl1'], 250)
                if fl < 15:
                    Vstall_to = opf['aero']['Vstall_to']
                    # C_v_min and Vd_cl are the same for 'Turboprop'
                    # and 'Piston'
                    C_v_min = gpf['C_v_min', 'turbo', 'cl']
                    if fl < 5:
                        Vd_cl = gpf['V_cl_6', 'turbo', 'cl']
                    elif fl < 10:
                        Vd_cl = gpf['V_cl_7', 'turbo', 'cl']
                    else:
                        Vd_cl = gpf['V_cl_8', 'turbo', 'cl']
                    # Ensure that CAS is not higher at lower altitudes
                    Vcas = min(Vcas, C_v_min * Vstall_to + Vd_cl)
            else:
                Vcas = apf[f'default_{mass}']['Vcl2']
    
    if phase == Phase.CRUISE:
        if eng_type == 'Jet':
            if fl < 30:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 170)
            elif fl < 60:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 220)
            elif fl < 140:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 250)
            else:
                Vcas = apf[f'default_{mass}']['Vcr2']
        else:
            # (Turboprop or Piston)
            if fl < 30:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 150)
            elif fl < 60:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 180)
            elif fl < 100:
                Vcas = min(apf[f'default_{mass}']['Vcr1'], 250)
            else:
                Vcas = apf[f'default_{mass}']['Vcr2']
    
    elif phase == Phase.DESCENT:
        if eng_type == 'Jet' or eng_type == 'Turboprop':
            if fl < 30:
                Vstall_lnd = opf['aero']['Vstall_lnd']
                # C_v_min and Vd_des are the same for 'Jet' and 'Turboprop'
                C_v_min = gpf['C_v_min', 'jet', 'des']
                if fl < 10:
                    Vd_des = gpf['V_des_1', 'jet', 'des']
                elif fl < 15:
                    Vd_des = gpf['V_des_2', 'jet', 'des']
                elif fl < 20:
                    Vd_des = gpf['V_des_3', 'jet', 'des']
                else:
                    Vd_des = gpf['V_des_4', 'jet', 'des']
                Vcas = C_v_min * Vstall_lnd + Vd_des
            elif fl < 60:
                Vcas = min(apf[f'default_{mass}']['Vdes1'], 220)
            elif fl < 100:
                Vcas = min(apf[f'default_{mass}']['Vdes1'], 250)
            else:
                Vcas = apf[f'default_{mass}']['Vdes2']
        else:
            # (Piston)
            if fl < 30:
                Vstall_lnd = opf['aero']['Vstall_lnd']
                C_v_min = gpf['C_v_min', 'piston', 'des']
                if fl < 5:
                    Vd_des = gpf['V_des_5', 'piston', 'des']
                elif fl < 10:
                    Vd_des = gpf['V_des_6', 'piston', 'des']
                else:
                    Vd_des = gpf['V_des_7', 'piston', 'des']
                Vcas = C_v_min * Vstall_lnd + Vd_des
            elif fl < 100:
                Vcas = apf[f'default_{mass}']['Vdes1']
            else:
                Vcas = apf[f'default_{mass}']['Vdes2']
    
    # [TODO]: calculate H as AGL instead of using FL
    H = fl * 100
    Vcas = max(Vcas, get_min_speed(ac, phase, H, Vcas))
    Vcas = min(Vcas, opf['envelope']['VMO'])
    
    return Vcas


def climb_fuel(ac, h_start, h_end, lat_s, lon_s, azi_s, m_s, t_s, simcfg,
               met=None, met_lats=None, met_lons=None, interpolate_ptf=False):
    """
    Calculate fuel burn while climbing from h_start to h_end

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    h_start : float
        Initial altitude above mean sea level in meters.
    h_end : float
        Final altitude above mean sea level in meters.
    lat_s : float
        Initial latitude in degrees north.
    lon_s : float
        Initial longitude in degress east.
    azi_s : float
        Initial azimuth in degrees clockwise from north (-180, +180].
    m_s : float
        Initial mass in kg.
    t_s : float
        Initial time in seconds.
    simcfg : main.SimConfig
        Configuration of simulation options.
    met : xr.Dataset or None, optional
        Meteorological conditions. Can be None if simulation is configured to
        not use them. The default is None.
    interpolate_ptf : bool, optional
        If True, interpolate ptf data for current FL. If False, simply take
        data from the first entry where FL is >= the current FL.
        The default is False.

    Raises
    ------
    Exception
        If initial FL or mass are outside of flight envelope.

    Returns
    -------
    emissions : list of main.EmissionSegment
        Calculated emissions.

    """
    # Verify that we are going up
    if h_end < h_start:
        raise Exception('h_end < h_start while climbing')
    fl = np.round(h_start / physics.FT_TO_METER / 100, 0)
    fl_end = np.round(h_end / physics.FT_TO_METER / 100, 0)
    
    # Verify that altitude is within modeled range
    if fl < ac.ptf['table'].index[0]:
        raise Exception('flight level is lower than lowest available in PTF')
    if fl > ac.ptf['hMO'] / 100:
        raise Exception('flight level is higher than max altitude')
    
    # Verify that mass is within modeled range
    if m_s < ac.ptf['mlo']:
        if simcfg.bada_clip_mlo:
            m_s = ac.ptf['mlo']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is lower than '
                   + '"low mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    if m_s > ac.ptf['mhi'] + MASS_TOLERANCE:
        if simcfg.bada_clip_mhi:
            m_s = ac.ptf['mhi']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is higher than '
                   + '"high mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    
    # Get performance at current flight level
    if interpolate_ptf:
        ptf_temp = ac.ptf['table'].copy(deep=True)
        if fl not in ptf_temp.index.values:
            ptf_temp.loc[fl] = np.nan
            ptf_temp = ptf_temp.interpolate(method='index')
        perf_at_fl = ptf_temp.loc[fl]
    else:
        perf_at_fl = (
            ac.ptf['table'].iloc[ac.ptf['table'].index.searchsorted(fl)]
        )
    
    # Interpolate ROC in ptf for current mass
    if m_s > ac.ptf['mnom']:
        frac = (m_s - ac.ptf['mnom']) / (ac.ptf['mhi'] - ac.ptf['mnom'])
        roc = (
            (1 - frac) * perf_at_fl['ROCDnom_cl']
            + frac * perf_at_fl['ROCDhi_cl']
        )
    else:
        frac = (m_s - ac.ptf['mlo']) / (ac.ptf['mnom'] - ac.ptf['mlo'])
        roc = (
            (1 - frac) * perf_at_fl['ROCDlo_cl']
            + frac * perf_at_fl['ROCDnom_cl']
        )
    roc *= physics.FPM_TO_MPS
    tas = perf_at_fl['Vcl'] * physics.KT_TO_MPS
    
    # Step time
    dt = (h_end - h_start) / roc
    if np.isclose(roc, 0):
        warnings.warn(f'roc ~ 0 for {ac.typecode} at {h_start:.0f} -> {h_end:.0f} m')
    
    # Calculate ground speed
    tas_hor = np.sqrt(tas**2 - roc**2)
    
    if hasattr(simcfg, 'wind') and simcfg.wind == 'fixed-field':
        ilev = met.h_edge.values.searchsorted(h_start) - 1
        ilev = max(ilev, 0)
        
        ilat_h = min(len(met_lats) - 1, met_lats.searchsorted(lat_s))
        ilat_l = max(0, ilat_h - 1)
        if met_lats[ilat_h] - lat_s < lat_s - met_lats[ilat_l]:
            ilat = ilat_h
        else:
            ilat = ilat_l
        
        ilon_h = min(len(met_lons) - 1, met_lons.searchsorted(lon_s))
        ilon_l = max(0, ilon_h - 1)
        if met_lons[ilon_h] - lon_s < lon_s - met_lons[ilon_l]:
            ilon = ilon_h
        else:
            ilon = ilon_l
        ilon %= len(met_lons) - 1
                
        gridcell = met.isel(lev=ilev, lat=ilat, lon=ilon)
        ws = gridcell.WS.values
        wdir = gridcell.WDIR.values
        
        alpha = min((wdir - azi_s) % 360, (azi_s - wdir) % 360)
        ws_parallel = ws * np.cos(alpha*np.pi/180)
        ws_perpendicular = ws * np.sin(alpha*np.pi/180)
        
        beta = np.arcsin(ws_perpendicular/tas_hor)
        
        gs = tas_hor * np.cos(beta) + ws_parallel
    else:
        gs = tas_hor
    
    # Horizontal distance travelled
    dist = gs * dt
        
    # Fuel burn
    fuel_burn = perf_at_fl['fnom_cl'] * dt / 60
    
    # Add fuel to accelerate to a new CAS
    Vcas = scheduled_CAS(ac, Phase.CLIMB, fl, mass='HI')
    Vcas_end = scheduled_CAS(ac, Phase.CLIMB, fl_end, mass='HI')
    if Vcas_end != Vcas:
        Vcas *= physics.KT_TO_MPS
        Vcas_end *= physics.KT_TO_MPS
        # First approximation using starting mass
        est_m = m_s - fuel_burn
        accel_energy = (est_m * Vcas_end**2 - m_s * Vcas**2) / 2
        est_m -= accel_energy / physics.FUEL_LHV / ENG_OVERALL_EFF
        # Second approximation considering mass reduction
        accel_energy = (est_m * Vcas_end**2 - m_s * Vcas**2) / 2
        fuel_burn += accel_energy / physics.FUEL_LHV / ENG_OVERALL_EFF
        
    lat_e, lon_e, azi_e = physics.direct_geodesic_prob(
        lat_s, lon_s, azi_s, dist)
    emissions = EmissionSegment(lat_s, lon_s, h_start,
                                lat_e, lon_e, h_end)
    emissions.azi_e = azi_e
    emissions.t_s = t_s
    emissions.t_e = t_s + dt
    emissions.tas = tas
    emissions.dist = dist
    emissions.phasenum = Phase.CLIMB
    emissions.fuelburn = fuel_burn
    
    return emissions
    

def cruise_fuel(ac, h, lat_s, lon_s, lat_e, lon_e, dist, m_s, t_s, simcfg,
                met=None, met_lats=None, met_lons=None, add_to_ptf=True):
    """
    Calculate fuel burn while cruising at h

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    h : float
        Cruise altitude above mean sea level in meters.
    lat_s : float
        Initial latitude in degrees north.
    lon_s : float
        Initial longitude in degress east.
    lat_e : float
        Final latitude in degrees north.
    lon_e : float
        Final longitude in degress east.
    dist : float
        Horizontal length of the segment in meters.
    m_s : float
        Initial mass in kg.
    t_s : float
        Initial time in seconds.
    simcfg : main.SimConfig
        Configuration of simulation options.
    met : xr.Dataset or None, optional
        Meteorological conditions. Can be None if simulation is configured to
        not use them. The default is None.
    add_to_ptf : bool, optional
        If True, missing performance data interpolated for this flight level
        are added to the aircraft's PTF table. The default is True.

    Raises
    ------
    Exception
        If initial FL or mass are outside of flight envelope.

    Returns
    -------
    emissions : list of main.EmissionSegment
        Calculated emissions.

    """
    fl = np.round(h / physics.FT_TO_METER / 100, 0)
    # Verify that altitude is within modeled range
    if fl < ac.ptf['table'].index[0]:
        raise Exception('flight level is lower than lowest available in PTF')
    if fl > ac.ptf['hMO'] / 100:
        raise Exception('flight level is higher than max altitude')
    
    # Verify that mass is within modeled range
    if m_s < ac.ptf['mlo']:
        if simcfg.bada_clip_mlo:
            m_s = ac.ptf['mlo']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is lower than '
                   + '"low mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    if m_s > ac.ptf['mhi'] + MASS_TOLERANCE:
        if simcfg.bada_clip_mhi:
            m_s = ac.ptf['mhi']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is higher than '
                   + '"high mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    
    # Get performance at current flight level
    ptf_table = ac.ptf['table']
    if fl not in ptf_table.index.values:
        if not add_to_ptf:
            ptf_table = ac.ptf['table'].copy(deep=True)
        ptf_table.loc[fl] = np.nan
        ptf_table = ptf_table.interpolate(method='index')
    # Since ptf's index is sorted this is a bit faster than using .loc[]
    ifl = ptf_table.index.searchsorted(fl)
    perf_at_fl = ptf_table.iloc[ifl]
    
    # Interpolate fuel flow in ptf for current mass
    if m_s > ac.ptf['mnom']:
        frac = (m_s - ac.ptf['mnom']) / (ac.ptf['mhi'] - ac.ptf['mnom'])
        f = (
            (1 - frac) * perf_at_fl['fnom_cr']
            + frac * perf_at_fl['fhi_cr']
        )
    else:
        frac = (m_s - ac.ptf['mlo']) / (ac.ptf['mnom'] - ac.ptf['mlo'])
        f = (
            (1 - frac) * perf_at_fl['flo_cr']
            + frac * perf_at_fl['fnom_cr']
        )
    f /= 60
    tas = perf_at_fl['Vcr'] * physics.KT_TO_MPS
    
    # Calculate ground speed
    if hasattr(simcfg, 'wind') and simcfg.wind == 'fixed-field':
        ilat_h = min(len(met_lats) - 1, met_lats.searchsorted(lat_s))
        ilat_l = max(0, ilat_h - 1)
        if met_lats[ilat_h] - lat_s < lat_s - met_lats[ilat_l]:
            ilat = ilat_h
        else:
            ilat = ilat_l
        
        ilon_h = min(len(met_lons) - 1, met_lons.searchsorted(lon_s))
        ilon_l = max(0, ilon_h - 1)
        if met_lons[ilon_h] - lon_s < lon_s - met_lons[ilon_l]:
            ilon = ilon_h
        else:
            ilon = ilon_l
        ilon %= len(met_lons) - 1
        
        gridcell = met.isel(lat=ilat, lon=ilon)
        ws = gridcell.WS.values
        wdir = gridcell.WDIR.values
        
        azi, _ = physics.distance(lat_s, lon_s, lat_e, lon_e)
        
        alpha = min((wdir - azi) % 360, (azi - wdir) % 360)
        ws_parallel = ws * np.cos(alpha*np.pi/180)
        ws_perpendicular = ws * np.sin(alpha*np.pi/180)
        
        beta = np.arcsin(ws_perpendicular/tas)
        
        gs = tas * np.cos(beta) + ws_parallel
    else:
        gs = tas
    
    # Step time
    dt = dist / gs
    
    # Fuel burn
    fuel_burn = f * dt
    
    emissions = EmissionSegment(lat_s, lon_s, h, lat_e, lon_e, h)
    emissions.t_s = t_s
    emissions.t_e = t_s + dt
    emissions.tas = tas
    emissions.phasenum = Phase.CRUISE
    emissions.fuelburn = fuel_burn
    
    return emissions

    
def descent_fuel(ac, h_start, h_end, lat_s, lon_s, azi_s, m_s, t_s, simcfg,
                 met=None, met_lats=None, met_lons=None,
                 interpolate_ptf=False):
    """
    Calculate fuel burn while descending from h_start to h_end

    Parameters
    ----------
    ac : bada.Aircraft
        Modeled aircraft.
    h_start : float
        Initial altitude above mean sea level in meters.
    h_end : float
        Final altitude above mean sea level in meters.
    lat_s : float
        Initial latitude in degrees north.
    lon_s : float
        Initial longitude in degress east.
    azi_s : float
        Initial azimuth in degrees clockwise from north (-180, +180].
    m_s : float
        Initial mass in kg.
    t_s : float
        Initial time in seconds.
    simcfg : main.SimConfig
        Configuration of simulation options.
    met : xr.Dataset or None, optional
        Meteorological conditions. Can be None if simulation is configured to
        not use them. The default is None.
    interpolate_ptf : bool, optional
        If True, interpolate ptf data for current FL. If False, simply take
        data from the first entry where FL is >= the current FL.
        The default is False.

    Raises
    ------
    Exception
        If initial FL or mass are outside of flight envelope, or h_s > h_e

    Returns
    -------
    emissions : list of main.EmissionSegment
        Calculated emissions.

    """
    # Verify that we are going down
    if h_end > h_start:
        raise Exception('h_end > h_start while descending')
    fl = np.round(h_start / physics.FT_TO_METER / 100, 0)
    
    # Verify that altitude is within modeled range
    if fl < ac.ptf['table'].index[0]:
        raise Exception('flight level is lower than lowest available in PTF')
    if fl > ac.ptf['hMO'] / 100:
        raise Exception('flight level is higher than max altitude')
    
    # Verify that mass is within modeled range
    if m_s < ac.ptf['mlo']:
        if simcfg.bada_clip_mlo:
            m_s = ac.ptf['mlo']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is lower than '
                   + '"low mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    if m_s > ac.ptf['mhi'] + MASS_TOLERANCE:
        if simcfg.bada_clip_mhi:
            m_s = ac.ptf['mhi']
        else:
            msg = (f'mass ({m_s:,.0f} kg) is higher than '
                   + '"high mass" defined in PTF')
            if hasattr(ac, 'typecode'):
                msg += f' (typecode={ac.typecode})'
            raise Exception(msg)
    
    # Get performance at current flight level
    if interpolate_ptf:
        ptf_temp = ac.ptf['table'].copy(deep=True)
        if fl not in ptf_temp.index.values:
            ptf_temp.loc[fl] = np.nan
            ptf_temp = ptf_temp.interpolate(method='index')
        perf_at_fl = ptf_temp.loc[fl]
    else:
        perf_at_fl = (
            ac.ptf['table'].iloc[ac.ptf['table'].index.searchsorted(fl)]
        )
    
    rod = perf_at_fl['ROCDnom_des'] * physics.FPM_TO_MPS
    tas = perf_at_fl['Vdes'] * physics.KT_TO_MPS
    
    # Step time
    dt = (h_start - h_end) / rod
    
    # Calculate ground speed
    tas_hor = np.sqrt(tas**2 - rod**2)
    
    if hasattr(simcfg, 'wind') and simcfg.wind == 'fixed-field':
        ilev = met.h_edge.values.searchsorted(h_start) - 1
        ilev = max(ilev, 0)
        
        ilat_h = min(len(met_lats) - 1, met_lats.searchsorted(lat_s))
        ilat_l = max(0, ilat_h - 1)
        if met_lats[ilat_h] - lat_s < lat_s - met_lats[ilat_l]:
            ilat = ilat_h
        else:
            ilat = ilat_l
        
        ilon_h = min(len(met_lons) - 1, met_lons.searchsorted(lon_s))
        ilon_l = max(0, ilon_h - 1)
        if met_lons[ilon_h] - lon_s < lon_s - met_lons[ilon_l]:
            ilon = ilon_h
        else:
            ilon = ilon_l
        ilon %= len(met_lons) - 1
                
        gridcell = met.isel(lev=ilev, lat=ilat, lon=ilon)
        ws = gridcell.WS.values
        wdir = gridcell.WDIR.values
        
        alpha = min((wdir - azi_s) % 360, (azi_s - wdir) % 360)
        ws_parallel = ws * np.cos(alpha*np.pi/180)
        ws_perpendicular = ws * np.sin(alpha*np.pi/180)
        
        beta = np.arcsin(ws_perpendicular/tas_hor)
        
        gs = tas_hor * np.cos(beta) + ws_parallel
    else:
        gs = tas_hor
    
    # Horizontal distance travelled
    dist = gs * dt
        
    # Fuel burn
    fuel_burn = perf_at_fl['fnom_des'] * dt / 60
    
    # No fuel saved from deceleration is subtracted, as it is assumed that
    # the engine will already idle during descent
        
    lat_e, lon_e, azi_e = physics.direct_geodesic_prob(
        lat_s, lon_s, azi_s, dist)
    emissions = EmissionSegment(lat_s, lon_s, h_start,
                                lat_e, lon_e, h_end)
    emissions.azi_e = azi_e
    emissions.t_s = t_s
    emissions.t_e = t_s + dt
    emissions.tas = tas
    emissions.dist = dist
    emissions.phasenum = Phase.DESCENT
    emissions.fuelburn = fuel_burn
    
    return emissions
    

##
# Main routine
##

if __name__ == '__main__':
    # Load model data
    print('Loading BADA...', end=' ')
    activebada = Bada(bada_dir)
    print('ok')
