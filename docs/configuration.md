# Configuration options

This file describes the options that can be set in a SimConfig object. Each option is saved as an attribute of the SimConfig object.

## Scope of the run

- *fly_nonlto* : bool
    - If True, simulate the non-LTO portion of flights. The default is `True`.
    
- *fly_international* : str
    - If 'no', disregard international flights. If 'only', disregard domestic flights. The default is `'yes'`.

- *acs_allow* : list of str
    - Allowlist of aircraft types. Only ICAO type codes in the list will be included in the run. If the list is empty, no allowlist is applied. The default is `[]`.

- *acs_block* : list of str
    - Blocklist of aircraft types. ICAO type codes in the list will be excluded from the run. If the list is empty, no blocklist is applied. The default is `[]`.

- *apt_cco_allow* : list of str
    - Allowlist of countries of departure. Only flights departing from airports with an ISO 2-letter code in the list will be included in the run. If the list is empty, no allowlist is applied. The default is `[]`.

- *apt_cco_block* : list of str
    - Blocklist of countries of departure. Flights departing from airports with an ISO 2-letter code in the list will be excluded from the run. If the list is empty, no blocklist is applied. The default is `[]`.

- *apt_ccd_allow* : list of str
    - Allowlist of countries of arrival. Only flights arriving at airports with an ISO 2-letter code in the list will be included in the run. If the list is empty, no allowlist is applied. The default is `[]`.

- *apt_ccd_block* : list of str
    - Blocklist of countries of arrival. Flights arriving at airports with an ISO 2-letter code in the list will be excluded from the run. If the list is empty, no blocklist is applied. The default is `[]`.

  
## Results returned

- *return_flight_durations* : bool
    - If True, returns a pd.DataFrame containing the duration and fuel burn of each flight. The default is `False`.

- *split_job* : 
    - How the run is split into "subtasks". This determines how the work is spread across multiple processes if multithreading is True, and determines which "sub-results" are returned|saved if return_splits|save_splits is True. Possible values are 'ac' for a split by aircraft type, 'cc_o' for a split by country of departure, 'cc_d' for a split by country of arrival. The default is `'ac'`.

- *return_splits* : bool
    - If True, return emissions per aircraft type, country of departure, or country of arrival, according to `split_job`. The default is `False`.


## Output files

- *save_emissions* : bool
    - If True, save the gridded emissions in a netCDF4 file. The default is `True`.

- *save_cfg_in_nc4* : bool
    - If True, the simulation configuration is recorded in the emissions dataset's attributes. The default is `True`.

- *output_name* : str
    - Base name of the output file, without extension. The default is `'run_output'`.

- *outputdir* : str
    - Path to directory where the gridded emissions will be saved. The default is `r'./output/'`.

- *splitdir* : str
    - Path to directory where the emission splits (emissions per aircraft type or country of origin or destination) will be saved. The default is `r'./output/'`.

- *save_splits* : bool
    - If True and `save_emissions` is also True, save emissions per aircraft type, country of departure, or country of arrival, according to `split_job`. The default is `True`.

- *merge_saved_splits* : bool
    - If True and both `save_emissions` and `save_splits` are also True, concatenate emissions splits into a single .nc4 file along a dimension named after `split_job`. The default is `True`.


## Multiprocessing
- *multiprocessing* : bool
    - If True, split run into multiple processes (by aircraft, country of origin, or country of destination, depending on `split_job`) . The default is `False`.

- *nthreads* : int
    - Number of processes between which the simulation run will be distributed. It is ignored if `multiprocessing` is False. The default is `15`.


## Pre-processing

- *dryrun* : bool
    - If True, stop run just after loading input data. The default is `False`.

- *precheck_apts* : bool
    - If True, verify that data for all airports needed are present. The default is `True`.

- *precheck_acs* : bool
    - If True, verify that data for all aircraft needed are present. The default is `True`.

- *precheck_engs* : bool
    - If True, verify that data for all engines needed are present. The default is `True`.

- *ignore_missing_apts* : bool
    - If True, disregard flights involving airports with data missing and continue the simulation. The default is `False`.
    
- *ignore_unknown_actypes* : bool
    - If True, disregard flights involving aircraft type codes that are not recognized. The default is `True`.
    
- *ignore_nonairplanes* : bool
    - If True, disregard flights involving aircraft types not recognized as airplanes. The default is `True`.

- *ignore_unsupported_acs* : bool
    - If True, disregard flights involving aircraft types not supported by the aircraft performance model. The default is `True`.

- *filter_nonairplanes* : bool
    - If True, filter out aircraft types not corresponding to airplanes. The default is `True`.

- *ignore_military* : bool
    - If True, disregard flights involving aircraft types that are listed on `aircraft/military.csv`. The default is `True`.

- *ignore_missing_engs* : bool
    - If True, disregard flights involving engines with data missing and continue the simulation. The default is `False`.

- *ignore_missing_apu* : bool
    - If True, disregard flights involving APUs with data missing and continue the simulation. The default is `False`.

- *apu_warn_missing* : bool
    - If True, raise a warning when APU data is missing. The default is `True`.
    
- *remove_same_od* : bool
    - If True, disregard flights in which origin and destination are the same. The default is `True`.
    
- *ac_replacements* : bool
    - If True, use a map from r'aircraft/typecode_replacements.csv' to replace aircraft type codes. The default is `True`.

- *use_replacement_eng* : bool
    - If True, when engine data is missing, try to load data for the engine assigned to the aircraft type representing this aircraft in the aircraft performance model. The default is `True`.
    
- *use_replacement_apu* : bool
    - If True, when APU data is missing, try to load data for the APU assigned to the aircraft type representing this aircraft in the aircraft performance model. The default is `True`.


## Airport database

- *apt_source* : str
    - Which airport databases to import. Possible values are 'openflights', 'ourairports', 'all'. The file paths are set in `datasources.FILEPATH_OPENFLIGHTS_APTS` and `datasources.FILEPATH_OURAIRPORTS_APTS`. The default is `'all'`.
    
- *apt_closed* : bool
    - If True, load airports marked as "closed". The default is `False`.

- *apt_heliport* : bool
    - If True, load airports marked as "heliport". The default is `False`.


## Aircraft engines and APUs

- *eng_properties_path* : str or None
    - Path to CSV file containing engine fuel rates and emission indices. If None, the default path set in `datasources.FILEPATH_ENG_PROPERTIES` is used. The default is `None`.

 - *eng_allocations_path* : str or None
    - Path to CSV file containing engine and APU allocation for each aircraft type. If None, the default path set in `datasources.FILEPATH_ENG_ALLOCATIONS` is used. The default is `None`.

- *apu_properties_path* : str or None
    - Path to CSV file containing APU fuel rates and emission indices. If None, the default path set in `datasources.FILEPATH_APU_PROPERTIES` is used. The default is `None`.
    
- *apu_tim* : str
    - Time-in-mode used for the APU. Possible values are 'ICAOadv', 'ICAOsimple', 'ATAmid', 'AEDT', 'AEIC', 'none'. The default is `'ICAOadv'`.
    
- *apu_eigas* : str or list of str
    - Source of gaseous emission indices for the APU. If a list, try each source in order until one succeeds. Possible values for the strings are 'specific', 'ICAOadv', 'ICAOsimple', 'ACRP', 'none'. The default is `['specific', 'ICAOadv']`.

- *apu_eipm* : str or list of str
    - Source of particulate matter emission indices for the APU. If a list, try each source in order until one succeeds. Possible values for the strings are 'specific', 'ICAOadv', 'ICAOsimple', 'ACRP', 'none'. The default is `['specific', 'ICAOadv']`.

- *apu_pm10_as_nvpm* : bool
    - If True, use PM<sub>10</sub> emission indices as nvPM EIs for APUs. The default is `False`.
    
- *apu_businessjet* : bool
    - If False, disregard APU emissions from business jets. The default is `True`.
    

## LTO

- *ltocycle* : str
    - LTO cycle model. Possible values are 'Stettler', 'ICAO', and 'none'. The default is `'Stettler'`.

- *reverse_rand_occurence* : bool
    - If True, in the Stettler LTO cycle, use a random number generator for each type-origin-destination triple to choose whether reverse thrust is applied. The default is `False`.


## Aircraft mass

- *yyyymm* : str or None
    - Year and month of simulation. Currently used to choose a month-specific payload mass fraction, unless `yyyymm` is None (in which case the value from `payload_m_frac` is used). The default is `None`.

- *payload_m_frac* : float or None
    - Payload mass fraction, i.e. percentage of maximum payload that will be considered for all flights. If None, a value will be chodsen from aircraft/load_factors.csv based on the year-month given by `yyyymm`. The default is `0.700`.

- *bada_clip_mlo* : 
    - If True, limit the minimum aircraft mass during flights to the minimum operating mass for that aircraft type. If False, an error is raised when the mass is lower than the minimum. The default is `True`.

- *bada_clip_mhi* : 
    - If True, limit the maximum aircraft mass during flights to the maximum operating mass for that aircraft type. If False, an error is raised when the mass is higher than the maximum.  The default is `False`.


## Non-LTO

- *lat_inefficiency* : str
    - Model of lateral inefficiency to use. Possible values are 'FEAT', 'AEIC', 'none'. The default is `'FEAT'`.

- *lat_ineff_cr_max* : float
    - Maximum limit of the lateral inefficiency correction factor to any single segment of emissions. This limit serves to avoid issues with the factor tending to infinity as the length of a segment tends to zero. The default is `2.0`.

- *cruise_fl* : str
    - Source of aircraft type specific cruise flight level. Possible values are 'openavem', 'bada-sage', 'bada'. The default is `'openavem'`.

- *hstep_cl* : float
    - Simulation altitude step during climb, in meters. The default is `1000 * physics.FT_TO_METER`.

- *hstep_des* : float
    - Simulation altitude step during descent, in meters. The default is `1000 * physics.FT_TO_METER`.

- *sstep_cr* : float
    - Simulation horizontal step during cruise, in meters. The default is `125 * physics.NM_TO_METER`.

- *wind* : str
    - Method of wind modeling. Possible values are 'fixed-field' and 'none'. The default is `'none'`.

- *wind_filepath* : str
    - Path to the file containing wind data to be used if `wind` is 'fixed-field'. The default is `openavem.core.FILEPATH_WIND`.


## nvPM

- *nvpm_lto* : str or list of str
    - Method to estimate nvPM mass during LTO. Possible values are 'measured', 'FOA4', 'SCOPE11', 'FOA3', 'FOX', 'constantEI', 'none'. If a list, try each method successively until one succeeds. The default is `['measured', 'FOA4', 'none']`.

- *nvpm_lto_EI* : float
    - nvPM mass emission index, in mg/kg, for the 'constantEI' method during LTO. The default is `30`.

- *nvpmnum_lto* : str or list of str
    - Method to estimate nvPM particle numbers during LTO. Possible values are 'measured', 'FOA4', 'SCOPE11', 'constantEI', 'none'. If a list, try each method successively until one succeeds. The default is `['measured', 'FOA4', 'none']`.

- *nvpmnum_lto_EI* : float
    - nvPM number emission index, in 1e14/kg, for the 'constantEI' method during LTO. The default is `0`.

- *nvpm_interp* : str
    - Method of interpolating nvPM EI by fuel flow. Possible values are 'PCHIP' and 'linear'. The default is `'PCHIP'`.

- *nvpm_cruise* : str or list of str
    - Method to estimate nvPM mass during non-LTO. Possible values are 'AEDT-FOA4', 'FOX', 'constantEI', 'none'. If a list, try each method successively until one succeeds. The default is `['AEDT-FOA4', 'constantEI', 'none']`.

- *nvpm_cruise_EI* : float
    - nvPM mass emission index, in mg/kg, for the 'constantEI' method during non-LTO. The default is `30`.

- *nvpmnum_cruise_EI* : float
    -  nvPM number emission index, in 1e14/kg, for the 'constantEI' method during non-LTO. The default is `0`.

- *nvpm_cruise_piston* : str or list of str
    - Method to estimate nvPM mass during LTO. Possible values are 'measured', 'FOA4', 'SCOPE11', 'FOA3', 'FOX', 'constantEI', 'none'. If a list, try each method successively until one succeeds. The default is `['AEDT-FOA4', 'constantEI', 'none']`.


## Grid

- *grid_vertical* : str
    - Type of vertical grid structure. Possible values are 'uniform' for 90 levels of 500 ft height, 'GC#' for the lower # levels of the [GEOS-FP / MERRA-2 72-layer grid](http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids#72-layer_vertical_grid), 'GC#r' for the lower # levels of the [GEOS-Chem reduced 47-layer grid](http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids#47-layer_reduced_vertical_grid), 'none' for 2-D grids. The default is `'GC33'`.

- *grid_everyflight* : bool
    - If True, emissions segments are regridded after every flight is calculated, which could reduce RAM usage at the cost of performance. The default is `False`.

- *grid_method* : str
    - The method used to spread emissions segments into grid values. Possible values are 'supersampling', 'nearestneighbor'. The default is `'supersampling'`.

- *transform_alt* : bool
    - If True, apply the altitude coordinate transformation such that non-LTO emissions transition to pressure levels. The default is `True`.


## Logging and errors

- *logfile* : str
    - Path to file where the log will be saved. The default is `'./logs/log.log'`.

- *loglvl* : int
    - [Level of logging](https://docs.python.org/3/library/logging.html#logging-levels) for the log that will go in logfile. The default is `logging.INFO`.

- *loglvl_stream* : int
    - [Level of logging](https://docs.python.org/3/library/logging.html#logging-levels) for the log that will go to stderr. The default is `logging.WARNING`.

- *continue_on_error* : bool
    - If True, ignore errors and continue to the next aircraft or next country in the simulation. The default is `True`.


## Debugging

- *debug_flist* : bool
    - If True, print fuel burn of every flight during the run. The default is `False`.

- *debug_m_est* : bool
    - If True, print starting mass (payload, fuel, and total) for every flight. The default is `False`.

- *debug_climb_step* : bool
    - If True, print fuel burn and altitude at every step during climb for every flight. The default is `False`.

- *debug_cruise_step* : bool
    - If True, print fuel burn and horizontal distance at every step during cruise for every flight. The default is `False`.

- *debug_descent_step* : bool
    - If True, print fuel burn and altitude at every step during descent for every flight. The default is `False`.

- *verify_flightfuel* : float
    - If larger than zero, give a warning if any flight if `fuel_burned > verify_flightfuel * (ac_max_mass - ac_min_mass)`, where ac_max_mass and ac_min_mass are the maximum and minimum operating weight of the aircraft type. The default is `0`.