# Open Aviation Emissions (openAVEM)

*openAVEM is a Python package that takes a list of flights as input and calculates fuel burn and atmospheric emissions*: oxides of nitrogen (NO<sub>x</sub>), hydrocarbons (HC), CO, non-volatile particulate matter (nvPM). Landing and takeoff (LTO) is modeled with a time-in-mode approach, while the non-LTO portions of flight (climb, cruise, descent) are modeled using EUROCONTROL's BADA aircraft performance model. A description of the methodology used by openAVEM will appear in the following article:

> Quadros, F. D. A., Sun, J., Snellen, M., and Dedoussi, I. C. “Global civil aviation emissions estimates for 2017–2020”. In preparation.

openAVEM was developed at the Aircraft Noise and Climate Effects section (ANCE) of the Faculty of Aerospace Engineering at Delft University of Technology, in the Netherlands. Flávio Quadros maintains the project.

Currently, openAVEM relies on the *BADA 3* (Base of aircraft data) aircraft performance model, which must be acquired directly from [EUROCONTROL](https://www.eurocontrol.int/model/bada).


## Code setup

### Example

```
$ git clone https://github.com/flavioquadros/openavem.git
$ cd openavem
$ conda create --name openavem --file requirements.txt
$ conda activate
```

### Required dependencies

See `requirements.txt` for version numbers

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [xarray](https://xarray.pydata.org/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [scipy](https://scipy.org/scipylib/)
- [geographiclib](https://geographiclib.sourceforge.io/html/python/)
- [matplotlib](https://matplotlib.org/) (needed for plotting)
- [cartopy](https://scitools.org.uk/cartopy/) (needed for plotting)
- [colorcet](https://colorcet.holoviz.org/) (needed for plotting)


## Data setup

### Aircraft performance model

After acquiring a valid license from EUROCONTROL for the BADA 3 model, the model files (\*.APF, \*.GPF, \*.NEW, \*.OPF, \*.PTD, \*.PTF) need to be added to a folder in the project root called `BADA`. A different path can be set by changing the variable `bada_dir` in `openavem/bada.py`.

### Airport data

Airport properties in the format of the database from [OpenFlights](https://openflights.org/data.html) should be saved as `./airports/airports.dat`. An additional airport database, based on [OurAirports](https://ourairports.com/), is distributed with openAVEM in `./airports/OurAirports/`. Different paths can be set by changing the variables `FILEPATH_OPENFLIGHTS_APTS` and `FILEPATH_OURAIRPORTS_APTS` in `openavem/datasources.py`.

For the "Stettler" LTO cycle, airport specific time-in-mode (TIM) values can be defined in `./airports/tim_stettler.csv`. In the absence of specific TIMs, the taxiing time of an airport in the "Stettler" LTO cycle can be estimated from the number of yearly aircraft movements as defined in the file `./airports/yearly_movements.csv`. Different paths can be set by changing the variables `FILEPATH_TIM_STETTLER` and `FILEPATH_MOVEMENTS` in `openavem/datasources.py`.

### Wind data

NetCDF-4 files containing gridded wind speed (`WS`) and direction (`WDIR`) can be added to the `./met` directory, using the format exemplified by the file `met/wind_dummy.nc4`.

### Flight movement data

Lists of flights to be simulated can be added to the `./flightlists` directory, in CSV format. Four columns are required, as exemplified by `flightlists/sample/fcounts_sample.csv`:

- `typecode`
: ICAO aircraft type designator

- `origin`
: ICAO or IATA airport designator for the origin

- `destination`
: ICAO or IATA airport designator for the destination

- `count`
: Number of times this aircraft-origin-destination triple occurs (emissions will be multiplied by this number)

Note that a file can contain either only ICAO or only IATA airport codes. It is not possible to mix them.


## Usage

### Obtaining emissions from a single flight

```python
import openavem
ds, emissions = openavem.fly.testflight()
```

### Using an input file with flight counts

```python
import openavem
inputfile = './flightlists/sample/fcounts_sample.csv'
subds, ds = openavem.fly.run_fcounts(inputfile)
```

### Configuring a simulation run

The configurations of a run are stored in a `SimConfig` object, defined in `openavem/core.py`. All the configuration options are described in `docs/configuration.md`

```python
import openavem
inputfile = './flightlists/sample/fcounts_sample.csv'
simcfg = openavem.SimConfig(
    fly_nonlto=False,
    return_splits=True
)
subds, ds = openavem.fly.run_fcounts(inputfile, simcfg=simcfg)
```

### Obtaining flight durations

```python
import openavem
inputfile = './flightlists/sample/fcounts_sample.csv'
simcfg = openavem.SimConfig(
    return_flight_durations=True
)
subds, ds, df = openavem.fly.run_fcounts(inputfile, simcfg=simcfg)
```


## Citing openAVEM

(article is in preparation)


## Contributing

Contributions to the code via pull requests are very welcome! Please use the discussion board for comments and suggestions, and report any bugs to the issue tracker.


## License, credits and references

> Technische Universiteit Delft hereby disclaims all copyright interest in the program "openAVEM" written by the Author(s).
>
> Henri Werij, Dean of Faculty of Aerospace Engineering
>
> © 2021, Flávio D. A. Quadros
>
> This work is licensed under an Apache v2.0 OSS license

openAVEM was created as part of my PhD research, supervised by Prof. Mirjam Snellen and Dr. Irene C. Dedoussi.

The general concept of the model was inspired by the [Aviation Emissions Inventory Code v2.1 (AEIC)](https://lae.mit.edu/codes/) from the MIT Laboratory for Aviation and the Environment, described in the papers by [Stettler et al. 2011](https://doi.org/10.1016/j.atmosenv.2011.07.012), [Simone et al. 2013](https://doi.org/10.1016/j.trd.2013.07.001), [Stettler et al. 2013](https://doi.org/10.1021/es401356v).

The airport and runway database provided with openAVEM is derived from public domain data released by [OurAirports](https://ourairports.com/).

Engine properties are based on data from the [ICAO Aircraft Engine Emissions Databank](https://www.easa.europa.eu/domains/environment/icao-aircraft-engine-emissions-databank), data from the [Swiss Federal Office of Civil Aviation (FOCA)](https://www.bazl.admin.ch/bazl/en/home/specialists/regulations-and-guidelines/environment/pollutant-emissions/aircraft-engine-emissions/report--appendices--database-and-data-sheets.html), the [Procedures for Emission Inventory Preparation - Volume IV: Mobile Sources report](https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=P1009ZEK.PDF) by the U.S. Environmental Protection Agency (EPA), and on [Stettler et al. 2011](https://doi.org/10.1016/j.atmosenv.2011.07.012).

Payload mass fractions are based on the [Airline Industry Economic Performance](https://www.iata.org/economics/) data from the International Air Transport Association.

Aircraft ICAO type code designations are derived from [ICAO DOC 8643 - Aircraft Type Designators](https://www.icao.int/publications/DOC8643/Pages/Search.aspx).

The LTO cycle is modeled according to [Stettler et al. 2011](https://doi.org/10.1016/j.atmosenv.2011.07.012).

APU emissions are modeled according to the [ICAO Airport Air Quality Manual, 2<sup>nd</sup> edition](https://www.icao.int/publications/Documents/9889_cons_en.pdf), [Stettler et al. 2011](https://doi.org/10.1016/j.atmosenv.2011.07.012), and the [Handbook for Evaluating Emissions and Costs of APUs and Alternative Systems](https://www.nap.edu/catalog/22797) by the U.S. National Academies of Sciences, Engineering, and Medicine.

APU model specific emission indices are based on data from the [Expansion of Hong Kong International Airport into a Three-Runway System, Environmental Impact Assessment Report](https://www.epd.gov.hk/eia/english/alpha/aspd_651.html) by the Environmental Protection Department (EPD) and the Airport Authority Hong Kong.

Cruise flight levels were defined based on data provided by [Flightradar24 AB](https://www.flightradar24.com). Limits on cruise flight level for short flights were defined based on data from [Kim et al. 2005](https://www.faa.gov/about/office_org/headquarters_offices/apl/research/models/sage/media/FAA-EE-2005-01_SAGE-Technical-Manual.pdf).

NO<sub>x</sub>, CO, and HC emissions are calculated using the method described by [Baughcum et al. 1996](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19960038445.pdf) and [Kim et al. 2005](https://www.faa.gov/about/office_org/headquarters_offices/apl/research/models/sage/media/FAA-EE-2005-01_SAGE-Technical-Manual.pdf).

nvPM emissions are calculated from smoke number according to the FOA4.0 method described by [ICAO Airport Air Quality Manual, 2<sup>nd</sup> edition](https://www.icao.int/publications/Documents/9889_cons_en.pdf) and [Agarwal et al. 2019](https://doi.org/10.1021/acs.est.8b04060). The FOA3 method is described by [Wayson et al. 2009](https://doi.org/10.3155/1047-3289.59.1.91). The FOX method is described by [Stettler et al. 2003](https://doi.org/10.1021/es401356v).

Lateral inneficiency is modeled according to [Seymour et al. 2020](https://doi.org/10.1016/j.trd.2020.102528) and [Reynolds 2008](https://doi.org/10.2514/6.2008-8865).