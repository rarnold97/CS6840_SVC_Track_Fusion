from pathlib import Path
import typing
from dataclasses import dataclass
from collections import namedtuple
import pickle as pkl
from cachetools import cached

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.sampling import EpochsArray
from poliastro.constants import J2000
from poliastro.util import time_range
from poliastro.plotting import OrbitPlotter3D
from poliastro.czml.extract_czml import CZMLExtractor
from astropy import units
import numpy as np
from matplotlib import colors

from config.data_structures import StateVector
from config.settings import START_TIME, NUM_TIMES, CACHE_PATH, CROSS_START_TIME, CROSS_END_TIME, DATA_PATH
from config import LOGGER

__all__ = ["generate_data", "cache_state_data", "load_cached_state_data", "SatelliteStates", "export_cesium"]


@dataclass
class SatConditions:
    name: str
    color: typing.Tuple
    r: units.km
    v: units.km / units.s
    orbit: Orbit=None
    
    def __post_init__(self):
        self.orbit = Orbit.from_vectors(Earth, self.r, self.v, epoch=START_TIME)
        
@cached(cache={})
def get_cached_satellites()->typing.Tuple[Orbit]:
    return (
        SatConditions('sat_1', colors.to_rgba('red'), r=[703644.0, -5694458.0, 4624978.0]<<units.m, v=[-983.0, 4545.0, 5713.0]<<units.m/units.s),
        SatConditions('sat_2', colors.to_rgba('blue'), r=[802919.0, -5681310.0, 4624978.0]<<units.m, v=[-2317.0, 4279.0, 5626.0]<<units.m/units.s),
        SatConditions('sat_3', colors.to_rgba('green'), r=[604155.0, -5705871.0, 4624978.0]<<units.m, v=[374.0, 4626.0, 5626.0]<<units.m/units.s)
    )

SATELLITES = get_cached_satellites()

SatelliteStates = namedtuple('SatelliteStates', [sat.name for sat in SATELLITES], defaults=(None,) * len(SATELLITES))
def get_orbit_states(debug_mode=False)->SatelliteStates:
    
    frame: OrbitPlotter3D = None if not debug_mode else OrbitPlotter3D()
    if frame:
        frame.set_attractor(Earth)
        
    state_list = []

    for sat in SATELLITES:
        orb = sat.orbit
        ephem = orb.to_ephem(strategy=EpochsArray(
            epochs=time_range(start=CROSS_START_TIME, end=CROSS_END_TIME, periods=NUM_TIMES)))

        states_cart = ephem.sample(ephem.epochs)
        times_rel = np.empty(len(ephem.epochs), dtype=np.float64)
        times_utc_str = np.empty(len(ephem.epochs), dtype=object)
        times_datetime = np.empty(len(ephem.epochs), dtype=object)
        
        
        for ii, epoch in enumerate(ephem.epochs):
            times_rel[ii] = epoch.datetime.second + epoch.datetime.microsecond / 1e6
            times_utc_str[ii] = epoch.value
            times_datetime[ii] = epoch.datetime
        
        times_rel = times_rel * units.s

        state_vector = StateVector(time_rel_sec=times_rel, 
                                   time_utc_str=times_utc_str, 
                                   time_datetime=times_datetime,
                                   pos_x_km=states_cart.x,
                                   pos_y_km=states_cart.y,
                                   pos_z_km=states_cart.z
                                   )
        state_vector.cart2geod()
        state_vector.normalize()
        
        state_list.append(state_vector)
        
        if frame is not None:
            frame.plot(orb, label=sat.name)
        
    return SatelliteStates(*state_list)


def cache_state_data(states: SatelliteStates, cache_filename: Path = CACHE_PATH / "satellite_traj.pkl"):
    with open(cache_filename, 'wb') as file:
        pkl.dump(states, file)
        
def load_cached_state_data(cache_filename: Path = CACHE_PATH / "satellite_traj.pkl")->SatelliteStates:
    with open(cache_filename, 'rb') as file:
        return pkl.load(file)
    
def export_cesium(czml_filename:Path = DATA_PATH / "satellite_trajectories.czml"):
    
    extractor = CZMLExtractor(CROSS_START_TIME, CROSS_END_TIME, NUM_TIMES)
    for sat in SATELLITES:
        orb = sat.orbit
        extractor.add_orbit(orb, id_name=sat.name, label_text=sat.name, path_color=sat.color, path_width=2)
        
    if extractor:
        try:
            with open(czml_filename, 'w') as file:
                file.write(extractor.get_document().dumps())
            LOGGER.info(f'Wrote czml file {czml_filename}')
        except Exception as ex:
            LOGGER.error(f'Could not export cesium file: {czml_filename}.  Error msg: {ex}')
        

def generate_data()->SatelliteStates:
    cache_filename: Path = CACHE_PATH / "satellite_traj.pkl"

    LOGGER.info('Loading Satellite Simulated States ...')
    if not cache_filename.is_file():
        states = get_orbit_states()
        cache_state_data(states, cache_filename=cache_filename)
        LOGGER.info('Finished Satellite Simulation Calculations.')  
        return states
    else:
        LOGGER.info('Finished Satellite Simulation load.')  
        return load_cached_state_data(cache_filename=cache_filename)
      

if __name__ == "__main__":
    states = generate_data()
    export_cesium()
    pass