from dataclasses import dataclass, fields, asdict
import json
import typing
from pathlib import Path
from sklearn import svm
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.units.quantity import Quantity
from astropy import units
from datetime import datetime
from collections import namedtuple

from config.settings import CACHE_PATH, RANDOM_STATE

__all__ = ['SvmParams', 'RADAR_ERRORS']

# had weird issues making this a dataclass for some reason
RadarErrors = namedtuple('RadarErrors', ['radar_1', 'radar_2'])
RADAR_ERRORS = RadarErrors(1*units.km, 5*units.km)


@dataclass
class SvmParams:
    kernel: str = 'rbf'
    C: float = 1.0
    decision_function_shape: str = 'ovo'
    gamma: float = 1.0
    
    def __init__(self, best_params: dict = None):
        # error handle
        
        if best_params is not None:
            for field in fields(SvmParams):
                assert field.name in best_params.keys(), f'{field.name} not found in optimized hyper parameters ...'

            self.C = best_params['C']
            self.gamma = best_params['gamma']
            self.kernel = best_params['kernel']
    
    def cache(self, cache_filename: Path = CACHE_PATH/"SvmOptimalParams.json"):
        with open(cache_filename, "w") as file:
            json.dump(asdict(self), file)
            
    
    @classmethod
    def load_from_cache(cls, cache_filename: Path = CACHE_PATH/"SvmOptimalParams.json"):
        try:
            with open(cache_filename, 'r') as file:
                data_dict: dict = json.load(file)
                return cls(data_dict)
        except Exception as err:
            raise IOError(f'could not read cache file due to following error: {err}')
    
    @property
    def svm_clf(self)->svm.SVC:
        return svm.SVC(
            C=self.C,
            kernel=self.kernel,
            decision_function_shape=self.decision_function_shape,
            gamma=self.gamma,
            random_state=RANDOM_STATE,
            probability=True
        )
    

@dataclass
class StateVector:
    time_rel_sec: Quantity
    time_utc_str: np.ndarray
    time_datetime: np.ndarray

    pos_x_km: Quantity
    pos_y_km: Quantity
    pos_z_km: Quantity
    
    range_km: Quantity = None
    shape: typing.Tuple[int] = (0,)

    lat_deg: Quantity = np.empty(0, dtype=np.float64) << units.deg
    lon_deg: Quantity = np.empty(0, dtype=np.float64) << units.deg
    alt_km: Quantity = np.empty(0, dtype=np.float64) << units.km
    
    def __post_init__(self):
        self.range_km = np.ones(self.pos_x_km.shape)
        self.shape = self.pos_x_km.shape
        
    def _is_normalized(self)->bool:
        # this is a very hacky way of checking this, needs to be adjusted and audited in future work
        rng = self.range_km.value if type(self.range_km) is Quantity else self.range_km
        if not rng.any():
            return False

        return not np.isclose(rng.sum(), float(max(rng.shape)))
    
    def cart2geod(self):
        assert self.pos_x_km.shape == self.pos_y_km.shape and self.pos_x_km.shape == self.pos_z_km.shape
        
        dims = self.pos_x_km.shape
        
        if self.lat_deg.shape != dims:
            self.lat_deg = np.empty(shape=dims, dtype=np.float64) * units.deg
        if self.lon_deg.shape != dims:
            self.lon_deg = np.empty(shape=dims, dtype=np.float64) * units.deg
        if self.alt_km.shape != dims:
            self.alt_km = np.empty(shape=dims, dtype=np.float64) * units.m
    
        for i, pos_geoc in enumerate(zip(self.pos_x_km, self.pos_y_km, self.pos_z_km)):
            px, py, pz = pos_geoc
            loc = EarthLocation.from_geocentric(px, py, pz)
            geod = loc.to_geodetic()
            self.lat_deg[i] = geod.lat
            self.lon_deg[i] = geod.lon
            self.alt_km[i] = geod.height
            
    def normalize(self):
        if not self._is_normalized():
            self.range_km = np.sqrt(np.square(self.pos_x_km) + np.square(self.pos_y_km) + np.square(self.pos_z_km))
            self.pos_x_km /= self.range_km.value
            self.pos_y_km /= self.range_km.value
            self.pos_z_km /= self.range_km.value
    
    def denormalize(self):
        if self._is_normalized():
            self.pos_x_km *= self.range_km.value
            self.pos_y_km *= self.range_km.value
            self.pos_z_km *= self.range_km.value
            self.range_km = np.ones(self.shape) * units.km
        
    def get_zulu_time(self, index:int=0):
        assert index >=-1 and index < max(self.shape)
        return self.time_datetime[index].strftime(r'%Y-%m-%dT%H:%M:%SZ')