from dataclasses import dataclass
import typing 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_generation import SatelliteStates

from data_generation.format_data import manip_data
from config import LOGGER

@dataclass
class DataSet:
    df: pd.DataFrame = None
    distances_to_cetroid: pd.DataFrame = None
    scaled: pd.DataFrame = None
    
    min_intersect_idx: int = 0
    max_intersect_idx: int = 0
    
    num_observations:int = 0

    def __post_init__(self):
        if self.df is None:
            self.df = manip_data()
        # using satellite 1 as the reference point
        sat_names = self._get_sat_names()
        
        l = int(max(self.df.shape) / len(sat_names))
        self.num_observations = l
        
        n = max(self.df.shape)
        distances = np.empty(n, dtype=np.float64)
        keys = np.empty(n, dtype=object)
        observations = np.tile(np.arange(0, l, 1, dtype=int), len(sat_names))

        try:
            for i, sat_name in enumerate(sat_names):
                data = self._filter_by_satellite(sat_name)
                square = lambda key: np.square(data[key].values - data[key].mean()).astype(np.float64)
                distances[i*l: (i+1)*l] = np.sqrt(square('x_km') + square('y_km') + square('z_km'))
                keys[i*l: (i+1)*l] = np.tile(np.array([sat_name], dtype=object), l)
        except AssertionError as err:
            LOGGER.error(f'Could not calculate distances. Error msg: {err}')
        
        self.distances_to_cetroid = pd.DataFrame.from_dict(
            {'observation_number': observations, 'label': keys, 'distance': distances})
        
        self._get_intersect_region()
        
    def _filter_by_satellite(self, filter: str)->pd.DataFrame:
        return self.df.loc[self.df['satellite_id']==filter]
    
    def _get_sat_names(self)->np.ndarray:
        return self.df.satellite_id.unique()
    
    def scale_data(self):
        scaler = MinMaxScaler()
        self.scaled = scaler.fit_transform(self.df[['x_km', 'y_km', 'z_km']])
        
    def _get_intersect_region(self):
        idx_total: int = 0
        
        names = self._get_sat_names()
        for sat_name in names:
            subset = self.distances_to_cetroid.loc[self.distances_to_cetroid['label']==sat_name]
            idx_total += subset.distance.argmin()

        mean_idx = np.around(idx_total/len(names)).astype(int)
        half_window = np.around(self.num_observations/8).astype(int)
        
        # use a quarter of the dataset where the min distances are (e.g., intersects)
        self.min_intersect_idx = max(mean_idx - half_window, int(0))
        self.max_intersect_idx = min(mean_idx + half_window, int(self.num_observations-1))
        
        
if __name__ == "__main__":
    data = DataSet()
    pass