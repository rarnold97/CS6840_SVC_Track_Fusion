from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy import units
from sklearn.preprocessing import MinMaxScaler

from data_generation.simulate_orbits import generate_data, SatelliteStates
from config.data_structures import RADAR_ERRORS

__all__ = ["manip_data"]


def manip_data()->pd.DataFrame:
    states = generate_data()
    assert len(states)
    ind = np.random.randint(len(RADAR_ERRORS._fields), size=states[0].shape)
    
    for i, field in enumerate(RADAR_ERRORS._fields):
        # assuming the states are uniformly dimensioned
        
        epsilon: float = getattr(RADAR_ERRORS, field, 0.0)
        noise = np.zeros(states[0].shape) * units.km
        
        noise_ind = ind == i
        num_err = noise_ind.sum()
        noise[noise_ind] = epsilon * np.random.randn(num_err)
        
        for state in states:
            # attempt to normalize
            state.normalize()
            state.range_km += noise
    
    sat_id, radar_id = pd.Series(dtype=str), pd.Series(dtype=str)
    time_rel, x, y, z = pd.Series(dtype=np.float64), pd.Series(dtype=np.float64), pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)
    
    for sat_name, state in zip(states._fields, states):
        sat_id = pd.concat((sat_id, pd.Series([sat_name]*max(state.shape), dtype=str)))
        rdr_id = np.array(['null']*max(state.shape), dtype=object)
        
        # record which radar made the observations for said satellite
        for i, radar_name in enumerate(RADAR_ERRORS._fields):
            rdr_id[ind==i] = radar_name
        radar_id = pd.concat((radar_id, pd.Series(rdr_id, dtype=str)))
        
        time_rel = pd.concat((time_rel, pd.Series(state.time_rel_sec.value, dtype=np.float64)))
        x = pd.concat((x, pd.Series(state.pos_x_km.value * state.range_km.value, dtype=np.float64)))
        y = pd.concat((y, pd.Series(state.pos_y_km.value * state.range_km.value, dtype=np.float64)))
        z = pd.concat((z, pd.Series(state.pos_z_km.value * state.range_km.value, dtype=np.float64)))
        
    frame_template = {
        "satellite_id": sat_id,
        "radar_id": radar_id,
        "time_rel_s": time_rel,
        "x_km": x,
        "y_km": y,
        "z_km": z
    }
    
    return pd.DataFrame(frame_template)


if __name__ == "__main__":
    df = manip_data()
    pass