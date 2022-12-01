import plotly.express as px
import pandas as pd

from data_generation.simulate_orbits import SatelliteStates
from config.settings import PLOT_WIDTH, PLOT_HEIGHT


__all__ =["plot_intersect", "plot_distances"]


def plot_intersect(df: pd.DataFrame):
    # going to plot by satellite identifier
    fig = px.scatter_3d(df, x='x_km', y='y_km', z='z_km', color='satellite_id', symbol='radar_id')
    fig.update_layout(title='Satellite Cartesian Spatial Data [km]', autosize=False, width=PLOT_WIDTH, height=PLOT_HEIGHT)
    fig.show()

    return fig

def plot_distances(df: pd.DataFrame):
    fig = px.scatter(df, x='observation_number', y='distance', color='label')
    fig.update_layout(title='Observation Distances to Centroid [km]', autosize=False, width=PLOT_WIDTH, height=PLOT_HEIGHT)
    fig.show()
    
    return fig
    
    
if __name__=="__main__":
    plot_intersect()
    plot_distances()
    pass