import plotly.express as px
import pandas as pd

from data_generation.simulate_orbits import SatelliteStates


__all__ =["plot_intersect", "plot_distances"]


def plot_intersect(df: pd.DataFrame):
    # going to plot by satellite identifier
    fig = px.scatter_3d(df, x='x_km', y='y_km', z='z_km', color='satellite_id', symbol='radar_id',
                        title='Satellite Cartesian Spatial Data [km]')
    fig.show()

    return fig

def plot_distances(df: pd.DataFrame):
    fig = px.scatter(df, x='observation_number', y='distance', color='label',
                     title='Track Distances from Centroid')
    fig.show()
    
    return fig
    
    
if __name__=="__main__":
    plot_intersect()
    plot_distances()
    pass