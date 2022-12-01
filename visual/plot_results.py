from pathlib import Path

import pandas as pd
import plotly.express as px

from config.settings import PLOT_WIDTH, PLOT_HEIGHT

__all__=["plot_fusion_results"]


def plot_fusion_results(results: pd.DataFrame):
    
    plot_data = pd.DataFrame(columns=['x_km', 'y_km', 'z_km', 'result'])
    plot_data[['x_km', 'y_km', 'z_km']] = results[['x_km', 'y_km', 'z_km']].copy()
    plot_data.loc[results['predict']==results['satellite_id'], 'result'] = 'Correctly Classified'
    plot_data.loc[results['predict']!=results['satellite_id'], 'result'] = 'Incorrectly Classified'
    
    fig = px.scatter_3d(plot_data, x='x_km', y='y_km', z='z_km', color='result',
                        color_discrete_sequence=['green', 'crimson'])
    fig.update_layout(title='Fused Classification Results', autosize=False, width=PLOT_WIDTH, height=PLOT_HEIGHT)
    fig.show()
    
    return fig
    
    
if __name__ == "__main__":
    plot_fusion_results()
    