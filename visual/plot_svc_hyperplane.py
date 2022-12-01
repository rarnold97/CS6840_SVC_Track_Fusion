from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn import svm
import plotly.express as px
from config.settings import PLOT_WIDTH, PLOT_HEIGHT

__all__ = ["plot_hyperplane"]


def plot_hyperplane(clf: svm.SVC, dataset: DataFrame)->px:
    
    fig = px.scatter_3d(x=clf.support_vectors_[:,0], y=clf.support_vectors_[:,1], z=clf.support_vectors_[:,2])
    fig.show()
    return fig
    