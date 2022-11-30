from pathlib import Path
import sys

import matplotlib.pyplot as plt

from data_generation import SatelliteStates
from shallow_learning.ml_models import learn
from data_generation import export_cesium
from config.settings import DATA_PATH, DO_ANALYSIS
from config import LOGGER


def run():
    try:
        czml_filename: Path = DATA_PATH / 'satellite_trajectories.czml'
        if not czml_filename.is_file():
            export_cesium()

        _ = learn()
        # execute show here to render confusion matrices
        plt.show()

        LOGGER.info('Project Executed Successfully!  Now Exiting.')
    except Exception as ex:
        LOGGER.error(f'Exception Occured in final project code: {ex}')
        sys.exit(-1)
    
    sys.exit(0)
    

if __name__ == "__main__":
    run()