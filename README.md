# Wright State University: CS6840 Final Project
**Authored by Ryan Arnold**

## Intro
The intent of this project is to demonstrate that a state vector machine classifier (SVC) can be used to identify target tracks in a scene with overlapping trajectories.  This is further used to implement a fusion algorithm, where the scene is observed by 2 different radars that contain different range errors.

The target tracks were assumed to be satellite orbites, in order to simplify the scenario.  The trajectory calculations were performed using the `poliastro` library.  After the trajectories were simulated, the tracks were split into different radar observations by randomly generating indeces to assign to each radar.  In this project, two radars were used, so half the data was randomly assigned to radar 1 and the other radar 2.  The range error for radar 1 was 1km and the range error for radar 2 was 5km.

To distinguish the radar tracks, training data was passed into a SVC model that was hyper trained using the `GridSearchCV` method from the `sklearn.model_selection` module.  The model was then compared against a `RandomForestClassifier` from the `sklearn.ensemble` module.  Finally, a voting classifier was used to pick between the SVC and RandomForest for each radar data subset.  In both cases, the SVC model was higher performing.

An additional capability this project contains is the ability to export the satellite trajectories to a `czml` file, which can be rendered in a cesium project.

## Primary Packages
- astropy
- czml3
- jupyter
- matplotlib
- numpy
- pandas
- plotly
- poliastro
- scikit-learn

## Usage
1. Install the provided wheel file
    - `pip install rarnold_cs6840_final_project-0.0.0-py3-none-any.whl`
    - If the wheel is not provided, do so by entering the following:
    - `python setup.py bdist_wheel -d <desired-wheel-dir>`
    - `pip install <desired-wheel-dir>/rarnold_cs6840_final_project-0.0.0-py3-none-any.whl`
    - once the wheel file is downloaded, the project can be ran via the command line utility:
    - `$ run_cs6840_final`

2. (Requires Step 1) Implement the project after installing the wheel.
    - jupyer notebook is a good way to visualize this.
    - *NOTE*: To enable or disable plotting directly through the `ml_models.learn()` method, set the following module variable:
    - `ml_models.__SHOW_PLOTS = [True/False]`
    - An example jupyer notebook file is provided in `./final_project/jupyter/rarnold_cs6840_final.ipynb`

2. Via the main script in `project_entry` module
    - run the `__main__.py` script, after downloading the packages in the PACKAGE_LIST tuple of the setup.py.
    - make sure that when using this method that the project root is appended to `PYTHON_PATH`

## Extra
- A vscode debugging environment workspace file is provided in `final_project/.vscode/cs6840_final_project.code-workspace`