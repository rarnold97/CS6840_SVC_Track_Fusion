import typing
from pathlib import Path
from collections import namedtuple
from cachetools import cached
import pickle as pkl

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import SatelliteStates
from config.settings import *
from config.logger import LOGGER
from shallow_learning.data_loader import DataSet
from visual import plot_fusion_results, plot_intersect, plot_distances


__all__=["learn", "generate_plots"]

__SHOW_PLOTS__ = [True]

# cache for faster load times
@cached(cache={})
def load_dataset():
    dataset = DataSet()
    dataset.scale_data()
    return dataset

# define global data structures that are constant
Data = load_dataset()
RADAR_NAMES = Data.df.radar_id.unique()
Classifiers = namedtuple('Classifiers', [*RADAR_NAMES])


def fit_svc(X_train: np.ndarray, y_train: np.ndarray, svc_cache_filename:Path) -> GridSearchCV:

    # electing to not analyze degree due to computational overhead incurred using poly kernel
    C_range: np.ndarray = [0.1, 1, 10, 100]
    gammas: np.ndarray = [1,0.1,0.01,0.001]
    
    model: GridSearchCV = None

    if not svc_cache_filename.is_file():
    
        # possibly separate this into two dictionaries for the polynomial kernel degree hyperparameter
        params_grid = {'C': C_range, 'gamma': gammas,'kernel': ['rbf', 'sigmoid'],
                    'decision_function_shape':['ovo'], 'random_state': [RANDOM_STATE], 'probability': [True]}
        
        ## :todo determine a proper scoring key
        grid_search = GridSearchCV(
            svm.SVC(),
            params_grid,
            cv=5,
            n_jobs=-1,  # run jobs in parallel to speed things up
            ## todo: investigate different scoring options
            #scoring='neg_mean_squared_error',
            return_train_score=True,
            verbose=3
        )
        
        model = grid_search.fit(X_train, y_train)
        
        with open(svc_cache_filename, 'wb') as file:
            pkl.dump(model, file)
        LOGGER.info(f'Cached SVC model to: {svc_cache_filename}')
    else:
        with open(svc_cache_filename, 'rb') as file:
            model = pkl.load(file)
        LOGGER.info(f'loded cached SVC model from: {svc_cache_filename}')
        
    # return optimal hyper parameters for caching purposes.
    return model


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, cache_filename: Path)->GridSearchCV:
    model: GridSearchCV = None
    if not cache_filename.is_file():
        params_rf = {'n_estimators':[50, 100, 200]}
        rf_grid_search = GridSearchCV(RandomForestClassifier(), params_rf, cv=5, n_jobs=-1, verbose=3)
        model = rf_grid_search.fit(X_train, y_train)
        with open(cache_filename, 'wb') as file:
            pkl.dump(model, file)
        LOGGER.info(f'Cached Random Forest model to {cache_filename}')
    else:
        with open(cache_filename, 'rb') as file:
            model = pkl.load(file)
        LOGGER.info(f'Loaded Random Forest Model from cache: {cache_filename}')

    return model

    
def test_with_toy_data():
    
    iris = datasets.load_iris()
    X = iris.data[:, :3]
    Y = iris.target
    
    X_bin = X[np.logical_or(Y==0, Y==1)]
    Y_bin = Y[np.logical_or(Y==0, Y==1)]
    
    cache_filename = CACHE_PATH/'toy_cache.json'
    clf: svm.SVC = None
    
    if not cache_filename.is_file():
        params = optimize_hyper_parameters(X_bin, Y_bin)
        params.cache(CACHE_PATH/'toy_cache.json')
    else:
        params = SvmParams.load_from_cache(cache_filename)
    
    svm_clf = params.svm_clf
    # before ada boost
    model = svm_clf.fit(X_bin, Y_bin)
    
    # after ada boost
    clf_boosted = AdaBoostClassifier(
        svm_clf,
        n_estimators=500,
        learning_rate=0.1,
        algorithm='SAMME.R',
        random_state=RANDOM_STATE
    )
    
    model_boost = clf_boosted.fit(X_bin, Y_bin)

def analysis(clf, X_test: pd.DataFrame, y_test: pd.DataFrame, label:str):
    predictions = clf.predict(X_test)
    acc_score = accuracy_score(predictions, y_test)
    LOGGER.info(f"Accuracy Score = {100*acc_score}%")
    
    confm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm, display_labels=clf.classes_)
    disp.plot()
    disp.ax_.set_title(f'{label} - Accuracy Score: {acc_score*100:.2f}%')

def apply_models(subset: pd.DataFrame, svc_cache_filename: Path, 
                 rf_cache_filename: Path, 
                 final_cache_filename: Path)->typing.Tuple[svm.SVC, pd.DataFrame, pd.DataFrame]:
    """apply the state vector machine classifier models to the dataset

    Parameters
    ----------
    subset : pd.DataFrame
        radar filtered dataset

    Returns
    -------
    svm.SVC
        the fitted SVC model to be later analyzed
    """ 
    clf: VotingClassifier = None
    
    # retain only the spatial data
    X = subset.drop(columns=['satellite_id', 'radar_id', 'time_rel_s'])
    y = subset.satellite_id
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    if not final_cache_filename.is_file():
        clf_svc = fit_svc(X_train, y_train, svc_cache_filename)
        clf_rf = fit_random_forest(X_train, y_train, rf_cache_filename)
        
        estimators = [('svc', clf_svc), ('rf', clf_rf)]
        clf = VotingClassifier(estimators, voting='hard')
        clf.fit(X_train, y_train)
        
        with open(final_cache_filename, 'wb') as file:
            pkl.dump(clf, file)
        LOGGER.info(f'cached final model to: {final_cache_filename}')
    else:
        with open(final_cache_filename, 'rb') as file:
            clf = pkl.load(file)
        LOGGER.info(f'Loaded final model from: {final_cache_filename}')
    # using the 0 value because it is assumed apriori that the dataset should be one radar only
    LOGGER.info(f'Model Fitting complete for subset: {subset.radar_id.values[0]}')
    return clf, X_test, y_test


def fusion(classifiers: Classifiers, cache_filename: Path)->pd.DataFrame:
    results: pd.DataFrame = None
    
    if not cache_filename.is_file():
        result_cols = Data.df.columns.to_list()
        result_cols.append('predict')
        results = pd.DataFrame(columns=result_cols)
        
        results[Data.df.columns.to_list()] = Data.df.copy()
        
        for radar in RADAR_NAMES:
            subset = results.loc[results['radar_id']==radar]
            X = subset.drop(columns=['satellite_id', 'radar_id', 'time_rel_s', 'predict'])
            clf= getattr(classifiers, radar)[0]
            results.loc[results['radar_id']==radar, 'predict'] = clf.predict(X)
        results.to_pickle(cache_filename)
        LOGGER.info(f'cached results to: {cache_filename}')
    else:
        results = pd.read_pickle(cache_filename)
        LOGGER.info(f'Loaded results from : {cache_filename}')
        
    fused_accuracy = accuracy_score(results.satellite_id, results.predict)
    LOGGER.info(f'Fused Model Accuracy: {fused_accuracy*100:.2f}%')
    
    return results

def generate_plots(results: pd.DataFrame):
    return (plot_intersect(Data.df), plot_distances(Data.distances_to_cetroid),
        plot_fusion_results(results=results))

def learn()->pd.DataFrame:
    
    clf_sets=[]
    for radar in RADAR_NAMES:
        svc_cache_name = CACHE_PATH / f"svc_model_{radar}.pkl"
        rf_cache_name = CACHE_PATH / f"rf_model_{radar}.pkl"
        final_cache_name = CACHE_PATH / f"final_model_{radar}.pkl"
        
        clf_sets.append(apply_models(Data.df.loc[Data.df['radar_id']==radar],
                                         svc_cache_name, rf_cache_name, final_cache_name))
        
    classifications = Classifiers(*clf_sets)
    
    fusion_results_cache_name = CACHE_PATH / "satellite_fusion_results.pkl"
    results = fusion(classifications, fusion_results_cache_name)
    
    # first do an analysis of SVC if desired in settings script
    if DO_ANALYSIS:
        # set contains a classifier, X_test, and y_test
        for clf_sets, label in zip(classifications, classifications._fields):
            # client needs to call plt.show() to render confusion matrices
            analysis(*clf_sets, label)
        
    if __SHOW_PLOTS__[0]:
        _ = generate_plots(results)
        
    return results


if __name__ == "__main__":
    _ = learn()
