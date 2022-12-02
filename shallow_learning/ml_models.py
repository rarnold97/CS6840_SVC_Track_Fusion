from __future__ import annotations

import typing
from pathlib import Path
from collections import namedtuple
from cachetools import cached
import pickle as pkl
from tabulate import tabulate
from dataclasses import dataclass

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, hinge_loss, log_loss, f1_score
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
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

@dataclass
class Report:
    name: str
    accuracy: float
    loss: float
    f1_score: float
    
    def __add__(self, other: Report):
        self.accuracy += other.accuracy
        self.loss += other.loss
        self.f1_score += other.f1_score
        return self
    
    def __truediv__(self, val:float):
        self.accuracy /= val
        self.loss /= val
        self.f1_score /= val
        return self
        
def create_table_from_reports(reports: typing.Tuple):
    table = [[r.name, r.accuracy, r.loss, r.f1_score] for r in reports]
    return tabulate(table, headers=["Classifier", "Accuracy", "Loss [hinge-SVC/log-RF]", "Macro F1 Score"], tablefmt='fancy_grid')

def fit_svc(X_train: np.ndarray, y_train: np.ndarray, svc_cache_filename:Path) -> GridSearchCV:

    # electing to not analyze degree due to computational overhead incurred using poly kernel
    C_range: np.ndarray = [0.1, 1, 10, 100]
    gammas: np.ndarray = [1,0.1,0.01,0.001]
    
    model: GridSearchCV = None

    if not svc_cache_filename.is_file():
    
        # possibly separate this into two dictionaries for the polynomial kernel degree hyperparameter
        params_grid = {'C': C_range, 'gamma': gammas,'kernel': ['rbf', 'sigmoid'],
                    'decision_function_shape':['ovo'], 'random_state': [RANDOM_STATE], 'probability': [True]}
        
        ## TODO: determine a proper scoring key
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
        params_rf = {'n_estimators':[50, 100, 200], 'criterion': ['entropy']}
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

    clf: svm.SVC = fit_svc(X_bin, Y_bin)
    
    # after ada boost
    clf_boosted = AdaBoostClassifier(
        clf,
        n_estimators=500,
        learning_rate=0.1,
        algorithm='SAMME.R',
        random_state=RANDOM_STATE
    )
    
    model_boost = clf_boosted.fit(X_bin, Y_bin)

def analysis(clf: VotingClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, experiment_label:str):
    # retrieve the classifier predictions
    predictions = clf.predict(X_test)
    
    # compute accuracy
    acc_score = accuracy_score(predictions, y_test)    
    # apply transform to compare random forest to SVC

    enc = LabelEncoder().fit(clf.classes_)
    y_test_enc = enc.transform(y_test)
    
    pred_svc = clf.named_estimators_['svc'].predict(X_test)
    pred_rf = clf.named_estimators_['rf'].predict(X_test)
    
    scores = (accuracy_score(pred_svc, y_test_enc), accuracy_score(pred_rf, y_test_enc))
    
    winner_idx: int = np.argmax(scores)
    
    winner: str = 'SVC' if winner_idx == 0 else 'Random Forest'
    loser: str = 'SVC' if winner_idx == 1 else 'Random Forest'
    
    LOGGER.info(f"Optimal Model for {experiment_label} was: {winner}")
    
    loss_hinge = hinge_loss(pred_svc, clf.named_estimators_['svc'].decision_function(X_test), 
                            labels=enc.transform(enc.classes_))
    loss_log = log_loss(y_test_enc, clf.named_estimators_['rf'].predict_proba(X_test), 
                        labels=enc.transform(enc.classes_))
    
    # hinge loss for SVC, log loss for random forest
    loss_winner: float = loss_hinge if winner == 'SVC' else loss_log
    loss_loser: float = loss_log if loser == 'Random Forest' else loss_hinge
    
    # compute macro f1 scores
    f1_score_winner = f1_score(y_test, predictions, average='macro')
    pred_loser = pred_rf if winner == 'SVC' else pred_svc
    f1_score_loser = f1_score(y_test_enc, pred_loser, average='macro')
    
    # gather confusion matrices and display them
    confm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm, display_labels=clf.classes_)
    disp.plot()
    disp.ax_.set_title(f'{experiment_label}: Model={winner} - Accuracy Score: {acc_score*100:.2f}%')
    
    # create report metrics
    report_winner = Report(winner, scores[winner_idx], loss_winner, f1_score_winner)
    report_loser = Report(loser, scores[int(not winner_idx)], loss_loser, f1_score_loser)
    
    return report_winner, report_loser


def folds_experiment(X: pd.DataFrame, y: pd.DataFrame, radar_lbl: str)->str:
    n_splits = 10
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=RANDOM_STATE)
    run_num: int = 1
    
    svc_wins: int = 0
    rf_wins: int = 0

    avg_svc = Report('SVC Averaged Results', 0, 0, 0)
    avg_rf = Report('Random Forest Averaged Results', 0, 0, 0)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        LOGGER.info(f'~~~ Fitting and Analyzing Models for stratified split number: {run_num} ~~~')
        
        svc_cache_filename = CACHE_PATH / f"svc_experiment_run_{run_num}.pkl"
        rf_cache_filename = CACHE_PATH / f"rf_experiment_run_{run_num}.pkl"
        selected_model_cache_filename = CACHE_PATH / f"final_model_experiment_{run_num}.pkl"
        
        if not selected_model_cache_filename.is_file():
            # only recalculate if really needed
            if not svc_cache_filename.is_file():
                clf_svc = fit_svc(X_train, y_train, svc_cache_filename)
            else:
                with open(svc_cache_filename, 'rb') as file:
                    clf_svc = pkl.load(file)
                
            if not rf_cache_filename.is_file():
                clf_rf = fit_random_forest(X_train, y_train, rf_cache_filename)
            else:
                with open(rf_cache_filename, 'rb') as file:
                    clf_rf = pkl.load(file)
                
            estimators = [('svc', clf_svc), ('rf', clf_rf)]
            clf = VotingClassifier(estimators, voting='hard')
            clf.fit(X_train, y_train)
            with open(selected_model_cache_filename, 'wb') as file:
                pkl.dump(clf, file)
        else:
            with open(selected_model_cache_filename, 'rb') as file:
                clf = pkl.load(file)
        
        # the first report is the winner
        reports = analysis(clf, X_test, y_test, radar_lbl)
        
        svc_idx: int = 0 if reports[0].name =='SVC' else 1
        rf_idx: int = 0 if reports[0].name == 'Random Forest' else 1
        
        if reports[0].name == 'SVC':
            svc_wins += 1
        elif reports[0].name == 'Random Forest':
            rf_wins += 1
        
        #table_str = create_table_from_reports(reports)
        avg_svc = avg_svc + reports[svc_idx]
        avg_rf = avg_rf + reports[rf_idx]
        
        LOGGER.info(f'~~~ END of experiment run: {run_num}')
        run_num += 1
    
    avg_svc = avg_svc / float(n_splits)
    avg_rf = avg_rf / float(n_splits)

    print(create_table_from_reports((avg_svc, avg_rf)), flush=True)
    
    final_res_table = [
        ['Experiment:', radar_lbl],
        ['Total Times SVC Chosen', 'Total Times RF Chosen'],
        [svc_wins, rf_wins]
    ]
    
    return tabulate(final_res_table, headers='firstrow', tablefmt='fancy_grid')

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
    fold_tables=[]
    for radar in RADAR_NAMES:
        svc_cache_name = CACHE_PATH / f"svc_model_{radar}.pkl"
        rf_cache_name = CACHE_PATH / f"rf_model_{radar}.pkl"
        final_cache_name = CACHE_PATH / f"final_model_{radar}.pkl"
        clf_sets.append(apply_models(Data.df.loc[Data.df['radar_id']==radar],
                                         svc_cache_name, rf_cache_name, final_cache_name))
        # do the folds experiment
        if DO_ANALYSIS:
            # TODO: make this less redundant.  This is done in apply_models already, too much copying
            # this blunder was made due to having a fire under my ass as the deadline approached :)
            table = folds_experiment(Data.df.loc[Data.df['radar_id']==radar].drop(columns=['satellite_id', 'radar_id', 'time_rel_s']),
                             Data.df.loc[Data.df['radar_id']==radar].satellite_id, 
                             radar_lbl=radar
                             )
            fold_tables.append(table)
    
    if fold_tables:
        for table in fold_tables:
            print(table, flush=True)
            
    classifications = Classifiers(*clf_sets)
    
    fusion_results_cache_name = CACHE_PATH / "satellite_fusion_results.pkl"
    results = fusion(classifications, fusion_results_cache_name)
    return results, classifications

def final_result_analysis(results: pd.DataFrame, classifications: Classifiers)->None:
    # first do an analysis of SVC if desired in settings script
    if DO_ANALYSIS:
        # set contains a classifier, X_test, and y_test
        for clf_sets, label in zip(classifications, classifications._fields):
            # client needs to call plt.show() to render confusion matrices
            reports: typing.Tuple[Report] = analysis(*clf_sets, label)
            # display and cache table
            report_str = create_table_from_reports(reports)
            print(report_str, flush=True)
            

    if __SHOW_PLOTS__[0]:
        _ = generate_plots(results)


if __name__ == "__main__":
    _ = learn()
