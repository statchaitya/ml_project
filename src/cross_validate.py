from sklearn import ensemble
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
import os


def import_data(project_home_path):
    '''
    Imports data and does a few manipulations
    * function - i.e. function which needs to be modified a bit when dataset changes
    '''
    df = pd.read_csv(os.path.join(project_home_path, "input/train_folds.csv"))
    df['target'] = np.where(df.y == "yes", 1, 0)
    df.drop('y', axis=1, inplace=True)
    return(df)

def import_test(project_home_path):
    '''
    Imports test data
    * function
    '''
    test = pd.read_csv(os.path.join(project_home_path, "input/test.csv"))
    test['target'] = np.where(test.y == 'yes', 1, 0)
    test.drop('y', axis=1, inplace=True)
    return(test)

def data_pre_processing(df):
    '''
    Preprocessing for train and test sets
    Removal of columns, imputing missing values etc..
    '''
    df.drop('kfold', axis=1, inplace=True)
    df = pd.get_dummies(df)
    return(df)

class SearchBestParameters():
    '''
    Only works for binary classifiers right now

    Input: df, target, model name, parameter list for gridsearch
    Output: best cv score, best params
    '''
    def __init__(self, df, target_name, MODEL, model, param_dict, cv, use_fold_column):

        self._df = df
        self._target_name = target_name
        self._model = model
        self._param_dict = param_dict
        self._use_fold_column = use_fold_column
        self._cv = cv
        self._MODEL = MODEL

        self._data_preprocessed = False

        self._df_preprocessed = None
        self._results_dict = None
        self._results_df = None

        # Class invariants
        if not isinstance(self._df, pd.DataFrame):
            raise ValueError("First arguement should be a pandas dataframe")

        

    def _data_preprocessing(self):
        '''
        Manipulates the features of the object and returns a new copy
        As of now we are only dropping 1 feature - duration
        '''
        if not self._data_preprocessed:
            
            if self._use_fold_column == False:
                self._df.drop('kfold', axis=1, inplace=True)
            
            self._df.drop('duration', axis=1, inplace=True)
            self._df = pd.get_dummies(self._df)

            self._data_preprocessed = True

            print(self._df.shape)
            return(self._df)

    def _grid_search(self):
        
        self._df_preprocessed = self._data_preprocessing()

        if not isinstance(self._df_preprocessed, pd.DataFrame):
            print("PreProcessing false")
            print(f"Type of df_preprocessed is {type(self._df_preprocessed)}")

        y_train = self._df_preprocessed[self._target_name]
        X_train = self._df_preprocessed.drop(self._target_name, axis=1)

        results_string = "_".join([self._MODEL, "CV", str(self._cv)])
        if not os.path.exists(os.path.join("cv_output", results_string+".csv")):
            clf = model_selection.GridSearchCV(self._model, self._param_dict, cv=model_selection.StratifiedKFold(n_splits=self._cv), scoring='roc_auc').fit(X_train, y_train)
            results_df = pd.DataFrame(clf.cv_results_)
            results_df.to_csv(os.path.join("cv_output", results_string+".csv"))
        else:
            raise Exception("Cross validation already complete for these set of inputs, params and model")







