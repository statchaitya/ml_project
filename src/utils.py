import os
import pandas as pd
import parameters
import numpy as np
from sklearn import metrics

general_integer_parameters = ['param_max_depth', 'param_n_estimators']

def getBestParams(output_path):
    '''
    Gets best parameters from an output path containing outputs of gridsearchcvs of various models

    Input: directory path
    Output: best params
    '''
    best_params_output = {}

    for gridsearch_result in os.listdir(output_path):
        # Extract the required values
        result_model = gridsearch_result.split("_")[0]
        params_dict = parameters.PARAMS_DICT[result_model]
        params_names = list(params_dict.keys())
        params_names_prefixed = ["_".join(["param", i]) for i in params_names]
        result_df = pd.read_csv(os.path.join(output_path, gridsearch_result))
        best_params_df = result_df[result_df['rank_test_score'] == 1]
        best_params_df = best_params_df[params_names_prefixed]
        best_params_dict = best_params_df.to_dict(orient='records')[0]
        
        for key, value in best_params_dict.items():
            if key in general_integer_parameters:
                best_params_dict[key] = np.int64(value)
        print(type(best_params_dict['param_n_estimators']))
        best_params_dict = {i.split("_", 1)[1]:j for i, j in best_params_dict.items()}

        # Append to the output dict
        best_params_output[result_model] = best_params_dict
    
    return best_params_output



def getWorstParams(output_path):
    '''
    Gets worst parameters from an output path containing outputs of gridsearchcvs of various models

    Input: directory path
    Output: worst params
    '''
    worst_params_output = {}

    for gridsearch_result in os.listdir(output_path):
        # Extract the required values
        result_model = gridsearch_result.split("_")[0]
        params_dict = parameters.PARAMS_DICT[result_model]
        params_names = list(params_dict.keys())
        params_names_prefixed = ["_".join(["param", i]) for i in params_names]
        result_df = pd.read_csv(os.path.join(output_path, gridsearch_result))

        worst_params_df = result_df[result_df['rank_test_score'] == max(result_df['rank_test_score'])]
        worst_params_df = worst_params_df[params_names_prefixed]
        worst_params_dict = worst_params_df.to_dict(orient='records')[0]
        for key, value in worst_params_dict.items():
            if key in general_integer_parameters:
                worst_params_dict[key] = np.int64(value)
        print(type(worst_params_dict['param_n_estimators']))
        worst_params_dict = {i.split("_", 1)[1]:j for i, j in worst_params_dict.items()}

        # Append to the output dict
        worst_params_output[result_model] = worst_params_dict
    
    return worst_params_output


def evaluateMetrics(best_results_dict, worst_results_dict):
    '''
        Given a 2 results_dict of the form {'RF': {'predicted':[],
                                                 'actual':[] }}
        Outputs a dataframe of the form
        | model | params | metric1 | metric2 | metric3 | metric4 ...
        |  RF   | best   |  0.8    | 0.45    | etc..
    '''
    dfcols = ['model', 'paramtype', 'precision', 'recall', 'auc', 'acc', 'f1score']
