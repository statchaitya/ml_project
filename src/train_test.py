import os
from sklearn import ensemble
import parameters
from cross_validate import import_data, import_test, data_pre_processing
from utils import getBestParams, getWorstParams, evaluateMetrics

project_home = "C:/DataScience/Github/ml_project"

# Get train and test sets
train = data_pre_processing(import_data(project_home))
test = data_pre_processing(import_test(project_home))

train_columns = list(train.columns)
test_columns = list(test.columns)

# For those categories which weren't in test, create 0 columns
# Best approach can be taken later
missing_features = [i for i in train_columns if i not in test_columns]
for mf in missing_features:
    test[mf] = 0

train_y = train['target']
train.drop('target', axis=1, inplace=True)

# Most times test will not have a target
# Need to modify the code for such cases
test_y = test['target']
test.drop('target', axis=1, inplace=True)

model_keys = ['ET', 'RF']

# Best params first
test_results_best = {model:{} for model in model_keys}
test_results_worst = {model:{} for model in model_keys}

for model in model_keys:
    # extract best params
    best_params = getBestParams(os.path.join(project_home, "cv_output"))
    rf_best = parameters.MODELS[model]
    rf_best.set_params(**best_params[model])
    rf_best.fit(train, train_y)
    test_results_best[model]['predicted'] = rf_best.predict(test)
    test_results_best[model]['predicted_proba'] = rf_best.predict_proba(test)
    test_results_best[model]['actual'] = test_y
    
    # extract worst params
    worst_params = getWorstParams(os.path.join(project_home, "cv_output"))
    rf_worst = parameters.MODELS[model]
    rf_worst.set_params(**worst_params[model])
    rf_worst.fit(train, train_y)
    test_results_worst[model]['predicted'] = rf_worst.predict(test)
    test_results_worst[model]['predicted_proba'] = rf_worst.predict_proba(test)
    test_results_worst[model]['actual'] = test_y


eval_results = evaluateMetrics(test_results_best, test_results_worst)
eval_results.to_csv(os.path.join(project_home, "cv_output", "results_df.csv"), index=False)