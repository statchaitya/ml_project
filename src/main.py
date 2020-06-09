import cross_validate
import parameters

MODEL = "ET"

model = parameters.MODELS[MODEL]
params_dict = parameters.PARAMS_DICT[MODEL]

# import data
df = cross_validate.import_data()

# Random forest
# search_rf_best_params = cross_validate.SearchBestParameters(df, 'target', MODEL, model, params_dict, cv=10, use_fold_column=False)
# search_rf_best_params._grid_search()

# Extra trees
search_et_best_params = cross_validate.SearchBestParameters(df, 'target', MODEL, model, params_dict, cv=10, use_fold_column=False)
search_et_best_params._grid_search()