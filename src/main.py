import cross_validate

df = cross_validate.import_data()

search_rf_best_params = cross_validate.SearchBestParameters(df, 'target', cross_validate.MODEL, cross_validate.model, cross_validate.params_dict, cv=10, use_fold_column=False)
search_rf_best_params._grid_search()