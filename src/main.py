import train

df = train.import_data()

search_rf_best_params = train.SearchBestParameters(df, 'target', train.model, train.params_dict, use_fold_column=False)
search_rf_best_params._grid_search()