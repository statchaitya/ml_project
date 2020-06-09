from sklearn import ensemble

MODELS = {
    "RF": ensemble.RandomForestClassifier(n_jobs=-1),
    "ET": ensemble.ExtraTreesClassifier(n_jobs=-1),
    "GBM": ensemble.GradientBoostingClassifier()
}

PARAMS_DICT = {
    "RF": {'n_estimators': [200, 400, 700, 1000],
    "max_depth": [2, 4, 6, 8, 10, 12],
    "max_features": [0.3, 0.4, 0.5, 0.6]}
}