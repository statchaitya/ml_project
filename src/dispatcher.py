from sklearn import ensemble

MODELS = {
    "RF": ensemble.RandomForestClassifier(n_jobs=-1),
    "ET": ensemble.ExtraTreesClassifier(n_jobs=-1),
    "GBM": ensemble.GradientBoostingClassifier()
}

PARAMS_DICT = {
    "RF": {'n_estimators': [100, 300, 500],
    "max_depth": [2, 5, 8, 11],
    "max_features": [0.3, 0.4, 0.5, 0.6]}
}