from sklearn import ensemble

MODELS = {
    "RF": ensemble.RandomForestClassifier(n_jobs=-1),
    "ET": ensemble.ExtraTreesClassifier(n_jobs=-1),
    "GBM": ensemble.GradientBoostingClassifier()
}