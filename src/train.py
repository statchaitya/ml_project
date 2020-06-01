from sklearn import ensemble
import sys
import pandas as pd
import numpy as np
from . import dispatcher
from sklearn import metrics

FOLD = int(sys.argv[1])
NUM_TREES = int(sys.argv[2])
MODEL = sys.argv[3]

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":

    df = pd.read_csv("input/train_folds.csv")
    df.drop('duration', axis=1, inplace=True)
    
    df['target'] = np.where(df.y == "yes", 1, 0)
    df.drop('y', axis=1, inplace=True)
    df = pd.get_dummies(df)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df['kfold'] == FOLD]

    ytrain = train_df['target']
    train_df.drop('target', axis=1, inplace=True)

    yvalid = valid_df['target']
    valid_df.drop('target', axis=1, inplace=True)

    # train the model
    model = dispatcher.MODELS[MODEL]
    model.fit(train_df, ytrain)
    preds = model.predict(valid_df)

    # print metric
    print(metrics.roc_auc_score(yvalid, preds))




