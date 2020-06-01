import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("input/bank-additional-full.csv", sep=";")
    df['kfold'] = -1

    # df.sample returns 'frac' proportion of random rows
    # frac=1 will return a shuffled dataset
    # We also reset index and do not force it as a column in df which is the natural behavior of reset_index()
    df = df.sample(frac=1).reset_index(drop=True)

    # StratifiedKFold object who method creates a generator yielding appropriate indices of train/val sets
    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=123)
    
    # enumerate() takes an iterator, returns an additional number of the number of current iteration (fold) in our case
    # Each iteration yeilds a set of indices in train_idx and val_idx
    # Since val_idx indices are unqiue, we accomodate them in 1 column 'kfold'
    # Train indices can then be automatically inferred while modelling process
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df.y.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    # Saving data back to the folder
    df.to_csv("input/train_folds.csv", index=False)