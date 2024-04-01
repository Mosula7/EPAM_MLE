import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import sys
import os

def split_data(df: pd.DataFrame, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
    """
    returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not val_size:
        val_size = test_size / (1 - test_size)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

    X_train = train[train.columns.drop(target)]
    X_val = val[val.columns.drop(target)]
    X_test = test[test.columns.drop(target)]

    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

dataset_name = sys.argv[1]
df = pd.read_csv(os.path.join('data', dataset_name))
target = 'output'

X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target=target, test_size=.2)


params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'eval_metric': 'logloss',
    'n_estimators': 200,
    'verbose': -100
}

model = lgbm.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgbm.early_stopping(10, verbose=0)])

model_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
model.booster_.save_model(os.path.join('models', f'model_{model_name}.txt'))


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('train accuracy', accuracy_score(y_train, pred_train))
print('test accuracy', accuracy_score(y_test, pred_test))
