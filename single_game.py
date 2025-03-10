import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# TODO: incorporate this data into the neural network, maybe use for a reccurrent network?
df = pd.read_csv(r"Data/Clean/matchups.csv")

data = df.loc[df['Year'] != 2024]
holdout = df.loc[df['Year'] == 2024]

X = data.drop(['Year', 'School_1', 'School_2', 'Winner'], axis=1).reset_index(drop=True)
y = data.Winner.reset_index(drop=True)

params = {
    'n_estimators': 3000,
    'max_depth': 3,
    'learning_rate': 0.001,
    'verbosity': 0,
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'gamma': 0.1,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 2024,
    'importance_type': 'gain',
    'device': 'cpu',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 10
}

# A parameter grid for XGBoost
param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.001, 0.01, 0.1],
        'gamma': [0, 0.01, 0.1],
        'subsample': [0.5, 0.75, 1],
        'colsample_bytree': [0.5, 0.75, 1],
        'reg_alpha': [0, 0.5],
        'reg_lambda': [0, 0.5, 1],
        }

# TODO:
# gs = GridSearchCV(, param_grid)

K = 21
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=2024)
rskf = RepeatedStratifiedKFold(n_splits=K, n_repeats=5, random_state=2024)
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    bst = xgb.XGBClassifier(**params)
    bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    results = bst.evals_result()
    train_logloss = results['validation_0']['logloss']
    test_logloss = results['validation_1']['logloss']
    epochs = len(train_logloss)
    plt.plot(range(epochs), train_logloss, label='Train Logloss')
    plt.plot(range(epochs), test_logloss, label='Test Logloss')
    plt.xlabel('Epochs')
    plt.ylabel('Logloss')
    plt.title('Fold {}'.format(fold))
    plt.legend()
    plt.show()

bst.best_score
bst.best_iteration
bst.score(X_test, y_test)

preds = bst.predict(holdout[X.columns])
y_t = holdout['Winner']
accuracy_score(y_t, preds)
confusion_matrix(y_t, preds)

imp = pd.Series(bst.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
imp[:50]
