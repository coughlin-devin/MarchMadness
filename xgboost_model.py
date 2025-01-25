import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_csv(r"Data/Clean/clean_aggregate.csv")

data = df.loc[df['Year'] != 2024].copy()
holdout = df.loc[df['Year'] == 2024].copy()

y = data.WINS
X = data.drop(['School', 'Year', 'WINS'], axis=1)

y_holdout = holdout.WINS
X_holdout = holdout.drop(['School', 'Year', 'WINS'], axis=1)

# convert to numpy array to match X after scaling in cross validation loop
y = np.array(y)
y_holdout = np.array(y_holdout)

# TODO: grid search for hyper-parameter tuning reg_alpha, reg_lambda, max_depth, objective, subsample, colsample_bytree
# WARNING: Overfitting: reduce feature set? grid search hyperparameter tuning
# Set parameters
params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'verbosity': 0,
    'objective': 'reg:squaredlogerror',  # Use 'reg:squaredlogerror' for non-negative target values
    'booster': 'gbtree',
    'tree_method': 'hist',
    'gamma': 0.1,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 10,
    'reg_lambda': 5,
    'base_score': 0,
    'random_state': 2024,
    'importance_type': 'gain',
    'device': 'cpu',
    'eval_metric': ['rmse', 'rmsle'],  # Evaluation metric
    'early_stopping_rounds': 10
}

# TODO: hyper parameter tuning
# grid = GridSearchCV(estimator=bst, param_grid=param_grid, scoring='neg_root_mean_squared_log_error', cv=skf.split(X, y))
#
# # A parameter grid for XGBoost
# param_grid = {
#         'max_depth': [3, 4, 5, 6],
#         'learning_rate': [0.001, 0.01, 0.1],
#         'objective': ['reg:squarederror', 'reg:squaredlogerror'],
#         'gamma': [0, 0.1, 0.5],
#         'subsample': [3, 4, 5],
#         'colsample_bytree': [],
#         'reg_alpha': [],
#         'reg_lambda': [],
#         }

skf = StratifiedKFold(n_splits=26, shuffle=True, random_state=2024)

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    scaler = MinMaxScaler()
    X_train, X_test = scaler.fit_transform(X)[train_index], scaler.transform(X)[test_index]
    y_train, y_test = y[train_index], y[test_index]
    bst = xgb.XGBRegressor(**params)
    bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    results = bst.evals_result()
    train_rmse = results['validation_0']['rmse']
    test_rmse = results['validation_1']['rmse']
    epochs = len(train_rmse)
    plt.plot(range(epochs), train_rmse, label='Train RMSE')
    plt.plot(range(epochs), test_rmse, label='Test RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Fold {}'.format(fold))
    plt.legend()
    plt.show()


# NOTE: get_fscore uses weight by default, excludes features not used to split
# pd.Series(xgb_r.get_booster().get_fscore()).sort_values(ascending=False)[:50]
# pd.Series(xgb_r.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)[:50]
# imp = pd.DataFrame(xgb_r.feature_importances_, index=X_train.columns, columns=['GAIN'])
# imp.sort_values(by='GAIN', ascending=False)[:50]

# xgb.plot_importance(xgb_r)

# xgb.plot_tree(xgb_r, num_trees=2) # NOTE: requires graphviz library

# TODO: learn how to use R^2 value to determine how good model is
# R^2 value
bst.score(X_test, y_test)
pred = bst.predict(X_test)

# clf.save_model("clf.json")
