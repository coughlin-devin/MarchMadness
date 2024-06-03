# dataset from https://www.kaggle.com/datasets/nishaanamin/march-madness-data?select=KenPom+Barttorvik.csv

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# IDEA: weight training by single score and double score, ex. x/64 and /192
# IDEA: play with using target values rounds as 0-6, 1-7, and 64,32,16,8,4,2,1
# IDEA: create a different model for the first round, second round, etc.
# IDEA: try a model to predict the probability of a single game (logistic regression)

#%% markdown
# ### Preprocessing

#%% codecell
# set random state
def set_random_seed(seed):
    """Set random seed for all potential libraries"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def clean(data, year, first_four_out):
    """Clean data to remove round of 68 losers and seperate current year tournament teams to predict."""
    # remove first four losers
    data = data.loc[data['ROUND'] != 68]

    # remove round of 68 play-in losers
    data = data.loc[~((data['YEAR'] == 2024) & (data['TEAM'].isin(first_four_out)))]

    # hold out 2024 data for final prediction
    current_year = data.loc[data['YEAR'] == year].drop('ROUND', axis=1)
    data = data.loc[data['YEAR'] != 2024]

    # change ROUND to round number 64->1, 32->2, 16->3, etc...
    round_map = {
        64:1,
        32:2,
        16:3,
        8:4,
        4:5,
        2:6,
        1:7
    }
    data.ROUND = [round_map[x] for x in data.ROUND]

    return (data, current_year)

#%% markdown
# ### Train-Test Split

#%% codecell
def train_test_split(year):
    """Split data into training and test sets. Test set is one year worth of NCAA Tournament data"""
    X_train = data.loc[data['YEAR'] != year]
    y_train = data.loc[data['YEAR'] != year, 'ROUND']
    X_test = data.loc[data['YEAR'] == year]
    y_test = data.loc[data['YEAR'] == year, 'ROUND']

def normalize(data):
    # normalize continuos data
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized

#%% markdown
# ### Feature Selection

#%% codecell
# # correlation matrix
# data, cy = clean(data, 2024, first_four_out)
#
# cm = data.loc[data['YEAR'] != 2024].corr()
# cm.style.background_gradient(cmap='coolwarm').set_precision(2)

# # # group 2024 teams by seed number
# gb_seed24 = data24.groupby('SEED').mean()
# gb_seed24.style.background_gradient(cmap='coolwarm').set_precision(2)
# #
# # # group teams by round made in tournament
# gb_round = data.drop('YEAR', axis=1).groupby('ROUND').mean()
# gb_round.style.background_gradient(cmap='coolwarm').set_precision(2)

#%% markdown
# ### Neural Network

# TODO: experiment with smarter model, maybe not neural network, maybe change to trying to predict each game tournament style rather than expected rounds and comparing?
#%% codecell
class Net(nn.Module):
    """docstring for Net."""

    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(21, 16),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 8),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        logits = self.fully_connected(x)
        return logits

#%% markdown
# ### Cross Validation

#%% codecell
def train_loop(X, y, model, loss_fn, optimizer):
    model.train()
    logits = model(X)
    loss = torch.sqrt(loss_fn(logits, y)) # NOTE this is RMSE not MSE

    # backpropogation
    optimizer.zero_grad() # zero gradients so they don't add up
    loss.backward()
    optimizer.step()
    return loss

def test_loop(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        loss = torch.sqrt(loss_fn(predictions, y))
    return (predictions, loss)

def fold(data, batch_size, f):
    """Split training data into training and validation folds."""
    X_train = data[hcf].drop("ROUND", axis=1)
    y_train = data['ROUND']

    # training fold
    X = normalize(X_train.drop(X_train.index[f:f+batch_size], axis=0))
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y = y_train.drop(X_train.index[f:f+batch_size], axis=0)
    y = torch.tensor(y.values, dtype=torch.float32, requires_grad=True)
    y = torch.unsqueeze(y,1)

    # validation fold
    X_validation = normalize(X_train.iloc[f:f+batch_size])
    X_validation = torch.tensor(X_validation, dtype=torch.float32, requires_grad=False)
    y_validation = y_train.iloc[f:f+batch_size]
    y_validation = torch.tensor(y_validation.values, dtype=torch.float32, requires_grad=False)
    y_validation = torch.unsqueeze(y_validation,1)

    return (X, y, X_validation, y_validation)

# TODO: ensemble model to reduce variance, look into rotating test year or doing cross validation in a smarter way
def cross_validation():
    pass

def test(X_test, y_test):
    """Evaluate model on test set data."""
    X = normalize(X_test)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
    y = y_test.values
    y = torch.tensor(y, dtype=torch.float32, requires_grad=False)
    y = torch.unsqueeze(y,1)
    test_predictions, test_rmse = test_loop(X, y)
    return (test_predictions, test_rmse)

# WARNING: broken
def plot_loss(train_errors, validation_erros, test_errors):
    plt.plot(range(K*epochs), train_error, label="Training RMSE")
    plt.plot(range(K*epochs), validation_error, label="Validation RMSE")
    plt.plot(range(K*epochs), rmse*torch.ones(K*epochs), label="Test RMSE")
    plt.xlabel("Cross Folds")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

# TODO save best model based on test data before prediction data

def predict(model, data, hcf):
    """Predict the number of March Madness wins for each team."""
    X = data.drop('TEAM', axis=1)[hcf]
    X = normalize(X)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        model.eval()
        nn_predictions = model(X)
    nn_predictions = [p.item() for p in nn_predictions]
    prediction = data
    prediction.loc[:,'PRED'] = nn_predictions
    prediction = prediction.sort_values(by='PRED', ascending=False)
    return prediction

# WARNING: broken
def main():
    set_random_seed(2024)
    # import data
    kenpom = pd.read_csv("Data_1.0/KenPom Barttorvik.csv")
    # public = pd.read_csv("Data_1.0/Public Picks.csv")
    # ap = pd.read_csv("Data_1.0/Preseason Votes.csv")
    # shooting = pd.read_csv("Data_1.0/Shooting Splits.csv")

    first_four_out = ["Howard", "Virginia", "Boise St.", "Montana St."]
    data, current_year = clean(kenpom, 2024, first_four_out)

    # reduce colinearity among features
    # combine average and effective height metrics
    data.loc[:, 'HGT'] = data['AVG HGT'] + data['EFF HGT']
    current_year.loc[:, 'HGT'] = current_year['AVG HGT'] + current_year['EFF HGT']

    # use PCA components
    pca = PCA(n_components=10, random_state=2024)
    features = ['SEED', 'WIN%', 'EFG%', 'EFG%D', 'FTR', 'FTRD', 'TOV%', 'OREB%', '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'BLKED%', 'HGT', 'EXP', 'PPPO', 'PPPD']
    # create principal components
    pca_data = normalize(data[features])
    components = pca.fit_transform(pca_data)

    # convert to dataframe
    component_names = [f"PC{i+1}" for i in range(10)]
    components = pd.DataFrame(components[:,:10], columns=component_names)
    data.reset_index(drop=True, inplace=True)
    data = pd.concat((data,components), axis=1)
    components_current_year = pca.transform(normalize(current_year.drop(['TEAM', 'AVG HGT', 'EFF HGT'], axis=1)[features]))
    components_current_year = pd.DataFrame(components_current_year[:,:10], columns=component_names)
    components_current_year.index = current_year.index
    current_year = pd.concat((current_year,components_current_year), axis=1)

    features.extend(component_names)
    # select fewer more highly correlated features to ROUND
    hcf = ['ROUND', 'SEED', 'WIN%', 'EFG%', 'EFG%D', 'FTR', 'FTRD', 'TOV%', 'OREB%', '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'BLKED%', 'HGT', 'EXP', 'PPPO', 'PPPD', 'PC1', 'PC3', 'PC5']

    K = len(data.YEAR.unique()) - 1
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    device = "cpu"
    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_error = []
    validation_error = []
    for f in range(K):
        X, y, X_validation, y_validation = fold(data[hcf], batch_size, f*batch_size)

        for ep in range(epochs):
            train_rmse = train_loop(X, y, model, loss_fn, optimizer)
            predictions, validation_rmse = test_loop(X_validation, y_validation, model, loss_fn)
            train_error.append(train_rmse.item())
            validation_error.append(validation_rmse.item())

    pred = predict(model, current_year, ['SEED', 'WIN%', 'EFG%', 'EFG%D', 'FTR', 'FTRD', 'TOV%', 'OREB%', '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'BLKED%', 'HGT', 'EXP', 'PPPO', 'PPPD', 'PC1', 'PC3', 'PC5'])
    return pred

pred = main()
pred[['TEAM','PRED']]
# P24.loc[(P24['TEAM'] == "Creighton") | (P24['TEAM'] == "Saint Mary's"), ['TEAM','PRED']]
# P24.loc[(P24['SEED'] == 5) | (P24['SEED'] == 12), ['TEAM', 'PRED']]
#
# P24[['TEAM', 'SEED', 'PRED']].iloc[0:50].style.background_gradient(cmap='coolwarm').set_precision(2)
