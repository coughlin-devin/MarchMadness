import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# NOTE: batch size of 64 for 64 teams each year in the tournament, gaurantees the correct distribution of rounds in each batch
batch_size = 64
learning_rate = 0.001
epochs = 5000
patience = 70
device = "cpu"
loss_fn = nn.MSELoss()
weighted_loss =  nn.MSELoss(reduction='none')

class Net(nn.Module):
    """docstring for Net."""

    def __init__(self):
        super().__init__()
        # TODO: play around with different number and size of layers, larger first layer, 2,3,4 hidden layers, all same size layers, progressively smaller layers, etc.

        # NOTE: good models so far:
        # 61->32 32->32 32->1 no dropout, 600 epochs, ~0.935
        # 61->32 32->32 32->1 0.5 dropout, 800 epochs, ~0.926
        # TODO: try 32 32 32
        # 61->64 64->32 32->1, no dropout or 0.5 dropout, 700 epochs, ~0.945
        # 61->64 64->32 32->16 16->1, test dropout, 700 epochs

        self.layer_1 = nn.Sequential(
            nn.Linear(61, 32),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            # nn.Dropout(0.5)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # NOTE: multiply sigmoid output [0-1] by 6 (maximum allowed output of model)
        x = x*6
        return x

# TODO: quickly redo feature selection, 40-70 features, reduce colinearity when possible
# TODO: make a model to incorporate bracket structure
# TODO: generate synthetic data for each class of number of wins to even out the distribution, SMOTE algorithm, GAN or VAE
# TODO: create function to fill out bracket given my ranking and that years bracket, and calculate stats like games correct and points for each round and total
# TODO: create evaluation metric that considers bracket score more, not just individual teams wins
# IDEA: make a model to predict the probability of a single game (logistic regression)

#%% codecell
# set random state
def set_random_seed(seed):
    """Set random seeds for libraries in use.

    Set initial random seed for each library in use which uses random seeds to help keep results as reproducible as possible.

    Parameters
    ----------
    seed : int
        Integer number acting as the random seed.

    Returns
    -------
    None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

def train_loop(X, y, model, optimizer):
    """Training loop for the model.

    This function implements one round of model training by doing a forward pass, backpropogation, and an optimization step of the loss gradient.

    Parameters
    ----------
    X : tensor
        Matrix of model inputs.
    y : tensor
        Vector of true target values.
    model: Torch nn.Module
        The model being trained.
    optimizer: torch.optim algorithm
        The optimization algorithm used to reduce the loss function.

    Returns
    -------
    float
        Returns the model loss.
    """
    model.train()
    logits = model(X)
    rmse = torch.sqrt(loss_fn(logits, y))
    # NOTE: train using MSE, but evaluate test data with RMSE
    # loss = loss_fn(logits, y)
    loss = weighted_loss(logits, y)

    # TODO: try weighting so that each round is worth 2x more than previous round, currently weighting so that each round should be even
    # multiply losses by weight proportional to 1/frequency of the target value, ie. target values of 6 will have a weight of 64
    mse_weights = torch.where(y < 5, torch.pow(2, y), 32)
    loss = torch.mean(loss*mse_weights)

    # # TODO: explore this penalty term some more
    # # NOTE: added penalty term squared difference between sum(y_pred) and sum(y)
    lambda_coef = 0.005
    sum_penalty = lambda_coef * torch.abs(torch.sum(logits-y))
    loss = loss + sum_penalty

    # backpropogation
    optimizer.zero_grad() # zero gradients so they don't add up
    loss.backward()
    optimizer.step()
    return rmse

# WARNING: # BUG: using different loss for test vs train may negatively affect early stopping, either making it stop prematurely or too late
def test_loop(X, y, model):
    """Test loop for the model.

    This function implements one round of testing for the model by getting predictions and calulating the loss.

    Parameters
    ----------
    X : tensor
        Matrix of model inputs.
    y : tensor
        Vector of true target values.
    model: Torch nn.Module
        The model being trained.
    optimizer: torch.optim algorithm
        The optimization algorithm used to reduce the loss function.

    Returns
    -------
    np.array, float
        Returns the predictions and loss.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        # NOTE: evaluate test data with RMSE even though training using MSE
        loss = torch.sqrt(loss_fn(predictions, y))
        # importance_weights = torch.pow(2, y)
        # loss = torch.sqrt(torch.mean(weighted_loss(predictions, y)*importance_weights))
    return (predictions, loss)

def plot_fold_loss(ax, train_error, test_error, fold):
    x_range = len(train_error)
    train_error = torch.sqrt(torch.tensor(train_error))
    ax.plot(range(x_range), train_error, label="Training RMSE")
    ax.plot(range(x_range), test_error, label="Test RMSE")
    ax.plot(range(x_range), np.ones(x_range))
    ax.plot(range(x_range), np.ones(x_range)*0.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Wins")
    ax.set_title("Fold {} RMSE".format(fold))

def plot_test_loss(ax, test_error, fold):
    # plot each folds training and test error
    # ax_single.plot(range(x_range), train_error, label="Fold {} Training RMSE".format(fold))
    ax.plot(range(len(test_error)), test_error, label="Fold {} Test RMSE".format(fold))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Wins")
    ax.legend()
    ax.set_title("K-Fold Test RMSE")

# TODO: plot custom weighted loss to see if weights are working
def plot_loss_distribution(dfs, ax, type):
    df = pd.concat(dfs, axis=0)
    y_mean = df.mean(axis=0).values
    y_min = y_mean - df.min(axis=0).values
    y_max = df.max(axis=0).values - y_mean
    ax.plot(df.columns, y_mean)
    ax.errorbar(df.columns, y_mean, yerr=[y_min, y_max], fmt ='o', capsize=10)
    ax.set_xlabel("Number of Wins")
    ax.set_ylabel("RMSE")
    ax.set_title("Average {} RMSE by Number of Wins Across All Folds".format(type))

def k_fold_cross_validation(data, repeats=1, shuffle=False):

    X = data.drop(['School', 'Year', 'WINS'], axis=1)
    y = data['WINS']

    grouped_average_loss = []
    grouped_total_loss = []

    # number of folds, equal to the number of years of data excluding the holdout year for final predictions
    K = len(data.Year.unique())

    # TODO: update plotting
    # create figs and axes for plotting trainnig and test loss
    ax_const = K*repeats
    rows = int(np.floor(ax_const ** 0.5))
    cols = int(np.ceil(ax_const / rows))
    fig_fold, ax_fold = plt.subplots(rows, cols)
    fig_fold.set_size_inches(24,12)
    fig_test, ax_test = plt.subplots()
    fig_test.set_size_inches(16,12)
    fig_avg, ax_avg = plt.subplots()
    fig_total, ax_total = plt.subplots()

    # store final test error of each fold
    final_epoch_test_error = []

    if repeats > 1:
        skf = RepeatedStratifiedKFold(n_splits=K, n_repeats=repeats)
    else:
        skf = StratifiedKFold(n_splits=K, shuffle=shuffle)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        # early stopping variables
        best_test_loss = float('inf')
        epochs_since_improve = 0
        early_stop = 0

        # NOTE: important to normalize each fold seperately
        # create scaler
        scaler = MinMaxScaler()

        # training fold
        X_train = scaler.fit_transform(X.loc[train_index])
        X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
        y_train = y[train_index]
        y_train = torch.tensor(y_train.values, dtype=torch.float32, requires_grad=True)
        y_train = torch.unsqueeze(y_train,1)

        # validation fold
        # NOTE: transform X_validaiton on already fit scaler
        X_test = scaler.transform(X.loc[test_index])
        X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=False)
        y_test = y[test_index]
        y_test = torch.tensor(y_test.values, dtype=torch.float32, requires_grad=False)
        y_test = torch.unsqueeze(y_test,1)

        # store train and test errors of each train/test fold
        train_error = []
        test_error = []

        # NOTE NEED to initialize and train a new instance of the model and optimizer each fold to avoid leaking parameter information
        model = Net().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01)

        # for each training/testing loop epoch
        for epoch in range(epochs):
            train_loss = train_loop(X_train, y_train, model, optimizer)
            predictions, test_loss = test_loop(X_test, y_test, model)
            train_error.append(train_loss.item())
            test_error.append(test_loss.item())

            # Early stopping check
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if epochs_since_improve == patience:
                early_stop = epoch + 1
                break

        # plot each test loss of each fold on a single axis
        plot_test_loss(ax_test, test_error, fold)
        # plot train and test loss for each fold on a subplots figure
        plot_fold_loss(ax_fold[fold//cols][fold%cols], train_error, test_error, fold)
        # append loss from final training epoch to list for getting average loss across all folds later
        final_epoch_test_error.append(test_error[-1])

        # seperate loss by wins
        with torch.no_grad():
            model.eval()
            predictions = model(X_test)
        df = data.loc[test_index, ['WINS']]
        wins = torch.tensor(df['WINS'].to_numpy())
        print(geometric_mean_score(wins, predictions, average=None))
        df['RMSE'] = torch.sqrt(weighted_loss(wins, predictions.squeeze(1)))
        # df['WMSE'] = weighted_loss(wins, predictions.squeeze(1))*torch.where(wins < 5, torch.pow(2, wins), 32)
        average_rmse = df.groupby('WINS').mean()
        total_rmse = df.groupby('WINS').sum()
        grouped_average_loss.append(average_rmse.T)
        grouped_total_loss.append(total_rmse.T)

    # plot mean and total loss by wins
    plot_loss_distribution(grouped_average_loss, ax_avg, 'Mean')
    plot_loss_distribution(grouped_total_loss, ax_total, 'Total')
    fig_fold.tight_layout()
    plt.show()
    mean_loss = np.mean(final_epoch_test_error)
    return mean_loss

# retrain full model on all data without k fold cross validation
def retrain(data):

    best_test_loss = float('inf')
    epochs_since_improve = 0
    early_stop = 0

    X = data.drop(['School', 'Year', 'WINS'], axis=1)
    y = data['WINS']

    scaler = MinMaxScaler()

    # training fold
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y.values, dtype=torch.float32, requires_grad=True)
    y = torch.unsqueeze(y,1)

    # NOTE NEED to initialize and train a new instance of the model and optimizer each fold to avoid leaking parameter information
    model = Net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.01)

    # for each training/testing loop epoch
    for epoch in range(epochs):
        train_loss = train_loop(X, y, model, optimizer)
        predictions, test_loss = test_loop(X, y, model)

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve == patience:
            early_stop = epoch + 1
            break

    return model, scaler

# TODO save best model based on test data before prediction data
def predict(model, data, scaler):
    """Predict the number of March Madness wins for each team."""
    schools = data['School']
    X = data.drop('School', axis=1)
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        model.eval()
        nn_predictions = model(X)
    predictions = pd.DataFrame({'School':schools, 'Predicted Wins':nn_predictions.squeeze(1)}).sort_values(by='Predicted Wins', ascending=False)
    return predictions

# TODO: goal is to get RMSE below 0.5 so the average error is less than half a win off
def main(target_year):
    # set random seeds for reproducible results
    set_random_seed(target_year)

    # import data
    df = pd.read_csv(r"Data/Clean/features.csv")

    # hold out target year data for final prediction
    data = df.loc[df['Year'] != target_year]
    holdout = df.loc[df['Year'] == target_year]

    # remove year and target variable from holdout data
    holdout = holdout.drop(['Year', 'WINS'], axis=1)

    # K-Fold Cross Validation
    mean_loss = k_fold_cross_validation(data, repeats=1, shuffle=True)
    print(mean_loss)

    # retrained model predictions on full data set
    retrained_model, scaler = retrain(data)

    # get predictions
    predictions = predict(retrained_model, holdout, scaler)

    return predictions

# TODO: build a logistic regression for win loss probability using my ranking as the only feature to get likelihood of team A beating Team B based on their respective rankings

# pred = main(2024)
predictions = main(2024)

predictions[:32]

predictions['Predicted Wins'].sum()

# pred.loc[(pred['School'] == "Colorado State") | (pred['School'] == "Texas"), ['School','PRED']]

# NOTE: model 1.0 (kenpom 2008-2024) on 2024 picks: 41 total correct picks (24,11,3,1,1,1) 114 pts
# NOTE: model 1.1 (bs4 1997-2024) minus 1-2 features on 2024 picks and adjusted training length: 43 correct picks (21,12,5,2,2,1) 145 pts
# NOTE: model 2.0 (team_rankings ~60 feature 2002-2024) on 2024 picks: 49 total correct picks (24,15,5,2,2,1) 154 pts, 77% correct picks, 80% of points on double score system
