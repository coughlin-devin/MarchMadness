import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

roster = pd.read_csv(r"Data/Clean/clean_roster.csv")
player = pd.read_csv(r"Data/Raw/player_table.csv")
per_40 = pd.read_csv(r"Data/Raw/per_40_table.csv")
team_opp = pd.read_csv(r"Data/Raw/team_opp_table.csv")

# NOTE: removes 120 players from 2024 who were most likely all redshirts; I confirmed some players were redshirts
df = roster.merge(player, how='left', on=['School', 'Year', 'Player'])
df40 = roster.merge(per_40, how='left', on=['School', 'Year', 'Player'])

# add total team games column
team_games = team_opp[['School', 'Year', 'G']]
team_games.columns = ['School', 'Year', 'TG']
df = df.merge(team_games, on=['School', 'Year'], how='left')
df40 = df40.merge(team_games, on=['School', 'Year'], how='left')

# add total minutes column
total_mins = per_40[['School', 'Year', 'Player', 'MP']]
total_mins.columns = ['School', 'Year', 'Player', 'TMP']
df = df.merge(total_mins, on=['School', 'Year', 'Player'], how='left')
df40 = df40.merge(total_mins, on=['School', 'Year', 'Player'], how='left')

# drop unnecessary rank column
df.drop(['Rk'], axis=1, inplace=True)
df40.drop(['Rk'], axis=1, inplace=True)

def correct_percentages(df):
    # set FG%, 2P%, and 3P% to NaN if zero FGA
    fga_mask = df.loc[(df['FGA'] == 0)]
    df.loc[fga_mask.index, ['FG%', '2P%', '3P%']] = float('NaN')

    _2pa_mask = df.loc[(df['2PA'] == 0)]
    df.loc[_2pa_mask.index, '2P%'] = float('NaN')

    _3pa_mask = df.loc[(df['3PA'] == 0)]
    df.loc[_3pa_mask.index, '3P%'] = float('NaN')

    fta_mask = df.loc[(df['FTA'] == 0)]
    df.loc[fta_mask.index, 'FT%'] = float('NaN')

    return df

# imputation to fill missing values
def knn_imputation(df, k, start_year, end_year):
    for year in range(start_year, end_year+1):
        if year != 2020:
            # create DataFrame for KNNImputer
            X = df.loc[df['Year'] == year].drop(['School', 'Player', 'RSCI Top 100'], axis=1)

            # keep track of indexes of NaN values
            class_mask = X.loc[X['Class'].isna()]
            pos_mask = X.loc[X['Pos'].isna()]

            # if there are no missing values to impute this year, skip to next year
            if (len(class_mask) == 0) & (len(pos_mask) == 0):
                continue

            # normalize data before using KNNImputer
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(X)

            # do KNN imputation
            knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='nan_euclidean')
            fit = knn.fit(X)
            imputer = KNNImputer(n_neighbors=k, weights='distance')
            transformed = imputer.fit_transform(normalized)

            # revert back to original scales of data
            inversed = scaler.inverse_transform(transformed)
            y = pd.DataFrame(inversed, columns=X.columns, index=X.index)

            # round imputed values to nearest integer
            y.loc[class_mask.index, 'Class'] = round(y.loc[class_mask.index, 'Class'])
            y.loc[pos_mask.index, 'Pos'] = round(y.loc[pos_mask.index, 'Pos'])

            df.loc[class_mask.index, 'Class'] = df.loc[class_mask.index, 'Class'].fillna(value=y.loc[class_mask.index, 'Class'])
            df.loc[pos_mask.index, 'Pos'] = df.loc[pos_mask.index, 'Class'].fillna(value=y.loc[pos_mask.index, 'Class'])

    return df

def nan_euclidean(y,X):
    difference = X.sub(y, axis=1)
    square = difference**2
    sum = square.sum(axis=1)
    distances = np.sqrt(sum)
    return distances

def knn_imputation(df, target, k, start_year, end_year, mode=False):
    for year in range(start_year, end_year+1):
        if year != 2020:
            X = df.loc[df['Year'] == year].drop(['School', 'Year', 'Player', 'RSCI Top 100', 'TG', 'TMP'], axis=1)

            # scale data before using KNNImputer
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(X)
            scaled = pd.DataFrame(scaled, columns=X.columns, index=X.index)

            Y = scaled.loc[scaled[target].isna()].drop(target, axis=1)
            num_players = len(Y)
            if num_players == 0:
                continue
            X = scaled.loc[~scaled[target].isna()].drop(target, axis=1)

            distances = Y.apply(nan_euclidean, axis=1, args=[X])
            kth = np.argpartition(distances, kth=k, axis=-1).iloc[:,:k]
            indices = np.array([distances.iloc[i].iloc[kth.iloc[i]].index for i in range(num_players)])

            if mode:
                # NOTE: important to use .loc[] here NOT .iloc[] and take the first (lower) class in cases of tied mode
                modes = [df.loc[indices[i], target].mode()[0].item() for i in range(num_players)]
                df.loc[Y.index, target] = modes
            else:
                means = [df.loc[indices[i], target].mean().item() for i in range(num_players)]
                df.loc[Y.index, target] = means

    return df

df = correct_percentages(df)
df = knn_imputation(df, 'Class', 30, 1997, 2024, mode=True)
df = knn_imputation(df, 'Pos', 30, 1997, 2024, mode=True)

df40 = correct_percentages(df40)
df40 = knn_imputation(df40, 'Class', 30, 1997, 2024, mode=True)
df40 = knn_imputation(df40, 'Pos', 30, 1997, 2024, mode=True)

# for use in creating features by position?
df = knn_imputation(df, 'TOV', 30, 1997, 2024)
df = knn_imputation(df, 'PF', 30, 1997, 2024)
df = knn_imputation(df, 'STL', 30, 1997, 2024)
df = knn_imputation(df, 'BLK', 30, 1997, 2024)

df40 = knn_imputation(df40, 'TOV', 30, 1997, 2024)
df40 = knn_imputation(df40, 'PF', 30, 1997, 2024)
df40 = knn_imputation(df40, 'STL', 30, 1997, 2024)
df40 = knn_imputation(df40, 'BLK', 30, 1997, 2024)

# TODO: make integer restricted values int64 types
df.isna().sum()
df40.isna().sum()

# TODO: for per_40 missing values, recalculate from per_game by Value * Games * 40 / MP
m = df.merge(df40, on=['School', 'Year', 'Player'], how='left')
m.loc[m['2P_y'].isna(), ['2P_x', '2P_y', 'MP_x', 'G_y']]

# IDEA: rebounds could get from doing something with TRB = ORB + DRB

# IDEA: don't waste time filling in missing games started, just create the varaince of starters and impute the average variance for the missing teams
