import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# TODO: look into MissForest imputation, https://arxiv.org/html/2403.14687v1#S2, bookmarked some medium articles also

# TODO: docstrings

# NOTE: manhatten distance may be better because it isn't as affected by large single stat outliers in determining distance between KNN targets. In euclidean distance the difference between a large outlier is squared, while in manhatten it isn't, so a similar datapoint might be ruled out with euclidean distance and kept with manhatten distance
# function to find manhatten distances between values in vector y and matrix X while ignoring NaN values
def nan_manhatten(y,X):
    difference = X.sub(y, axis=1)
    absolute = difference.abs()
    distances = absolute.sum(axis=1)
    return distances

# function to find euclidean distances between values in vector y and matrix X while ignoring NaN values
def nan_euclidean(y,X):
    difference = X.sub(y, axis=1)
    square = difference**2
    sum = square.sum(axis=1)
    distances = np.sqrt(sum)
    return distances

# TODO: implement option for nan_manhatten
# NOTE: only using players in the same year to impute on to keep 'era' intact and also because I'm treating each season as its own system
def knn_imputer(df, target, k, start_year, end_year, mode=False):
    for year in range(start_year, end_year+1):
        if year != 2020:
            # remove non-numeric columns
            X = df.loc[df['Year'] == year].drop(['School', 'Year', 'RSCI Top 100'], axis=1)

            # scale data before using KNNImputer
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(X)
            scaled = pd.DataFrame(scaled, columns=X.columns, index=X.index)

            # seperate target column
            Y = scaled.loc[scaled[target].isna()].drop(target, axis=1)
            num_players = len(Y)
            if num_players == 0:
                continue
            X = scaled.loc[~scaled[target].isna()].drop(target, axis=1)

            distances = Y.apply(nan_euclidean, axis=1, args=[X])
            kth = pd.DataFrame(np.argpartition(distances, kth=k, axis=-1)).iloc[:,:k]
            indices = np.array([distances.iloc[i].iloc[kth.iloc[i]].index for i in range(num_players)])

            if mode:
                # NOTE: important to use .loc[] here NOT .iloc[] and take the first (lower) class in cases of tied mode
                modes = [df.loc[indices[i], target].mode()[0].item() for i in range(num_players)]
                df.loc[Y.index, target] = modes
            else:
                means = [df.loc[indices[i], target].mean().item() for i in range(num_players)]
                df.loc[Y.index, target] = means

    return df

# TODO: add random noise using residuals to mean imputation for better results
def position_mean_imputer(df, column, start_year, end_year):
    """Fill in missing stat of players with the mean of their position group.

    Replaces NaN values with the mean average of thier position in the specific year. For players without a listed position, the mean of all tournament players that year is used. For years without any values listed like weights from the years 1997-2001, the next closest years average is used (2002).

    Parameters
    ----------
    df : DataFrame
        The DataFrame with missing NaN values to fill.
    column : string
        Column name specifying whether to replace height or weight.
    start_year : int
        The first year to start filling NaN values.
    end_year : int
        The year last to end filling NaN values.

    Returns
    -------
    DataFrame
        DataFrame with NaN values filled.
    """
    group = df.drop('School', axis=1).groupby(['Year', 'Pos']).mean().reset_index()
    for year in range(start_year, end_year+1):
        # years missing all weights
        if year in [1997, 1998, 1999, 2000, 2001] and column=='Weight':
            # NOTE: using only 2002 data for mean imputation here because it is closest to the years with missing data and many of the players from these years will play in 2002
            avg_weight = df.loc[df['Year'] == 2002, 'Weight'].mean()
            df.loc[df['Year'] == year, 'Weight'] = df.loc[df['Year'] == year, 'Weight'].fillna(value=avg_weight)
        elif year != 2020:
            # get players in each position missing the column's stat
            center_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (df['Pos'] == 3)]
            forward_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (df['Pos'] == 2)]
            guard_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (df['Pos'] == 1)]

            # impute missing ORB and DRB stats using TRB stat to fill in with 1/3-2/3 splits for centers and forwards, but 1/4-3/4 split for guards
            if column == 'ORB':
                df.loc[center_mask.index, column] = df.loc[center_mask.index, column].fillna(value=df.loc[center_mask.index, 'TRB'] / 3)
                df.loc[forward_mask.index, column] = df.loc[forward_mask.index, column].fillna(value=df.loc[forward_mask.index, 'TRB'] / 3)
                df.loc[guard_mask.index, column] = df.loc[guard_mask.index, column].fillna(value=df.loc[guard_mask.index, 'TRB'] / 4)

            elif column == 'DRB':
                df.loc[center_mask.index, column] = df.loc[center_mask.index, column].fillna(value=df.loc[center_mask.index, 'TRB']*2/3)
                df.loc[forward_mask.index, column] = df.loc[forward_mask.index, column].fillna(value=df.loc[forward_mask.index, 'TRB']*2/3)
                df.loc[guard_mask.index, column] = df.loc[guard_mask.index, column].fillna(value=df.loc[guard_mask.index, 'TRB']*3/4)

            else:
                # get mean of column stat by position for the year
                center = group.loc[(group['Year'] == year) & (group['Pos'] == 3), column].item()
                forward = group.loc[(group['Year'] == year) & (group['Pos'] == 2), column].item()
                guard = group.loc[(group['Year'] == year) & (group['Pos'] == 1), column].item()

                # mean imputation of average stat by position for the year
                df.loc[center_mask.index, column] = df.loc[center_mask.index, column].fillna(value=center)
                df.loc[forward_mask.index, column] = df.loc[forward_mask.index, column].fillna(value=forward)
                df.loc[guard_mask.index, column] = df.loc[guard_mask.index, column].fillna(value=guard)
    return df

# generate imputed values from gaussian distribution with global mean and standard deviation
def gaussian_mean_imputation(df, columns):
    # TODO: docstring
    # df = df.drop('School', axis=1)
    means = df.drop('School', axis=1).mean()
    stds = df.drop('School', axis=1).std()
    for column in columns:
        num_missing = df[column].isna().sum()
        gaussian = pd.Series(np.random.normal(means[column], stds[column], size=num_missing))
        gaussian.index = df[column].loc[df[column].isna()].index
        df.loc[:, column] = df.loc[:, column].fillna(value=gaussian)
    return df
