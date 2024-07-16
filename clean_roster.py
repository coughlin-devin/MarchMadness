import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import random
# # set random seed
# random.seed(2024)

# TODO: docstrings

def get_data():
    # pull csv table
    roster_df = pd.read_csv(r"Data/Raw/roster_table.csv")
    df = roster_df[['School', 'Year', 'Player', 'Class', 'Pos', 'Height', 'Weight', 'Summary', 'RSCI Top 100']].copy()

    # remove players who aren't in the player per game or player per 40 minutes tables because they are likely redshirts and did not play
    player_df = pd.read_csv(r"Data/Raw/player_table.csv")
    diff = roster_df.merge(player_df, on=['School', 'Year', 'Player'], how='left', indicator=True)
    diff_mask = diff.loc[diff['_merge'] == 'left_only']
    df.drop(diff_mask.index, inplace=True)

    # remove 5 outliers with no games played
    no_games_mask = player_df.loc[(player_df['G'].isna()) | (player_df['G'] == 0)]
    df.drop(no_games_mask.index, axis=0, inplace=True)
    df.reset_index()

    return df

df = get_data()

def height_in_inches(heights):
    """Convert heights in foot-inches format to inches.

    Extended description of function.

    Parameters
    ----------
    heights : pandas.core.series.Series (string)
        Pandas Series of heights as strings.

    Returns
    -------
    pandas.core.series.Series (float64)
        Reutrns a pandas Series of heights in inches as float64.
    """
    feet_inches = heights.str.split('-')
    feet = feet_inches.str.get(0)
    inches = feet_inches.str.get(1)
    feet = feet.astype('float64')
    inches = inches.astype('float64')
    feet = feet*12
    in_inches = feet+inches
    return in_inches

# convert height in feet and inches to inches
df.Height = height_in_inches(df.Height)

# mean stats by position for each year
centers = df.groupby(['Year', 'Pos']).mean().round(1).xs('C', level=1, drop_level=True)
forwards = df.groupby(['Year', 'Pos']).mean().round(1).xs('F', level=1, drop_level=True)
guards = df.groupby(['Year', 'Pos']).mean().round(1).xs('G', level=1, drop_level=True)

# NOTE: encoding positions as ordinal because there is some ordering: Guard -> <- Forward -> <- Center
def ordinal_encoding(df):
    class_map = {
        'FR':1,
        'SO':2,
        'JR':3,
        'SR':4
    }
    ordinal_class = pd.Series([class_map[i] if i in class_map else float('nan') for i in df.Class], index=df.index, dtype='float64')
    df.Class = ordinal_class

    pos_map = {
        'C':3,
        'F':2,
        'G':1
    }
    ordinal_pos = pd.Series([pos_map[i] if i in pos_map else float('nan') for i in df.Pos], index=df.index, dtype='float64')
    df.Pos = ordinal_pos

    return df

# WARNING: years 1997-2001 all players missing weight
def fill_height_weight(df, column, start_year, end_year):
    """Fill in missing height or weight of players with the mean.

    Replaces NaN values with the average of thier position in the specific year, rounded to one significant figure. For players without a listed position, the mean of all tournament players that year is used. For years without any values listed like weights in 1997-2001, the next years average is used.

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
    for year in range(start_year, end_year+1):
        # years missing all weights
        if year in [1997, 1998, 1999, 2000, 2001] and column=='Weight':
            # NOTE: using only 2002 data for mean imputation here because it is closest to the years with missing data and many of the players from these years will play in 2002
            avg_weight = df.loc[df['Year'] == 2002, 'Weight'].mean().round(1)
            df.loc[df['Year'] == year, 'Weight'] = df.loc[df['Year'] == year, 'Weight'].fillna(value=avg_weight)
        elif year != 2020:
            # get players in each position missing height or weight
            center_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (~df['Pos'].isna()) & (df['Pos'] == 'C')]
            forward_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (~df['Pos'].isna()) & (df['Pos'] == 'F')]
            guard_mask = df.loc[(df[column].isna()) & (df['Year'] == year) & (~df['Pos'].isna()) & (df['Pos'] == 'G')]

            # mean height or weight of all tournament players
            mean = df.loc[df['Year'] == year, column].mean().round(1)

            # mean imputation of average height or weight by position for the year
            df.loc[center_mask.index, column] = df.loc[center_mask.index, column].fillna(value=centers.loc[year, column])
            df.loc[forward_mask.index, column] = df.loc[forward_mask.index, column].fillna(value=forwards.loc[year, column])
            df.loc[guard_mask.index, column] = df.loc[guard_mask.index, column].fillna(value=guards.loc[year, column])

            # for players missing a position replace height and weight with mean height of all tournament players that year
            positionless_mask = df.loc[(df[column].isna()) & (df['Year'] == year)]
            df.loc[positionless_mask.index, column] = df.loc[positionless_mask.index, column].fillna(value=mean)
    return df

def fill_RSCI(df):
    top100 = df.loc[~df['RSCI Top 100'].isna(), 'RSCI Top 100'].str.split(' ')
    top100 = top100.str.get(0)
    top100 = top100.astype('int64')
    top100 = 101 - top100
    df.loc[top100.index, 'RSCI Top 100'] = top100
    df['RSCI Top 100'].fillna(value=0, inplace=True)
    return df

def fill_summary(df):
    mask = df.loc[df['Summary'].isna()]
    zero_summary = "0.0 Pts, 0.0 Reb, 0.0 Ast"
    df.loc[mask.index, 'Summary'] = df.loc[mask.index, 'Summary'].fillna(value=zero_summary)
    return df

# WARNING: depends on correct spacing in summary column
def summary_to_stats(df):
    """Split statistical summary into seperate columns.

    Create points, rebounds, and assists columns from one summary column.

    Parameters
    ----------
    df : DataFrame
        DataFrame with the summary column and where to add new columns.

    Returns
    -------
    DataFrame
        Returns DataFrame with new stats columns.
    """
    split = df.Summary.str.split(',')
    points = split.str.get(0)
    rebounds = split.str.get(1)
    assists = split.str.get(2)
    points = points.str.split(' ').str.get(0)
    rebounds = rebounds.str.split(' ').str.get(1)
    assists = assists.str.split(' ').str.get(1)
    points = points.astype('float64')
    rebounds = rebounds.astype('float64')
    assists = assists.astype('float64')
    df['PTS'] = points
    df['REB'] = rebounds
    df['AST'] = assists
    df.drop('Summary', axis=1, inplace=True)
    return df

def knn_imputation(df, start_year, end_year):
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
            imputer = KNNImputer(n_neighbors=3, weights='distance')
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

def imputation(df, start_year, end_year):
    # encode categorical variables
    df = ordinal_encoding(df)
    df = fill_height_weight(df, 'Height', start_year, end_year)
    df = fill_height_weight(df, 'Weight', start_year, end_year)
    df = fill_RSCI(df)

    # create columns for points, rebounds, and asists from summary column
    df = fill_summary(df)
    df = summary_to_stats(df)

    # KNN imputation for missing values of class and position
    df = knn_imputation(df, start_year, end_year)

    # drop summary stats PTS, AST, REB, because they match player stats PTS, AST, TRB
    df = df.drop(['PTS', 'AST', 'REB'], axis=1)

    return df

df = imputation(df, 1997, 2024)
df.to_csv(r"Data/Clean/clean_roster.csv", index=False)

# # NOTE: previous methods of imputation
# # NOTE: helper for random_fill_position()
# def predict_positions(year, height):
#     # get mean height by position
#     center_height = centers.loc[year, 'Height']
#     forward_height = forwards.loc[year, 'Height']
#     guard_height = guards.loc[year, 'Height']
#
#     # find distances between height and average position height
#     center_distance = abs(height-center_height)
#     forward_distance = abs(height-forward_height)
#     guard_distance = abs(height-guard_height)
#
#     # predict position by smallest distance
#     pred = np.argmin([center_distance, forward_distance, guard_distance], axis=0)
#     pos_map = {
#         0:'C',
#         1:'F',
#         2:'G'
#     }
#     positions = pd.Series(data=[pos_map[pos] for pos in pred], index=height.index, dtype='str')
#     return positions
#
# # NOTE: old method of filling position by using height when available, otherwise picking randomly weighted by distribution
# def random_fill_position(df, start_year, end_year):
#     for year in range(start_year, end_year+1):
#         if year != 2020:
#             # for players with a position
#             mask = df.loc[(df['Pos'].isna()) & (~df['Height'].isna()) & (df['Year'] == year)]
#             positions = predict_positions(year, mask.Height)
#             df.loc[mask.index, 'Pos'] = df.loc[mask.index, 'Pos'].fillna(value=positions)
#
#             # for players without a position
#             mask = df.loc[(df['Pos'].isna()) & (df['Year'] == year)]
#
#             # get count of positions for each year
#             group = df.groupby(['Year','Pos']).count()
#             centers = group.loc[(year, 'C'), 'Player']
#             forwards = group.loc[(year, 'F'), 'Player']
#             guards = group.loc[(year, 'G'), 'Player']
#
#             # create probability of being randomly selected based on the distribution of positions
#             total = sum([centers, forwards, guards])
#             c_rate = centers/total
#             f_rate = forwards/total
#             g_rate = guards/total
#
#             # create series of randomly chosen positions
#             positions = ['C', 'F', 'G']
#             weights = [c_rate, f_rate, g_rate]
#             choices = random.choices(positions, weights, k=len(mask))
#             choices = pd.Series(data=choices, index=mask.index, dtype='str')
#
#             df.loc[mask.index, 'Pos'] = df.loc[mask.index, 'Pos'].fillna(value=choices)
#
# # NOTE: old method of filling class by picking randomly from weighted by distribution
# def random_fill_class(df, start_year, end_year):
#     for year in range(start_year, end_year+1):
#         if year != 2020:
#             mask = df.loc[(df['Class'].isna()) & (df['Year'] == year)]
#
#             # get count of positions for each year
#             group = df.groupby(['Year','Class']).count()
#             freshman = group.loc[(year, 'FR'), 'Player']
#             sophomores = group.loc[(year, 'SO'), 'Player']
#             juniors = group.loc[(year, 'JR'), 'Player']
#             seniors = group.loc[(year, 'SR'), 'Player']
#
#             # create probability of being randomly selected based on the distribution of positions
#             total = sum([freshman, sophomores, juniors, seniors])
#             fr_rate = freshman/total
#             so_rate = sophomores/total
#             jr_rate = juniors/total
#             sr_rate = seniors/total
#
#             # create series of randomly chosen positions
#             classes = ['FR', 'SO', 'JR', 'SR']
#             weights = [fr_rate, so_rate, jr_rate, sr_rate]
#             choices = random.choices(classes, weights, k=len(mask))
#             choices = pd.Series(data=choices, index=mask.index, dtype='str')
#
#             df.loc[mask.index, 'Class'] = df.loc[mask.index, 'Class'].fillna(value=choices)
#
#     return df
