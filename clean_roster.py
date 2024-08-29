import pandas as pd

# TODO: docstrings

def get_data():
    # pull csv table
    roster_df = pd.read_csv(r"Data/Raw/roster_table.csv")
    df = roster_df[['School', 'Year', 'Player', 'Class', 'Pos', 'Height', 'Weight', 'RSCI Top 100']].copy()

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
    # encode class
    class_map = {
        'FR':1,
        'SO':2,
        'JR':3,
        'SR':4
    }
    ordinal_class = pd.Series([class_map[i] if i in class_map else float('nan') for i in df.Class], index=df.index, dtype='float64')
    df.Class = ordinal_class

    # encode position
    pos_map = {
        'C':3,
        'F':2,
        'G':1
    }
    ordinal_pos = pd.Series([pos_map[i] if i in pos_map else float('nan') for i in df.Pos], index=df.index, dtype='float64')
    df.Pos = ordinal_pos

    # encode RSCI Top 100 rankings
    top100 = df.loc[~df['RSCI Top 100'].isna(), 'RSCI Top 100'].str.split(' ')
    top100 = top100.str.get(0)
    top100 = top100.astype('int64')
    top100 = 101 - top100
    df.loc[top100.index, 'RSCI Top 100'] = top100
    df['RSCI Top 100'].fillna(value=0, inplace=True)

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

def clean(df, start_year, end_year):
    # encode categorical variables
    df = ordinal_encoding(df)
    df = fill_height_weight(df, 'Height', start_year, end_year)
    df = fill_height_weight(df, 'Weight', start_year, end_year)
    return df

# TODO: create features
def avg_height():
    """Calculate average team height."""
    pass

def interior_height():
    """Calculate average team height of Centers and Forwards."""
    pass

def exterior_height():
    """Calculate average team height of Gaurds."""
    pass

# IDEA: weight each players height by some metric like minutes played, possessions played, games played etc
def avg_weight():
    """Calculate average team weight."""
    pass

def interior_weight():
    """Calculate average team weight of Centers and Forwards."""
    pass

def exterior_weight():
    """Calculate average team weight of Gaurds."""
    pass



df = clean(df, 1997, 2024)
df.to_csv(r"Data/Clean/clean_roster.csv", mode='w', index=False)
