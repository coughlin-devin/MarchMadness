import pandas as pd

def get_data():
    """Pull data from file.

    Fetch the roster data from a csv file into a pandas DataFrame.

    Parameters
    ----------

    Returns
    -------
    pandas DataFrame
        Returns a DataFrame of tabular Roster data.
    """
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

    return df

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

# NOTE: encoding positions as ordinal because there is some ordering: Guard -> <- Forward -> <- Center
def ordinal_encoding(df):
    """Encode categorical data into ordinal data.

    Encode Class, Position, and RSCI Top 100 data to ordinal from categorical data. Class and Position are encoded to ordinal and not dummy encoding because I believe there is an ordering to Class in the way that Freshman are first years, Sophomores second years and so on. Freshman < Sophomore < Junior < Senior.

    The same thinking applies to Positions. A guard is more similar in play style to forward than a center, and center is similarly more similar to forward than guard. So there is a G < F < C ordering.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containg the categorical data.

    Returns
    -------
    pandas DataFrame
        Returns a DataFrame with the categorical data now encoded to ordinal data.
    """
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

df = get_data()

# convert height in feet and inches to inches
df.Height = height_in_inches(df.Height)

# encode categorical variables to ordinal variables
df = ordinal_encoding(df)

df.to_csv(r"Data/Clean/clean_roster.csv", mode='w', index=False)
