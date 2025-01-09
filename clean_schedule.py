import pandas as pd

# import data
schedule = pd.read_csv(r"Data/Raw/schedule_table.csv", low_memory=False)

def prepare_frame(frame):
    """Clean the DataFrame so that is ready to work with.

    Rename unnamed columns, create a workable DataFrame, remove repeated header rows, remove NCAA games from schedule to keep data integrity when and where possible.

    Parameters
    ----------
    frame : pandas DataFrame
        Raw DataFrame needing to be prepared to work with.

    Returns
    -------
    DataFrame
        Returns cleaned and workable DataFrame.
    """
    # rename unnamed and duplicated columns
    frame.rename(columns={"Unnamed: 3": "Venue", "Unnamed: 4": "Venue2"}, inplace=True)
    frame.rename(columns={"Unnamed: 7": "Result", "Unnamed: 8": "Result2"}, inplace=True)

    # create working DataFrame
    df = frame[['School', 'Year', 'Type', 'Venue', 'Venue2', 'Opponent', 'SRS', 'Result', 'Result2', 'Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak']]

    # remove header rows that are just column names
    headers = df.loc[df['Type'] == 'Type']
    df = df.drop(headers.index)

    # remove NCAA games so they don't corrupt training features data
    ncaa_games = df.loc[df['Type'] == 'NCAA']
    df = df.drop(ncaa_games.index)

    # remove unnecessary columns
    df = df.drop(['Type', 'Opponent'], axis=1)

    # remove 13 (as of 2024) games with null wins/losses bc they were cancelled or postponed
    df.drop(df.loc[df['W'].isna()].index, inplace=True)

    return df

def venue(df):
    """Create dummy variables for home, away and neutral site venues.

    Encodes H, A, and N game sites to dummy variables.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to add dummy variables to.

    Returns
    -------
    DataFrame
        Returns input DataFrame with added dummy variables.
    """
    venue = df.Venue.fillna('H') + df.Venue2.fillna('H')
    map_venue = {
        'HH':'H',
        'NH':'N',
        'HN':'N',
        '@H':'A',
        'H@':'A'
    }
    df.Venue = [map_venue[x] for x in venue]
    df = pd.concat([df, pd.get_dummies(df.Venue, dtype='int64')], axis=1)
    df.drop(['Venue', 'Venue2'], axis=1, inplace=True)
    return df

def result(df):
    """Convert result column to numeric type.

    Encode wins as 1, and losses as 0.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to encode game results to.

    Returns
    -------
    DataFrame
        Returns input DataFrame with encoded game results.
    """
    results = df.Result.fillna("L") + df.Result2.fillna("L")
    df.Result = [1 if 'W' in x else 0 for x in results]
    df.drop(['Result2'], axis=1, inplace=True)
    return df

def overtime(df):
    """Set the number of overtime games to numeric type.

    Encode overtime games as numeric type.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to encode overtime games to.

    Returns
    -------
    DataFrame
        Returns input DataFrame with encoded overtime game counts.
    """
    overtime = df.loc[~df['OT'].isna()]
    ot = [1 if x == 'OT' else int(x[0]) for x in overtime.OT]
    df.loc[~df['OT'].isna(), 'OT'] = ot
    df.loc[df['OT'].isna(), 'OT'] = df.loc[df['OT'].isna(), 'OT'].astype('float64').fillna(value=0)
    return df

def streaks(df):
    """Set streak to integer type.

    Encode winning streaks as positive integer values and losing streaks as negative integer values.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to encode streaks to.

    Returns
    -------
    DataFrame
        Returns input DataFrame with streaks encoded to integer representation.
    """
    df.loc[df['Streak'].str.contains('W'), 'Streak'] = df.loc[df['Streak'].str.contains('W'), 'Streak'].str.replace('W ', '')
    df.loc[df['Streak'].str.contains('L'), 'Streak'] =  df.loc[df['Streak'].str.contains('L'), 'Streak'].str.replace('L ', '-')
    return df

def set_types(df):
    """Set data types.

    Set each column to the correct data type, either integer or float. Using default int64 and float64.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to change dtypes in.

    Returns
    -------
    DataFrame
        Returns input DataFrame with corrected dtypes.
    """
    # set numeric integer columns to int64 type
    df = df.astype({'Tm':'int64', 'Opp':'int64', 'OT':'int64', 'G':'int64', 'W':'int64', 'L':'int64', 'Streak':'int64', 'A':'int64', 'H':'int64', 'N':'int64'})

    # set numeric float columns to float64 type
    df = df.astype({'SRS':'float64'})

    return df

def create_features(df):
    """Create schedule features.

    Create features from schedule data for exploration and use in the model.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to add features to.

    Returns
    -------
    DataFrame
        Returns input DataFrame with schedule features added.
    """
    # create empty DataFrame for features
    features = pd.DataFrame()

    # create groupby objects to extract features from
    means = df.groupby(['Year', 'School']).mean().reset_index()
    meds = df.groupby(['Year', 'School']).median().reset_index()
    maxes = df.groupby(['Year', 'School']).max().reset_index()
    sums = df.groupby(['Year', 'School']).sum().reset_index()

    # create features from groupby objects
    features['School'] = means.School
    features['Year'] = means.Year
    features['SRS_OPP'] = means.SRS
    features['PTS_Mean'] = means.Tm
    features['OPP_Mean'] = means.Opp
    features['MV_Median'] = means.Tm - means.Opp
    features['MV_Median'] = meds.Tm - meds.Opp
    features['G'] = maxes.G
    features['W'] = maxes.W
    features['L'] = maxes.L
    features['W%'] = (maxes.W / maxes.G)
    features['WS'] = maxes.Streak
    features['WS6'] = [1 if x >=6 else 0 for x in maxes.Streak]
    features['OT'] = sums.OT
    features['HG%'] = (sums.H / maxes.G)
    features['AG%'] = (sums.A / maxes.G)
    features['NG%'] = (sums.N / maxes.G)

    return features

def features_by_venue(df, features, venue):
    """Create fetures split by which venue the game is played at.

    Create features split by home games, away games, and neutral site games.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing schedule data to make features from.
    features : pandas DataFrame
        DataFrame to add features to.
    venue : char
        Character indicating the game site. A = Away, H = Home, N = Neutral.

    Returns
    -------
    DataFrame
        DataFrame with features.
    """
    means = df.loc[df[venue] == 1].groupby(['Year', 'School']).mean()
    meds = df.loc[df[venue] == 1].groupby(['Year', 'School']).median()
    games = df.loc[df[venue] == 1].groupby(['Year', 'School']).sum()

    if venue == 'N':
        # fix indexes for teams without nuetral site games
        mask = df.groupby(['Year', 'School']).sum().reset_index()
        mask = mask.loc[mask['N'] != 0]
        index = mask.index

        means.index = index
        meds.index = index
        games.index = index
    else:
        # flatten group objects
        means.reset_index(inplace=True)
        meds.reset_index(inplace=True)
        games.reset_index(inplace=True)

    features[venue+'_PTS_Mean'] = means.Tm
    features[venue+'_OPP_Mean'] = means.Opp
    features[venue+'_MV_Mean'] = means.Tm - means.Opp
    features[venue+'_MV_Median'] = meds.Tm - meds.Opp
    features[venue+'G'] = games[venue]
    features[venue+'W%'] = (games['Result'] / games[venue])

    return features

def features_not_home(df, features):
    """Create features of Home vs Not Home games.

    Split features by played at home, and not played at Home. Essentially combining away and neutral site games into one category.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the base data.
    features : DataFrame
        DataFrame to build the features into.

    Returns
    -------
    DataFrame
        Returns a DataFrame with the features.
    """
    # create features for combined non-home games
    means = df.loc[df['H'] != 1].groupby(['Year', 'School']).mean().reset_index()
    meds = df.loc[df['H'] != 1].groupby(['Year', 'School']).median().reset_index()
    games = df.loc[df['H'] != 1].groupby(['Year', 'School']).sum().reset_index()

    features['XH_PTS_Mean'] = means.Tm
    features['XH_OPP_Mean'] = means.Opp
    features['XH_MV_Mean'] = means.Tm - means.Opp
    features['XH_MV_Median'] = meds.Tm - meds.Opp
    features['XHG'] = games.A + games.N
    features['XHW%'] = (games.Result / (games.A + games.N))

    # fill null values from teams without nuetral site games with 0 except NW%
    features.loc[features['NW%'].isna(), ['N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'N_MV_Median', 'NG']] = features.loc[features['NW%'].isna(), ['N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'N_MV_Median', 'NG']].fillna(value=0)

    return features

df = prepare_frame(schedule)
df = venue(df)
df = result(df)
df = overtime(df)
df = streaks(df)
df = set_types(df)
features = create_features(df)
features = features_by_venue(df, features, 'A')
features = features_by_venue(df, features, 'H')
features = features_by_venue(df, features, 'N')
features = features_not_home(df, features)

# impute neutral site winning percentage with 0 for teams without neutral site games
features['NW%'] = features['NW%'].fillna(0)

# round features to 1 significant figure
features.loc[:, ['SRS_OPP', 'PTS_Mean', 'OPP_Mean', 'A_PTS_Mean', 'A_OPP_Mean', 'A_MV_Mean', 'H_PTS_Mean', 'H_OPP_Mean', 'H_MV_Mean', 'N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'XH_PTS_Mean', 'XH_OPP_Mean', 'XH_MV_Mean']] = features.loc[:, ['SRS_OPP', 'PTS_Mean', 'OPP_Mean', 'A_PTS_Mean', 'A_OPP_Mean', 'A_MV_Mean', 'H_PTS_Mean', 'H_OPP_Mean', 'H_MV_Mean', 'N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'XH_PTS_Mean', 'XH_OPP_Mean', 'XH_MV_Mean']].round(1)

# round percentages to 3 significant figures
features.loc[:, ['W%', 'HG%', 'AG%', 'NG%', 'AW%', 'HW%', 'NW%', 'XHW%']] = features.loc[:, ['W%', 'HG%', 'AG%', 'NG%', 'AW%', 'HW%', 'NW%', 'XHW%']].round(3)

# fix features data types
features.loc[:, 'NG'] = features.loc[:, 'NG'].astype('int64')

features.to_csv(r"Data/Clean/clean_schedule.csv", mode='w', index=False)
