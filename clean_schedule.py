import pandas as pd

# import data
schedule = pd.read_csv(r"Data/Raw/schedule_table.csv", low_memory=False)

def prepare_frame(frame):
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

    # remove 13 (as of 2024) games with null wins/losses bc they were cancelled or postponed
    df.drop(df.loc[df['W'].isna()].index, inplace=True)

    return df

# clean Venue column so that home games are 1 and non-home games are 0 and combine split venue columns
def venue(df):
    venue = df.Venue.fillna('H') + df.Venue2.fillna('H')
    map_venue = {
        'HH':'H',
        'NH':'N',
        'HN':'N',
        '@H':'A',
        'H@':'A'
    }
    df.Venue = [map_venue[x] for x in venue]
    df = pd.concat([df, pd.get_dummies(df.Venue)], axis=1)
    df.drop(['Venue', 'Venue2'], axis=1, inplace=True)
    return df

# clean Result column so that Wins are 1, Losses are 0, and combine split win columns
def result(df):
    results = df.Result.fillna("L") + df.Result2.fillna("L")
    df.Result = [1 if 'W' in x else 0 for x in results]
    df.drop(['Result2'], axis=1, inplace=True)
    return df

# clean OT column to remove W and L labels
def overtime(df):
    overtime = df.loc[~df['OT'].isna()]
    ot = [1 if x == 'OT' else int(x[0]) for x in overtime.OT]
    df.loc[~df['OT'].isna(), 'OT'] = ot
    df.loc[df['OT'].isna(), 'OT'] = df.loc[df['OT'].isna(), 'OT'].fillna(value=0)
    return df

# set streak to integer type where win streak is positive and losing streak is negative
def streaks(df):
    df.loc[df['Streak'].str.contains('W'), 'Streak'] = df.loc[df['Streak'].str.contains('W'), 'Streak'].str.replace('W ', '')
    df.loc[df['Streak'].str.contains('L'), 'Streak'] =  df.loc[df['Streak'].str.contains('L'), 'Streak'].str.replace('L ', '-')
    return df

def set_types(df):
    # set numeric integer columns to int64 type
    df.loc[:, ['Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak', 'A', 'H', 'N']] = df.loc[:, ['Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak', 'A', 'H', 'N']].astype('int64')

    # set numeric float columns to float64 type
    df.loc[:, 'SRS'] = df.loc[:, 'SRS'].astype('float64')

    return df

def create_features(df):
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

# function to create features split by home, away, or nuetral site games
def features_by_venue(df, features, venue):
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

# round features to 1 significant figure
features.loc[:, ['SRS_OPP', 'PTS_Mean', 'OPP_Mean', 'A_PTS_Mean', 'A_OPP_Mean', 'A_MV_Mean', 'H_PTS_Mean', 'H_OPP_Mean', 'H_MV_Mean', 'N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'XH_PTS_Mean', 'XH_OPP_Mean', 'XH_MV_Mean']] = features.loc[:, ['SRS_OPP', 'PTS_Mean', 'OPP_Mean', 'A_PTS_Mean', 'A_OPP_Mean', 'A_MV_Mean', 'H_PTS_Mean', 'H_OPP_Mean', 'H_MV_Mean', 'N_PTS_Mean', 'N_OPP_Mean', 'N_MV_Mean', 'XH_PTS_Mean', 'XH_OPP_Mean', 'XH_MV_Mean']].round(1)

# round percentages to 3 significant figures
features.loc[:, ['W%', 'HG%', 'AG%', 'NG%', 'AW%', 'HW%', 'NW%', 'XHW%']] = features.loc[:, ['W%', 'HG%', 'AG%', 'NG%', 'AW%', 'HW%', 'NW%', 'XHW%']].round(3)

# fix features data types
features.loc[:, 'NG'] = features.loc[:, 'NG'].astype('int64')

features.to_csv(r"Data/Clean/clean_schedule.csv", mode='w', index=False)
