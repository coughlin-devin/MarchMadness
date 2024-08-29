import pandas as pd

schedule = pd.read_csv(r"Data/Raw/schedule_table.csv", low_memory=False)
schedule.rename(columns={"Unnamed: 3": "Venue", "Unnamed: 4": "Venue2"}, inplace=True)
schedule.rename(columns={"Unnamed: 7": "Result", "Unnamed: 8": "Result2"}, inplace=True)


df = schedule[['School', 'Year', 'Type', 'Venue', 'Venue2', 'Opponent','Result', 'Result2', 'Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak']]

# remove header rows that are just column names
headers = df.loc[df['Type'] == 'Type']
df = df.drop(headers.index)

# remove NCAA games so they don't corrupt training features data
ncaa_games = df.loc[df['Type'] == 'NCAA']
df = df.drop(ncaa_games.index)

# remove 13 (as of 2024) games with null wins/losses bc they were cancelled or postponed
df.drop(df.loc[df['W'].isna()].index, inplace=True)

# clean Venue column so that home games are 1 and non-home games are 0 and combine split venue columns
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

# clean Result column so that Wins are 1, Losses are 0, and combine split win columns
results = df.Result.fillna("L") + df.Result2.fillna("L")
df.Result = [1 if 'W' in x else 0 for x in results]
df.drop(['Result2'], axis=1, inplace=True)

# clean OT column to remove W and L labels
overtime = df.loc[~df['OT'].isna()]
ot = [1 if x == 'OT' else int(x[0]) for x in overtime.OT]
df.loc[~df['OT'].isna(), 'OT'] = ot
df.loc[df['OT'].isna(), 'OT'] = df.loc[df['OT'].isna(), 'OT'].fillna(value=0)

# set streak to integer type where win streak is positive and losing streak is negative
df.loc[df['Streak'].str.contains('W'), 'Streak'] = df.loc[df['Streak'].str.contains('W'), 'Streak'].str.replace('W ', '')
df.loc[df['Streak'].str.contains('L'), 'Streak'] =  df.loc[df['Streak'].str.contains('L'), 'Streak'].str.replace('L ', '-')

# set numeric integer columns to int64 type
df.loc[:, ['Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak', 'A', 'H', 'N']] = df.loc[:, ['Tm', 'Opp', 'OT', 'G', 'W', 'L', 'Streak', 'A', 'H', 'N']].astype('int64')

# create empty DataFrame for features
features = pd.DataFrame()

# create features
means = df.groupby(['Year', 'School']).mean().round(1)
meds = df.groupby(['Year', 'School']).median()
maxes = df.groupby(['Year', 'School']).max()
sums = df.groupby(['Year', 'School']).sum()

means.reset_index(inplace=True)
meds.reset_index(inplace=True)
maxes.reset_index(inplace=True)
sums.reset_index(inplace=True)

features['School'] = means.School
features['Year'] = means.Year
features['MPTS'] = means.Tm
features['MOPP'] = means.Opp
features['MNMV'] = means.Tm - means.Opp
features['MDMV'] = meds.Tm - meds.Opp
features['G'] = maxes.G
features['W'] = maxes.W
features['L'] = maxes.L
features['W%'] = (maxes.W / maxes.G).round(3)
features['WS'] = maxes.Streak
features['WS6'] = [1 if x >=6 else 0 for x in maxes.Streak]
features['OT'] = sums.OT
features['HG%'] = (sums.H / maxes.G).round(3)
features['AG%'] = (sums.A / maxes.G).round(3)
features['NG%'] = (sums.N / maxes.G).round(3)

def features_by_venue(df, features, venue):
    means = df.loc[df[venue] == 1].groupby(['Year', 'School']).mean().round(1)
    meds = df.loc[df[venue] == 1].groupby(['Year', 'School']).median()
    games = df.loc[df[venue] == 1].groupby(['Year', 'School']).sum()

    if venue == 'N':
        mask = df.groupby(['Year', 'School']).sum().reset_index()
        mask = mask.loc[mask['N'] != 0]
        index = mask.index

        means.index = index
        meds.index = index
        games.index = index
    else:
        means.reset_index(inplace=True)
        meds.reset_index(inplace=True)
        games.reset_index(inplace=True)

    features[venue+'MPTS'] = means.Tm
    features[venue+'MOPP'] = means.Opp
    features[venue+'MNMV'] = means.Tm - means.Opp
    features[venue+'MDMV'] = meds.Tm - meds.Opp
    features[venue+'G'] = games[venue]
    features[venue+'W%'] = (games['Result'] / games[venue]).round(3)
    return features

features = features_by_venue(df, features, 'A')
features = features_by_venue(df, features, 'H')
features = features_by_venue(df, features, 'N')

# create features for non-home games
means = df.loc[df['H'] != 1].groupby(['Year', 'School']).mean().round(1)
meds = df.loc[df['H'] != 1].groupby(['Year', 'School']).median()
games = df.loc[df['H'] != 1].groupby(['Year', 'School']).sum()
means.reset_index(inplace=True)
meds.reset_index(inplace=True)
games.reset_index(inplace=True)
features['XHMPTS'] = means.Tm
features['XHMOPP'] = means.Opp
features['XHMNMV'] = means.Tm - means.Opp
features['XHMDMV'] = meds.Tm - meds.Opp
features['XHG'] = games.A + games.N
features['XHW%'] = (games.Result / (games.A + games.N)).round(3)

# fill null values from teams without nuetral site games with 0 except NW%
features.loc[features['NW%'].isna(), ['NMPTS', 'NMOPP', 'NMNMV', 'NMDMV', 'NG']] = features.loc[features['NW%'].isna(), ['NMPTS', 'NMOPP', 'NMNMV', 'NMDMV', 'NG']].fillna(value=0)

features.to_csv(r"Data/Clean/clean_schedule.csv", mode='w', index=False)
