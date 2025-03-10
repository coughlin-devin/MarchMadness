import pandas as pd
import re

team_opp_df = pd.read_csv(r"Data/Raw/team_opp_table.csv")

# NOTE: GR (rank of how many games the team payed) and MPR were mistakenly created while building the DataFrame. Just remove these since they are useless.
df = team_opp_df.drop(['GR', 'MPR'], axis=1)

# drop opponnet team stats and rankings
opp_columns = [col for col in df.columns if '_OP' in col]
df = df.drop(opp_columns, axis=1)

# renmae all _OP stats to be _OPP
# df.columns = [col + 'P' if '_OP' in col else col for col in df.columns]

# rename rank stats to say Rank
df.columns = [col + 'ank' if col[-1] == 'R' else col for col in df.columns]
df = df.rename(columns={'PIRank': 'PIR', 'CTRank':'CTR'})

# put school and year at beginning of df
school = df.pop('School')
year = df.pop('Year')
df.insert(0, 'Year', year)
df.insert(0, 'School', school)

# fill NaN minutes played with 40 minutes per game
df.loc[:, 'MP'] = df.loc[:, 'MP'].fillna(40)

# TODO: docstring
def fill_mean(df, feature, index):
    for i in index:
        year = df.loc[i, 'Year']
        df.loc[i, feature] = df[['Year', feature]].groupby('Year').mean().loc[year].item()
    return df

# get indexes of teams missing features
orb_idx = df.loc[df['ORB'].isna()].index
drb_idx = df.loc[df['DRB'].isna()].index
tov_idx = df.loc[df['TOV'].isna()].index

# fill missing values with mean value from that year
df = fill_mean(df, 'ORB', orb_idx)
df = fill_mean(df, 'DRB', drb_idx)
df = fill_mean(df, 'TOV', tov_idx)

# round to 3 sig figs if percentage otherwise 1 sig fig
percentages = [col for col in df.drop(['School', 'Year'], axis=1).columns if '%' in col]
df.loc[:, percentages] = df.loc[:, percentages].round(3)
df.loc[:, ~df.columns.isin(percentages)] = df.loc[:, ~df.columns.isin(percentages)].round(1)

# NOTE: rankings might be useful as a way to eliminate influence from yearly trends, for example if average ORB was higher 20 years ago and has trended down, bad teams 20 years a go could have better ORB than the best teams today
# get view of ranks
ranks = df.loc[:, df.dtypes == 'object'].drop('School', axis=1)
# convert to float
ranks = ranks.replace('[^0-9]', '', regex=True)
df.loc[:, ranks.columns] = ranks

# NOTE: fill in missing rank approximately based on where mean value of that year would place them
# St. John's 2000
df.loc[(df['School'] == 'St. John\'s (NY)') & (df['Year'] == 2000), 'ORBRank'] = 147
df.loc[(df['School'] == 'St. John\'s (NY)') & (df['Year'] == 2000), 'DRBRank'] = 99
# Florida A&M 2004
df.loc[(df['School'] == 'Florida A&M') & (df['Year'] == 2004), 'ORBRank'] = 148
df.loc[(df['School'] == 'Florida A&M') & (df['Year'] == 2004), 'DRBRank'] = 98
# Louisianna 2000
df.loc[(df['School'] == 'Louisiana') & (df['Year'] == 2000), 'TOVRank'] = 207

# set columns to appropriate data types
type_dict = {}
for col in ranks.columns:
    type_dict.update({col:'int64'})
df = df.astype(type_dict)

# calculate possessions, also called pace or tempo
def possessions(df, weight_ft=0.475, weight_reb=1.07, opponent=False):
    """
    Estimate the number of offensive possesions for a team.
    FGA - Field Goal Attempts
    FG - Field Goals Made
    ORB - Offensive Rebounds
    DRB - Deffensive Rebounds
    TOV - Turn Overs
    FTA - Free Throw Attempts
    _OPP - Suffix indicating opponent team's stat.
    weight_ft - Probability of the free throw attempt being a made last free throw, or being a missed free throw with a defensive rebound.
            This is calculated from college game data but may not be accurate/optimal.
    weight_reb - Weight placed on percentage of missed field goals that result in offensive rebounds.
    """
    if not opponent:
        # basic formula for estimating number of possessions for a single team
        simple = df.FGA - df.ORB + df.TOV + (0.475*df.FTA)

        # # parts of surgical calclation
        # team_half = df.FGA + weight_ft*df.FTA - weight_reb*(df.ORB / (df.ORB + df.DRB_OPP)) * (df.FGA-df.FG) + df.TOV
        # opp_half = df.FGA_OPP + weight_ft*df.FTA_OPP - weight_reb*(df.ORB_OPP / (df.ORB_OPP + df.DRB)) * (df.FGA_OPP-df.FG_OPP) + df.TOV_OPP
        #
        # # theoretically more precise formula for estimating number of possesions from basketball-reference.com
        # surgical = 0.5 * (team_half + opp_half)
    else:
        simple = df.FGA_OPP - df.ORB_OPP + df.TOV_OPP + (0.475*df.FTA_OPP)

    simple = round(simple, 1)
    # surgical = round(surgical, 1)

    return simple

# calculate number of possesions per game
simple_pos = possessions(df)

df['Simple_POS'] = simple_pos
# df['Surgical_POS'] = surgical_pos

def per_possession_stats(df, surgical=False, opponent=False):
    if surgical:
        pos = 'Surgical_POS'
    else:
        pos = 'Simple_POS'

    stats = ['FG', 'FGA', '2P', '2PA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    if opponent:
        stats = [x + '_OPP' for x in stats]
    for stat in stats:
        df[stat + '_per_' + pos] = round(df[stat] / df[pos], 3)
    return df

df = per_possession_stats(df)
# df = per_possession_stats(df, opponent=True)

# drop games here because more accurate games in schedule table
df = df.drop(['G'], axis=1)

# replace tournament rounds with tournament wins
df['WINS'] = df['Rounds'] - 1
df = df.drop('Rounds', axis=1)

df.to_csv(r"Data/Clean/clean_team_opp.csv", mode='w', index=False)
