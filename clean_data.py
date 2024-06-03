import pandas as pd

# TODO: add gitignore for html folders and data folder
# TODO: delete data and html folders from github

basic_stats_df = pd.read_csv(r"Data/basic_stats_table.csv")
advanced_stats_df = pd.read_csv(r"Data/advanced_stats_table.csv")
roster_df = pd.read_csv(r"Data/roster_table.csv")
team_opp_df = pd.read_csv(r"Data/team_opp_table.csv")
player_df = pd.read_csv(r"Data/player_table.csv")
per_40_df = pd.read_csv(r"Data/per_40_table.csv")
schedule_df = pd.read_csv(r"Data/schedule_table.csv")
ap_poll_df = pd.read_csv(r"Data/ap_poll_table.csv")

team_opp_df['Seed']

# TODO: flatten table indexes df.index = df.index.to_flat_index() and rename columns

# NOTE: Stats Tables
# TODO: change year column to integer type instead of float on stats tables after getting rid of the NaN above
# TODO: remove null rows where column labels repeatedly come up

# NOTE: team_opp_df
# TODO: for teams in conferences without a conference tournament, set their number of conference rounds to zero if not already

basic_stats_df.loc[pd.isna(basic_stats_df.iloc[:,2])]
basic_stats_df.info()
basic_stats_df.describe(include='all')

def clean_roster(roster_data):
    """
        Fill in missing Height with Pos average if possible, else total average.
        Fill in missing Pos with best guess?
        Fill in missing Weigth with Pos average if possible, else total average.
        Turn RSCI into team stat about recruiting class strength.
        Check previous and following seasons to match name to fill in Class.
    """

    pass

def clean_team(team_data):
    """
        replace missing data with average from team from previouis past years?
        replace missing data with average from other teams from same year?
    """
    pass

# TODO: use "chalk" brackets (always pick higher seed) as one of the baseline tests for how well model does
# IDEA: make a model to predict the seed of a team given their season stats and if they won conference tournament etc, use that seed prediction as a feature in the model
# WARNING: need to clean bracket data better and extract tournament seed, # rounds advanced, if won conference tournament, if won conference season,
2019-2011 68
2010-2001 65
roster_data.reset_index(drop=True).info()
team_data.loc[team_data["Year"] == 1997]
team_data.reset_index(drop=True).info()

player_data.reset_index(drop=True).info()
per_40_data.reset_index(drop=True).info()
