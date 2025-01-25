import pandas as pd

schedule = pd.read_csv(r"Data/Clean/clean_schedule.csv")
ap = pd.read_csv(r"Data/Clean/clean_ap_poll.csv")

player_pg_all = pd.read_csv(r"Data/Clean/clean_player_per_game_complete.csv")
player_pg_pruned = pd.read_csv(r"Data/Clean/clean_player_per_game_pruned.csv")

advanced = pd.read_csv(r"Data/Clean/clean_school_advanced.csv")
basic_opp = pd.read_csv(r"Data/Clean/clean_basic_opponent.csv")
advanced_opp = pd.read_csv(r"Data/Clean/clean_advanced_opponent.csv")

team_opp = pd.read_csv(r"Data/Clean/clean_team_opp.csv")

# merge schedule and ap
schedule_ap = schedule.merge(ap, how='left', on=['School', 'Year'])

# TODO: revisit clean_player for added per_40 and per_100 data and to make sure imputation methods are concrete and can be done per year, add knn manhatten distance
# merge player_pg_all and player_pg_pruned
player = player_pg_all.merge(player_pg_pruned, how='left', on=['School', 'Year'], suffixes=['', '_Pruned'])

# merge basic_opp and advanced_opp
school_opp = basic_opp.merge(advanced_opp, how='left', on=['School', 'Year'])

# merge advanced stats with schedule and ap
advanced_schedule_ap = advanced.merge(schedule_ap, how='left', on=['School', 'Year'])

# merge player stats with advanced stats and schedule and ap
advanced_schedule_ap_player = advanced_schedule_ap.merge(player, how='left', on=['School', 'Year'])

df = team_opp.merge(advanced_schedule_ap_player, how='left', on=['School', 'Year'])

# NOTE: Four Factors: eFG% (40%), TOV% (25%), ORB% & DRB% (20%), FT/FGA (15%)
# eFG%
df['EFG%'] = round((df['2P'] + 1.5*df['3P']) / df['FGA'], 3)
df['EFG%_OPP'] = round((df['2P_OPP'] + 1.5*df['3P_OPP']) / df['FGA_OPP'], 3)

# TOV%
df['TOV%']
df['TOV%_OPP'] = round(df['TOV_OPP'] / (df['FGA_OPP'] + (0.475*df['FTA_OPP']) + df['TOV_OPP']), 3)

# ORB% & DRB%
df['ORB%']
df['DRB%'] = round(df['DRB'] / (df['ORB_OPP'] + df['DRB']), 3)

# FTA / FGA
df['FTr']
df['FTr_OPP'] = round(df['FTA_OPP'] / df['FGA_OPP'])

# TODO: weight these features to use in the model? Or will the model decide on appropriate weights itself?
four_factors = df.loc[:, ['School', 'Year', 'EFG%', 'EFG%_OPP', 'TOV%', 'TOV%_OPP', 'ORB%', 'DRB%', 'FTr', 'FTr_OPP']]

df.to_csv(r"Data/Clean/clean_aggregate.csv", mode='w', index=False)
