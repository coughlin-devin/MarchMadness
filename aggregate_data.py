import pandas as pd

schedule = pd.read_csv(r"Data/Clean/clean_schedule.csv")
ap = pd.read_csv(r"Data/Clean/clean_ap_poll.csv")

player_pg_all = pd.read_csv(r"Data/Clean/clean_player_per_game_complete.csv")
player_pg_pruned = pd.read_csv(r"Data/Clean/clean_player_per_game_pruned.csv")

advanced = pd.read_csv(r"Data/Clean/clean_school_advanced.csv")
basic_opp = pd.read_csv(r"Data/Clean/clean_basic_opponent.csv")
advanced_opp = pd.read_csv(r"Data/Clean/clean_advanced_opponent.csv")

# merge schedule and ap
schedule_ap = schedule.merge(ap, how='left', on=['School', 'Year'])

# TODO: revisit clean_player for added per_40 and per_100 data and to make sure imputation methods are concrete and can be done per year, add knn manhatten distance
# merge player_pg_all and player_pg_pruned
player = player_pg_all.merge(player_pg_pruned, how='left', on=['School', 'Year'], suffixes=['', '_Pruned'])

# merge basic_opp and advanced_opp
schhool_opp = basic_opp.merge(advanced_opp, how='left', on=['School', 'Year'])
