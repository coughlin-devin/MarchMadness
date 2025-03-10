import pandas as pd

schedule = pd.read_csv(r"Data/Clean/clean_schedule.csv")
ap = pd.read_csv(r"Data/Clean/clean_ap_poll.csv")
player_pg_pruned = pd.read_csv(r"Data/Clean/clean_player_per_game_pruned.csv")
team_opp = pd.read_csv(r"Data/Clean/clean_team_opp.csv")
team_rankings = pd.read_csv(r"Data/Clean/team_rankings.csv")
kenpom = pd.read_csv(r"Data/Clean/kenpom.csv")
# TODO: incorporate this for single game model, need to attach team and opponnent features to df, maybe go back and include round of the matchups
matchups = pd.read_csv(r"Data/Clean/clean_matchups.csv")

df = schedule.merge(ap, how='left', on=['School', 'Year'])
df = df.merge(player_pg_pruned, how='left', on=['School', 'Year'])
df = df.merge(team_opp, how='left', on=['School', 'Year'])
df = df.loc[df['Year'] >= 1998]
df = df.merge(team_rankings, how='left', on=['School', 'Year'])
df = df.loc[df['Year'] >= 2002]
df = df.merge(kenpom, how='left', on=['School', 'Year'])

# df.to_csv(r"Data/Clean/clean_aggregate.csv", mode='w', index=False)

# TODO: try excluding Pre, RSCI, and Seed with a head to head model
features = ['School', 'Year', 'WINS']
ap = ['Pre']
schedule = ['SRS_OPP']
player = ['Class_MW', 'Pos_MW', 'Height_MW', 'FG%_MW', 'Cohesion', 'Guard_RSCI Top 100_Mean', 'Forward_RSCI Top 100_Mean', 'Center_RSCI Top 100_Mean', 'Freshman_RSCI Top 100_Mean', 'Sophomore_RSCI Top 100_Mean', 'Junior_RSCI Top 100_Mean', 'Senior_RSCI Top 100_Mean']
team_opp = ['Seed', 'PIR', 'CTR', 'CSC', 'CTC', 'SRS', 'Simple_POS']
# TODO: TOV% uses 0.44 for free throw constant, calculate my own using 0.475
team_rankings = ['average-scoring-margin', 'average-2nd-half-margin', 'floor-percentage', 'opponent-floor-percentage', 'win-pct-all-games', 'effective-possession-ratio', 'extra-chances-per-game', 'games-played', 'field-goals-made-per-game', 'assist--per--turnover-ratio', 'turnover-pct', 'assists-per-possession', 'assists-per-fgm', 'total-rebounding-percentage', 'offensive-rebounding-pct', 'block-pct', 'steal-pct', 'opponent-steal-pct', 'personal-foul-pct', 'percent-of-points-from-2-pointers', 'percent-of-points-from-3-pointers', 'percent-of-points-from-free-throws', 'opponent-percent-of-points-from-2-pointers', 'opponent-percent-of-points-from-3-pointers', 'opponent-percent-of-points-from-free-throws', 'effective-field-goal-pct', 'opponent-effective-field-goal-pct', 'three-point-pct', 'two-point-pct', 'free-throw-pct', 'true-shooting-percentage', 'opponent-true-shooting-percentage', 'fta-per-fga', 'opponent-fta-per-fga', 'points-from-2-pointers']
kenpom = ['AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo', 'Pyth']

features.extend(ap)
features.extend(schedule)
features.extend(player)
features.extend(team_opp)
features.extend(team_rankings)
features.extend(kenpom)

features = df[features]
# features.to_csv(r"Data/Clean/features.csv", mode='w', index=False)
