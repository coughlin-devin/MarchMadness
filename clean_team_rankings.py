import pandas as pd
import re
import json

df = pd.read_csv(r"Data/Raw/team_rankings.csv")
ap = pd.read_csv(r"Data/Clean/clean_ap_poll.csv")
a = ap.loc[ap['Year'] > 1997]
m = a.merge(df, how='inner', on=['School', 'Year'])

# import alternate school name map
with open("alternate_school_names.json", 'r') as f:
        schools = json.load(f)

# func to simplify school names for matching
def format_school_name(school):
    # replace - with space
    school = school.replace('-', ' ')
    # strip non-alpha-numeric characters
    school = re.sub('[^a-zA-Z ]', '', school)
    # make it lowercase
    school = school.lower()
    return school

# create school name columns for merging
df['School'] = [schools[format_school_name(x)] if format_school_name(x) in schools.keys() else x for x in df['School']]
df = a.merge(df, how='inner', on=['School', 'Year'])
df = df.drop(['Pre', 'AP_Mean', 'AP_3WMean', 'AP_5WMean','AP_10WMean', 'AP_Min', 'AP_Max', 'AP_Last'], axis=1)

# replace '--' place holder with NaN
df = df.replace('--', float('nan'))

# remove % sign and divide by 100 for 3 sig figs in range [0-1]
percent = []
for col in df.columns:
    val = df[col].iloc[0]
    if '%' in str(val):
        percent.append(col)
df.loc[:, percent] = df.loc[:, percent].replace('%', '', regex=True).astype('float64') / 100

df = df.drop(['win-pct-close-games Last 3', 'win-pct-close-games Last 1', 'win-pct-close-games Home', 'win-pct-close-games Away', 'opponent-win-pct-close-games Last 3', 'opponent-win-pct-close-games Last 1', 'opponent-win-pct-close-games Home', 'opponent-win-pct-close-games Away'], axis=1)

# fill Kansas 2003 missing Last 1 stats with the average of its Last 3 and season mean stats
for i, column in enumerate(df.loc[(df['School']=='Kansas') & (df['Year'] == 2003)]):
    if df.loc[(df['School']=='Kansas') & (df['Year'] == 2003), column].isna().any():
        last3 = float(df.loc[(df['School']=='Kansas') & (df['Year'] == 2003)].iloc[:,i-1].item())
        mean = float(df.loc[(df['School']=='Kansas') & (df['Year'] == 2003)].iloc[:,i-2].item())
        fill_value = round((last3 + mean) / 2, 3)
        df.loc[(df['School']=='Kansas') & (df['Year'] == 2003), column] = df.loc[(df['School']=='Kansas') & (df['Year'] == 2003), column].fillna(str(fill_value))

# fill in missing assist--per--turnover-ratio Last 1 and opponent-assist--per--turnover-ratio Last 1 values
ast_to = df.loc[df['assist--per--turnover-ratio Last 1'].isna(), ['assist--per--turnover-ratio', 'assist--per--turnover-ratio Last 3']].astype('float64').mean(axis=1).round(3).astype('str')
opp_ast_to = df.loc[df['opponent-assist--per--turnover-ratio Last 1'].isna(), ['opponent-assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio Last 3']].astype('float64').mean(axis=1).round(3).astype('str')
df.loc[df['assist--per--turnover-ratio Last 1'].isna(), 'assist--per--turnover-ratio Last 1'] = df.loc[df['assist--per--turnover-ratio Last 1'].isna(), 'assist--per--turnover-ratio Last 1'].fillna(ast_to)
df.loc[df['opponent-assist--per--turnover-ratio Last 1'].isna(), 'opponent-assist--per--turnover-ratio Last 1'] = df.loc[df['opponent-assist--per--turnover-ratio Last 1'].isna(), 'opponent-assist--per--turnover-ratio Last 1'].fillna(opp_ast_to)

# got 0.754 by assuming the ration of last previous years win% to previous years close-game win% is the same
df.loc[df['win-pct-close-games'].isna(), 'win-pct-close-games'] = df.loc[df['win-pct-close-games'].isna(), 'win-pct-close-games'].fillna(str(0.754))

df.loc[df['opponent-win-pct-close-games'].isna(), 'opponent-win-pct-close-games'] = df.loc[df['opponent-win-pct-close-games'].isna(), 'opponent-win-pct-close-games'].fillna(str(0.174))

# correct dtypes
round_columns = list(df.columns)
round_columns.remove('School')
round_columns.remove('Year')
type_map = {}
for col in round_columns:
    type_map.update({col:'float64'})
df = df.astype(type_map)

# round everything to a maximum of 3 decimal places
df = df.apply(lambda x: round(x,3))

df.to_csv(r"Data/Clean/team_rankings.csv", mode='w', index=False)

# NOTE: used to add school mappings to alternate_school_names.json
# missing = []
# for x in a.School.unique():
#     if x not in df.Team.unique():
#         missing.append(x)
#         print(x)
#
# def jaccard_similarity(x,y):
#   """ returns the jaccard similarity between two lists """
#   intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
#   union_cardinality = len(set.union(*[set(x), set(y)]))
#   return intersection_cardinality/float(union_cardinality)
#
# def find_school(l1, l2):
#     couples = pd.DataFrame(columns=['Target Name', 'Name', 'Jaccard'])
#     for i in l1:
#         max = 0
#         school = ''
#         for j in l2:
#             jac = jaccard_similarity(i,j)
#             if jac > max:
#                 max = jac
#                 school = j
#         couples.loc[len(couples)] = (i, school, max)
#     return couples.sort_values(by='Jaccard', ascending=False).reset_index(drop=True)
#
# couples = find_school(missing, df.Team.unique())
# fifty = couples[:50]
# index = [4, 8, 9, 10, 12, 14, 16]
# add = fifty.drop(index)
# add.apply(lambda x: schools.update({format_school_name(x['Name']):x['Target Name']}), axis=1)
