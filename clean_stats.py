import pandas as pd
import json
import re
from imputation import gaussian_mean_imputation

# TODO: opponent stats
def get_data(table):
    data = pd.read_csv(r"Data/Raw/{}.csv".format(table))
    return data.copy()

adv = get_data("advanced_stats_table")
opp = get_data("basic_opp_table")
adv_opp = get_data("advanced_opp_table")

def clean_frame(df, column_key):
    # drop unnecessary columns
    keep_cols = [col for col in df.columns if column_key in col]
    keep_cols.insert(0, 'Year')
    keep_cols.insert(0, 'Unnamed: 1_level_0')
    df = df[keep_cols]

    # rename columns
    df.columns = df.iloc[0]
    df.columns.values[1] = 'Year'

    # remove label rows
    rank_mask = df.loc[df['School'].isna()].index
    df = df.drop(rank_mask, axis=0)
    df = df.drop(0, axis=0)

    # remove non-NCAA tournament schools
    non_tourney = df.loc[~df['School'].str.contains('NCAA')].index
    df = df.drop(non_tourney, axis=0).reset_index(drop=True)

    # NOTE: remove NCAA from school names, split by special character \xa0 instead of space
    df.loc[:, 'School'] = [x.split('\xa0')[0] for x in df.loc[:, 'School']]

    return df

adv = clean_frame(adv, 'Advanced')
opp = clean_frame(opp, 'Opponent')
adv_opp = clean_frame(adv_opp, 'Opponent Advanced')

def fix_dtypes(df):
    # set columns to appropriate data types
    df = df.astype({'Year':'int64'})
    df[df.columns[2:]] = df[df.columns[2:]].astype('float64')

    # divide % stats by 100 to make range [0-1]
    df.loc[:, ['TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'ORB%']] = df.loc[:, ['TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'ORB%']] / 100

    return df

adv = fix_dtypes(adv)
opp = opp.astype({'Year':'int64'})
adv_opp = fix_dtypes(adv_opp)

# TODO: recreate pace stats instead of gaussian imputation
# 40 * ((Tm Poss + Opp Poss) / (2 * (Tm MP / 5)))

# fill in NaN values with random values from a gaussian (normal) distribution of all values across all teams in all years
nan_columns = adv.loc[:, adv.isna().any()].columns
adv = gaussian_mean_imputation(adv, nan_columns)

# get table without first four teams
schedule = pd.read_csv(r"Data/Clean/clean_schedule.csv")
with open(f"alternate_school_names.json", 'r', encoding='utf-8') as f:
    school_mapping = json.load(f)

# func to simplify school names for matching
def format_school_name(school):
    # replace - with space
    school = school.replace('-', ' ')
    # strip non-alpha-numeric characters
    school = re.sub('[^a-zA-Z ]', '', school)
    # make it lowercase
    school = school.lower()
    return school

# NOTE: needed to manually add some other alternate school names to the dictionary
# df.loc[df['Team'].isna()]
# school_mapping.update({'uc santa barbara':'UCSB'})
# school_mapping.update({'uc irvine':'UC-Irvine'})
# school_mapping.update({'uc davis':'UC-Davis'})
# json_object = json.dumps(school_mapping)
# with open(r"alternate_school_names.json", 'w', encoding='utf-8') as f:
#     f.write(json_object)

# TODO: remove schdule and extra name columns
def remove_first_four(df):
    # merge df with schedule to get only schools in 64 team tournament
    df['Team'] =  [x if x in schedule.School.values else school_mapping.get(format_school_name(x)) for x in df.School]
    merged = df.merge(schedule, how='inner', left_on=['Team', 'Year'], right_on=['School', 'Year'])

    # remove redundant school name columns
    drop_cols = schedule.columns
    drop_cols = drop_cols.drop(['School', 'Year'])
    merged = merged.drop(drop_cols, axis=1)
    merged = merged.drop(['School_x', 'Team'], axis=1)
    merged = merged.rename(columns={'School_y': 'School'})
    school = merged.School
    merged.drop('School', axis=1, inplace=True)
    merged.insert(0, 'School', school)

    return merged

adv = remove_first_four(adv)
opp = remove_first_four(opp)
adv_opp = remove_first_four(adv_opp)

adv.to_csv(r"Data/Clean/clean_school_advanced.csv", mode='w', index=False)
opp.to_csv(r"Data/Clean/clean_basic_opponent.csv", mode='w', index=False)
adv_opp.to_csv(r"Data/Clean/clean_advanced_opponent.csv", mode='w', index=False)
