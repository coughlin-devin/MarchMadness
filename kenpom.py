import pandas as pd
import json
import re

df = pd.read_csv(r"KenPom/DEV _ March Madness.csv")

# drop rankings
rankings = [col for col in df.columns if 'Rank' in col]
df = df.drop(rankings, axis=1)
drop = ['Short Conference Name', 'Mapped Conference Name', 'Current Coach', 'Active Coaching Length Index', 'Region', 'Post-Season Tournament Sorting Index', 'Full Team Name', 'Since', 'Seed', 'Post-Season Tournament', 'Correct Team Name?', 'DFP', 'NSTRate', 'OppNSTRate', 'CenterPts', 'PFPts', 'SFPts', 'SGPts', 'PGPts', 'CenterOR', 'PFOR', 'SFOR', 'SGOR', 'PGOR', 'CenterDR', 'PFDR', 'SFDR', 'SGDR', 'PGDR']
df = df.drop(drop, axis=1)

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
df['School'] = [schools[format_school_name(x)] if format_school_name(x) in schools.keys() else x for x in df['Mapped ESPN Team Name']]

# rename some teams before merging
df.loc[df['School'] == 'McNeese', ['School']] = 'McNeese State'
df.loc[df['School'] == 'Miami', ['School']] = 'Miami (FL)'
df.loc[df['School'] == 'St. John\'s', ['School']] = 'St. John\'s (NY)'
df.loc[df['School'] == 'IU Indianapolis', ['School']] = 'IUPUI'
df.loc[df['School'] == 'SE Louisiana', ['School']] = 'Southeastern Louisiana'
df.loc[df['School'] == 'UAlbany', ['School']] = 'Albany (NY)'
df.loc[df['School'] == 'American University', ['School']] = 'American'
df.loc[df['School'] == 'Loyola Maryland', ['School']] = 'Loyola (MD)'
df.loc[df['School'] == 'Loyola Chicago', ['School']] = 'Loyola (IL)'
df.loc[df['School'] == 'Charleston', ['School']] = 'College of Charleston'

# merge with schedule to remove non March Madness Teams
schedule = pd.read_csv(r"Data/Clean/clean_schedule.csv")
gt = schedule.loc[schedule['Year'] >= 2002]
df = gt.merge(df, how='inner', left_on=['School', 'Year'], right_on=['School', 'Season'])

columns = ['School', 'Year',
       'eFGPct', 'TOPct', 'ORPct', 'FTRate', 'OffFT', 'Off2PtFG', 'Off3PtFG',
       'DefFT', 'Def2PtFG', 'Def3PtFG', 'Tempo', 'AdjTempo', 'OE', 'AdjOE',
       'DE', 'AdjDE', 'AdjEM', 'FG2Pct', 'FG3Pct', 'FTPct', 'BlockPct',
       'OppFG2Pct', 'OppFG3Pct', 'OppFTPct', 'OppBlockPct', 'FG3Rate',
       'OppFG3Rate', 'ARate', 'OppARate', 'StlRate', 'OppStlRate', 'Net Rating', 'Active Coaching Length']

df = df[columns]

df.loc[:, 'Active Coaching Length'] = df['Active Coaching Length'].replace('[^0-9]', '', regex=True)

df = df.astype({'Active Coaching Length':'float32'})
df.loc[:, 'Active Coaching Length'] = df.loc[:, 'Active Coaching Length'].fillna(df['Active Coaching Length'].median())
df = df.astype({'Active Coaching Length':'int64'})

df = df.loc[:, ['School', 'Year', 'AdjEM', 'AdjOE', 'AdjDE', 'AdjTempo']]
df.loc[:, 'AdjTempo'] = round(df.loc[:, 'AdjTempo'], 3)
df.loc[:, ['AdjEM', 'AdjOE', 'AdjDE']] = round(df.loc[:, ['AdjEM', 'AdjOE', 'AdjDE']] / 100, 3)
pythagorean_exp = 10.25
df['Pyth'] = round(df['AdjOE']**pythagorean_exp / (df['AdjOE']**pythagorean_exp + df['AdjDE']**pythagorean_exp), 3)

df.to_csv(r"Data/Clean/kenpom.csv", mode='w', index=False)
