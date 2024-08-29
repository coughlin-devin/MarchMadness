import pandas as pd
import numpy as np

ap = pd.read_csv(r"Data/Raw/ap_poll_table.csv")

# reorder columns
cols = ap.columns.tolist()
cols.remove('Year')
cols.insert(3, 'Year')

df = ap[cols]
df.replace('-', np.nan, inplace=True)
df.loc[:, df.columns != 'School'] = df.loc[:, df.columns != 'School'].apply(pd.to_numeric, axis=1)
df.loc[:, (df.columns != 'School') & (df.columns != 'Year')] = 26 - df.loc[:, (df.columns != 'School') & (df.columns != 'Year')]
df.fillna(value=0, inplace=True)

df['APAVG'] = 0
df['APMAVG'] = 0
df['APMIN'] = 0
df['APMAX'] = 0
df['APD'] = df.Final - df.Pre

def add_features(df, start_year, end_year):
    for year in range(start_year, end_year+1):
        this_year = df.loc[df['Year'] == year]
        cols = [x for x in df.columns if str(year) in x]
        df.loc[df['Year'] == year, 'APAVG'] = df.loc[df['Year'] == year, cols].mean(axis=1).round(1)
        df.loc[df['Year'] == year, 'APMIN'] = df.loc[df['Year'] == year, cols].min(axis=1)
        df.loc[df['Year'] == year, 'APMAX'] = df.loc[df['Year'] == year, cols].max(axis=1)
        cols.append('Final')
        df.loc[df['Year'] == year, 'APMAVG'] = df.loc[df['Year'] == year, cols[-5:]].mean(axis=1)
    return df

df = add_features(df, 1997, 2024)
df = df[['School', 'Year', 'Pre', 'Final', 'APAVG', 'APMAVG', 'APMIN', 'APMAX', 'APD']]

df.to_csv(r"Data/Clean/clean_ap_poll.csv", mode='w', index=False)
