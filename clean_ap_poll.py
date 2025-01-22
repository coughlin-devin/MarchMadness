import pandas as pd
import numpy as np

ap = pd.read_csv(r"Data/Raw/ap_poll_table.csv")

# reorder columns so Year is first column
cols = ap.columns.tolist()
cols.remove('Year')
cols.insert(0, 'Year')

# convert ap rankings to numeric type
df = ap[cols]
df.replace('-', np.nan, inplace=True)
df.loc[:, df.columns != 'School'] = df.loc[:, df.columns != 'School'].apply(pd.to_numeric, axis=1)
df.loc[:, (df.columns != 'School') & (df.columns != 'Year')] = 26 - df.loc[:, (df.columns != 'School') & (df.columns != 'Year')]
school = ap.School
df = df.loc[:, df.columns != 'School'].astype('float64')
df = df.fillna(value=0)
df.insert(0, value=school, column='School')
# # WARNING: Final ap ranking comes out after tournament is over
df = df.drop('Final', axis=1)

# create columns for new features
df['AP_Mean'] = 0.0
df['AP_3WMean'] = 0.0
df['AP_5WMean'] = 0.0
df['AP_10WMean'] = 0.0
df['AP_Min'] = 0
df['AP_Max'] = 0
df['AP_Last'] = 0

def add_features(df, start_year, end_year):
    """Add features to the DataFrame.

    Calculate and add new features related to AP poll to the DataFrame.

    Parameters
    ----------
    df : panda DataFrame
        DataFrame of AP poll data.
    start_year : int
        First year of data to use in computation.
    end_year : int
        Last year of data to use in computation.

    Returns
    -------
    pandas DataFrame
        Returns DataFrame with new AP poll features.
    """
    for year in range(start_year, end_year+1):
        if year != 2020:
            # this_year = df.loc[df['Year'] == year]
            cols = [x for x in df.columns if str(year) in x]
            df.loc[df['Year'] == year, 'AP_Mean'] = df.loc[df['Year'] == year, cols].mean(axis=1).round(1)
            df.loc[df['Year'] == year, 'AP_3WMean'] = df.loc[df['Year'] == year, cols[-3:]].mean(axis=1).round(1)
            df.loc[df['Year'] == year, 'AP_5WMean'] = df.loc[df['Year'] == year, cols[-5:]].mean(axis=1).round(1)
            df.loc[df['Year'] == year, 'AP_10WMean'] = df.loc[df['Year'] == year, cols[-10:]].mean(axis=1).round(1)
            df.loc[df['Year'] == year, 'AP_Min'] = df.loc[df['Year'] == year, cols].min(axis=1)
            df.loc[df['Year'] == year, 'AP_Max'] = df.loc[df['Year'] == year, cols].max(axis=1)
            df.loc[df['Year'] == year, 'AP_Last'] = df.loc[df['Year'] == year, cols[-1:]].astype('int64').iloc[:,0]
    return df

df = add_features(df, 1997, 2024)

# get slice
df = df[['School', 'Year', 'Pre', 'AP_Mean', 'AP_3WMean', 'AP_5WMean', 'AP_10WMean', 'AP_Min', 'AP_Max', 'AP_Last']]

# fix types
df = df.astype({'Year':'int64', 'Pre':'int64'})

df.to_csv(r"Data/Clean/clean_ap_poll.csv", mode='w', index=False)
