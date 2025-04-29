import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_kenpom(start_year, end_year, prev_seasons):
    # list of dataframes to concatenate after scraping
    seasons = prev_seasons

    for year in range(start_year, end_year+1):
        # wait to avoid flagging rate limiting
        time.sleep(10)

        # ignore Covid year when there was no NCAA tournament
        if year != 2020:

            # open Firefox
            driver = webdriver.Firefox()

            # go to Kenpom web page
            page = r"https://kenpom.com/index.php/?y={}".format(year)
            driver.get(page)

            # scrape html into DataFrame
            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//table[@id='ratings-table']"))).get_attribute('outerHTML')
            df = pd.read_html(StringIO(str(element)), flavor='bs4')[0]

            # exit Firefox
            driver.quit()

            # format DataFrame
            df.insert(0, 'Year', year)
            df.columns = df.columns.to_flat_index()
            df = df.drop(df.columns[[7,9,11,13,15,17,19,21]], axis=1)
            df.columns = ['Year', 'Rk', 'Team', 'Conf', 'W-L', 'AdjEM', 'AdjOE', 'AdjDE', 'AdjT', 'Luck', 'SOS', 'AdjOE_Opp', 'AdjDE_Opp', 'NCOS']
            df = df.drop(['Rk', 'Conf', 'W-L', 'NCOS'], axis=1)

            # only keep NCAA tournament teams
            ncaa = [x for x in df.Team if re.search(r'\d', x) is not None]
            df = df.loc[df['Team'].isin(ncaa)]
            df.loc[:, 'Team'] = [re.sub(r'\d+', '', x)[:-1] for x in df.Team if re.search(r'\d', x) is not None]

            # remove seeds and rename Team column to School
            schools = [x for x in df.Team]
            df.insert(0, 'School', schools)
            df = df.drop('Team', axis=1).reset_index(drop=True)
            df = df.sort_values(by=['Year', 'School'])

            # calculate Pythagorean formula for winning percentage
            # NOTE: pythagorean_exp taken from Kenpom blog post
            pythagorean_exp = 10.25
            df['Pyth'] = round(df['AdjOE']**pythagorean_exp / (df['AdjOE']**pythagorean_exp + df['AdjDE']**pythagorean_exp), 3)

            # add DataFrame to list of other years DataFrames
            seasons.append(df)

        # show progress
        print(year)

    # combine each years DataFrames into one
    df = pd.concat(seasons, axis=0)

    return df

df = pd.read_csv(r"Data/Clean/kenpom.csv")
df = scrape_kenpom(2026, 2026, [df])
