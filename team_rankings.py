import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = r"https://www.teamrankings.com{}"

# NOTE: includes March Madness games
full_seasons = {
    1998: "1998-03-31",
    1999: "1999-03-30",
    2000: "2000-04-04",
    2001: "2001-04-03",
    2002: "2002-04-02",
    2003: "2003-04-08",
    2004: "2004-04-06",
    2005: "2005-04-05",
    2006: "2006-04-04",
    2007: "2007-04-03",
    2008: "2008-04-08",
    2009: "2009-04-07",
    2010: "2010-04-06",
    2011: "2011-04-04",
    2012: "2012-04-02",
    2013: "2013-04-08",
    2014: "2014-04-07",
    2015: "2015-04-06",
    2016: "2016-04-05",
    2017: "2017-04-04",
    2018: "2018-04-03",
    2019: "2019-04-09",
    2021: "2021-04-06",
    2022: "2022-04-05",
    2023: "2023-04-04",
    2024: "2024-04-09",
}

# NOTE: excludes March Madness games
regular_seasons = {
    1998: "1998-03-11",
    1999: "1999-03-10",
    2000: "2000-03-15",
    2001: "2001-03-12",
    2002: "2002-03-11",
    2003: "2003-03-17",
    2004: "2004-03-15",
    2005: "2005-03-14",
    2006: "2006-03-13",
    2007: "2007-03-12",
    2008: "2008-03-21",
    2009: "2009-03-16",
    2010: "2010-03-15",
    2011: "2011-03-14",
    2012: "2012-03-12",
    2013: "2013-03-18",
    2014: "2014-03-17",
    2015: "2015-03-16",
    2016: "2016-03-14",
    2017: "2017-03-13",
    2018: "2018-03-12",
    2019: "2019-03-18",
    2021: "2021-03-20",
    2022: "2022-03-15",
    2023: "2023-03-14",
    2024: "2024-03-19",
}

scoring = ["/ncaa-basketball/stat/points-per-game","/ncaa-basketball/stat/average-scoring-margin","/ncaa-basketball/stat/offensive-efficiency","/ncaa-basketball/stat/floor-percentage","/ncaa-basketball/stat/1st-half-points-per-game","/ncaa-basketball/stat/2nd-half-points-per-game","/ncaa-basketball/stat/overtime-points-per-game","/ncaa-basketball/stat/average-1st-half-margin","/ncaa-basketball/stat/average-2nd-half-margin","/ncaa-basketball/stat/average-overtime-margin","/ncaa-basketball/stat/points-from-2-pointers","/ncaa-basketball/stat/points-from-3-pointers","/ncaa-basketball/stat/percent-of-points-from-2-pointers","/ncaa-basketball/stat/percent-of-points-from-3-pointers","/ncaa-basketball/stat/percent-of-points-from-free-throws"]

shooting = ["/ncaa-basketball/stat/shooting-pct","/ncaa-basketball/stat/effective-field-goal-pct","/ncaa-basketball/stat/three-point-pct","/ncaa-basketball/stat/two-point-pct","/ncaa-basketball/stat/free-throw-pct","/ncaa-basketball/stat/true-shooting-percentage","/ncaa-basketball/stat/field-goals-made-per-game","/ncaa-basketball/stat/field-goals-attempted-per-game","/ncaa-basketball/stat/three-pointers-made-per-game","/ncaa-basketball/stat/three-pointers-attempted-per-game","/ncaa-basketball/stat/free-throws-made-per-game","/ncaa-basketball/stat/free-throws-attempted-per-game","/ncaa-basketball/stat/three-point-rate","/ncaa-basketball/stat/two-point-rate","/ncaa-basketball/stat/fta-per-fga","/ncaa-basketball/stat/ftm-per-100-possessions","/ncaa-basketball/stat/free-throw-rate","/ncaa-basketball/stat/non-blocked-2-pt-pct"]

rebounding = ["/ncaa-basketball/stat/offensive-rebounds-per-game","/ncaa-basketball/stat/defensive-rebounds-per-game","/ncaa-basketball/stat/total-rebounds-per-game","/ncaa-basketball/stat/offensive-rebounding-pct","/ncaa-basketball/stat/defensive-rebounding-pct","/ncaa-basketball/stat/total-rebounding-percentage"]

blocks_steals = ["/ncaa-basketball/stat/blocks-per-game","/ncaa-basketball/stat/steals-per-game","/ncaa-basketball/stat/block-pct","/ncaa-basketball/stat/steals-perpossession","/ncaa-basketball/stat/steal-pct"]

assists_turnovers = ["/ncaa-basketball/stat/assists-per-game","/ncaa-basketball/stat/turnovers-per-game","/ncaa-basketball/stat/turnovers-per-possession","/ncaa-basketball/stat/assist--per--turnover-ratio", "/ncaa-basketball/stat/assists-per-fgm","/ncaa-basketball/stat/assists-per-possession","/ncaa-basketball/stat/turnover-pct"]

fouls = ["/ncaa-basketball/stat/personal-fouls-per-game","/ncaa-basketball/stat/personal-fouls-per-possession","/ncaa-basketball/stat/personal-foul-pct"]

scoring_defense = ["/ncaa-basketball/stat/opponent-points-per-game","/ncaa-basketball/stat/opponent-average-scoring-margin","/ncaa-basketball/stat/defensive-efficiency","/ncaa-basketball/stat/opponent-floor-percentage","/ncaa-basketball/stat/opponent-1st-half-points-per-game","/ncaa-basketball/stat/opponent-2nd-half-points-per-game","/ncaa-basketball/stat/opponent-overtime-points-per-game","/ncaa-basketball/stat/opponent-points-from-2-pointers","/ncaa-basketball/stat/opponent-points-from-3-pointers","/ncaa-basketball/stat/opponent-percent-of-points-from-2-pointers","/ncaa-basketball/stat/opponent-percent-of-points-from-3-pointers","/ncaa-basketball/stat/opponent-percent-of-points-from-free-throws"]

shooting_defense = ["/ncaa-basketball/stat/opponent-shooting-pct","/ncaa-basketball/stat/opponent-effective-field-goal-pct","/ncaa-basketball/stat/opponent-three-point-pct","/ncaa-basketball/stat/opponent-two-point-pct","/ncaa-basketball/stat/opponent-free-throw-pct","/ncaa-basketball/stat/opponent-true-shooting-percentage","/ncaa-basketball/stat/opponent-field-goals-made-per-game","/ncaa-basketball/stat/opponent-field-goals-attempted-per-game","/ncaa-basketball/stat/opponent-three-pointers-made-per-game","/ncaa-basketball/stat/opponent-three-pointers-attempted-per-game","/ncaa-basketball/stat/opponent-free-throws-made-per-game","/ncaa-basketball/stat/opponent-free-throws-attempted-per-game","/ncaa-basketball/stat/opponent-three-point-rate","/ncaa-basketball/stat/opponent-two-point-rate","/ncaa-basketball/stat/opponent-fta-per-fga","/ncaa-basketball/stat/opponent-ftm-per-100-possessions","/ncaa-basketball/stat/opponent-free-throw-rate","/ncaa-basketball/stat/opponent-non-blocked-2-pt-pct"]

opponnet_rebounding = ["/ncaa-basketball/stat/opponent-offensive-rebounds-per-game","/ncaa-basketball/stat/opponent-defensive-rebounds-per-game","/ncaa-basketball/stat/opponent-team-rebounds-per-game","/ncaa-basketball/stat/opponent-total-rebounds-per-game","/ncaa-basketball/stat/opponent-offensive-rebounding-pct","/ncaa-basketball/stat/opponent-defensive-rebounding-pct"]

opponnet_blocks_steals = ["/ncaa-basketball/stat/opponent-blocks-per-game","/ncaa-basketball/stat/opponent-steals-per-game","/ncaa-basketball/stat/opponent-block-pct","/ncaa-basketball/stat/opponent-steals-perpossession","/ncaa-basketball/stat/opponent-steal-pct"]

opponnet_assists_turnovers = ["/ncaa-basketball/stat/opponent-assists-per-game","/ncaa-basketball/stat/opponent-turnovers-per-game","/ncaa-basketball/stat/opponent-assist--per--turnover-ratio","/ncaa-basketball/stat/opponent-assists-per-fgm","/ncaa-basketball/stat/opponent-assists-per-possession","/ncaa-basketball/stat/opponent-turnovers-per-possession","/ncaa-basketball/stat/opponent-turnover-pct"]

opponnet_fouls = ["/ncaa-basketball/stat/opponent-personal-fouls-per-game","/ncaa-basketball/stat/opponent-personal-fouls-per-possession","/ncaa-basketball/stat/opponent-personal-foul-pct"]

other = ["/ncaa-basketball/stat/games-played","/ncaa-basketball/stat/possessions-per-game","/ncaa-basketball/stat/extra-chances-per-game","/ncaa-basketball/stat/effective-possession-ratio","/ncaa-basketball/stat/opponent-effective-possession-ratio"]

winning_percentage = ["/ncaa-basketball/stat/win-pct-all-games","/ncaa-basketball/stat/win-pct-close-games","/ncaa-basketball/stat/opponent-win-pct-all-games","/ncaa-basketball/stat/opponent-win-pct-close-games"]

stats = scoring + shooting + rebounding + blocks_steals + assists_turnovers + fouls + scoring_defense + shooting_defense + opponnet_rebounding + opponnet_blocks_steals + opponnet_assists_turnovers + opponnet_fouls + other + winning_percentage

seasons = []
for year in range(1998, 2024+1):
    # ignore Covid year when there was no NCAA tournament
    if year != 2020:
        frames = []
        driver = webdriver.Firefox()
        # os.mkdir(r"Team Rankings/Scoring/{}".format(year))
        for stat in stats:
            page = url.format(stat + "?date={}".format(regular_seasons[year]))
            driver.get(page)
            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//table[@id='DataTables_Table_0']"))).get_attribute('outerHTML')
            df = pd.read_html(StringIO(str(element)), flavor='bs4')[0]
            df = df.drop(['Rank', df.columns[-1]], axis=1)
            col_name = stat.split('/')[-1]
            df = df.rename(columns={df.columns[1]:col_name, 'Last 3':col_name+' Last 3', 'Last 1':col_name+' Last 1', 'Home':col_name+' Home', 'Away':col_name+' Away'})
            frames.append(df)
        driver.quit()
        df = pd.concat(frames, axis=1)
        df.insert(0, 'Year', year)
        seasons.append(df)
    print(year)

df = pd.concat(seasons, axis=0)
schools = df.iloc[:,1]
df.insert(0, 'School', schools)
df = df.drop('Team',axis=1).reset_index(drop=True)
df = df.sort_values(by=['Year', 'School'])

df.to_csv(r"Data/Raw/team_rankings.csv", mode='w', index=False)
