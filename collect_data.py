import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

# WARNING: delay needs to be more than 3 seconds to avoid 20 requests per minute for sportsreference website
def scrape(url):
    """Scrape web page html with delay to avoid rate limiting."""
    page = requests.get(url, timeout=5)
    time.sleep(3.1)
    return page

def scrape_brackets(url, year):
    """Scrape NCAA bracket."""
    page = scrape(url.format(year))
    with open("Brackets/bracket_{}.html".format(year), "w+") as f:
        f.write(page.text)

def get_tournament_teams(tournament_teams, year):
    """Get the urls to tournament teams."""
    with open(r"Brackets/bracket_{}.html".format(year), "r") as f:
        page = f.read()
    soup = BeautifulSoup(page, "html.parser")
    brackets = soup.find(id="brackets")
    links = brackets.find_all("a")
    team_urls = set()
    for link in links:
        if "schools" in link["href"]:
            team_urls.add(link["href"])
    tournament_teams.update({year:team_urls})
    return tournament_teams

def scrape_season(url, team_url, year):
    """Scrape the team info for the season."""
    school = team_url.split(r"/")[3]
    page = scrape(url.format(team_url))
    if not os.path.exists("Seasons/{}".format(year)):
        os.mkdir("Seasons/{}".format(year))
    with open("Seasons/{}/{}.html".format(year, school), "w+", encoding="utf-8") as f:
        f.write(page.text)

def main():
    # create folders for web scraped html data
    if not os.path.exists("Brackets"):
        os.mkdir("Brackets")
    if not os.path.exists("Seasons"):
        os.mkdir("Seasons")

    # website source of data
    url = r"https://www.sports-reference.com{}"
    bracket_url = r"https://www.sports-reference.com/cbb/postseason/men/{}-ncaa.html"

    # dictionary for {year : [tournament team urls]}
    tournament_teams = {}

    # NOTE: start from year 1997 bc that is the first year offensive and defensive rebounds are tracked in the data source
    start_year = 2024
    end_year = 2024
    for year in range(start_year, end_year+1):
        # ignore Covid year when there was no NCAA tournament
        if year != 2020:
            scrape_brackets(bracket_url, year)
            tournament_teams = get_tournament_teams(tournament_teams, year)
            for team_url in tournament_teams[year]:
                scrape_season(url, team_url, year)

if __name__ == "__main__":
    main()
