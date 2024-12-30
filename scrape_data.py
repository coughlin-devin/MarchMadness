import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json

def get_ncaa_schools(map_names, year):
    """Create a list of each school in the NCAA tournament in a given year.

    Parameters
    ----------
    map_names : dictionary of {string : string}
        Dictionary mapping alternate school name to common school name.
    year : int
        Year of tournament.

    Returns
    -------
    tuple
        Returns a tuple containing a list of <a> html tags with a link to the teams page and the name of the school and a dictionary of school name mappings.
    """
    with open(r"NCAA Tournament/bracket_{}.html".format(year), 'r', encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    brackets = soup.find_all(id='bracket')
    ncaa_schools = set()
    for bracket in brackets:
        links = bracket.find_all('a')
        for link in links:
            if 'schools' in link['href']:
                ncaa_schools.add(link)
    for link in ncaa_schools:
        school_name = link.get_text()
        alt_name = link['href'].split('/')[3]
        alt_name_split = alt_name.split('-')
        name_parts = [s.lower() for s in alt_name_split]
        alt_name = ' '.join(name_parts)
        map_names.update({alt_name:school_name})
    return (ncaa_schools, map_names)

def scrape(url, delay=3.1):
    """Send an HTTP request to a web page with a time delay.

    Add a timed delay after sending an HTTP request to avoid rate limiting. For https://www.sports-reference.com the delay should be more than 3.0 seconds to avoid reaching 20 requests per minute.

    Parameters
    ----------
    url : string
        Website url.
    arg2 : float
        Number of seconds to sleep after sending HTTP request.

    Returns
    -------
    requests.Response()
        Returns a requests.Resonse Object which will contain the web page content on success.
    """
    page = requests.get(url, timeout=5)
    time.sleep(delay)
    return page

def scrape_ncaas(url, ncaa_url, year):
    """Scrape NCAA tournament brackets from a web page.

    Pull NCAA tournament brackets and save them as html files.

    Parameters
    ----------
    url : string
        Main website url.
    ncaa_url : string
        Subdirectories of the url leading to the page contining the bracket.
    year : int
        The tournament year.

    Returns
    -------
    None
        Writes to file.
    """
    page = scrape(url.format(ncaa_url))
    with open(r"NCAA Tournament/bracket_{}.html".format(year), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_basic_stats(url, stats_url, year):
    """Scrape basketball team statistics from a table on a web page.

    Pull a table of all Division 1 mens basketball program season basic team stats and save them as html files.

    Parameters
    ----------
    url : string
        Main website url.
    stats_url : string
        Subdirectories of the url leading to the page contining the table of stats.
    year : int
        The tournament year.

    Returns
    -------
    None
        Writes to file.
    """
    page = scrape(url.format(stats_url))
    with open(r"School Stats/Basic/basic_{}.html".format(year), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_basic_opp_stats(url, stats_opp_url, year):
    """Scrape basketball team opponent statistics from a table on a web page.

    Pull a table of all Division 1 mens basketball program season basic team stats and save them as html files.

    Parameters
    ----------
    url : string
        Main website url.
    stats_opp_url : string
        Subdirectories of the url leading to the page contining the table of opponent stats.
    year : int
        The tournament year.

    Returns
    -------
    None
        Writes to file.
    """
    page = scrape(url.format(stats_opp_url))
    with open(r"School Stats/Basic/basic_opp_{}.html".format(year), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_advanced_stats(url, advanced_url, year):
    """Scrape basketball team statistics from a table on a web page.

    Pull a table of all Division 1 mens basketball program season advanced team stats and save them as html files.

    Parameters
    ----------
    url : string
        Main website url.
    advanced_url : string
        Subdirectories of the url leading to the page contining the table of stats.
    year : int
        The tournament year.

    Returns
    -------
    None
        Writes to file.
    """
    page = scrape(url.format(advanced_url))
    with open(r"School Stats/Advanced/advanced_{}.html".format(year), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_advanced_opp_stats(url, advanced_opp_url, year):
    """Scrape basketball team statistics from a table on a web page.

    Pull a table of all Division 1 mens basketball program season advanced team stats and save them as html files.

    Parameters
    ----------
    url : string
        Main website url.
    advanced_opp_url : string
        Subdirectories of the url leading to the page contining the table of opponent stats.
    year : int
        The tournament year.

    Returns
    -------
    None
        Writes to file.
    """
    page = scrape(url.format(advanced_opp_url))
    with open(r"School Stats/Advanced/advanced_opp_{}.html".format(year), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_roster_stats(url, school, year):
    """Scrape roster infromation and team stats from a table on a web page.

    Save a web page with roster information, team and opponent stats, and player stats to an html file.

    Parameters
    ----------
    url : string
        Website url.
    school : bs4.element.tag
        An html <a> tag containing a link to the schools page and the name of the school.
    year : int
        The year of the season the team played.

    Returns
    -------
    None
        Writes to file.
    """
    school_name = school.get_text()
    school_url = school['href']
    page = scrape(url.format(school_url))
    if not os.path.exists(r"Roster & Stats/{}".format(year)):
        os.mkdir(r"Roster & Stats/{}".format(year))
    with open(r"Roster & Stats/{}/{}.html".format(year, school_name), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_schedule_results(url, school, year):
    """Scrape schedule and game results.

    Save a web page containing the team's schedule, results, and win streak information.

    Parameters
    ----------
    url : string
        Website url.
    school : bs4.element.tag
        An html <a> tag containing a link to the schools page and the name of the school.
    year : int
        The year of the season the team played.

    Returns
    -------
    None
        Writes to file.
    """
    school_name = school.get_text()
    school_url = school['href']
    schedule_url = school_url.replace('.html', '-schedule.html')
    page = scrape(url.format(schedule_url))
    if not os.path.exists(r"Schedule & Results/{}".format(year)):
        os.mkdir(r"Schedule & Results/{}".format(year))
    with open(r"Schedule & Results/{}/{}.html".format(year, school_name), 'w', encoding='utf-8') as f:
        f.write(page.text)

def scrape_data(start_year, end_year):
    """Scrape all the data for each year from https://www.sports-reference.com.

    Get data from each page from school stats, team and player stats, roster, schedule and ncaa bracket information.

    Parameters
    ----------
    start_year : int
        First year to begin collecting data from.
    end_year : int
        Last year to begin collecting data from.

    Returns
    -------
    None
        Writes to file.
    """

    # create folders for web scraped html data
    if not os.path.exists(r"NCAA Tournament"):
        os.mkdir(r"NCAA Tournament")
    if not os.path.exists(r"School Stats"):
        os.mkdir(r"School Stats")
        os.mkdir(r"School Stats/Basic")
        os.mkdir(r"School Stats/Advanced")
    if not os.path.exists(r"Roster & Stats"):
        os.mkdir(r"Roster & Stats")
    if not os.path.exists(r"Schedule & Results"):
        os.mkdir(r"Schedule & Results")

    # website urls of source of data
    url = r"https://www.sports-reference.com{}"
    ncaa_url = r"/cbb/postseason/men/{}-ncaa.html"
    stats_url = r"/cbb/seasons/men/{}-school-stats.html"
    advanced_url = r"/cbb/seasons/men/{}-advanced-school-stats.html"
    stats_opp_url = r"/cbb/seasons/men/{}-opponent-stats.html"
    advanced_opp_url = r"/cbb/seasons/men/{}-advanced-opponent-stats.html"

    # dictionary object to map alternate school names to common school names
    map_names = {}

    for year in range(start_year, end_year+1):
        # ignore Covid year when there was no NCAA tournament
        if year != 2020:
            # scrape per 100 possessions and opponent advanced stats starting from first availabel season in 2010
            if year >= 2010:
                scrape_basic_opp_stats(url, stats_opp_url.format(year), year)
                scrape_advanced_opp_stats(url, advanced_opp_url.format(year), year)
            scrape_basic_stats(url, stats_url.format(year), year)
            scrape_advanced_stats(url, advanced_url.format(year), year)
            scrape_ncaas(url, ncaa_url.format(year), year)
            ncaa_schools, map_names = get_ncaa_schools(map_names, year)
            for school in ncaa_schools:
                scrape_roster_stats(url, school, year)
                scrape_schedule_results(url, school, year)

    # write school name mapping dictionary to json file
    json_object = json.dumps(map_names)
    with open(r"alternate_school_names.json", 'w', encoding='utf-8') as f:
        f.write(json_object)

# NOTE: 1997 is the first year offensive and defensive rebounds are tracked, which is important for calculating advanced stats
scrape_data(1997, 2024)
