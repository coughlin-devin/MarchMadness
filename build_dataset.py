import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os

def add_keys(df, school, year):
    """Add keys school and year to a DataFrame.

    Use the school name and the year as keys for the DataFrames.

    Parameters
    ----------
    df : DataFrame
        The Datarfame to add the keys to.
    school : string
        The school name.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Return the Datarfame with the added keys.
    """
    df['School'] = school
    df['Year'] = int(year)
    return df

# NOTE: find html tag by attribute using dictionary {attr : name}
def get_seed(soup):
    """Get the NCAA tournament seed of a school.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.

    Returns
    -------
    int
        Returns the seed number.
    """
    summary = soup.find(attrs={'data-template':'Partials/Teams/Summary'})
    seed = re.search("#[0-9]+ seed", summary.get_text()).group(0)
    seed = re.search("[0-9]+", seed).group(0)
    return seed

# NOTE: check NCAA opponent in first NCAA game to check if it was a play-in game
def get_rounds(soup, year):
    """Get number of rounds played in NCAA and conference tournament.

    Get the number of rounds each team progresses in the NCAA tournament and their conference tournament.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    year : int
        The tournament year.

    Returns
    -------
    tuple of (int, int, int)
        Returns a tuple of three integers for the number of rounds.
    """
    schedule = soup.find('table', id='schedule')
    schedule = pd.read_html(str(schedule), flavor='bs4')[0]

    # NCAA tournament rounds
    ncaa_games = schedule.loc[(schedule['Type'] == 'NCAA')]
    if '{}.html'.format(ncaa_games.iloc[0].Opponent) in os.listdir(r"Roster & Stats/{}".format(year)):
        # wins and losses are in the 8th column which doesn't have good name
        ncaa_wins = ncaa_games.loc[(schedule.iloc[:,8] == 'W')]
        play_in_rounds = 0
    else:
        ncaa_games = ncaa_games.iloc[1:]
        ncaa_wins = ncaa_games.loc[(schedule.iloc[:,8] == 'W')]
        play_in_rounds = 1
    ncaa_rounds = len(ncaa_wins) + 1

    # conference tournament rounds
    conf_games = schedule.loc[schedule['Type'] == 'CTOURN']
    if len(conf_games) > 0:
        conf_wins = conf_games.loc[conf_games.iloc[:,8] == 'W']
        conf_rounds = len(conf_wins) + 1
    else:
        conf_rounds = 0
    return (ncaa_rounds, play_in_rounds, conf_rounds)

# NOTE: don't count ncaa games towards longest win streak because ncaa games are in the test set
def get_win_streak(soup):
    """Get a teams longest win streak.

    Get the longest win streak of a team up to their first NCAA game or NCAA play-in game. Conference tournament games are counted in the win streak.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.

    Returns
    -------
    int
        Returns the longest win streak prior to the first NCAA game or NCAA play-in game.
    """
    schedule = soup.find('table', id='schedule')
    schedule = pd.read_html(str(schedule), flavor='bs4')[0]
    streaks = schedule.loc[schedule['Type'] != 'NCAA', 'Streak']
    win_streaks = streaks.loc[streaks.str.contains('W', na=False)]
    win_streak = int(re.search("[0-9]+", win_streaks.max()).group(0))
    return win_streak

def is_conference_champion(soup):
    """Check if a team won their cofnerence regular season and conference tournament.

    Determine whether a team is their conferences regular season champion and conference tournament champion.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.

    Returns
    -------
    tuple of (boolean, boolean)
        Returns a tuple of two boolean values 0 or 1 as integers indicating if a team is their conference regular season or tournament champion.
    """
    reg_season = 0
    tourney = 0
    bling = soup.find(id="bling")
    if bling is not None:
        titles = bling.find_all("a")
        for title in titles:
            if 'Reg Season' in title.get_text():
                reg_season = 1
            if 'Tourney' in title.get_text():
                tourney = 1
    return (reg_season, tourney)

def build_roster(soup, school, year):
    """Create a DataFrame containing roster information.

    Converts an HTML table object into a DataFrame. The DataFrame will contain roster information like player position, height, and weight.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of roster information.
    """
    roster = soup.find('table', id="roster")
    roster = pd.read_html(str(roster), flavor='bs4')[0]
    roster = add_keys(roster, school, year)
    return roster

# WARNING: 2021 VCU is missing seed (10 seed)
def build_per_game_team_opp(soup, school, year):
    """Create a DataFrame containing basic team and opponent stats and rankings.

    Converts an HTML table object into a DataFrame. The DataFrame will contain stats like shooting percentages, offensive and defensive rebounds, and points scored per game for both the team and opponent.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of basic team and opponent stats and rankings.
    """
    # select table of team and opponent stats
    season_total_per_game = soup.find('table', id="season-total_per_game")
    season_total_per_game = pd.read_html(str(season_total_per_game), flavor='bs4')[0]

    # seperate each row
    team = pd.DataFrame(season_total_per_game.iloc[0, 1:]).T
    team_rank = pd.DataFrame(season_total_per_game.iloc[1, 1:]).T.reset_index(drop=True)
    opponent = pd.DataFrame(season_total_per_game.iloc[2, 1:]).T.reset_index(drop=True)
    opponent_rank = pd.DataFrame(season_total_per_game.iloc[3, 1:]).T.reset_index(drop=True)

    # rename columns before combining into a single row
    team_rank.columns = [col+'r' for col in team_rank.columns]
    opponent.columns = [col+"_OP" for col in opponent.columns]
    opponent_rank.columns = [col+"R_OP" for col in opponent_rank.columns]

    # combine into a single row
    team = pd.concat([team, team_rank], axis=1)
    opponent = pd.concat([opponent, opponent_rank], axis=1)
    per_game_team_opp = pd.concat([team, opponent], axis=1)

    # add other feature columns
    per_game_team_opp = add_keys(per_game_team_opp, school, year)

    # VCU missing seed in 2021
    if (school == 'VCU') and (year == 2021):
        per_game_team_opp['Seed'] = 10
    else:
        per_game_team_opp['Seed'] = get_seed(soup)

    with open(r"Schedule & Results/{}/{}.html".format(year, school), 'r', encoding='utf-8') as f:
        page = f.read()
    stew = BeautifulSoup(page, 'html.parser')
    ncaa_rounds, play_in_rounds, conf_rounds = get_rounds(stew, year)
    per_game_team_opp['Rounds'] = ncaa_rounds
    per_game_team_opp['PIR'] = play_in_rounds
    per_game_team_opp['CTR'] = conf_rounds

    season_champ, conf_champ = is_conference_champion(soup)
    per_game_team_opp['CSC'] = season_champ
    per_game_team_opp['CTC'] = conf_champ

    win_streak = get_win_streak(stew)
    per_game_team_opp['WS'] = win_streak

    return per_game_team_opp

def build_per_game_player(soup, school, year):
    """Create a DataFrame containing player per game statistics.

    Converts an HTML table object into a DataFrame. The DataFrame will contain stats like player shooting percentages, assists, steals, blocks, rebounds and points scored per game.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of basic per game player statistics.
    """
    per_game = soup.find('table', id='per_game')
    per_game = pd.read_html(str(per_game), flavor='bs4')[0]
    per_game = add_keys(per_game, school, year)
    return per_game

# WARNING: south carolina state 1998 is missing table
def build_per_40(soup, school, year):
    """Create a DataFrame containing player per 40 minute statistics.

    Converts an HTML table object into a DataFrame. The DataFrame will contain stats like player shooting percentages, assists, steals, blocks, rebounds and points scored per 40 minutes.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of basic per 40 minutes player statistics.
    """
    per_min = soup.find('table', id='per_min')
    if per_min is not None:
        per_min = pd.read_html(str(per_min), flavor='bs4')[0]
        per_min = add_keys(per_min, school, year)
        return per_min
    else:
        return None

def build_schedule(soup, school, year):
    """Create a DataFrame containing team schedule information.

    Converts an HTML table object into a DataFrame. The DataFrame will contain information like win streaks and simple rating system scores.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of team schedule information.
    """
    schedule = soup.find('table', id='schedule')
    schedule = pd.read_html(str(schedule), flavor='bs4')[0]
    schedule = add_keys(schedule, school, year)
    return schedule

def build_ap_poll(soup, school, year):
    """Create a DataFrame containing AP poll ranings.

    Converts an HTML table object into a DataFrame. The DataFrame will contain AP rankings for the team.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of AP poll rankings.
    """
    ap_poll = soup.find('table', id='polls')
    ap_poll = pd.read_html(str(ap_poll), flavor='bs4')[0]
    ap_poll = add_keys(ap_poll, school, year)
    return ap_poll

def build_basic_stats(year):
    """Create a DataFrame containing basic team statistics.

    Converts an HTML table object into a DataFrame. The DataFrame will contain basic teams stats like shooting% and rebounding for all Division 1 teams.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of basic team statistics.
    """
    with open(r"School Stats/Basic/basic_{}.html".format(year), 'r', encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    basic_school_stats = soup.find('table', id='basic_school_stats')
    basic_school_stats = pd.read_html(str(basic_school_stats), flavor='bs4')[0]
    basic_school_stats['Year'] = int(year)
    return basic_school_stats

def build_advanced_stats(year):
    """Create a DataFrame containing advanced team statistics.

    Converts an HTML table object into a DataFrame. The DataFrame will contain advanced teams stats like shooting efficiency for all Division 1 teams.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        BeautifulSoup parsed HTML object.
    school : string
        The name of the school.
    year : int
        The tournament year.

    Returns
    -------
    DataFrame
        Returns a DataFrame of advanced team statistics.
    """
    with open(r"School Stats/Advanced/advanced_{}.html".format(year), 'r', encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    advanced_school_stats = soup.find('table', id='adv_school_stats')
    advanced_school_stats = pd.read_html(str(advanced_school_stats), flavor='bs4')[0]
    advanced_school_stats['Year'] = int(year)
    return advanced_school_stats

def build_DataFrames(start_year, end_year):
    """Build DataFrames from HTML data.

    Parameters
    ----------
    start_year : int
        First year of data to use.
    end_year : int
        Last year of data to use.

    Returns
    -------
    tuple of 8 DataFrames
        Returns a tuple of 8 DataFrames and writes to file.
    """
    # create folder for the data
    if not os.path.exists(r"Data"):
        os.mkdir(r"Data")

    # lists to store tables
    basic_school_stats = []
    advanced_school_stats = []
    rosters = []
    team_opp = []
    per_game = []
    per_min = []
    schedules = []
    ap_polls = []

    for year in range(start_year, end_year+1):
        if year != 2020:

            # build tables from School Stats page
            basic_stats = build_basic_stats(year)
            advanced_stats = build_advanced_stats(year)
            basic_school_stats.append(basic_stats)
            advanced_school_stats.append(advanced_stats)

            # ncaa_64 = get_64(year)
            for file in os.listdir(r"Roster & Stats/{}".format(year)):
                school, ext = os.path.splitext(file)

                # build tables from Roster & Stats page
                with open(r"Roster & Stats/{}/{}.html".format(year, school), 'r', encoding='utf-8') as f:
                    page = f.read()
                soup = BeautifulSoup(page, 'html.parser')
                roster = build_roster(soup, school, year)
                per_game_team_opp = build_per_game_team_opp(soup, school, year)
                per_game_player = build_per_game_player(soup, school, year)
                per_40 = build_per_40(soup, school, year)
                rosters.append(roster)
                team_opp.append(per_game_team_opp)
                per_game.append(per_game_player)
                if per_40 is not None:
                    per_min.append(per_40)

                # build tables from Schedule and Polls page
                with open(r"Schedule & Results/{}/{}.html".format(year, school), 'r', encoding='utf-8') as f:
                    page = f.read()
                soup = BeautifulSoup(page, 'html.parser')
                schedule = build_schedule(soup, school, year)
                ap_poll = build_ap_poll(soup, school, year)
                schedules.append(schedule)
                ap_polls.append(ap_poll)

    # concatenate into DataFrames
    basic_stats_df = pd.concat(basic_school_stats).reset_index(drop=True)
    advanced_stats_df = pd.concat(advanced_school_stats).reset_index(drop=True)
    roster_df = pd.concat(rosters).reset_index(drop=True)
    team_opp_df = pd.concat(team_opp).reset_index(drop=True)
    player_df = pd.concat(per_game).reset_index(drop=True)
    per_40_df = pd.concat(per_min).reset_index(drop=True)
    schedule_df = pd.concat(schedules).reset_index(drop=True)
    ap_poll_df = pd.concat(ap_polls).reset_index(drop=True)

    # save to csv files
    basic_stats_df.to_csv(r"Data/basic_stats_table.csv", mode='w')
    advanced_stats_df.to_csv(r"Data/advanced_stats_table.csv", mode='w')
    roster_df.to_csv(r"Data/roster_table.csv", mode='w')
    team_opp_df.to_csv(r"Data/team_opp_table.csv", mode='w')
    player_df.to_csv(r"Data/player_table.csv", mode='w')
    per_40_df.to_csv(r"Data/per_40_table.csv", mode='w')
    schedule_df.to_csv(r"Data/schedule_table.csv", mode='w')
    ap_poll_df.to_csv(r"Data/ap_poll_table.csv", mode='w')

    return (basic_stats_df, advanced_stats_df, roster_df, team_opp_df, player_df, per_40_df, schedule_df, ap_poll_df)

basic_stats_df, advanced_stats_df, roster_df, team_opp_df, player_df, per_40_df, schedule_df, ap_poll_df = build_DataFrames(1997, 2024)
