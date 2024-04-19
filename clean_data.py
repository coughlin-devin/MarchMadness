import pandas as pd
from bs4 import BeautifulSoup
import os

# TODO: redo function headers for consistency and include input and return values

# ids of interest
# bling
# meta
# roster
# season-total_per_game
# per game

def possessions(FGA, FG, ORB, DRB, TOV, FTA, FGA_OP, FG_OP, ORB_OP, DRB_OP, TOV_OP, FTA_OP, weight_ft=0.475, weight_reb=1.07):
    """
    Estimate the number of offensive possesions for a team.
    FGA - Field Goal Attempts
    FG - Field Goals Made
    ORB - Offensive Rebounds
    DRB - Deffensive Rebounds
    TOV - Turn Overs
    FTA - Free Throw Attempts
    _OP - Suffix indicating opponent team's stat.
    weight_ft - Probability of the free throw attempt being a made last free throw, or being a missed free throw with a defensive rebound.
            This is calculated from college game data but may not be accurate/optimal.
    weight_reb - Weight placed on percentage of missed field goals that result in offensive rebounds.
    """

    # basic formula for estimating number of possessions for a single team
    simple = FGA - ORB + TOV + (0.475*FTA)

    # parts of surgical calclation
    team_half = FGA + weight_ft*FTA - weight_reb*(ORB / (ORB + DRB_OP)) * (FGA-FG) + TOV
    opp_half = FGA_OP + weight_ft*FTA_OP - weight_reb*(ORB_OP / (ORB_OP + DRB)) * (FGA_OP-FG_OP) + TOV_OP

    # theoretically more precise formula for estimating number of possesions from basketball-reference.com
    surgical = 0.5 * (team_half + opp_half)

    return (simple, surgical)

# TODO: keep track of special school names like UCLA to correctly format them into more common form ex. brigham-boung -> BYU
def format_school_name(school):
    """Format school name into common team name."""
    team = school.replace(".html", "")
    team = team.replace("-", " ")
    # team = team.title()
    return team

def add_team_year(df, team, year):
    df["Team"] = team
    df["Year"] = int(year)
    return df

# IDEA: weight each players height by some metric like minutes played, possessions played, games played etc
def avg_height():
    """Calculate average team height."""
    pass

def interior_height():
    """Calculate average team height of Centers and Forwards."""
    pass

def exterior_height():
    """Calculate average team height of Gaurds."""
    pass

# IDEA: weight each players height by some metric like minutes played, possessions played, games played etc
def avg_weight():
    """Calculate average team weight."""
    pass

def interior_weight():
    """Calculate average team weight of Centers and Forwards."""
    pass

def exterior_weight():
    """Calculate average team weight of Gaurds."""
    pass

def build_roster(soup, team_name, year):
    """Turn roster table html into a DataFrame."""
    html = soup.find("table", id="roster")
    roster = pd.read_html(str(html), flavor="bs4")[0]
    roster = add_team_year(roster, team_name, year)
    return roster

def get_seed(soup):
    pass

def tournament_performance(soup):
    pass

def is_conference_champion(soup):
    pass

def is_conference_tournament_champion(soup):
    pass

# TODO: seed, # of rounds progressed, conference tournament champ? conference season champ?
def build_team_per_game(soup, team_name, year):
    """Combine team and opponnet html tables into a DataFrame."""
    html = soup.find("table", id="season-total_per_game")
    table = pd.read_html(str(html), flavor="bs4")[0]
    team = pd.DataFrame(table.iloc[0, 1:]).T
    team_rank = pd.DataFrame(table.iloc[1, 1:]).T.reset_index(drop=True)
    team_rank.columns = [col+"R" for col in team_rank.columns]
    opponent = pd.DataFrame(table.iloc[2, 1:]).T.reset_index(drop=True)
    opponent.columns = [col+"_OP" for col in opponent.columns]
    opponent_rank = pd.DataFrame(table.iloc[3, 1:]).T.reset_index(drop=True)
    opponent_rank.columns = [col+"R_OP" for col in opponent_rank.columns]
    team = pd.concat([team, team_rank], axis=1)
    opponent = pd.concat([opponent, opponent_rank], axis=1)
    team_per_game = pd.concat([team, opponent], axis=1)
    team_per_game = add_team_year(team_per_game, team_name, year)
    return team_per_game

def build_player_per_game(soup, team_name, year):
    """Turn player per game html table into a DataFrame."""
    html = soup.find("table", id="per_game")
    player_per_game = pd.read_html(str(html), flavor="bs4")[0]
    player_per_game = add_team_year(player_per_game, team_name, year)
    return player_per_game

def build_per_40(soup, team_name, year):
    # WARNING: south carolina state 1998 missing table
    """Turn player per 40 minutes html table into a DataFrame."""
    html = soup.find("table", id="per_min")
    if html is not None:
        per_40 = pd.read_html(str(html), flavor="bs4")[0]
        per_40 = add_team_year(per_40, team_name, year)
        return per_40
    else:
        return None

# TODO: need to collect seed and how many wins from brackets
def build_dataframes(start_year, end_year):
    rosters = []
    team_per_games = []
    player_per_games = []
    players_per_40s = []
    for year in range(start_year, end_year+1):
        if year != 2020:
            for school in os.listdir("Seasons/{}".format(year)):
                team_name = format_school_name(school)
                with open("Seasons/{}/{}".format(year, school), "r") as f:
                    page = f.read()
                soup = BeautifulSoup(page, "html.parser")
                roster = build_roster(soup, team_name, year)
                team_per_game = build_team_per_game(soup, team_name, year)
                player_per_game = build_player_per_game(soup, team_name, year)
                per_40 = build_per_40(soup, team_name, year)
                rosters.append(roster)
                team_per_games.append(team_per_game)
                player_per_games.append(player_per_game)
                if per_40 is not None:
                    players_per_40s.append(per_40)

            roster_data = pd.concat(rosters).reset_index(drop=True)
            team_data = pd.concat(team_per_games).reset_index(drop=True)
            player_data = pd.concat(player_per_games).reset_index(drop=True)
            per_40_data = pd.concat(players_per_40s).reset_index(drop=True)

    roster_data.to_csv("Data/roster_table.csv", mode="w")
    team_data.to_csv("Data/team_table.csv", mode="w")
    player_data.to_csv("Data/player_table.csv", mode="w")
    per_40_data.to_csv("Data/per_40_table.csv", mode="w")

    return (roster_data, team_data, player_data, per_40_data)

# TODO: function to pull data from csv files instead now that they are made
roster_data, team_data, player_data, per_40_data = build_dataframes(1997, 2024)

# NOTE on RSCI (Risky) top 100 rankings: https://sites.google.com/site/rscihoops/home Link to experts: https://sites.google.com/site/rscihoops/home/the-experts
# The formula for calculating the RSCI rankings is quite simple and completely objective. Here’s how it works:
#     The process begins with a single top 100 list from one of the experts.
#     The players listed are assigned points based on their position on that list. The top ranked player is given 100 points, #2 gets 99 points, #3 gets 98, and so on with #100 getting 1 point.
#     Repeat step 2 for each of the top 100 lists.
#     Finally, add up the scores based on all the lists and sort the players by their score in descending order.
#
# It’s not exactly rocket science but it does achieve the desired effect of providing a more unbiased, consensus ranking, which one might argue, is even more accurate than any of the single top 100 lists alone.
#
# However, this process is not without its pitfalls:
#     The RSCI formula is objective but the underlying ratings it is based on are not. This subjective aspect should never be underestimated.
#     Some experts include 5th year and prep school players (denoted by “*” in the RSCI rankings) in their top 100 lists and other don’t.
#     This means that a really great 5th year player might be ranked #10 by one expert and not listed at all by the others, thereby dropping his RSCI ranking dramatically. In other words, RSCI rankings of 5th year players aren’t worth much.
#     By its very nature the RSCI rankings get less and less accurate the further down the list you move. The reason is twofold.
#     First, the affect of a single, high rating from one expert can effectively override the prevailing opinion of the other experts that may have left the player off their lists entirely.
#     Also, a player that just narrowly misses making 1 or more top 100 lists receives no points from those lists and is not effectively distinguished from all the others that were not ranked.
#     For example, a guy ranked #101 gets the same zero points as a guy ranked #250 even though they clearly aren’t that close. Stated simply: “a miss is as good as a mile.”

def clean_roster(roster_data):
    """
        Fill in missing Height with Pos average if possible, else total average.
        Fill in missing Pos with best guess?
        Fill in missing Weigth with Pos average if possible, else total average.
        Turn RSCI into team stat about recruiting class strength.
        Check previous and following seasons to match name to fill in Class.
    """

    pass

def clean_team(team_data):
    """
        replace missing data with average from team from previouis past years?
        replace missing data with average from other teams from same year?
    """
    pass

# WARNING: need to clean bracket data better and extract tournament seed, # rounds advanced, if won conference tournamnent, if won conference season,
2019-2011 68
2010-2001 65
team_data.loc[team_data["Year"] == 1997]
team_data.reset_index(drop=True).info()
