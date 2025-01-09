import pandas as pd
from imputation import gaussian_mean_imputation

team_opp_df = pd.read_csv(r"Data/Raw/team_opp_table.csv")
team_opp_df.isna().sum()[50:]

# BUG: GR (rank of how many games the team payed) and GR_OP (same but averaged for opposing teams) were mistakenly created while building the DataFrame. Just remove these since they are useless.
df = team_opp_df.drop(['GR', 'GR_OP'], axis=1)

# columns of team ranking in regards to specific stats
rank_columns = ['MPR', 'FGR', 'FGAR', 'FG%R', '2PR', '2PAR', '2P%R', '3PR', '3PAR', '3P%R', 'FTR', 'FTAR', 'FT%R', 'ORBR', 'DRBR', 'TRBR', 'ASTR', 'STLR', 'BLKR', 'TOVR', 'PFR', 'PTSR', 'MPR_OP', 'FGR_OP', 'FGAR_OP', 'FG%R_OP', '2PR_OP', '2PAR_OP', '2P%R_OP', '3PR_OP', '3PAR_OP', '3P%R_OP', 'FTR_OP', 'FTAR_OP', 'FT%R_OP', 'ORBR_OP', 'DRBR_OP', 'TRBR_OP', 'ASTR_OP', 'STLR_OP', 'BLKR_OP', 'TOVR_OP', 'PFR_OP', 'PTSR_OP']

# NOTE: rankings might be useful as a way to eliminate influence from yearly trends, for example if average ORB was higher 20 years ago and has trended down, bad teams 20 years a go could have better ORB than the best teams today
# drop rankings, just use stats.
df = df.drop(rank_columns, axis=1)

df.isna().sum()
# fill NaN minutes played with 40 minutes per game
df.loc[:, 'MP'] = df.loc[:, 'MP'].fillna(40)
df.loc[:, 'MP_OP'] = df.loc[:, 'MP_OP'].fillna(40)

# fill in NaN values with random values from a gaussian (normal) distribution of all values across all teams in all years
nan_columns = df.loc[:, df.isna().any()].columns
df = gaussian_mean_imputation(df, nan_columns)

# put school and year at beginning of df
school = df.pop('School')
year = df.pop('Year')
df.insert(0, 'Year', year)
df.insert(0, 'School', school)

# renmae all _OP stats to be _OPP
df.columns = [col + 'P' if '_OP' in col else col for col in df.columns]

# round to 3 sig figs if percentage otherwise 1 sig fig
percentages = [col for col in df.drop(['School', 'Year'], axis=1).columns if '%' in col]
df.loc[:, percentages] = df.loc[:, percentages].round(3)
df.loc[:, ~df.columns.isin(percentages)] = df.loc[:, ~df.columns.isin(percentages)].round(1)

# set columns to appropriate data types
df = df.astype({'G': 'int64', 'G_OPP': 'int64'})

df

# TODO: look at historical correct brackets and check for patterns in them, are they symetric acros each region?

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

# TODO: create advanced features and per possesions

# TODO: clean per possession stats for 2010-2024 so i dont have to calculate those
