# TODO: check how rank stats are calculated
# TODO: scrape margin of victory/result/record data
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

# IDEA: longest winning streak
# IDEA: a boolean marking if the team had a 6 win streak or longer (number of wins needed to win the NCAA tournament)

# IDEA: do some graph analysis to see which teams have more distributed scoring like in the soccermatics project

# IDEA: explore margin of victory

# IDEA: look into SRS simple rating system
