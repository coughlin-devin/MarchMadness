import pandas as pd
from imputation import knn_imputer, position_mean_imputer

# TODO: per_40
# TODO: per_100

def get_data(per_game=True):
    roster = pd.read_csv(r"Data/Clean/clean_roster.csv")
    team_opp = pd.read_csv(r"Data/Raw/team_opp_table.csv")
    if per_game:
        player_data = pd.read_csv(r"Data/Raw/player_table.csv")
    else:
        player_data = pd.read_csv(r"Data/Raw/per_40_table.csv")

    # NOTE: removes 120 players from 2024 who were most likely all redshirts; I confirmed some players were redshirts
    df = roster.merge(player_data, how='left', on=['School', 'Year', 'Player'])

    # NOTE: append teams total games and calculate teams total minutes on the season.
    # WARNING: This unavoidably including NCAA games.
    df = df.merge(team_opp[['Year', 'School', 'G', 'MP']], how='left', on=['School', 'Year'], suffixes=['_P', '_T'])
    df.loc[:, 'MP_T'] = df.loc[:, 'MP_T'].fillna(value=40) * df['G_T']

    # create a column for the percentage of total team player minutes minutes a player plays (how many minutes does the player play over team games * 40min * 5players)
    df['MP%'] = df['G_P'] * df['MP_P'] / (df['MP_T'] * 5)

    # drop unnecessary columns
    df.drop(['Rk', 'Player'], axis=1, inplace=True)

    return df

def correct_percentages(df):
    """Set shooting percentage stats to NaN from 0 if the player has not taken any shots.

    If a player has not taken any FG, set there FG% to NaN instead of 0. Do the same thing for 2P, 3P, and FT. This is because a shooting % of 0 is not accurate if there haven't been any shots taken. FG% = FG made / FG taken, which is undefined for 0 FG taken.

    Parameters
    ----------
    arg1 : df
        pandas DataFrame of player stats.

    Returns
    -------
    pandas DataFrame
        Returns the input DataFrame with updated shooting %.
    """
    fga_mask = df.loc[(df['FGA'] == 0)]
    df.loc[fga_mask.index, ['FG%', '2P%', '3P%']] = float('NaN')

    _2pa_mask = df.loc[(df['2PA'] == 0)]
    df.loc[_2pa_mask.index, '2P%'] = float('NaN')

    _3pa_mask = df.loc[(df['3PA'] == 0)]
    df.loc[_3pa_mask.index, '3P%'] = float('NaN')

    fta_mask = df.loc[(df['FTA'] == 0)]
    df.loc[fta_mask.index, 'FT%'] = float('NaN')

    return df

# IDEA: don't waste time filling in missing games started, just create the varaince of starters and impute the average variance for the missing teams

def drop_bench(df, percent_minutes=0.05):
    """Remove players who don't meet a minutes played criteria from the data.

    Exclude players who fail to meet a minimum percentage of their teams minutes over the season. These players are likely bench players who don't make strong impacts on games and rarely play.

    Parameters
    ----------
    arg1 : pandas DataFrame
        pandas DataFrame of player stats.
    arg2 : float
        float between 0-1.0 indicating the minimum percentage of team minutes played to be included in the data.

    Returns
    -------
    pandas DataFrame
        Returns a DataFrame without players who played fewer than some minimum percentage of their team's minutes.
    """
    bench = df.loc[df['MP%'] < percent_minutes]
    df = df.drop(bench.index).reset_index(drop=True)
    return df

def cohesion(minutes, playing_time=40, group=5):
    """Calculate a metric describing how much the n players who play the most play together.

    Teams with smaller groups of players with higher minutes per game have a higher cohesion rating. Teams with larger groups of players with fewer minutes per game have a lower cohesion metric. The range of the cohesion metric is [0-1].

    Parameters
    ----------
    minutes : float
        Average minutes played per game.
    playing_time : int
        Total minutes per game.
    group : int
        NUmber of players in the core group being measured for cohesion.

    Returns
    -------
    float
        Returns a float between [0-1] indicating the cohesion score.
    """
    # sort minutes from highest to lowest
    minutes.sort_values(ascending=False)
    core_squad = minutes[:group]
    return core_squad.mean() / playing_time

def add_mean_features_by_filter(group, features, filter_column, columns, filter_number, filter_name):
    """Helper function to add mean features to a DataFrame.

    Format and merge mean features to a DataFrame.

    Parameters
    ----------
    group : pandas DataFrame
        Description of arg1.
    features : pandas DataFrame
        DataFrame to add the features to.
    filter_column : String
        Column to filter group by.
    columns : list of Strings
        List of columns to build features from.
    filter_number : int
        Integer indicating which value to filter the filter column. Either 1-4 for class or 1-3 for position.
    filter_name : String
        Name of value corresponding to feature number. Either class name ie. Freshman or position name ie. Center.

    Returns
    -------
    DataFrame
        Returns features DataFrame with added mean features.
    """
    # drop non-feature columns
    group = group.loc[:, columns]
    # create dataframe of players included in the filter
    filter = group.loc[group[filter_column] == filter_number]
    # rename columns to prepare for merging into features
    filter.columns = [filter_name + '_' + x + '_' + 'Mean' if x not in ['Year', 'School'] else x for x in filter.columns]
    features = features.merge(filter, how='left', on=['Year', 'School'])
    return features

def add_sum_features_by_filter(group, features, filter_column, columns, filter_number, filter_name):
    """Helper function to add sum features to a DataFrame.

    Format and merge sum features to a DataFrame.

    Parameters
    ----------
    group : pandas DataFrame
        Description of arg1.
    features : pandas DataFrame
        DataFrame to add the features to.
    filter_column : String
        Column to filter group by.
    columns : list of Strings
        List of columns to build features from.
    filter_number : int
        Integer indicating which value to filter the filter column. Either 1-4 for class or 1-3 for position.
    filter_name : String
        Name of value corresponding to feature number. Either class name ie. Freshman or position name ie. Center.

    Returns
    -------
    DataFrame
        Returns features DataFrame with added sum features.
    """
    # drop non-feature columns
    group = group.loc[:, columns]
    # create dataframe of players included in the filter
    filter = group.loc[group[filter_column] == filter_number]
    # rename columns to prepare for merging into features
    filter.columns = [filter_name + '_' + x + '_' + 'Sum' if x not in ['Year', 'School', 'Class', 'Pos'] else x for x in filter.columns]
    # drop Class and Pos columns to avoid duplicating them in features
    filter = filter.drop(['Class', 'Pos'], axis=1)
    features = features.merge(filter, how='left', on=['Year', 'School'])
    return features

def add_minute_weighted_features(df, features, mwc, pmwc):
    """Add features weighted by minutes played by the players.

    Multiply features by the number of minutes the player played on average per game and as part of the total player minutes of the team. A teams player minutes are the possible minutes a player can play per game (40 without overtime) multiplied by the number of players on the court (5 players per team).

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the feature data.
    features : pandas DataFrame
        DataFrame to add the features to.
    mwc : list of Strings
        Minute weighted feature columns for features weighted by minutes per game.
    pmwc : list of String
        Player minute weighted feature columns for features weighted by total team minutes per game.

    Returns
    -------
    DataFrame
        DataFrame with added features.
    """
    columns_mwc = []
    columns_pmwc = []
    # create a column for player-minute weighted stats
    for column in mwc:
        new_column_name = column + "_MW"
        df[new_column_name] = df[column] * df['MP%']
        columns_mwc.append(new_column_name)
    for column in pmwc:
        new_column_name = column + "_MW"
        df[new_column_name] = df[column] * df['MP%'] * 5
        columns_pmwc.append(new_column_name)

    # create a group by object to sum the minute weighted stats
    sums = df.groupby(['Year', 'School']).sum().reset_index()

    # add minute wieghted stats to features
    features[columns_mwc] = sums.loc[:, columns_mwc]
    features[columns_pmwc] = sums.loc[:, columns_pmwc]
    return features

def create_features(df):
    """Create all the player per game features.

    This function calls all the feature cretion helper functions and outputs the final feature Dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with the data to create features from.

    Returns
    -------
    DataFrame
        Returns a DataFrame of player per game features.
    """
    # create features dataframe
    features = pd.DataFrame()

    # create means groupby object
    means = df.groupby(['Year', 'School']).mean().reset_index()

    # add year, school, and team average class, height, and weight to features
    features['Year'] = means.Year
    features['School'] = means.School
    features['Class_Mean'] = means.Class
    features['Height_Mean'] = means.Height
    features['Weight_Mean'] = means.Weight

    # columns to create mean features from
    mean_feature_columns = ['School', 'Year', 'Class', 'Pos', 'Height', 'Weight', 'RSCI Top 100', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    sum_feature_columns = ['School', 'Year', 'Class', 'Pos', 'MP%']

    # create and flatten groups for each filter
    pos_means = df.groupby(['Year', 'School', 'Pos']).mean().reset_index()
    class_means = df.groupby(['Year', 'School', 'Class']).mean().reset_index()
    pos_sums = df.groupby(['Year', 'School', 'Pos']).sum().reset_index()
    class_sums = df.groupby(['Year', 'School', 'Class']).sum().reset_index()

    # add average features by position
    features = add_mean_features_by_filter(pos_means, features, 'Pos', mean_feature_columns, 1, 'Guard')
    features = add_mean_features_by_filter(pos_means, features, 'Pos', mean_feature_columns, 2, 'Forward')
    features = add_mean_features_by_filter(pos_means, features, 'Pos', mean_feature_columns, 3, 'Center')

    # add average features by class
    features = add_mean_features_by_filter(class_means, features, 'Class', mean_feature_columns, 1, 'Freshman')
    features = add_mean_features_by_filter(class_means, features, 'Class', mean_feature_columns, 2, 'Sophomore')
    features = add_mean_features_by_filter(class_means, features, 'Class', mean_feature_columns, 3, 'Junior')
    features = add_mean_features_by_filter(class_means, features, 'Class', mean_feature_columns, 4, 'Senior')

    # add sum features by position
    features = add_sum_features_by_filter(pos_sums, features, 'Pos', sum_feature_columns, 1, 'Guard')
    features = add_sum_features_by_filter(pos_sums, features, 'Pos', sum_feature_columns, 2, 'Forward')
    features = add_sum_features_by_filter(pos_sums, features, 'Pos', sum_feature_columns, 3, 'Center')

    # add sum features by class
    features = add_sum_features_by_filter(class_sums, features, 'Class', sum_feature_columns, 1, 'Freshman')
    features = add_sum_features_by_filter(class_sums, features, 'Class', sum_feature_columns, 2, 'Sophomore')
    features = add_sum_features_by_filter(class_sums, features, 'Class', sum_feature_columns, 3, 'Junior')
    features = add_sum_features_by_filter(class_sums, features, 'Class', sum_feature_columns, 4, 'Senior')

    # columns to create minute weighted features based on players percent of total minutes they could play
    minute_weighted_columns = ['Class', 'Pos', 'Height', 'Weight', 'FG%', '2P%', '3P%', 'FT%']
    # columns to create player-minuite weighted features based on players percent of total team minutes they could play (extra factor of 5 for 5 players)
    player_minute_weighted_columns = ['FG', 'FGA', '2P', '2PA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    features = add_minute_weighted_features(df, features, minute_weighted_columns, player_minute_weighted_columns)

    # cohesion metric measuring how tight the main playing squad is on a team
    features['Cohesion'] = features.apply(lambda x: cohesion(df.loc[(df['School'] == x.School) & (df['Year'] == x.Year), 'MP_P']), axis=1)

    return features

def clean_player_features(per_game=True, min_minutes=0):
    """Main logic to fetch data, clean it, and create features.

    Handles the main logic of the file including fetching the data, cleaning it through calls to KNNImputer, rounding features to correct sigfigs, etc.

    Parameters
    ----------
    per_game : bool
        Boolean indicating whether to fetch per_game data or per_40 minute data.
    min_minutes : float
        Float between[0-1] indicating the minimum percentage of a teams season minutes a player needs to have played to be included in the data for building features from.

    Returns
    -------
    DataFrame
        Returns a DataFrame of features built from player stats.
    """

    df = get_data(per_game)

    # discard players with fewer than n% minutes played
    df = drop_bench(df, min_minutes)

    # correct shooting percentages so that players with 0 attempts have NaN percentage
    df = correct_percentages(df)

    # mean stats by position for each year
    # NOTE: used a useful function xs() which helps get grouped rows inside a groupby object
    # centers = df.groupby(['Year', 'Pos']).mean().xs(3, level=1, drop_level=True)
    # forwards = df.groupby(['Year', 'Pos']).mean().xs(2, level=1, drop_level=True)
    # guards = df.groupby(['Year', 'Pos']).mean().xs(1, level=1, drop_level=True)

    # NOTE: years 1997-2001 all players missing weight, so these are MCAR because it is a data collection issue and the wieghts can't be predicted from things like height or age (no correlation)
    # BUG: need to run this before KNN Imputation so there is not an all NaN slice from missing Weights
    df = position_mean_imputer(df, 'Weight', 1997, 2001)

    # WARNING: # TODO: maybe copy df, impute each stat seperately, then recombine them?
    # NOTE: after imputing on one column, it can affect the next imputation. To mitigate this affect, I impute on the columns with fewer missing values first, and the columns with the most missing values last. This way during imputation of each column, the fewest generated values influence the computation. This is not likely to be a big issue in general because I am using a relatively low K value, so it is unlikely much of the data looked at for imputation is itself generated data.

    # NOTE: K is about the square root (29-30) of the # of players for each year (850-900). I went with 30 instead of 29 because 30 is also a population threshold for statistical power.
    K = 30

    # KNN impute missing stats
    df = knn_imputer(df, 'STL', K, 1997, 2024)
    df = knn_imputer(df, 'BLK', K, 1997, 2024)
    df = knn_imputer(df, 'Class', K, 1997, 2024, mode=True)
    df = knn_imputer(df, 'Pos', K, 1997, 2024, mode=True)

    # WARNING: it's likely that after imputing minutes played the total minutes played by a team is innacurate
    df = knn_imputer(df, 'MP_P', K, 1997, 2024)
    df = knn_imputer(df, 'GS', K, 1997, 2024, mode=True)
    df = knn_imputer(df, 'PF', K, 1997, 2024)
    df = knn_imputer(df, 'TOV', K, 1997, 2024)

    # NOTE: not MCAR data bc it is related to if players have taken certain shots ot not, MAR (missing at random but predictable, sometimes confused with MNAR missing not at random but not predictable) # WARNING: papers may use MNAR to mean random and predictable
    # NOTE: can pad by position average, or KNN impute these values
    # df = position_mean_imputer(df, 'FG%', 1997, 2024)
    # df = position_mean_imputer(df, '2P%', 1997, 2024)
    # df = position_mean_imputer(df, '3P%', 1997, 2024)
    # df = position_mean_imputer(df, 'FT%', 1997, 2024)
    df = knn_imputer(df, 'FG%', K, 1997, 2024)
    df = knn_imputer(df, '2P%', K, 1997, 2024)
    df = knn_imputer(df, 'FT%', K, 1997, 2024)
    df = knn_imputer(df, '3P%', K, 1997, 2024)

    # fill in missing stats with position average for each year
    # NOTE: not KNN imputing Height and Wieght because I believe they are independent of player statistical profile
    # NOTE: run position_mean_imputer() after imputing Pos so there are no players missing a Pos
    df = position_mean_imputer(df, 'Height', 1997, 2024)
    df = position_mean_imputer(df, 'Weight', 1997, 2024)
    df = position_mean_imputer(df, 'ORB', 1997, 2024)
    df = position_mean_imputer(df, 'DRB', 1997, 2024)

    # IDEA: before making features, remove all players who have not been in a game, played fewer than X mins, have not taken a shot, etc
    features = create_features(df)

    # missing values from teams without players in a certain year, or certain positions replaced with 0
    features = features.fillna(value=0)

    # round columns to 3 sig figs if a percentage or cohesion, otherwise round to 1 sig fig
    percentages = [col for col in features.columns if '%' in col]
    percentages.append('Cohesion')
    features.loc[:, percentages] = features.loc[:, percentages].round(3)
    features.loc[:, ~features.columns.isin(percentages)] = features.loc[:, ~features.columns.isin(percentages)].round(1)

    return features

clean_player_per_game_complete = clean_player_features()
clean_player_per_game_pruned = clean_player_features(min_minutes=0.05)

clean_player_per_game_complete.to_csv(r"Data/Clean/clean_player_per_game_complete.csv", mode='w', index=False)
clean_player_per_game_pruned.to_csv(r"Data/Clean/clean_player_per_game_pruned.csv", mode='w', index=False)

# BUG: per 40 data missing multiple columns present in per game data
# IDEA: using per 40 in this usecase may be irrelevent because college games are 40 minutes long, so on a team basis per 40 is equal to per game. I don't care about the "accuracy" boost to single player evaluation for player that may play well under constrained minutes
# clean_player_per_40_complete = clean_player_features(per_game=False)
# clean_player_per_40_pruned = clean_player_features(per_game=False, min_minutes=0.05)
# clean_player_per_40_complete.to_csv(r"Data/Clean/clean_player_per_40_complete.csv", mode='w', index=False)
# clean_player_per_40_pruned.to_csv(r"Data/Clean/clean_player_per_40_pruned.csv", mode='w', index=False)
