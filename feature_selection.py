import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv(r"Data/Clean/clean_aggregate.csv")

data = df.loc[df['Year'] < 2024].drop(['School', 'Year'], axis=1)
holdout = df.loc[df['Year'] == 2024].drop(['School', 'Year'], axis=1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)

y = scaled['WINS']
X = scaled.drop('WINS', axis=1)

mi = mutual_info_regression(X,y)
feat_importance = pd.Series(mi, X.columns[0: len(X.columns)]).sort_values(ascending=False)

feat_importance[:50]

pairs = ['SRS', 'Seed', 'AP_Mean', 'Pre', 'W', 'SRS_OPP', 'AW%', 'AG%', 'Guard_RSCI Top 100_Mean', 'PTS_per_Surgical_POS', 'MV_Mean', 'PIR', 'Forward_RSCI Top 100_Mean', 'FG%_MW_Pruned', 'Height_Mean_Pruned', 'Pos_MW_Pruned']

# TODO: plot pairwise scatter plots
fig, ax = plt.subplots(len(pairs), len(pairs))
fig.set_size_inches(24,24)
for i, var_i in enumerate(pairs):
    for j, var_j in enumerate(pairs):
        ax[i,j].scatter(scaled[var_i], scaled[var_j])
plt.show()

# NOTE: using Spearman correlation which doesn't assume normally distributed data and allows ordinal data
correlation = abs(scaled.corr(method='spearman')).sort_values(by='WINS', ascending=False)
correlation = correlation.reindex(columns=correlation.index)
idx = feat_importance[:50].index
idx = idx.insert(0, 'WINS')
correlation = correlation.loc[idx[:25], idx[:25]]
plt.figure(figsize=(24,24))
plt.imshow(correlation, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.tick_params(axis='x', labelrotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.tick_params(axis='y', labelrotation=0)
for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        text = plt.text(j, i, correlation.iloc[i, j].round(2),
                       ha="center", va="center", color="w")
plt.show()

# NOTE: point biserial tests the relationship between a binary variable and continuous variable
pointbiserialr(x=X['CTC'], y=y)
pointbiserialr(x=X['CSC'], y=y)
pointbiserialr(x=X['PIR'], y=y)


from statsmodels.stats.outliers_influence import variance_inflation_factor


# NOTE: Four Factors: eFG% (40%), TOV% (25%), ORB% & DRB% (20%), FT/FGA (15%)
# TODO: weight these features to use in the model? Or will the model decide on appropriate weights itself?
four_factors = df.loc[:, ['School', 'Year', 'EFG%', 'EFG%_OPP', 'TOV%', 'TOV%_OPP', 'ORB%', 'DRB%', 'FTr', 'FTr_OPP']]

# features = []
# # add primary key to features
# features.extend(['School', 'Year'])
# # add target variable to features
# features.append('WINS')
# # add narrowed down candidate features based on mutual information gain, and reducing colinear features based on spearman correlation
# features.extend(['SRS', 'SRS_OPP'])
# features.extend(['Seed'])
# features.extend(['AP_Mean', 'Pre'])
# features.extend(['AG%', 'AW%', 'W%'])
# features.append('PTS_per_Surgical_POS')
# features.append('MV_Mean')
# features.extend(['Guard_RSCI Top 100_Mean', 'Forward_RSCI Top 100_Mean', 'Center_RSCI Top 100_Mean'])
# features.extend([])


# TODO: reduce colinearity among most correlated features, and include statistical features that 'SHOULD' be important for winning games
features = df.loc[:, ['School', 'Year', 'WINS', 'SRS', 'SRS_OPP', 'Seed', 'AP_Mean', 'AP_Last', 'AP_Max', 'AP_5W_Mean', 'Pre', 'Guard_RSCI Top 100_Mean', 'Forward_RSCI Top 100_Mean', 'Center_RSCI Top 100_Mean', 'MV_Mean', 'AG%', 'W%', 'FG', 'FG%', 'FG%_OPP', 'CTC', 'Height_Mean_Pruned', 'TRB%', 'Pace', 'STL%', 'BLK%', 'FT/FGA', '2P%', '3P%', 'FT%', 'PF', 'FG_per_Surgical_POS', 'FGA_per_Surgical_POS', '2P_per_Surgical_POS',
       '2PA_per_Surgical_POS', '3P_per_Surgical_POS', '3PA_per_Surgical_POS',
       'FT_per_Surgical_POS', 'FTA_per_Surgical_POS', 'ORB_per_Surgical_POS',
       'DRB_per_Surgical_POS', 'TRB_per_Surgical_POS', 'AST_per_Surgical_POS',
       'STL_per_Surgical_POS', 'BLK_per_Surgical_POS', 'TOV_per_Surgical_POS',
       'PF_per_Surgical_POS', 'PTS_per_Surgical_POS',
       'FG_OPP_per_Surgical_POS', 'FGA_OPP_per_Surgical_POS',
       '2P_OPP_per_Surgical_POS', '2PA_OPP_per_Surgical_POS',
       '3P_OPP_per_Surgical_POS', '3PA_OPP_per_Surgical_POS',
       'FT_OPP_per_Surgical_POS', 'FTA_OPP_per_Surgical_POS',
       'ORB_OPP_per_Surgical_POS', 'DRB_OPP_per_Surgical_POS',
       'TRB_OPP_per_Surgical_POS', 'AST_OPP_per_Surgical_POS',
       'STL_OPP_per_Surgical_POS', 'BLK_OPP_per_Surgical_POS',
       'TOV_OPP_per_Surgical_POS', 'PF_OPP_per_Surgical_POS',
       'PTS_OPP_per_Surgical_POS',]]
