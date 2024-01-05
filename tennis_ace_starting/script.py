import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
# print(df.columns)

# perform exploratory analysis here:


for column in df:
    plt.scatter(df[column], df.Winnings)
    plt.title(f"Winnings Vs {column}")
#     plt.show()
#     plt.clf()

# Putting all the features that showed some correlation in the scatter plots into a features dataframe
features = df[['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed', 'Wins', 'Losses', 'Ranking']]

## perform single feature linear regressions here:

# Function that can take in as many variables needed for single or multi linear regression.
# type needs to be filled in as type = 'multi' to access model coefficients in multi linear regression. Can be left as 'single' otherwise.
def lr(df, dependant, *variable, type):
    y = df[dependant]
    X = df[[*variable]]
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size=0.2, random_state= 42)
    print(X.shape)
    slr = LinearRegression()
    slr.fit(x_train, y_train)
    y_predict = slr.predict(x_test)
    score = slr.score(x_test, y_test)
    print(f"The score for the linear relationship between {variable} and {dependant} is {score}")

    plt.scatter(y_test, y_predict, alpha = 0.4)
    plt.title(f"Predicted Vs Actual {dependant} for {variable}")
    # plt.show()

    if type.lower() == 'multi':
        return score, slr.coef_
    else:
        return score


score = {}
for column in features:
    score[column] = lr(df, 'Winnings', column, type = 'single')
# print(f"The {max(score)} variable has the greatest score of {max(score.values())} ")

## perform two feature linear regressions here:
lr(df, 'Winnings', 'Wins', 'Losses', type = 'multi')
lr(df, 'Winnings', "Ranking", 'Aces', type = 'multi')
# etc..


## perform multiple feature linear regressions here:
# The first set of variables produced the linear regression model with the highest coefficient of determination.
score, coef = lr(df, 'Winnings', 'Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed', 'Wins', 'Losses', 'Ranking', type = 'multi')
print(coef)

score, coef = lr(df, 'Winnings', 'Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed', 'Wins', 'Losses', type = 'multi')
print(coef)

score, coef = lr(df, 'Winnings', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ServiceGamesPlayed', 'Wins', 'Losses', type = 'multi')
print(coef)

score, coef = lr(df, 'Winnings', 'BreakPointsFaced', 'BreakPointsOpportunities', 'Wins', 'Losses', 'Ranking', type = 'multi')
print(coef)

score, coef = lr(df, 'Winnings', 'FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon', type = 'multi')
