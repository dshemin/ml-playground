import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import column_or_1d

df = pl.read_csv('./titanic.csv')
df = df.with_column((pl.col('Sex') == 'male').alias('male'))
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].to_numpy()
y = column_or_1d(df['Survived'].to_numpy())

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

print("best params:", gs.best_params_)
print("best score:", gs.best_score_)
