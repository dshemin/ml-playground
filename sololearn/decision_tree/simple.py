import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d

df = pl.read_csv('./titanic.csv')
df = df.with_columns([
    (pl.col('Sex') == 'male').alias('male')
])
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].to_numpy()
y = column_or_1d(df['Survived'].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.predict([[3, True, 22, 1, 0, 7.25]]))
