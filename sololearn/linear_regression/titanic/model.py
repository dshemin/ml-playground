#!/usr/bin/env python

from sklearn.utils import column_or_1d
import polars as pl
from sklearn.linear_model import LogisticRegression

df = pl.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df = df.with_columns(
    (pl.col('Sex') == 'male').alias('male'),
)

x = df[[
    'Pclass',
    'male',
    'Age',
    'Siblings/Spouses',
    'Parents/Children',
    'Fare',
]].to_numpy()
y = column_or_1d(df.select(['Survived']).to_numpy())

model = LogisticRegression()
model.fit(x, y)

res = model.predict([[3, True, 22.0, 1, 0, 7.25]])
print(res[0])
