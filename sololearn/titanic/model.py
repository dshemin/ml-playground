#!/usr/bin/env python

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

x = df[[
    'Pclass',
    'male',
    'Age',
    'Siblings/Spouses',
    'Parents/Children',
    'Fare',
]].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(x, y)

res = model.predict([[3, True, 22.0, 1, 0, 7.25]])
print(res[0])
