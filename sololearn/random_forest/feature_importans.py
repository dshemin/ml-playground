import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import column_or_1d

cancer_data = load_breast_cancer()

df = pl.DataFrame(cancer_data['data'], columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                                'mean smoothness', 'mean compactness', 'mean concavity',
                                                'mean concave points', 'mean symmetry', 'mean fractal dimension',
                                                'radius error', 'texture error', 'perimeter error', 'area error',
                                                'smoothness error', 'compactness error', 'concavity error',
                                                'concave points error', 'symmetry error', 'fractal dimension error',
                                                'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                                                'worst smoothness', 'worst compactness', 'worst concavity',
                                                'worst concave points', 'worst symmetry', 'worst fractal dimension',])
df = df.with_columns([
    pl.Series('target', cancer_data['target']),
])

X = df[cancer_data['feature_names']].to_numpy()
y = column_or_1d(df['target'].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf.fit(X_train, y_train)

ft_imp = pl.DataFrame([
    pl.Series('name', cancer_data['feature_names']),
    pl.Series('importance', rf.feature_importances_),
]).lazy().sort(by='importance', reverse=True).head(10).collect()

print(ft_imp)
