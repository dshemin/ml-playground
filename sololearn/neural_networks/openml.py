from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X5 = X[y <= '3']
y5 = y[y <= '3']
