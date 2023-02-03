from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X5 = X[y <= '3']
y5 = y[y <= '3']

mlp = MLPClassifier(
    hidden_layer_sizes=(6,),
    max_iter=200, alpha=1e-4,
    solver='sgd', random_state=2)

mlp.fit(X5, y5)
