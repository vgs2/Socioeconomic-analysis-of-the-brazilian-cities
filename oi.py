# Tratamos os dados segundo Godoy

#Treinando o nosso modelo
from sklearn.model_selection import train_test_split

SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=SEED)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)

# X_train.head()
# X_train.shape
# y_test.shape

# Regressao linear
# Verificaremos o nosso modelo com a métrica de mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

model = LinearSVR(random_state = SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)
mean_squared_error(y_validation, y_pred)

y_validation[:5]
plt.figure(figsize=(8,8))
sns.scatterplot(x=y_pred, y=y_validation)
plt.xlim((-50, 1050))
plt.ylim((-50,1050))


# Comparando o valor real com o quanto tirou a mais ou a menos, ou seja o erro
plt.figure(figsize=(8,8))
sns.scatterplot(x=y_test, y=y_validation-y_pred)


# DummyRegressor: regressor burro - quase um pior caso -
from sklearn.dummy import DummyRegressor

model_dummy = DummyRegressor()
model_dummy.fit(X_train, y_train)
dummy_prediction = model_dummy.predict(X_validation)
mean_squared_error(y_validation, dummy_prediction)

# Arvore de Decisao 
from sklearn.tree import DecisionTreeRegressor

model_tree = DecisionTreeRegressor(max_depth = 3)
model_tree.fit(X_train, y_train)
model_tree.score(X_validation, y_validation)


# KNN Regressor

from sklearn.neighbors import KNeighborsRegressor
model_knn = KNeighborsRegressor(n_neighbors=2)
model_knn.fit(X_train, y_train)
model_knn.score(X_validation, y_validation)
