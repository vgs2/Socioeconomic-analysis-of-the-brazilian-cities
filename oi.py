# Tratamos os dados segundo Godoy

#Treinando o nosso modelo
from sklearn.model_selection import train_test_split

SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=SEED)

X_train.head()
X_train.shape
y_test.shape

# Regressao linear

from sklearn.svm import LinearSVR

model = LinearSVR(random_state = SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test[:5]
plt.figure(figsize=(8,8))
sns.scatterplot(x=y_pred, y=y_test)
plt.xlim((-50, 1050))
plt.ylim((-50,1050))


# Comparando o valor real com o quanto tirou a mais ou a menos, ou seja o erro

plt.figure(figsize=(8,8))
sns.scatterplot(x=y_teste, y=y_test-y_pred)

# Criando um novo dataframe para visualizar nossos dados apos treinamento e os erros
result = pd.DataFrame()
result["Real"] = y_test
result["Prediction"] = y_pred
result["Diference"] = result["Real"] - result["Prediction"]
# Necessario para diferenciar positivos de negativos numa soma, pois nao temos
# como saber qual tipo de erro eh pior ja que nao sabemos o contexto
result["Squared_Diference"] = (result["Diference"])**2