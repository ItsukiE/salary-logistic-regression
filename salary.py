## Importando as bibliotecas necessárias

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## Carregar nosso dataset

df = pd.read_csv('adult11.csv')
df.head()

## Verificar o tipo de cada coluna, utilizaremos um modelo de regressão linear e devemos garantir que todos os dados possam ser lidos corretamente

df.info()

## Verificar a existência de nulos

df.isna().sum()

## Tratar os ? no dataset

df.replace(r'\s*\?\s*', np.nan, regex = True, inplace= True)

## Remove todas as linhas que contém nulos

df.dropna(inplace= True)
df.head()

## Converter a coluna de salários para 0 e 1 com base em ser <=50k ou >50k respectivamente.

df['salary_numeric'] = df['salary'].str.strip().map({ 
'<=50K': 0,
'>50K': 1
})

df = df.drop('salary', axis = 1)

## Transformação das Features Categóricas (get_dummies)

col_categ = df.select_dtypes(include=['object']).columns
df_tratado = pd.get_dummies(df, columns=col_categ, drop_first= True)

## Verificar shape

df_tratado.shape

## Separar X (features) e y (salário)

X = df_tratado.drop('salary_numeric', axis = 1)
y = df_tratado['salary_numeric']

## Dividir em treino/teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

## Feature Scaling
## Para colocar todas as features na mesma escala

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Criação e Treinamento do modelo

logistic = LogisticRegression(random_state= 42, max_iter= 1000)
logistic.fit(X_train_scaled, y_train)

print("Modelo treinado!")

## Avaliação do modelo

previsoes = logistic.predict(X_test_scaled)

## Calcular as métricas de classificação

accuracy = accuracy_score(y_test, previsoes)
print(f"Acurácia: {accuracy:.4f}")

print("\n--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

# Um relatório completo da performance

print(classification_report(y_test, previsoes, target_names=['<=50K (0)', '>50K (1)']))

## Análise das previsões

print("\n --- Exemplo de Previsões vs. Reais ---")

df_resultados = pd.DataFrame({'Valor_Real': y_test, 'Previsão_Final': previsoes})

print(df_resultados.head(10))