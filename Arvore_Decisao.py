# Utilizando Arvore de Decisão

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Carregar base de dados
data = pd.read_csv('ai4i2020.csv', delimiter=';')

# Obter o número de linhas
num_linhas = data.shape[0]
print(f'O arquivo CSV tem {num_linhas} linhas.')

# Ignorar as colunas Product Id e Type, pois não serão uteis.
data.drop(['Product ID', 'Type'], axis=1, inplace=True)

# Renomear as colunas no DataFrame para remover espaços em branco
data = data.rename(columns=lambda x: x.strip())

#Print para verificar se os nomes das colunas da base de dados estão corretos.
print(data.columns)

# Codificar a coluna "Machine failure" usando Label Encoding (De "sim ou não" para 1 ou 0).
label_encoder = LabelEncoder()
data['Machine failure'] = label_encoder.fit_transform(data['Machine failure'])

# Garantir que os nomes das colunas estejam corretos
expected_columns = ['UDI','Air temperature','Process temperature','Rotational speed','Torque','Tool wear','Machine failure','TWF','HDF','PWF','OSF','RNF']

# Se os nomes não estiverem corretos, da uma mensagem de erro .
if set(data.columns) != set(expected_columns):
    raise ValueError("Nomes das colunas do DataFrame não correspondem aos nomes esperados.")

# Selecionar as colunas no DataFrame
X = data[expected_columns[0:6] + expected_columns[7:]]
y = data['Machine failure']

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Prever a falha do computador/falha de máquina no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

#Resultados
linha_separar = '-' * 80
print(linha_separar)
print("Resultados:\n")
print(f'Acurácia do modelo: {accuracy}\n')
print('Matriz de Confusão:')
print(conf_matrix, end='\n\n')
print('Relatório de Classificação:')
print(classification_rep)
print(linha_separar)

