#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("titanic.csv")

# Tratamento
df.loc[df["Sex"] == "male", "Sex"] = 1
df.loc[df["Sex"] == "female", "Sex"] = 0

# Poderia ser feito tb com a biblioteca numpy
# df["sex"] = np.where(df["Sex"] == "male", 1, 0)

features = ["Pclass", "Sex", "Age", "Fare"]

X = df[features]
Y = df["Survived"]


# Divide os dados em teste e treino
# Test_Size, tamanho reservado para os dados de teste, nesse caso 20%
# Randon state é para pegar dados aleatórios, 42 serve para ser os mesmos valores aleatórios
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Retorna qual o Max_depth tem a melhor precisão, nesse caso 4
# for depth in range(1, 21):
#     arvore = tree.DecisionTreeClassifier(max_depth=depth)
#     scores = cross_val_score(arvore, X_train, Y_train, cv=5)  # 5-fold cross-validation
#     print(f"Max Depth: {depth}, Cross-Validation Accuracy: {scores.mean():.4f}")

arvore = tree.DecisionTreeClassifier(max_depth=4)


# Treinar a árvore de decisão no conjunto de treino
arvore.fit(X_train, Y_train)

# Calculando a acurácia do modelo.
y_pred = arvore.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%") # 77.53%

# Matriz de confusão
cm = confusion_matrix(Y_test, y_pred)
# Print Matriz de connfusão
labels = ['Não Sobreviveu', 'Sobreviveu']
cm_df = pd.DataFrame(cm, index=[f'Real: {label}' for label in labels],
                     columns=[f'Previsto: {label}' for label in labels])
print(cm_df)

class_names = ["Not Survived", "Survived"]

plt.figure(dpi=230)

tree.plot_tree(arvore,
               class_names=class_names,
               feature_names=features,
               filled=True)

plt.show()
# %%
