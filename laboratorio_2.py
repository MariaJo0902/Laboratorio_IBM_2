# Practica de laboratorio From Understanding to Preparation, exploración 
# de datos y preparación para modelarlos

import pandas as pd
import numpy as np 
import re
import random

pd.set_option('display.max_columns', None)

#import piplite
#await piplite.install(['skillsnetwork'])
#import skillsnetwork

recetas = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv") 
print("Data read into dataframe!")

column_names = recetas.columns.values
column_names[0] = "cocina"
recetas.columns = column_names

#Convirtiendo los nombres de cocina en minúsculas
recetas["cocina"] = recetas["cocina"].str.lower()

# Haciendo que los nombres de las cocinas sean consistentes

recetas.loc[recetas["cocina"] == "austria", "cocina"] = "austrian"
recetas.loc[recetas["cocina"] == "belgium", "cocina"] = "belgian"
recetas.loc[recetas["cocina"] == "china", "cocina"] = "chinese"
recetas.loc[recetas["cocina"] == "canada", "cocina"] = "canadian"
recetas.loc[recetas["cocina"] == "netherlands", "cocina"] = "dutch"
recetas.loc[recetas["cocina"] == "france", "cocina"] = "french"
recetas.loc[recetas["cocina"] == "germany", "cocina"] = "german"
recetas.loc[recetas["cocina"] == "india", "cocina"] = "indian"
recetas.loc[recetas["cocina"] == "indonesia", "cocina"] = "indonesian"
recetas.loc[recetas["cocina"] == "iran", "cocina"] = "iranian"
recetas.loc[recetas["cocina"] == "italy", "cocina"] = "italian"
recetas.loc[recetas["cocina"] == "japan", "cocina"] = "japanese"
recetas.loc[recetas["cocina"] == "israel", "cocina"] = "israeli"
recetas.loc[recetas["cocina"] == "korea", "cocina"] = "korean"
recetas.loc[recetas["cocina"] == "lebanon", "cocina"] = "lebanese"
recetas.loc[recetas["cocina"] == "malaysia", "cocina"] = "malaysian"
recetas.loc[recetas["cocina"] == "mexico", "cocina"] = "mexican"
recetas.loc[recetas["cocina"] == "pakistan", "cocina"] = "pakistani"
recetas.loc[recetas["cocina"] == "philippines", "cocina"] = "philippine"
recetas.loc[recetas["cocina"] == "scandinavia", "cocina"] = "scandinavian"
recetas.loc[recetas["cocina"] == "spain", "cocina"] = "spanish_portuguese"
recetas.loc[recetas["cocina"] == "portugal", "cocina"] = "spanish_portuguese"
recetas.loc[recetas["cocina"] == "switzerland", "cocina"] = "swiss"
recetas.loc[recetas["cocina"] == "thailand", "cocina"] = "thai"
recetas.loc[recetas["cocina"] == "turkey", "cocina"] = "turkish"
recetas.loc[recetas["cocina"] == "vietnam", "cocina"] = "vietnamese"
recetas.loc[recetas["cocina"] == "uk-and-ireland", "cocina"] = "uk-and-irish"
recetas.loc[recetas["cocina"] == "irish", "cocina"] = "uk-and-irish"

#Eliminando las cocinas con <50 recetas
recetas_counts = recetas["cocina"].value_counts()
cocina_indices = recetas_counts > 50

cocina_to_keep = list(np.array(recetas_counts.index.values)[np.array(cocina_indices)])
recetas = recetas.loc[recetas["cocina"].isin(cocina_to_keep)]

#Convertir todos los Sí en 1 y los No en 0
recetas = recetas.replace(to_replace="Yes", value=1)
recetas = recetas.replace(to_replace="No", value=0)

#Importar bibliotecasde  árboles de decisiones scikit-learn

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import itertools

recetas.head()

#Contruir un árbol de decisiones utilizando los datos relacionados con las
# cocinas asiáticas e india y llamemos a nuestro árbol de decisiones árbol_bambú

#Subconjuntos de cocinas

asian_indian_recetas = recetas[recetas.cocina.isin(["korean","japonese","chinese","thai","indian"])]
cocina = asian_indian_recetas["cocina"]
ingredientes = asian_indian_recetas.iloc[:,1:]

bamboo_tree = tree.DecisionTreeClassifier(max_depth=3)
bamboo_tree.fit(ingredientes, cocina)

print("Árbol de decisiones guardado en bamboo_tree!")

#Trazando el árbol de decisión y examinando como se ve


plt.figure(figsize=(40,20))  # perzonalizar según el tamaño del árbol
_ = tree.plot_tree(bamboo_tree,
                   feature_names = list(ingredientes.columns.values),
                   class_names=np.unique(cocina),filled=True,
                   node_ids=True,
                   impurity=False,
                   label="all",
                   fontsize=20, rounded = True)
plt.show()

# El árbol de decisión aprendío:
# Si una receta contiene comino y pescado y no contiene yogurt, lo más
# probable es que sea un receta tailandesa
# Si una receta contiene comino pero no pescado ni salsa de soja, lo más 
# probable es que sea una receta india

# Para evaluar nuestro modelo de cocinas asiática e india, dividiremos nuestro
# conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba.
# Construiremos el árbol de decisión utilizando el conjunto de entrenamiento. 
# Luego, probaremos el modelo en el conjunto de prueba y compararemos las
# cocinas que predice el modelo con las cocinas reales.

# Primero creamos un nuevo marco de datos utilizando solo los datos pertenecientes
# a las cocinas asiáticas e india
bamboo = recetas[recetas.cocina.isin(["korean", "japanese", "chinese", "thai", "indian"])]

#Veamos cuántas recetas existen para cada cocina
bamboo["cocina"].value_counts()

# Eliminemos 30 recetas de cada cocina para usarlas como conjunto de prueba y 
# llamemos a este conjunto de prueba bamboo_test
sample_n = 30

# Creemos un marco de datos que contenga 30 recetas de cada cocina, seleccionadas
# al azar
random.seed(1234) # set random seed
bamboo_test = bamboo.groupby("cocina", group_keys=False).apply(lambda x: x.sample(sample_n))

bamboo_test_ingredientes = bamboo_test.iloc[:,1:] # ingredientes
bamboo_test_cocinas = bamboo_test["cocina"] # cocinas

# Comprobar que existan 30 recetas para cada cocina
bamboo_test["cocina"].value_counts()

# A continuación, creemos el conjunto de entrenamiento elimando el conjunto de
# prueba del conjunto de datos de bambú y llamemos al conjunto de entrenamiento
# bamboo_train
bamboo_test_index = bamboo.index.isin(bamboo_test.index)
bamboo_train = bamboo[~bamboo_test_index]

bamboo_train_ingredientes = bamboo_train.iloc[:,1:] # ingredientes
bamboo_train_cocinas = bamboo_train["cocina"] # cocinas

# Comprobar que ahora hay 30 recetas menos para cada cocina
bamboo_train["cocina"].value_counts()

# Contruimos el árbol de decisión usando el conjunto de entrenamiento
# bamboo_train y nombremos el árbol generado bamboo_train_tree

bamboo_train_tree = tree.DecisionTreeClassifier(max_depth=15)
bamboo_train_tree.fit(bamboo_train_ingredientes, bamboo_train_cocinas)

print("Módelo de árbol de decisión guardado en bamboo_train_tree!")

# Tracemos el árbol de decisión

plt.figure(figsize=(40,20)) 
_ = tree.plot_tree(bamboo_train_tree, # perzonalizar según el tamaño del árbol
                   feature_names=list(bamboo_train_ingredientes.columns.values),
                   class_names=np.unique(bamboo_train_cocinas),
                   filled=True,
                   node_ids=True,
                   impurity=False,
                   label="all",
                   fontsize=10, rounded = True)
plt.show()

# Ahora probemos nuestro modelo con los datos de prueba
bamboo_pred_cocinas = bamboo_train_tree.predict(bamboo_test_ingredientes)

# Creemos la matriz de confusión sobre qué tan bien el árbol de decisión
# puede clasificar correctamente las recetas en bamboo_test

test_cocinas = np.unique(bamboo_test_cocinas)
bamboo_confusion_matrix = confusion_matrix(bamboo_test_cocinas, bamboo_pred_cocinas, labels = test_cocinas)
title = 'Matriz de confusión de bambú'
cmap = plt.cm.Blues

plt.figure(figsize=(8, 6))
bamboo_confusion_matrix = (
    bamboo_confusion_matrix.astype('float') / bamboo_confusion_matrix.sum(axis=1)[:, np.newaxis]
    ) * 100

plt.imshow(bamboo_confusion_matrix, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(test_cocinas))
plt.xticks(tick_marks, test_cocinas)
plt.yticks(tick_marks, test_cocinas)

fmt = '.2f'
thresh = bamboo_confusion_matrix.max() / 2.
for i, j in itertools.product(range(bamboo_confusion_matrix.shape[0]), range(bamboo_confusion_matrix.shape[1])):
    plt.text(j, i, format(bamboo_confusion_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if bamboo_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()




























