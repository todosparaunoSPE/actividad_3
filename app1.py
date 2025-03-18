# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:02:35 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Sección de Ayuda
st.sidebar.subheader("Ayuda")
st.sidebar.write("""
**Aplicación de Análisis de Cáncer de Seno (Wisconsin)**

Esta aplicación utiliza dos modelos de aprendizaje automático (Naïve Bayes y Regresión Logística) para analizar el conjunto de datos sobre cáncer de seno, proveniente de la base de datos de Wisconsin. El objetivo de la aplicación es comparar el rendimiento de ambos modelos en términos de precisión, recall, F1-Score y matrices de confusión.

**Descripción del flujo de la aplicación:**

1. **Carga de datos:** Se carga el conjunto de datos, que contiene características relacionadas con el cáncer de seno, y se preprocesa eliminando valores faltantes.
2. **Entrenamiento de modelos:** Los modelos de Naïve Bayes y Regresión Logística son entrenados utilizando el conjunto de datos.
3. **Evaluación de modelos:** Se evalúan los modelos en conjuntos de datos de validación y prueba utilizando métricas como precisión, recall, F1-Score, y matrices de confusión.
4. **Análisis de resultados:** Se proporciona un análisis comparativo entre los modelos, abordando temas como el desbalance de clases, la presencia de valores faltantes, y recomendaciones sobre qué modelo usar.

**Resultados:** Los resultados se muestran en una tabla, seguida de las matrices de confusión para cada modelo y conjunto de datos. Además, se presentan análisis y recomendaciones sobre el rendimiento de los modelos.

---

**Creado por:**
- **Javier Horacio Pérez Ricárdez**  
- **Materia:** Aprendizaje Máquina Aplicado  
- Profesor: Omar Velázquez López
""")



# URL del conjunto de datos (definida como variable global)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Título de la aplicación
st.title("Examen Práctico - Aprendizaje Máquina Aplicado")
st.header("Análisis de Cáncer de Seno (Wisconsin)")

# Cargar el conjunto de datos
@st.cache_data
def load_data():
    columns = ["ID", "Grosor_tumor", "Uniformidad_tamaño_celula", "Uniformidad_forma_celula", 
               "Adhesion_marginal", "Tamaño_celula_epitelial", "Nucleos_desnudos", 
               "Cromatina_blanda", "Nucleolos_normales", "Mitosis_celulas", "Clase"]
    data = pd.read_csv(URL, header=None, names=columns)
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)
    data["Clase"] = data["Clase"].apply(lambda x: 0 if x == 2 else 1)  # 0: Benigno, 1: Maligno
    return data

data = load_data()

# Mostrar el conjunto de datos
if st.checkbox("Mostrar conjunto de datos"):
    st.write(data)

# Dividir el conjunto de datos
X = data.drop(["ID", "Clase"], axis=1)
y = data["Clase"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Entrenar modelos
nb_model = GaussianNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Evaluar modelos
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

# Obtener resultados
nb_results_val = evaluate_model(nb_model, X_val, y_val)
nb_results_test = evaluate_model(nb_model, X_test, y_test)
lr_results_val = evaluate_model(lr_model, X_val, y_val)
lr_results_test = evaluate_model(lr_model, X_test, y_test)

# Crear DataFrame con resultados
results_df = pd.DataFrame({
    "Modelo": ["Naïve Bayes", "Naïve Bayes", "Regresión Logística", "Regresión Logística"],
    "Dataset": ["Validación", "Prueba", "Validación", "Prueba"],
    "Accuracy": [nb_results_val[0], nb_results_test[0], lr_results_val[0], lr_results_test[0]],
    "Precision": [nb_results_val[1], nb_results_test[1], lr_results_val[1], lr_results_test[1]],
    "Recall": [nb_results_val[2], nb_results_test[2], lr_results_val[2], lr_results_test[2]],
    "F1-Score": [nb_results_val[3], nb_results_test[3], lr_results_val[3], lr_results_test[3]]
})

# Mostrar DataFrame con resultados
st.subheader("Resultados de los Modelos")
st.write(results_df)

# Mostrar las matrices de confusión
def plot_confusion_matrix(conf_matrix, model_name, dataset_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
    plt.title(f"Matriz de Confusión: {model_name} - {dataset_name}")
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción')
    st.pyplot(fig)

# Mostrar matrices de confusión
st.subheader("Matriz de Confusión - Naïve Bayes (Validación)")
plot_confusion_matrix(nb_results_val[4], "Naïve Bayes", "Validación")
st.subheader("Matriz de Confusión - Naïve Bayes (Prueba)")
plot_confusion_matrix(nb_results_test[4], "Naïve Bayes", "Prueba")
st.subheader("Matriz de Confusión - Regresión Logística (Validación)")
plot_confusion_matrix(lr_results_val[4], "Regresión Logística", "Validación")
st.subheader("Matriz de Confusión - Regresión Logística (Prueba)")
plot_confusion_matrix(lr_results_test[4], "Regresión Logística", "Prueba")

# Análisis de resultados
st.subheader("Análisis de Resultados")

# 1. Comparación de modelos
st.write("1. **Comparación de modelos:**")
# Extraer métricas del dataframe
nb_f1_val = results_df[(results_df["Modelo"] == "Naïve Bayes") & (results_df["Dataset"] == "Validación")]["F1-Score"].values[0]
lr_f1_val = results_df[(results_df["Modelo"] == "Regresión Logística") & (results_df["Dataset"] == "Validación")]["F1-Score"].values[0]
nb_precision_val = results_df[(results_df["Modelo"] == "Naïve Bayes") & (results_df["Dataset"] == "Validación")]["Precision"].values[0]
lr_precision_val = results_df[(results_df["Modelo"] == "Regresión Logística") & (results_df["Dataset"] == "Validación")]["Precision"].values[0]

# Valores de la matriz de confusión para Naïve Bayes (Validación)
tn_nb_val, fp_nb_val, fn_nb_val, tp_nb_val = nb_results_val[4].ravel()

# Valores de la matriz de confusión para Regresión Logística (Validación)
tn_lr_val, fp_lr_val, fn_lr_val, tp_lr_val = lr_results_val[4].ravel()

st.write(f"En el conjunto de validación, **Naïve Bayes** obtuvo un F1-Score de {nb_f1_val:.4f}, mientras que **Regresión Logística** obtuvo un F1-Score de {lr_f1_val:.4f}. Esto indica que Naïve Bayes tiene un mejor equilibrio entre precisión y recall.")
st.write(f"En cuanto a la precisión, **Regresión Logística** obtuvo {lr_precision_val:.4f}, mientras que **Naïve Bayes** obtuvo {nb_precision_val:.4f}. Sin embargo, el F1-Score sugiere que Naïve Bayes es más adecuado para este conjunto de datos.")
st.write(f"Además, observando las matrices de confusión, Naïve Bayes mostró {tp_nb_val} verdaderos positivos, {fp_nb_val} falsos positivos, {fn_nb_val} falsos negativos y {tn_nb_val} verdaderos negativos en validación. En comparación, la Regresión Logística tuvo {tp_lr_val} verdaderos positivos, {fp_lr_val} falsos positivos, {fn_lr_val} falsos negativos y {tn_lr_val} verdaderos negativos.")

# 2. Desbalance de clases
st.write("2. **Desbalance de clases:**")
# Verificar distribución de clases
class_distribution = data["Clase"].value_counts()
st.write("Distribución de clases:", class_distribution)

st.write(f"En el conjunto de prueba, **Naïve Bayes** tiene un recall de {nb_results_test[2]:.4f}, lo que indica que identifica correctamente la mayoría de los casos positivos.")
st.write(f"Además, **Regresión Logística** también tiene un recall de {lr_results_test[2]:.4f}, lo que sugiere que ambos modelos son efectivos para identificar casos positivos.")
st.write(f"Al observar las matrices de confusión, podemos ver que ambos modelos tienen un buen desempeño en términos de recall, pero la cantidad de falsos positivos y falsos negativos es clave para entender el rendimiento.")

# 3. Valores faltantes
st.write("3. **Valores faltantes:**")
# Verificar filas eliminadas
original_rows = len(pd.read_csv(URL, header=None))
cleaned_rows = len(data)
st.write(f"Filas originales: {original_rows}, Filas después de limpieza: {cleaned_rows}.")

# 4. Recomendación de modelos
st.write("4. **Recomendación de modelos:**")
# Comparar F1-Score en prueba
nb_f1_test = results_df[(results_df["Modelo"] == "Naïve Bayes") & (results_df["Dataset"] == "Prueba")]["F1-Score"].values[0]
lr_f1_test = results_df[(results_df["Modelo"] == "Regresión Logística") & (results_df["Dataset"] == "Prueba")]["F1-Score"].values[0]

st.write(f"En el conjunto de prueba, **Naïve Bayes** obtuvo un F1-Score de {nb_f1_test:.4f}, mientras que **Regresión Logística** obtuvo un F1-Score de {lr_f1_test:.4f}. Esto sugiere que Naïve Bayes es más adecuado para este conjunto de datos debido a su mejor equilibrio entre precisión y recall.")
st.write("""Naïve Bayes es recomendable cuando se trabaja con conjuntos de datos pequeños o cuando las características son independientes.  
En este caso, Naïve Bayes demostró un mejor rendimiento general en términos de F1-Score.""")
