"""
Script de création d'une interface utilisateur web avec la lib Steamlit.

On va développer une interface web intéractive pour que le client puisse intéragir avec le modèle de scoring de crédit.

Avec Steamlite on utilise directement le langage Python, et non du html

"""

# 1- Import des librairies

import streamlit as st
import requests
import pandas as pd
import numpy as np



# 2- Définition de l'interface web

st.title("Scoring de crédit")

# 3- chargement du fichier csv contenant les données réelles

uploaded_file = st.file_uploader("upload your csv file", type="csv")

if uploaded_file:
    df_data = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df_data.head())

# 4- sélection de un ou plusieurs clients

# filtre sur le dataframe
st.subheader("Select clients to predict")
selected_indices = st.multiselect(
    "Choose row numbers (indices) to predict",
    df_data.index.tolist()
)

# formatage des données en format json pour l'api
if selected_indices:
    df_selected = df_data.loc[selected_indices]
    df_processed = df_selected.replace({np.nan: None, np.inf: None, -np.inf: None})
    st.write("Selected clients:")
    st.dataframe(df_processed)


# 5- envoyer une requête https pour interroger l'api et renvoyer la prédiction

if st.button("Get Predictions"):
    data_json = {"features": df_processed.to_dict(orient="records")}
    url = "http://127.0.0.1:8000/predict"  # localhost
    url_cloud = "https://client-scoring-model.onrender.com/predict"
    response = requests.post(url_cloud, 
                             headers={"Content-Type": "application/json"}, 
                             json=data_json)
    
    if response.status_code == 200:
        result = response.json()
        st.subheader("Predictions")
        st.write(result["prediction"])
        st.subheader("Probabilities")
        st.write(result["probabilite_1"])
        st.subheader("Top 5 features by SHAP")
        st.write(result["top_features"])
    else:
        st.error(f"Error: {response.status_code}")


""" 
```bash
python -m streamlit run app.py
```
"""


