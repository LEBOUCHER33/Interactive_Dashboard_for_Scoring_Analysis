"""
Script pour afficher les prédictions d'un client sélectionné via une requête POST
"""

# ////////////////////////////////////////
# 1- Import des librairies
# ////////////////////////////////////////
import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
import requests

# /////////////////////////////////////////
# 2- Paramètres
# /////////////////////////////////////////
# url
USE_RENDER = False  # False = local, True = Render
if USE_RENDER:
    API_URL = "https://client-scoring-model.onrender.com"
else:
    API_URL = "http://127.0.0.1:8000"

# Endpoints
url_predict = f"{API_URL}/predict"

# data
file_path = "./Data/Data_cleaned/application_test_final.csv"
data = pd.read_csv(file_path)

# /////////////////////////////////////////
# 3- Conception de la page 2
# /////////////////////////////////////////
st.set_page_config(page_title='Client Details', layout='wide')
st.title("Analyse client individuelle")

# Vérifier si un client a été sélectionné sur la page d'accueil
if "selected_client_id" not in st.session_state:
    st.warning("Veuillez sélectionner un client sur la page d'accueil")
    st.page_link("page_1_model_overview.py", label="Retour à la page d'accueil")
    st.stop()

# Obtenir l'ID du client sélectionné depuis l'état de la session
id_client = st.session_state["selected_client_id"]
st.write(f"Client sélectionné: {id_client}")

# Afficher les informations du client sélectionné dans une sidebar
st.sidebar.header("Informations du client")
client_data = data[data['SK_ID_CUST'].astype(str) == id_client]
if client_data.empty:
    st.sidebar.write("Aucun client trouvé avec cet ID")

# Faire la requête POST pour obtenir les prédictions
st.subheader("Prédictions pour le client sélectionné")
if st.button("Obtenir les prédictions"):
    if not client_data.empty:
        # Préparer les données pour la requête POST
        data_client = client_data.replace({np.nan: None, np.inf: None, -np.inf: None})
        try:
            response = requests.post(url_predict, 
                                     json={"client_data": data_client},
                                     headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                prediction = response.json()
                st.write("Résultats de la prédiction:")
                st.json(prediction)
            else:
                st.error(f"Erreur lors de la requête: {response.status_code}")
                st.write("Veuillez vérifier les paramètres de l'API.")
        except Exception as e:
            st.error(f"Une erreur est survenue: {str(e)}")
    else:
        st.warning("Aucun client trouvé avec cet ID")