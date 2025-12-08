"""
Script pour afficher les prédictions d'un client sélectionné via une requête GET
"""

# ////////////////////////////////////////
# 1- Import des librairies
# ////////////////////////////////////////

import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
import requests
import traceback
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import features_mapping





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
df = pd.read_csv(file_path)
try:
    logger.info("Données client chargées avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement des données client : {e}")
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
try:
    logger.info("Conversion de SK_ID_CURR en int réussie.")
except Exception as e:
    logger.error(f"Erreur lors de la conversion de SK_ID_CURR : {e}")

# /////////////////////////////////////////
# 3- Conception de la page 2
# /////////////////////////////////////////

st.set_page_config(page_title='Client Details', layout='wide')
st.title("Analyse client individuelle")
st.info("Détails et prédiction pour un client sélectionné")

# /////////////////////////////////////////
# 3-1 Sélection du client
# /////////////////////////////////////////

if "selected_client_id" not in st.session_state:
    st.session_state["selected_client_id"] = None


st.subheader("Sélection du client pour afficher les prédictions")

client_ids = df['SK_ID_CURR'].astype(str).tolist()  
selected_client_id = st.selectbox("Sélectionnez un ID client:", client_ids)

# sauvegarde de l'ID client sélectionné dans l'état de la session

st.session_state["selected_client_id"] = selected_client_id

# convertir en int pour le filtrage
client_id_int = int(selected_client_id)
client_data = df[df["SK_ID_CURR"] == client_id_int]

if client_data.empty:
    st.error("Aucun client trouvé dans la base.")
    st.stop()

st.write(f"Client sélectionné : **{selected_client_id}**")
st.success("Données du client chargées avec succès.")
#st.write(client_data.head())

# /////////////////////////////////////////
# conception de la side_bar
# /////////////////////////////////////////

st.sidebar.header("Indicateurs clés du client")

client_id = st.session_state["selected_client_id"]

# requête GET à l'API pour obtenir la prédiction
try:
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{url_predict}/{client_id}",
                             timeout=5000)
    response.raise_for_status()  # lève une exception pour les codes 4xx/5xx
    print(f"Réponse de l'API : {response.json()}, status code : {response.status_code}")
    logger.info("Requête GET envoyée avec succès à l'API pour la prédiction client.")
    data = response.json()
    #st.write(data)
except requests.exceptions.RequestException as e:
    logger.error(f"Erreur lors de la requête GET à l'API : {e}")
    st.write(traceback.format_exc(), str(e))
    st.stop()
except ValueError as e:
    logger.error(f"Erreur de décodage JSON de la réponse de l'API : {e}")
    st.write(traceback.format_exc())
    st.stop()
except Exception as e:
    st.error("Erreur lors de la récupération de la prédiction du client.")
    st.write(traceback.format_exc())
    st.stop()

st.subheader("Détails de la prédiction")

proba = float(data.get("prediction_proba"))
pred = int(data.get("prediction"))
pred_seuil = int(data.get("prediction_proba_seuil"))
top_features_list = data.get("top_features", []) or []



# ////////////////////////////////////////////////
# affichage des résultats dans la side_bar
# ////////////////////////////////////////////////

st.sidebar.subheader("Résultats de la prédiction")
st.sidebar.write(f"**ID Client** : {selected_client_id}")
st.sidebar.write(f"**Prédiction (0=Bon payeur, 1=Mauvais payeur)** : {data.get('prediction')}")
st.sidebar.write(f"**Probabilité d'être un mauvais payeur** : {data.get('prediction_proba')*100:.2f} %")
st.sidebar.write(f"**Prédiction avec seuil de risque** : {'Mauvais payeur' if proba >= 0.3 else 'Bon payeur'}")




# ////////////////////////////////////////////////
# affichage des résultats dans la page principale
# ////////////////////////////////////////////////

# MESSAGE PRINCIPAL
st.subheader("Décision du modèle pour ce client")

if pred == 1:
    st.error("Le modèle prédit que ce client est un **mauvais payeur**.")
else:
    st.success("Le modèle prédit que ce client est un **bon payeur**.")


# BARRE DE PROBABILITÉ
if pred_seuil < 0.3:
    color = "#4CAF50"
elif pred_seuil < 0.6:
    color = "#FF9800"
else:
    color = "#F44336"

st.markdown(
    f"""
    <div style="
        background-color:#eee;
        border-radius:10px;
        height:22px;
        width:100%;
        margin-top:10px;
        position:relative;">
        <div style="
            background-color:{color};
            width:{proba*100}%;
            height:100%;
            border-radius:10px;">
        </div>
    </div>

    <p style="text-align:center;font-weight:bold;margin-top:5px;">
        Probabilité de défaut : {proba*100:.2f}%
    </p>
    """,
    unsafe_allow_html=True
)


# EXPLAINABILITE

#st.write("### Top 5 features")

feat_pretty_names = [features_mapping(f) for f,_ in top_features_list]


# TOP FEATURES

st.subheader("Raisons principales de la décision")
col1, col2 = st.columns([1, 2])
# Left column: feature impacts
with col1:
    for pretty, val in top_features_list:
        color = "#F44336" if val > 0 else "#4CAF50"
        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                padding:10px;
                border-radius:8px;
                margin-bottom:8px;
            ">
                <b>{pretty}</b><br>
                Impact SHAP : <b style='color:{color};'>{val:+.4f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
# Right column: horizontal bar plot
with col2:
    features = [features_mapping(f) for f,_ in top_features_list]
    values = [v for _, v in top_features_list]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(features[::-1], values[::-1])  # reverse for nicer order
    ax.set_title("Impact SHAP sur la décision (Top 5)")
    ax.set_xlabel("Contribution")
    st.pyplot(fig)  # pass the figure



# implémentation d'un bouton pour accéder à des plots :
st.subheader("Analyse complémentaire")

# distributions des features du client vs population

if feat_pretty_names:
    selected_feat_pretty = st.selectbox(
        "Sélectionnez une feature pour afficher sa distribution :",
        feat_pretty_names,
        key=f"feature_select_dist_{selected_client_id}"
    )
else:
    selected_feat_pretty = ""



#st.write(client_data.head())

def get_top_feature_distributions(client_data, df, feature_name):
    """
    _Summary_: Affichage des distributions des features du client vs population
    Args:
        client_data (_type_): données du client sélectionné
        data (_type_): données de la population
        top_features (_type_): liste des features les plus importantes pour la prédiction du client
    _Returns_: None
    """
    st.write("Affichage des distributions des features du client vs population")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df[feature_name].dropna(), bins=30, alpha=0.7, label='Population', color='darkblue')
    serie = df[feature_name].replace({None: np.nan}).dropna()
    client_val= 
    ax.hist(pop, bins=30, alpha=0.7, label="Population")
    ax.axvline(client_val, color="red", linestyle="--", linewidth=2, label="Client")

    ax.set_title(f"Distribution : {feature_raw}")
    ax.legend()
    return fig

if feat_pretty_names:
    fig = get_top_feature_distributions(df, client_data, selected_feat_pretty)
    if fig:
        st.pyplot(fig)



