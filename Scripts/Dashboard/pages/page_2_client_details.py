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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)

from Scripts.App.utils import features_mapping




# /////////////////////////////////////////
# 2- Paramètres
# /////////////////////////////////////////

# url
USE_RENDER = True  # False = local, True = Render
if USE_RENDER:
    API_URL = "https://client-scoring-model.onrender.com"
else:
    API_URL = "http://127.0.0.1:8000"

# Endpoints
url_predict = f"{API_URL}/predict"

# data

DATA_PATH = "./Scripts/App/assets/data_sample.csv"
df = pd.read_csv(DATA_PATH)

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
    #session = requests.Session()
    #session.trust_env = False
    response = requests.get(f"{url_predict}/{client_id}",
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


# /////////////////////////////////////////
# Affichage top features SHAP
# /////////////////////////////////////////

def clean_feature_name(name: str) -> str:
    """Retire les préfixes du pipeline pour retrouver le nom brut."""
    if name.startswith("remainder__"):
        return name.replace("remainder__", "")
    if "__" in name:
        return name.split("__")[-1]
    return name


# TOP FEATURES — Nettoyage + mapping propre
top_features_cleaned = [
    (
        raw,                                           # nom pipeline
        features_mapping(clean_feature_name(raw)),     # nom joli propre
        float(val)                                     # valeur shap
    )
    for raw, _, val in top_features_list
]


st.subheader("Raisons principales de la décision")
col1, col2 = st.columns([1, 2])

with col1:
    for raw_feat, pretty_feat, val in top_features_cleaned:

        color = "#F44336" if val > 0 else "#4CAF50"

        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                padding:10px;
                border-radius:8px;
                margin-bottom:8px;
                border-left: 5px solid {color};
            ">
                <small style="opacity:0.7">{raw_feat}</small><br>
                <b>{pretty_feat}</b><br>
                Impact : <b style='color:{color};'>{val:+.4f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# Colonne droite : Bar plot SHAP
with col2:
    features_labels = [pretty for _, pretty, val in top_features_cleaned]
    values = [val for _, _, val in top_features_cleaned]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = range(len(features_labels))

    ax.barh(
        y_pos,
        values[::-1],
        color=["#F44336" if v > 0 else "#4CAF50" for v in values[::-1]],
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_labels[::-1])
    ax.set_title("Impact SHAP sur la décision (Top 5)")
    ax.set_xlabel("Contribution au score")

    st.pyplot(fig)


# ---------------------------------------------------------
# Analyse complémentaire (Distributions)
# ---------------------------------------------------------
st.subheader("Analyse de la position du client")

def plot_feature_distribution(client_data, df_population, feature_name, feature_pretty_name):
    fig, ax = plt.subplots(figsize=(8, 4))

    data_to_plot = df_population[feature_name].dropna()
    ax.hist(data_to_plot, bins=30, alpha=0.6, label='Population', color='#3f51b5', density=True)

    try:
        client_val = client_data[feature_name].values[0]
        ax.axvline(client_val, color="red", linestyle="--", linewidth=2,
                   label=f"Client ({client_val:.2f})")
    except KeyError:
        st.warning(f"Impossible de trouver la valeur brute pour {feature_name}")

    ax.set_title(f"Distribution : {feature_pretty_name}")
    ax.legend()
    return fig

valid_for_plot = [
    (raw, pretty)
    for raw, pretty, _ in top_features_cleaned
    if clean_feature_name(raw) in df.columns  
]

if valid_for_plot:
    options_map = {pretty: raw for raw, pretty in valid_for_plot}

    selected_pretty_name = st.selectbox(
        "Sélectionnez une variable à analyser :",
        list(options_map.keys())
    )

    raw_col_name = clean_feature_name(options_map[selected_pretty_name])

    fig = plot_feature_distribution(client_data, df, raw_col_name, selected_pretty_name)
    st.pyplot(fig)