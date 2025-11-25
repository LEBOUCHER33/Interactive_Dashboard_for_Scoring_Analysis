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
import traceback
from streamlit_extras.switch_page_button import switch_page
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../App')))
from Scripts.App.utils import features_mapping






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
try:
    logger.info("Données client chargées avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement des données client : {e}")
data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
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


st.subheader("Sélection du client pour afficher les prédictions")
client_ids = data['SK_ID_CURR'].astype(str).tolist()  
selected_client_id = st.selectbox("Sélectionnez un ID client:", client_ids)

# sauvegarde de l'ID client sélectionné dans l'état de la session
st.session_state['selected_client_id'] = selected_client_id

# extraction des données du client sélectionné
client_id_int = int(selected_client_id)
client_data = data[data["SK_ID_CURR"] == client_id_int]
if client_data.empty:
    st.error("Aucun client trouvé dans la base.")
    st.stop()

# affichage de l'ID client sélectionné
st.write(f"Client sélectionné : **{selected_client_id}**")
st.success("Données du client chargées avec succès.")
st.write(client_data.head())


# /////////////////////////////////////////
# conception de la side_bar
# /////////////////////////////////////////

st.sidebar.header("Indicateurs clés du client")


client_data = client_data.replace({np.nan: None, np.inf: None, -np.inf: None})
client_data_json = client_data.to_dict(orient='records')[0]  # convertir en dict pour l'API

# requête POST à l'API pour obtenir la prédiction
try:
    #st.write("JSON envoyé :", client_data_json)
    response = requests.post(url_predict, 
                             headers={"Content-Type": "application/json"}, 
                             json=client_data_json,
                             timeout=500)
    response.raise_for_status()  # lève une exception pour les codes 4xx/5xx
    print(f"Réponse de l'API : {response.json()}, status code : {response.status_code}")
    logger.info("Requête POST envoyée avec succès à l'API pour la prédiction client.")
    logger.debug(f"Donnees converties en format JSON pour l'API :\nclient_data_json : {client_data_json}")
    response = response.json()
except requests.exceptions.RequestException as e:
    logger.error(f"Erreur lors de la requête POST à l'API : {e}")
    st.write(traceback.format_exc(), str(e))
    st.stop()
except ValueError as e:
    logger.error(f"Erreur de décodage JSON de la réponse de l'API : {e}")
    st.write(traceback.format_exc())
    st.stop()
except Exception as e:
    logger.error(f"Erreur inattendue : {e}")
    st.write(traceback.format_exc())
    st.stop()
    st.error("Erreur lors de la récupération de la prédiction du client.")
    st.write(traceback.format_exc())
    st.stop()

st.subheader("Détails de la prédiction")

proba=float(response['probabilite_1'][0])
pred=int(response['prediction'][0])
top_features = response["top_features"][0] # liste (feat, shap_val)


# ////////////////////////////////////////////////
# affichage des résultats dans la side_bar
# ////////////////////////////////////////////////

st.sidebar.subheader("Résultats de la prédiction") 
st.sidebar.write(f"**ID Client** : {selected_client_id}")
st.sidebar.write(f"**Prédiction (0=Bon payeur, 1=Mauvais payeur)** : {response['prediction'][0]}")
st.sidebar.write(f"**Probabilité d'être un mauvais payeur** : {response['probabilite_1'][0]*100:.2f} %")
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
if proba < 0.3:
    color = "#4CAF50"  
elif proba < 0.6:
    color = "#FF9800"
else:
    color = "#F44336"

st.markdown(f"""
<div style="
    background-color: #eee;
    border-radius: 10px;
    height: 22px;
    width: 100%;
    position: relative;">
    <div style="
        background-color: {color};
        width: {proba*100}%;
        height: 100%;
        border-radius: 10px;">
    </div>
</div>

<p style="text-align:center; font-weight:bold;">
    Probabilité de défaut : {proba*100:.2f}%
</p>
""", unsafe_allow_html=True)

# TOP FEATURES

st.subheader("Raisons principales de la décision")
col1, col2 = st.columns([1, 1])
with col1:
    for feat, val in top_features:
        color = "#F44336" if val > 0 else "#4CAF50"  # rouge = augmente risque, vert = réduit risque
        st.markdown(
            f"""
            <div style="
                background-color:{color}22;
                padding:10px;
                border-radius:8px;
                margin-bottom:6px;
            ">
            <b>{feat}</b>  
            <br>Impact SHAP : <b style='color:{color};'>{val:+.4f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        features = [f for f,v in top_features]
        values = [v for f,v in top_features]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(features, values)
        ax.set_title("Impact SHAP sur la décision (Top 5)")
        ax.set_xlabel("Contribution")
        st.pyplot(fig)

# SHAP LOCAL PLOT

local_shap_path = f"./Metrics/shap_local_['{selected_client_id}'].png"
if os.path.exists(local_shap_path):
    st.image(local_shap_path,
         caption="Importance des features pour la prédiction du client selon SHAP")
else:
    st.info("Le graphique SHAP local pour ce client n'est pas disponible.")

# implémentation d'un bouton pour accéder à des plots :
st.subheader("Analyse complémentaire")

# distributions des features du client vs population

mapped_features = [utils.features_mapping(col) for col in data.columns]
features = [str(feat) for feat in mapped_features]
pretty_names = [feat for feat,_ in top_features]

selected_feat_pretty = st.selectbox(
    "Sélectionnez une feature pour afficher sa distribution :",
    pretty_names,
    key="feature_select"
)
 
raw_name = features[pretty_names.index(selected_feat_pretty)] 
client_value = client_data.iloc[raw_name].values[0]


st.write(f"Feature sélectionnée : **{selected_feat_pretty}** ({raw_name})")
st.write(f"Valeur client : **{client_value}**")

def get_top_feature_distributions(client_data, data, top_features):
    """
    _Summary_: Affichage des distributions des features du client vs population
    Args:
        client_data (_type_): données du client sélectionné
        data (_type_): données de la population
        top_features (_type_): liste des features les plus importantes pour la prédiction du client
    _Returns_: None
    """
    st.write("Affichage des distributions des features du client vs population")
    for feat, val in top_features:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(data[feat].dropna(), bins=30, alpha=0.7, label='Population', color='gray')
        ax.axvline(client_data[feat].values[0], color='red', linestyle='dashed', linewidth=2, label='Client')
        ax.set_title(f"Distribution de la feature : {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)

fig = get_top_feature_distributions(data, raw_name, client_value)
st.pyplot(fig)


"""

feature_names = [feat for feat, _ in top_features]
real_feat=response['top_real_names'][0]
real_names = [f for f in real_feat]
selected_feat_pretty = st.selectbox(
    "Sélectionnez une feature pour afficher sa distribution :",
    feature_names,
    key="feature_select"
)
client_value = client_data[selected_feat_pretty].values[0]


st.selectbox("Sélectionnez une feature pour afficher sa distribution :", top_features, key="feature_select")
for f in top_features:
    if st.session_state['feature_select'] == f:
        feat_index = f.index(f)
        feat_name = features[feat_index]
        client_value = client_data.iloc[0][f]
        st.write(f"Affichage de la distribution pour la feature sélectionnée : {f[feat_index]}")
        fig = get_top_feature_distributions(data, f, client_value)
        st.pyplot(fig)






# bouton pour revenir à la page 1
if st.button("Revenir à l'accueil // au tableau de bord du modèle"):
    switch_page("page_1_model_overview")


"""