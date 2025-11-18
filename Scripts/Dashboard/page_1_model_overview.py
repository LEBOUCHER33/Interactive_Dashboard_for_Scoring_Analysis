"""
Script d'implémentation d'un dashboard avec Streamlit.

Plan du Dasboard :

- Page 1 :
    1- descriptif clé du modèle de scoring :
        - indicateurs de performance du modèle : drift / risque moyen par client d'un FN / score_moyen global par client
        - affichage de l'explainabilité globale
        - affichage des premieres ID/features de la BD
        - caractéristiques du fichier clients : nbre de demande // proportion credits accordés/refusés
    2- sélection du client :
        - bouton_deroulant d'indexation sur le fichier client


- Page 2 : deux parties 
    1- side_bar : 
        - descriptif du client / éléménts clés :
            - score obtenu
            - probabilité d'obtention du crédit via un visuel colorimétrique
            - âge
            - niveau de revenus
            - niveau de dette
            - capacité d'endettement
        - descriptif de la demande / éléments clés :
            - montant du crédit demandé
            - durée d'endettement
            - taux d'endettement

            
Workflow :

- loading des data
- requêtes API /predict et /metrics 
- affichage des données et des résultats 
- organisation du dashboard


"""

# ////////////////////////////////////////
# 1- Import des librairies
# ////////////////////////////////////////

import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
import requests
from streamlit_extras.switch_page_button import switch_page 


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
url_metrics = f"{API_URL}/metrics"

# data
file_path = "./Data/Data_cleaned/application_test_final.csv"
data = pd.read_csv(file_path)
sample_size = len(data)



# /////////////////////////////////////////
# conception de la page 1
# /////////////////////////////////////////


# /////////////////////////////////////////
# Titre
# /////////////////////////////////////////

st.set_page_config(page_title='Scoring Credit Dashboard', layout='wide')
st.icon="📊"
st.title("📊Credit Scoring Dashboard")

st.header("Outil métier d'aide à la décision pour l'octroi d'un crédit à la consommation")

# //////////////////////////////////////////////////////////////////////
# calcul et affichage des métriques globales de performance du modèle
# //////////////////////////////////////////////////////////////////////

st.header("Tableau de bord - Indicateurs Globaux")

@st.cache_data
def get_global_metrics(refresh: bool = False):
    """
    _Summary_: Récupération des métriques globales du modèle via l'API.
    _Args_:
        refresh (bool): Recalcul des métriques. Par défaut False.
    _Returns_:
        dict: métriques globales ou None si erreur
    """
    try:
        params = {"refresh": True} if refresh else {}
        response = requests.get(url_metrics, params=params, timeout=300)
        response.raise_for_status()  # lève une exception pour les codes 4xx/5xx
        logger.info("Requête GET envoyée avec succès à l'API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la requête GET à l'API : {e}")
        return None

# Streamlit
refresh_button = st.button("Refresh global metrics")
if refresh_button:
    metrics = get_global_metrics(refresh=refresh_button)
else:
    metrics = get_global_metrics()


if metrics is not None:  
    # affichage
    col1, col_spacer, col2, col_spacer2, col3 = st.columns([1, 0.1, 1, 0.3, 1])
    with col1:
        st.subheader("Indicateurs de performance")
        st.metric("Risque moyen par client de non-solvabilité :", f"{metrics['risk_moy_fn']*100:.2f}%")
        st.metric("Score moyen global :", f"{metrics['score_moy']}")
        st.metric("data_drift :", "Stable")
        st.metric("seuil décisionnel :", 0.3)    
    with col2:
        st.subheader("Explainabilité Globale")
        st.image("./Metrics/global_shap.png",
                 caption="Importance globale des features selon SHAP")
    with col3:
        st.subheader("Données clients")
        st.metric("Nombre de demandes : ", f"{metrics['nb_clients']}")
        st.metric("Crédits accordés : ", f"{metrics['nb_accord']*100/metrics['nb_clients']:.2f}%")        
        st.metric("Crédits refusés : ", f"{metrics['nb_refus']*100/metrics['nb_clients']:.2f}%")
        st.metric("Taux d'accord moyen :", f"{metrics['taux_accord']*100:.2f}%")

# ////////////////////////////////////
# extrait du fichier client
# ////////////////////////////////////
st.subheader("Extrait du fichier client")
st.dataframe(data.head(10))

# ////////////////////////////////
# selection d'un client
# ///////////////////////////////

st.subheader("Sélection du client pour afficher les prédictions")
client_ids = data['SK_ID_CURR'].astype(str).tolist()  
selected_client_id = st.selectbox("Sélectionnez un ID client:", client_ids)
# sauvegarde de l'ID client sélectionné dans l'état de la session
st.session_state['selected_client_id'] = selected_client_id

# bouton pour aller à la page client

if st.button("Afficher les prédictions du client"):
    st.session_state["Go_to_client_page"] = True
    st.switch_page("page_2_client_details")
else:
    st.session_state["Go_to_client_page"] = False