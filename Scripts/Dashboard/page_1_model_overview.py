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
import os


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
url_metrics = f"{API_URL}/compute_metrics"

# data


DATA_PATH = "./Scripts/App/assets/data_sample.csv"
df = pd.read_csv(DATA_PATH)

try:
    logger.info("Données client chargées avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement des données client : {e}")



# /////////////////////////////////////////
# conception de la page 1
# /////////////////////////////////////////


# /////////////////////////////////////////
# Titre
# /////////////////////////////////////////

st.set_page_config(page_title='Scoring Credit Dashboard', layout='wide')
st.icon="📊"
st.title("📊 Tableau de bord du modèle de Scoring")

st.info("Outil métier d'aide à la décision pour l'octroi d'un crédit à la consommation")

# //////////////////////////////////////////////////////////////////////
# calcul et affichage des métriques globales de performance du modèle
# //////////////////////////////////////////////////////////////////////



def load_metrics_once():
    if "global_metrics" not in st.session_state:
        try:
            params = {"refresh": "false"}
            #session = requests.Session()
            #session.trust_env = False
            response = requests.post(url_metrics, params=params, timeout=600)
            response.raise_for_status()
            st.session_state.global_metrics = response.json()
        except Exception as e:
            st.error(f"Erreur API: {e}")
            st.stop()

    return st.session_state.global_metrics


if st.button("Recalculer les métriques"):
    params = {"refresh": "true"}
    session = requests.Session()
    session.trust_env = False
    response = session.post(url_metrics, params=params, timeout=600)
    st.session_state.metrics = response.json()
    st.success("Métriques recalculées !")


metrics = load_metrics_once()



st.header("Indicateurs clés du Modèle")

if metrics is None:
    st.error("Impossible de récupérer les métriques depuis l’API.")
    st.stop()

# Vérification structure
if "metrics" not in metrics:
    st.error("Structure des données renvoyées par l'API incorrecte.")
    st.stop()

kp_metrics = metrics["metrics"]
clients_data = metrics.get("client", {})

# affichage
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Indicateurs de performance")
    st.metric("Risque moyen par client de non-solvabilité :", f"{kp_metrics['risk_moy_fn']:.3f}%")
    st.metric("Score moyen global :", f"{kp_metrics['score_moy']*100:.2f}%")
    st.metric("Dérive des données :", "Stable")
    st.metric("Seuil décisionnel :", 0.3)    
with col2:
    st.subheader("Explainabilité Globale")
    shap_path = kp_metrics.get("shap_plot_path")
    if shap_path and os.path.exists(shap_path):
        st.image(shap_path, caption="Importance des features (SHAP)")
    else:
        st.warning("L’image SHAP globale n'est pas encore disponible.")


st.header("Caractéristiques du fichier client")
col_a, col_b, col_c, col_d = st.columns(4)
nb_clients = kp_metrics['nb_clients']
nb_accord = kp_metrics['nb_accord']
nb_refus = kp_metrics['nb_refus']

taux_accord_calc = (nb_accord / nb_clients) * 100 if nb_clients else 0
taux_refus_calc = (nb_refus / nb_clients) * 100 if nb_clients else 0

col_a.metric("Nombre de clients", nb_clients)
col_b.metric("Crédits Accordés", f"{taux_accord_calc:.1f}%")
col_c.metric("Crédits Refusés", f"{taux_refus_calc:.1f}%")
col_d.metric("Âge moyen", f"{kp_metrics['age_moye_client']} ans")


# ////////////////////////////////////
# extrait du fichier client
# ////////////////////////////////////
st.subheader("Extrait du fichier client")
if clients_data:
        # Transformation du dict {id: {data}} en liste de dicts [{data}, {data}]
        df_display = pd.DataFrame(list(clients_data.values()))
        cols_to_show = ["client_id", "prediction", "prediction_proba"]
        st.dataframe(df_display[cols_to_show].head(5))
else:
    st.warning("Aucune donnée client disponible.")