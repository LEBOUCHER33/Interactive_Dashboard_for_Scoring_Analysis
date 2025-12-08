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
USE_RENDER = False  # False = local, True = Render
if USE_RENDER:
    API_URL = "https://client-scoring-model.onrender.com"
else:
    API_URL = "http://127.0.0.1:8000"

# Endpoints
url_predict = f"{API_URL}/predict"
url_metrics = f"{API_URL}/compute_metrics"

# data
file_path = "./Data/Data_cleaned/application_test_final.csv"
data = pd.read_csv(file_path)
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

st.header("Indicateurs clés du Modèle")

METRICS_DIR = "./Metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False, ttl=5)
def get_global_metrics(refresh: bool = False):
    """
    _Summary_: Récupération des métriques globales du modèle via l'API.
    _Args_:
        refresh (bool): Recalcul des métriques. Par défaut False.
    _Returns_:
        dict: métriques globales ou None si erreur
    """
    session = requests.Session()
    session.trust_env = False
    try:
        params = {"refresh": str(refresh).lower()}
        response = session.post(url_metrics, params=params, timeout=600)

        if response.status_code != 200:
            st.error(f"Erreur API ({response.status_code}): {response.text}")
            return None
        response.raise_for_status() 
        logger.info("Requête GET envoyée avec succès à l'API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        return None

metrics = get_global_metrics(refresh=True)

#st.write("**Debug - Type reçu :**", type(metrics))

if metrics:
    #st.write("**Debug - Contenu reçu :**", metrics)# Cas 1 : Structure correcte (Celle qu'on veut)
    if "metrics" in metrics:
        kp_metrics = metrics['metrics']
        clients_data = metrics.get('client', {}) # .get pour éviter erreur si absent
        st.success("Données structurées chargées correctement.")


if metrics is not None:  
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
        image_filename = "global_shap.png"
        image_path = os.path.join(METRICS_DIR, image_filename)
    
    if os.path.exists(image_path):
        st.image(image_path, caption="Importance globale des features (SHAP)")
    else:
        st.info("L'image s'affichera ici après le calcul des métriques.")

st.header("Caractéristiques du fichier client")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Nombre de clients", kp_metrics['nb_clients'])
if kp_metrics['nb_clients'] > 0:
        taux_accord_calc = (kp_metrics['nb_accord'] / kp_metrics['nb_clients']) * 100
        taux_refus_calc = (kp_metrics['nb_refus'] / kp_metrics['nb_clients']) * 100
else:
    taux_accord_calc = 0
    taux_refus_calc = 0
# affichage des caractéristiques du fichier client       
col_b.metric("Crédits Accordés", f"{taux_accord_calc:.1f}%")
col_c.metric("Crédits Refusés", f"{taux_refus_calc:.1f}%")
col_d.metric("Age moyen", f"{kp_metrics['age_moye_client']} ans")


# ////////////////////////////////////
# extrait du fichier client
# ////////////////////////////////////
st.subheader("Extrait du fichier client")
if clients_data:
        # Transformation du dict {id: {data}} en liste de dicts [{data}, {data}]
        list_clients = list(clients_data.values())
        df_display = pd.DataFrame(list_clients)
        cols_to_show = ["client_id", "prediction", "prediction_proba"]
        st.dataframe(df_display[cols_to_show].head(5))
else:
    st.warning("Aucune donnée client disponible.")

