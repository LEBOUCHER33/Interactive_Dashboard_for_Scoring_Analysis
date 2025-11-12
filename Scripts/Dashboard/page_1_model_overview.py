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

st.title("Tableau de bord - Indicateurs Globaux")
if st.button("Rafraichir les métriques"):
    response = requests.get(url_metrics, refresh=True)
else:
    response = requests.get(url_metrics)
if response.status_code == 200:
    logger.info("Requete POST envoyee avec succes a l'API.")
    metrics = response.json()  
    st.json(metrics)
    # affichage
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Indicateurs de performance")
        st.metric("Risque moyen par client de non-solvabilité :", f"{metrics['risk_moy_fn']*100:.2f}%")
        st.metric("Score moyen global :", f"{metrics['score_moy']}")
        st.metric("data_drift :", "Stable")
        st.metric("seuil décisionnel :", 0.3)    
    with col2:
        st.image(f"{API_URL}/Metrics/global_shap.png",
                 caption="Features Importance Globale")
    with col3:
        st.subheader("Données clients")
        st.metric("Nombre de demandes : ", f"{metrics['nb_clients']}")
        st.metric("Crédits accordés : ", f"{metrics['nb_accord']*100/metrics['nb_clients']:.2f}%")        
        st.metric("Crédits refusés : ", f"{metrics['nb_refus']*100/metrics['nb_clients']:.2f}%")
        st.metric("Taux d'accord moyen :", f"{metrics['taux_accord']*100:.2f}%")

# ////////////////////////////////////
# extrait du fichier client
# ////////////////////////////////////

if st.button("Refresh global metrics"):
    response=requests.get(url_metrics, refresh=True)
else:
    response=requests.get(url_metrics)
st.markdown(f"Uploading de la Base de Données clients ({sample_size} clients)")
st.write(data.head(5))

# ////////////////////////////////
# selection d'un client
# ///////////////////////////////

st.subheader("Sélection du client pour afficher les prédictions")
client_ids = data['SK_ID_CUST'].astype(str).tolist()  
selected_client_id = st.selectbox("Sélectionnez un ID client:", client_ids)

st.session_state['selected_client_id'] = selected_client_id

# bouton pour aller à la page client

if st.button("Afficher les prédictions du client"):
    st.switch_page("page_2_client_details.py")