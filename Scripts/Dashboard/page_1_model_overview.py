"""
Script d'implémentation d'un dashboard avec Streamlit.

Plan du Dasboard :

- Page 1 :
    1- descriptif clé du modèle de scoring :
        - indicateurs de performance : drift / risque moyen par client d'un FN / score_moyen global par client
        - caractéristiques du fichier client : nbre de demande / proportion credits accordés/refusés
    2- sélection du client :
        - bouton_deroulant d'indexation su fichier client
        - affichage des premiers features du client

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
- définition des pages de l'interface utilisateur
- appelle de l'api
- affichage des données et des résultats associés à la prediction


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
# 2- Loading du fichier client
# /////////////////////////////////////////

data = pd.read_csv("../data_test.csv")


# /////////////////////////////////////////
# conception de la page 1
# /////////////////////////////////////////

st.set_page_config(page_title='Scoring Dashboard', layout='wide')
st.title("Credit Scoring Dashboard")

st.header("Outil métier d'aide à la décision pour l'octroi d'un crédit à la consommation")

col1, col2 = st.columns(2)
with col1:
    st.header("Indicateurs de performance")
    st.metric("data_drift :", "Stable")
    st.metric("seuil décisionnel :", 0.3)
    
with col2:
    st.header("Données clients")
    st.metric("Nombre de demandes : ", 40000)
    st.metric("Crédits accordés : ", 5000)
    st.metric("Crédits refusés : ", 35000)