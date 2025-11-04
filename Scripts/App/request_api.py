
"""
Script permettant de tester le deploiement de l'api de scoring sur un serveur cloud 
"""

# ///////////////////////////
# 1- Configuration du script
# ///////////////////////////

# importer les bibliotheques necessaires
import requests
import json
import pandas as pd
import numpy as np
from loguru import logger

# ajouter un logger d'information
logger.add("logs/request_api.log", format="{time} {level} {message}", level="INFO", retention="10 days", compression="zip")
logger.add("logs/request_api_error.log", format="{time} {level} {message}", level="ERROR", retention="10 days", compression="zip")
logger.add("logs/request_api_debug.log", format="{time} {level} {message}", level="DEBUG", retention="10 days", compression="zip")


# ///////////////////////////
# 2- Tester l'API de scoring
# ///////////////////////////


# definir l'url de l'api
url_cloud = "https://client-scoring-model.onrender.com/predict"
logger.info(f"Demarrage du script de test de l'api de scoring, URL de l'API: {url_cloud}")
url_local = "http://127.0.0.1:8000/predict"


# creer un dataframe d'exemple
data = pd.read_csv("./Data/application_test_final.csv")

# selectionner un echantillon de 5 lignes
data_sample = data.sample(n=1, random_state=42)
logger.info(f"Echantillon de donnees selectionne pour le test de l'API :\n{data_sample}")

# nettoyer les donnees
data_input = data_sample.replace({np.nan: None, np.inf: None, -np.inf: None})

# convertir le dataframe en liste de dictionnaires (format json attendu par l'api)
data_json = data_input.to_dict(orient="records")
logger.debug(f"Donnees converties en format JSON pour l'API :\n{data_json}")

# envoyer la requete POST a l'api
try:
    response = requests.post(url_local, 
                             headers={"Content-Type": "application/json"}, 
                             json=data_json)
    response.raise_for_status()  # lever une erreur pour les codes de statut 4xx/5xx
    logger.info("Requete POST envoyee avec succes a l'API.")
    print("Reponse de l'API :", response.json(), response.status_code)
except requests.exceptions.RequestException as e:
    logger.error(f"Erreur lors de l'envoi de la requete POST a l'API : {e}")
    raise
