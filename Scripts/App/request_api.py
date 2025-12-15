
"""
Script permettant de tester le fonctionnement de l'api :

1- en local
2- le deploiement de l'api de scoring sur un serveur cloud 

"""

# ///////////////////////////
# 1- Configuration du script
# ///////////////////////////

# importer les bibliotheques necessaires
import requests
import pandas as pd
import numpy as np
from loguru import logger
import traceback



# ajouter un logger d'information
logger.add("logs/request_api.log", format="{time} {level} {message}", level="INFO", retention="10 days", compression="zip")
logger.add("logs/request_api_error.log", format="{time} {level} {message}", level="ERROR", retention="10 days", compression="zip")
logger.add("logs/request_api_debug.log", format="{time} {level} {message}", level="DEBUG", retention="10 days", compression="zip")


# ///////////////////////////
# 2- Tester l'API de scoring
# ///////////////////////////


# /////////////////////////////////////////////
# 2-1 test de connexion à l'API 
# /////////////////////////////////////////////


# definir les url de l'api

USE_RENDER = False  # False = local, True = Render


if USE_RENDER:
    API_URL = "https://scoring-model-2z6o.onrender.com"
else:
    API_URL = "http://127.0.0.1:8000"

# Endpoints
url_predict = f"{API_URL}/predict"
url_metrics = f"{API_URL}/metrics"

# Test de connexion à l'API
try:
    healthcheck = requests.get(f"{API_URL}", timeout=300)
    if healthcheck.status_code in (200, 404):
        print(f"Connexion réussie à l API : {API_URL}")
    else:
        print(f"API a répondu avec le code {healthcheck.status_code}")
except requests.exceptions.RequestException:
    print("❌ Impossible de se connecter à l API.")
    print(traceback.format_exc())
    exit()


# /////////////////////////////////////////////
# 2-2 test du endpoint /predict 
# /////////////////////////////////////////////

# creer un dataframe d'exemple

# charger les donnees
data = pd.read_csv("./Data/Data_cleaned/application_test_final.csv")

# selectionner un echantillon de n ligne(s)
data_sample = data.sample(n=1, random_state=42)
logger.info(f"Echantillon de donnees selectionne pour le test de l'API :\n{data_sample}")

# nettoyer les donnees
data_input = data_sample.replace({np.nan: None, np.inf: None, -np.inf: None})

# convertir le dataframe en liste de dictionnaires (format json attendu par l'api)
data_json = data_input.to_dict(orient="records")
logger.debug(f"Donnees converties en format JSON pour l'API :\n{data_json}")


try:
    response = requests.post(url_predict, 
                             headers={"Content-Type": "application/json"}, 
                             json=data_json)
    response.raise_for_status()  # lever une erreur pour les codes de statut 4xx/5xx et afficher l'erreur
    logger.info("Requete POST envoyee avec succes a l'API.")
    print(f"Reponse de l'API : {response.json()}, status code : {response.status_code}")
except requests.exceptions.HTTPError as http_err:
    logger.error(f"Erreur HTTP lors de l'envoi de la requete POST a l'API : {http_err}")
    print(traceback.format_exc())
    raise
except requests.exceptions.RequestException as e:
    logger.error(f"Erreur lors de l'envoi de la requete POST a l'API : {e}")
    print(traceback.format_exc())
    raise


# /////////////////////////////////////////
# 2-2 Test du endpoint /metrics
# /////////////////////////////////////////


# configuration du logger

logger.add("logs/request_api_metrics.log", format="{time} {level} {message}", level="INFO", retention="10 days", compression="zip")
logger.add("logs/request_metrics_error.log", format="{time} {level} {message}", level="ERROR", retention="10 days")

file_path = "./Data/Data_cleaned/application_test_final.csv"
logger.info(f"Test du endpoint /metrics avec le fichier {file_path}")

sample_size = 50

# test du endpoint

try:
    response = requests.get(url_metrics, timeout=300)
    response.raise_for_status()
    logger.info("Requete GET envoyee avec succes a l'API.")
    results = response.json()
    # Affichage partiel des résultats
    logger.info("=== Résumé des métriques ===")
    for key, value in results.items():
        if key != "top_features":
            logger.info(f"{key}: {value}")

    logger.info("=== Exemple de top features ===")
    top_feats = results.get("top_features", [])
    if top_feats:
        logger.info(top_feats[:3])  # afficher les 3 premières lignes

except requests.exceptions.RequestException as e:
    logger.error(f"Erreur HTTP : {e}")
    print(traceback.format_exc())
    exit()
except ValueError as e:
    logger.error(f"Erreur de décodage JSON : {e}")
    print(traceback.format_exc())
    exit()
except Exception as e:
    logger.error(f"Erreur inattendue : {e}")






