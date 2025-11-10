
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
# 2-1 test du endpoint /predict 
# /////////////////////////////////////////////


# definir les url de l'api
url_cloud = "https://client-scoring-model.onrender.com"
url_local = "http://127.0.0.1:8000"

url_predict_local = f"{url_local}/predict"
url_predict_cloud = f"{url_cloud}/predict"

logger.info(f"Demarrage du script de test de l'api de scoring, URL de l'API: {url_cloud}")


try:
    requests.get("http://127.0.0.1:8000")
except requests.exceptions.ConnectionError:
    print("❌ Impossible de se connecter à l’API.")
    print(traceback.format_exc())
    exit()


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
    response = requests.post(url_predict_cloud, 
                             headers={"Content-Type": "application/json"}, 
                             json=data_json)
    response.raise_for_status()  # lever une erreur pour les codes de statut 4xx/5xx
    logger.info("Requete POST envoyee avec succes a l'API.")
    print(f"Reponse de l'API : {response.json()}, status code : {response.status_code}")
except requests.exceptions.RequestException as e:
    logger.error(f"Erreur lors de l'envoi de la requete POST a l'API : {e}")
    raise


# /////////////////////////////////////////
# 2-2 Test du endpoint /metrics
# /////////////////////////////////////////


# configuration du logger

logger.add("logs/request_api_metrics.log", format="{time} {level} {message}", level="INFO", retention="10 days", compression="zip")
logger.add("logs/request_metrics_error.log", format="{time} {level} {message}", level="ERROR", retention="10 days")

# définir les paramètres

url_metrics_cloud = f"{url_cloud}/metrics"
url_metrics_local = f"{url_local}/metrics"


file_path = "./Data/Data_cleaned/application_test_final.csv"
logger.info(f"Test du endpoint /metrics avec le fichier {file_path}")

sample_size = 50

# test du endpoint

try:
    with open(file_path, "rb") as file:
        files = {"file_csv": (file_path, file, "text/csv")}
        params = {"sample_size": sample_size}
        response = requests.post(url_metrics_cloud, files=files, params=params)
    response.raise_for_status()
    logger.info("Requete POST envoyee avec succes a l'API.")
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
except Exception as e:
    logger.error(f"Erreur inattendue : {e}")






