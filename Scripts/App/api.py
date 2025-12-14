"""
Script de création d'une API avec FastAPI.

Objectif : 

1-Fournir une interface pour interagir avec le modèle entrainé de prédiction de scoring de crédit.
    - recevoir des données en entrée : les features clients au format JSON
    - retourner 
        - la prédiction, 
        - la probabilité associée (format JSON)
        - les 5 principales features qui ont influencé la prédiction (explainabilité)

2- Calculer les indicateurs clés du modèle et de la BD pour le dashboard associé à l'api :
    - recevoir un fichier csv de données clients 
    - calculer les indicateurs globaux associés
    - générer les visualisations globales

Worflow :

- loader les data (BD clients)
- loader le pipeline de prédiction
- calculer les métriques globales
- définir l'explainabilité globale avec SHAP
- implémenter l'API avec FastAPI
- définir les endpoints pour les prédictions et pour les métriques

"""

# ///////////////////////////////////////////////
# import des bibliothèques nécessaires
# ///////////////////////////////////////////////

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import shap
from loguru import logger
from Scripts.App.utils import features_mapping, compute_metrics
from contextlib import asynccontextmanager
from requests import request
import numpy as np
import traceback
import os
import requests
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi.middleware.cors import CORSMiddleware



# //////////////////////////////////////////////////
# loading des data
# //////////////////////////////////////////////////

#https://drive.google.com/file/d/1O8nJnYQnTolRfoP4mFyc13OlZBeAm-Nv/view?usp=drive_link

FILE_ID = "1O8nJnYQnTolRfoP4mFyc13OlZBeAm-Nv"


url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


def load_df_sample(n=500):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(
        io.BytesIO(response.content),
        sep=",",
        engine="python",
        nrows=n
    )

df = load_df_sample()



# //////////////////////////////////////////////////
# loading du pipeline de prédiction
# //////////////////////////////////////////////////

with open("./Scripts/App/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)
try:
    logger.info("Pipeline de prédiction chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du pipeline de prédiction : {e}")

preprocessor = model_pipeline.named_steps['preprocessor']
model = model_pipeline.named_steps['model']
try:
    logger.info("Modèle de prédiction extrait du pipeline avec succès.")
except Exception as e:
    logger.error(f"Erreur lors de l'extraction du modèle du pipeline : {e}")    



CACHED_METRICS = None
GLOBAL_EXPLAINER = None


# ////////////////////////////////////////////////////////////////////////////////////
# création d'un endpoint compute_metrics pour lire les indicateurs globaux du modèle 
# ////////////////////////////////////////////////////////////////////////////////////

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CACHED_METRICS
    print("[STARTUP] Lancement de l'API...")
    
    # 1. Chargement/Calcul de l'Explainer
    
    # 2. Tentative de calcul des métriques (Petit échantillon)
    try:
        print("[STARTUP] Tentative de pré-calcul des métriques...")
        raw_metrics = compute_metrics(
            df=df, 
            model_pipeline=model_pipeline,
            explainer=None,
            features_mapping=features_mapping,
            sample_size=min(500,len(df))
        )
        CACHED_METRICS = jsonable_encoder(raw_metrics)
        print("[STARTUP] Métriques pré-calculées avec succès !")
    except Exception as e:
        print(f"[STARTUP] Échec du pré-calcul : {e}")
        CACHED_METRICS = None
    
    yield
    print("[SHUTDOWN] Arrêt de l'API.")

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://interactive-dashboard-for-scoring-3y6d.onrender.com"],  
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/compute_metrics")
async def compute_data(refresh:bool = False):
    global CACHED_METRICS , GLOBAL_EXPLAINER
    
    if refresh or CACHED_METRICS is None:
        logger.info(f"REQ REÇUE - Refresh demandé : {refresh}")
        try:  
            df_copy = df.copy()
            if GLOBAL_EXPLAINER is None:
                GLOBAL_EXPLAINER = shap.TreeExplainer(model)
            raw_metrics = compute_metrics(
                    df=df_copy,  
                    model_pipeline=model_pipeline,
                    explainer=GLOBAL_EXPLAINER,
                    features_mapping = features_mapping,
                    sample_size=len(df_copy)
                )
            CACHED_METRICS = jsonable_encoder(raw_metrics) 
            logger.info("Calcul des métriqes terminé.")

        except Exception as e:
            logger.error(f"Erreur lors du calcul {e}")
    return CACHED_METRICS

"""
pour debug :

@app.get("/compute_metrics")
async def compute_data_get(refresh: bool = False):
    return await compute_data(refresh)
"""


# /////////////////////////////////////////////////
# création d'un endpoint de prédiction
# /////////////////////////////////////////////////


@app.get("/predict/{client_id}")
async def predict(client_id: int):
    """
    _Summary_: Endpoint POST pour récupérer une prédiction de scoring client.
    _Args_:
        data (list[dict] | dict): données clients au format JSON (un ou plusieurs individus)
    _Return_: JSONResponse
    """
    global CACHED_METRICS

    print("[DEBUG] CACHED_METRICS keys:", list(CACHED_METRICS.keys()) if CACHED_METRICS else None)
    if CACHED_METRICS is None:
        return JSONResponse(status_code=503, content={"error": "Les prédictions ne sont pas encore prêtes."})
    try:
        client_id_str = str(client_id)
        client_data = CACHED_METRICS["client"].get(client_id_str)
        if not client_data:
            return JSONResponse(
                status_code=404, 
                content={"error": f"Client {client_id} introuvable dans l'échantillon chargé."}
            )

        logger.info(f"Données récupérées pour le client {client_id}.")
        return client_data

    except Exception as e:
        logger.error(f"Erreur lors de la récupération du client : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})




"""
Pour tester en local l'API, lancer le server local :
url = "http://127.0.0.1:8000"
```bash
uvicorn Scripts.App.api:app --host 127.0.0.1 --port 8000 --reload
```

lien url de l'api sur le cloud :
https://client-scoring-model.onrender.com

"""