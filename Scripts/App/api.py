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
import pickle
import pandas as pd
import shap
from loguru import logger
from Scripts.App.utils import features_mapping, compute_metrics
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
from requests import request
import numpy as np
import traceback
import os



# //////////////////////////////////////////////////
# loading des data
# //////////////////////////////////////////////////

df =  pd.read_csv("./Data/Data_cleaned/application_test_final.csv")
df = df.replace({np.nan: None, np.inf: None, -np.inf: None})



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




# ////////////////////////////////////////////////////////////////////////////
# calcul des predictions et des métriques globales sur le dataset complet
# ////////////////////////////////////////////////////////////////////////////

CACHED_PREDICTIONS = None
CACHED_METRICS = None
GLOBAL_EXPLAINER = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    _Summary_: Pré-calcul des prédictions et des métriques globales à l'initialisation de l'API.
    _Args_:
        app (FastAPI): instance de l'application FastAPI
    _Yields_:
        None
    """
    global CACHED_PREDICTIONS, CACHED_METRICS, GLOBAL_EXPLAINER
    print("Pré-calcul des prédictions sur le dataset complet...")
    try:
        CACHED_PREDICTIONS = df.copy()
    except Exception as e:
        print(f"Erreur lors de la copie du dataset pour les prédictions : {e}")
    # pré-calcul des prédictions
    try:
        predictions = model_pipeline.predict(CACHED_PREDICTIONS)
        prediction_proba = model_pipeline.predict_proba(CACHED_PREDICTIONS)[:,1]  # probabilité d'être un mauvais payeur (classe 1)
        prediction_proba_seuil = (prediction_proba>=0.3).astype(int) # inclus la notion de stringence avec un seuil pour minimiser les FN
        CACHED_PREDICTIONS['prediction'] = predictions
        CACHED_PREDICTIONS['prediction_proba'] = prediction_proba
        CACHED_PREDICTIONS['prediction_proba_seuil'] = prediction_proba_seuil
        logger.info("Prédictions pré-calculées avec succès.")
    except Exception as e:
        print(f"Erreur lors du pré-calcul des prédictions : {e}, {traceback.format_exc()}")
    # explainabilité globale avec SHAP
    try:
        GLOBAL_EXPLAINER = shap.TreeExplainer(model)
        logger.info("Explainer global calculé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'explainer global : {e}")
    # sauvegarde de l'explainer global
    with open("./Scripts/App/global_shap_explainer.pkl", "wb") as f:
        pickle.dump(GLOBAL_EXPLAINER, f)
    # calcul des métriques globales
    print("Pré-calcul des métriques globales...")
    try:
        CACHED_METRICS = compute_metrics(
            df=CACHED_PREDICTIONS,  
            model_pipeline=model_pipeline,
            explainer=GLOBAL_EXPLAINER,
            features_mapping = features_mapping,
            sample_size=len(df)
        )
        print("Métriques pré-calculées.")
    except Exception as e:
        print(f"Erreur lors du pré-calcul des métriques : {e}")
    print("API prête à recevoir les requêtes.")
    yield
app = FastAPI(lifespan=lifespan)




# /////////////////////////////////////////////////
# création d'un endpoint de prédiction
# /////////////////////////////////////////////////


@app.post("/predict")  
async def predict(payload: dict = Body(...)): 
    """
    _Summary_: Endpoint POST pour récupérer une prédiction de scoring client.
    _Args_:
        data (list[dict] | dict): données clients au format JSON (un ou plusieurs individus)
    _Return_: JSONResponse 
    """
    global CACHED_PREDICTIONS, GLOBAL_EXPLAINER
    if CACHED_PREDICTIONS is None:
        return JSONResponse(status_code=503, content={"error": "Les prédictions ne sont pas encore prêtes."})
    
    client_id = payload.get("SK_ID_CURR", None)
    if client_id is None:
        logger.warning("Aucun SK_ID_CURR fourni dans les données d'entrée.")
        return JSONResponse(status_code=404, content={"error": "Client ID non trouvé."})
    if client_id not in CACHED_PREDICTIONS["SK_ID_CURR"].values:
            return JSONResponse(status_code=404, content={"error": "Client ID non trouvé."})
    try:
        row_df = CACHED_PREDICTIONS.loc[CACHED_PREDICTIONS["SK_ID_CURR"] == client_id].reset_index(drop=True)
        row_input = row_df.drop(columns=["prediction",
                                      "prediction_proba", "prediction_proba_seuil"])
        row_input_transformed = preprocessor.transform(row_input)
        row_values = row_input_transformed[0]
        feat_names_trans = preprocessor.get_feature_names_out()
        # explanabilité locale avec SHAP
        shap_values = GLOBAL_EXPLAINER.shap_values(row_input_transformed)
        features_shap = dict(zip(feat_names_trans, shap_values[0].tolist()))
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        features_mapped = {features_mapping(f): v for f, v in top_5_features}
        shap.initjs()
        shap_plot_path = f"./Scripts/App/shap_plots/shap_plot_{client_id}.png"
        os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
        base_value = GLOBAL_EXPLAINER.expected_value
        mapped_feature_names = [features_mapping(f) for f in feat_names_trans]
        plt.figure()
        shap.force_plot(
            base_value,
            shap_values[0],
            features=row_input_transformed[0],
            feature_names=mapped_feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        # construction de la réponse
        response = {
            "client_id": int(row_df.loc[0, "SK_ID_CURR"]),
            "prediction": int(row_df.loc[0, "prediction"]),
            "prediction_proba": float(row_df.loc[0, "prediction_proba"]),
            "prediction_proba_seuil": int(row_df.loc[0, "prediction_proba_seuil"]),
            "top_features":  features_mapped,
            "shap_plot_path": shap_plot_path
        }
        logger.info("Prédiction client calculée avec succès.")
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        logger.error(f"Erreur lors de la construction de la réponse : {e}")
        return JSONResponse(status_code=500, content={"error": "Erreur interne du serveur."})

        

# ////////////////////////////////////////////////////////////////////////////////////
# création d'un endpoint metrics pour lire les indicateurs globaux du modèle 
# ////////////////////////////////////////////////////////////////////////////////////

@app.get("/metrics")
async def metrics(refresh:bool = False):
    """
    _Summary_: Endpoint GET pour récupérer les métriques globales du modèle et de la BD.
    _Args_ : refresh(bool) = recalcul des métriques. Par défaut, False
    _Return_ : JSONResponse 
    """
    global CACHED_METRICS
    global GLOBAL_EXPLAINER
    if CACHED_METRICS is None:
        return JSONResponse(status_code=503, content={"error": "Les métriques ne sont pas encore prêtes."})
    try:
        logger.info("Requête GET reçue pour les métriques globales.")
    except Exception as e:
        logger.error(f"Erreur lors de la réception de la requête GET pour les métriques : {e}")
    if refresh or CACHED_METRICS is None:
        print("Recalcul des métriques à la demande.")
        CACHED_METRICS = compute_metrics(df=df.sample(10000),  # pour accélérer le calcul
                                         model_pipeline=model_pipeline,
                                          explainer=GLOBAL_EXPLAINER,
                                          features_mapping=features_mapping,
                                          sample_size=len(df))
    return JSONResponse(status_code=200, content=CACHED_METRICS)







"""
Pour tester en local l'API, lancer le server local :
url = "http://127.0.0.1:8000"
```bash
uvicorn Scripts.App.api:app --host 127.0.0.1 --port 8000 --reload
```

lien url de l'api sur le cloud :
https://client-scoring-model.onrender.com

"""