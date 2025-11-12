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

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import shap
import io
from loguru import logger
from Scripts.App.utils import features_mapping, compute_metrics
import numpy as np
import traceback

# //////////////////////////////////////////////////
# loading des data
# //////////////////////////////////////////////////

df =  pd.read_csv("./Data/Data_cleaned/application_test_final.csv")


# //////////////////////////////////////////////////
# loading du pipeline de prédiction
# //////////////////////////////////////////////////


with open("./Scripts/App/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)


# /////////////////////////////////////////////////
# définition de l'explainabilité avec SHAP
# /////////////////////////////////////////////////

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])
# sauvegarde de l'explainer
with open("./Scripts/App/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# /////////////////////////////////////////////////
# implémentation de l'API avec FastAPI
# /////////////////////////////////////////////////

app = FastAPI()
                 
# création d'un endpoint de test de démarrage de l'api

@app.get("/")  
def read_root():        
    """
    _Summary_ : fonction de test qui retourne un message de bienvenue.
    _Returns_ :
        dict : dictionnaire contenant le message de bienvenue
    """
    return {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}



# ////////////////////////////////////////////////////////
# calcul des indicateurs clés du modèle et de la BD
# ////////////////////////////////////////////////////////

# calcul des metrics au start de l'api
CACHED_METRICS = None
@app.lifespan("startup")
async def startup_event():
    global CACHED_METRICS
    print("Pré-calcul des métriques globales...")
    CACHED_METRICS = compute_metrics(
        df=df,
        model_pipeline=model_pipeline,
        explainer=explainer,
        features_mapping = features_mapping,
        sample_size=len(df)
    )


# /////////////////////////////////////////////////
# création d'un endpoint de prédiction
# /////////////////////////////////////////////////


@app.post("/predict")  
async def predict(data : list[dict] | dict): 
    """
    _Summary_ : fonction de prédiction qui reçoit les données en format JSON et retourne 
        - la prédiction 
        - la probabilité de solvabilité associée 
        - les 5 features les plus influences 
    _Arguments_ :
        data : (list or dict)
        - un dictionnaire si un seul individu
        - une liste de dictionnaire si plusieurs individus où chaque ligne = 1 dict
    _Returns_ :
        dict : dictionnaire contenant les outputs
            - la prédiction 
            - la probabilité d'être classé "1", soit mauvais payeur
            - les 5 features les plus influentes sur le calcul
    """

    # 1- récupération des données JSON de la requête et conversion en DataFrame pandas pour pouvoir faire la prédiction
    if isinstance(data, dict):
            input_data = pd.DataFrame([data])  # un individu
    elif isinstance(data, list):
        input_data = pd.DataFrame(data)    # plusieurs individus
    else:
        return {"error": "Invalid input format"}
    # 3- faire la prédiction avec le pipeline chargé
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)[:,1]  # probabilité d'être un mauvais payeur (classe 1)
    prediction_proba_seuil = (prediction_proba>=0.3).astype(int) # inclus la notion de stringence avec un seuil pour minimiser les FN
    # 4- explainabilité avec SHAP
    data_transformed = model_pipeline.named_steps['preprocessor'].transform(input_data)  # on applique le préprocesseur aux données d'entrée
    shap_expl = explainer(data_transformed)  # on calcule les valeurs SHAP
    shap_values = shap_expl.values # on extrait seulement les valeurs numériques
    # on affiche les 5 features les plus importantes pour chaque individu
    explanation = []
    for i in range(len(input_data)):
        features_shap = dict(zip(input_data.columns, shap_values[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        features_shap_mapped = {features_mapping(f): v for f, v in features_shap.items()}
        top_5_features = sorted(features_shap_mapped.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        explanation.append(top_5_features)
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probabilite_1": prediction_proba.tolist(),
        "prediction_seuil" : prediction_proba_seuil.tolist(),
        "top_features": explanation 
    }


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
    if refresh or CACHED_METRICS is None:
        print("Recalcul des métriques à la demande.")
        CACHED_METRICS = compute_metrics(df=df, 
                                         model_pipeline=model_pipeline,
                                          explainer=explainer,
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