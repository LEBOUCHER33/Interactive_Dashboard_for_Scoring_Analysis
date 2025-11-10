"""
Script de création d'une API avec FastAPI.

Objectif : Fournir une interface pour interagir avec un modèle de prédiction de scoring de crédit.
    - recevoir des données en entrée : les features clients au format JSON
    - retourner la prédiction, la probabilité associée (format JSON) et les 5 principales features qui ont influencé la prédiction (explainabilité)
    - recevoir un fichier csv de données clients et calculer les indicateurs globaux associés


Worflow :

- loader le pipeline de prédiction
- définir l'explainabilité avec SHAP
- définir l'API avec FastAPI
- définir les endpoints pour les prédictions

"""

# ///////////////////////////////////////////////
# 1- import des bibliothèques nécessaires
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
# 2- chargement du pipeline de prédiction
# //////////////////////////////////////////////////


with open("./Scripts/App/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)


# /////////////////////////////////////////////////
# 3- définition de l'explainabilité avec SHAP
# /////////////////////////////////////////////////

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])
# sauvegarde de l'explainer
with open("./Scripts/App/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)



# /////////////////////////////////////////////////
# 4- définir l'API avec FastAPI
# /////////////////////////////////////////////////

app = FastAPI()


# app est l'instance de l'API, on va définir les endpoints en utilisant app

# ////////////////////////////////////////////////////////                  
# 5- création d'un endpoint de test de démarrage de l'api
# ////////////////////////////////////////////////////////

@app.get("/")  
def read_root():        
    """
    _Summary_ : fonction de test qui retourne un message de bienvenue.
    _Returns_ :
        dict : dictionnaire contenant le message de bienvenue
    """
    return {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}



# /////////////////////////////////////////////////
# 6- création d'un endpoint de prédiction
# /////////////////////////////////////////////////


@app.post("/predict")  # endpoint de prédiction : la fonction en dessous sera exécutée lorsqu'une requête POST est envoyée à /predict
# on prend en entrée de la fonction, les données formatées en JSON (dictionnaire ou liste de dictionnaires)
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
# 7- création d'un endpoint metrics pour calculer les indicateurs globaux du modèle 
# ////////////////////////////////////////////////////////////////////////////////////

@app.post("/metrics")
async def metrics(file_csv: UploadFile = File(...),
                  sample_size: int = None):
    """
    Endpoint pour calculer les métriques globales du modèle et de la BD.

    _Args_ :
        - file_csv (UploadFile): fichier csv de la BD clients
        - sample_size (int, optional): taille de l'échantillon à utiliser pour le calcul des indicateurs. Par défaut à None 
    
    - Indicateurs de performance du modèle :

        - score moyen global de prediction
        - taux d'accord moyen 
        - risque moyen par client d'un FN
        - drift
        - seuil de décisionnel
        - top_features explainer

    - Indicateurs caractéristiques de la BD clients :

        - nbre total de clients
        - âge moyen
        - nbre total accord / refus
        - taux d'endettement global

    """
    try:
        content = await file_csv.read()
        df = pd.read_csv(io.BytesIO(content))

        if df.empty :
            return JSONResponse(status_code=400, content={"error": "Le fichier est vide."})
        data = pd.DataFrame(df)
        data = data.replace({np.nan: None, np.inf: None, -np.inf: None})
        # calcul des métriques
        metrics = compute_metrics(df=data, 
                                  model_pipeline=model_pipeline,
                                  features_mapping=features_mapping,
                                  explainer=explainer,
                                  sample_size=sample_size)
        return JSONResponse(status_code=200, content=metrics)
    except Exception as e:
        print (traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
    











"""
Pour tester en local l'API, lancer le server local :
url = "http://127.0.0.1:8000/predict"
```bash
uvicorn Scripts.App.api:app --host 127.0.0.1 --port 8000 --reload
```

lien url de l'api sur le cloud :
https://client-scoring-model.onrender.com/predict

"""