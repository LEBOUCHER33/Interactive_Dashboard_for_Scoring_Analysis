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
from utils import features_mapping






# //////////////////////////////////////////////////
# 2- chargement du pipeline de prédiction
# //////////////////////////////////////////////////


with open("./pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)


# /////////////////////////////////////////////////
# 3- définition de l'explainabilité avec SHAP
# /////////////////////////////////////////////////

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])

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
    for i in range(len(input_data)):
        features_shap = dict(zip(input_data.columns, shap_values[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        features_mapped = [
            {"feature" : features_mapping([f])[0], "importance" : float(v)} for f, v in top_5_features
            ]

    
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probabilite_1": prediction_proba.tolist(),
        "prediction_seuil" : prediction_proba_seuil.tolist(),
        "top_features": features_mapped 
    }


# ////////////////////////////////////////////////////////////////////////////////////
# 7- création d'un endpoint metrics pour calculer les indicateurs globaux du modèle 
# ////////////////////////////////////////////////////////////////////////////////////

@app.post("/metrics")
async def metrics(file_csv: UploadFile = File(...)):
    """
    Endpoint pour calculer les métriques globales du modèle et de la BD.
    
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
    content = await file_csv.read()
    df = pd.read_csv(io.BytesIO(content))
    if df.empty :
        return JSONResponse(status_code=400, content={"error": "Le fichier est vide."})
    data = pd.DataFrame(df)
    # calcul des predictions
    prediction = model_pipeline.predict(data) # score (0, 1)
    prediction_proba = model_pipeline.predict_proba(data)[:,1] # proba d'être 1
    prediction_proba_seuil = (prediction_proba>=0.3).astype(int) # proba d'être 1 avec un seuil plus stringent
    # explainabilite 
    data_transformed = model_pipeline.named_steps['preprocessor'].transform(data)
    shap_expl = explainer(data_transformed)
    shap_values = shap_expl.values
    for i in range(len(data)):
        features_shap = dict(zip(data.columns, shap_values[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        # mapping
        features_mapped = [
            {"feature" : features_mapping([f])[0], "importance" : float(v)} for f, v in top_5_features
            ]
    # calcul des stats
    score_moy = float(prediction.mean())
    taux_accord = float(prediction_proba_seuil.mean())
    risk_moy_fn = float(1753/61503) # FN/total predictions
    seuil_decisionnel = float(0.3)
    drift = float(0.17)
    # stats BD
    nb_clients = int(len(data)) 
    age_moye_client = float(data["DAYS_BIRTH"].mean())
    nb_accord = int(len(prediction[prediction==1]))
    nb_refus = int(len(prediction[prediction==0]))
    global_amt_endettement_mean = float(data["APP_CREDIT_PERC"].mean())
    # resultats
    results = {
        "score_moy": round(score_moy,3),
        "taux_accord": round(taux_accord,2),
        "risk_moy_fn": round(risk_moy_fn,2),
        "drift": drift,
        "seuil_decisionnel": seuil_decisionnel,
        "nb_clients": nb_clients,
        "age_moye_client": round(age_moye_client,1),
        "nb_accord": nb_accord,
        "nb_refus": nb_refus,
        "global_amt_endettement_mean": round(global_amt_endettement_mean,2),
        "top_features": features_mapped
    }
    return JSONResponse(content=results)








"""
Pour tester en local l'API, lancer le server local :
url = "http://127.0.0.1:8000/predict"
```bash
uvicorn api:app --reload
```

lien url de l'api sur le cloud :
https://client-scoring-model.onrender.com/predict

"""