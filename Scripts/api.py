"""
Script de création d'une API avec FastAPI.

Objectif : Fournir une interface pour interagir avec un modèle de prédiction de scoring de crédit.
    - recevoir des données en entrée : les features clients au format JSON
    - retourner la prédiction, la probabilité associée (format JSON) et les 5 principales features qui ont influencé la prédiction (explainabilité)

Worflow :

- loader le pipeline de prédiction
- définir l'explainabilité avec SHAP
- définir l'API avec FastAPI
- définir les endpoints pour les prédictions
    - un endpoint de test
    - un endpoint de prédiction
    - un endpoint de calcul des métriques du modèle et de la BD pour alimenter le dashboard

"""

# //////////////////////////////////////////
# 1- import des bibliothèques nécessaires
# //////////////////////////////////////////


from fastapi import FastAPI
import pickle
import pandas as pd
import shap
import numpy as np
import requests





# //////////////////////////////////////////
# 2- chargement du pipeline de prédiction
# //////////////////////////////////////////


with open("./Scripts/App/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)



# ////////////////////////////////////////////////
# 3- définition de l'explainabilité avec SHAP
# ////////////////////////////////////////////////

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])

# ///////////////////////////////////////////////
# 4- définir l'API avec FastAPI
# ///////////////////////////////////////////////

app = FastAPI()

# création d'un endpoint de test

@app.get("/")  
def read_root():        
    """
    _Summary_ : fonction de test qui retourne un message de bienvenue.
    _Returns_ :
        dict : dictionnaire contenant le message de bienvenue
    """
    return {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}




# création d'un endpoint de prédiction

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
    explanations = []
    for i in range(len(input_data)):
        features_shap = dict(zip(input_data.columns, shap_values[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        explanations.append(top_5_features)

    
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probabilite_1": prediction_proba.tolist(),
        "prediction_seuil" : prediction_proba_seuil.tolist(),
        "top_features": explanations
    }

 
# création d'un endpoint de calcul de métriques et statistiques de la BD clients test

DATA_PATH = "./Data/application_test_final.csv"
url_api_local = "http://127.0.0.1:8000/predict"
url_api_cloud = "https://client-scoring-model.onrender.com/predict"

@app.get("/metrics")
def get_metrics():
     
    """
    _Summary_: fonction qui lit la BD clients de test et calcule les statistiques globales
        1- Indicateurs de performance du modèle :
            - score moyen global de prediction
            - taux d'accord moyen 
            - risque moyen par client d'un FN
            - drift
            - seuil de décisionnel
            - top_features explainer

        2-Indicateurs caractéristiques de la BD clients :
            - nbre total de clients
            - âge moyen
            - nbre total accord / refus
            - taux d'endettement global
    _Args_:
    _Returns_: dict : dictionnaire contenant les outputs

    """
    # loading des data
    data = pd.read_csv(DATA_PATH)
    data_input = data.replace({np.nan: None, np.inf: None, -np.inf: None})
    data_json = data_input.to_dict(orient="records")
    # predictions
    results = []
    for _, raw in data_input.iterrows():
        response = requests.post(url_api_local, json=data_json, headers={"Content-Type": "application/json"})
        result=response.json()
        results.append({
        "ID_Client": raw["SK_ID_CURR"],
        "Score": result["prediction"][0],
        "probabilite_accord": result["probabilite_1"][0],
        "top_features": result["top_features"][0]
    })
    # convertion en DataFrame
    df_results = pd.DataFrame(results)
    # calcul des indicateurs
    # indicateurs de performance du modèle
    score_moy = df_results["Score"].mean()
    taux_accord = df_results["probabilite_accord"].mean()
    risk_moy_fn = (1753/61503) # FN/total predictions
    seuil_decisionnel = 0.3
    drift = 0.17
    # statistiques de la BD clients
    nb_clients = len(data) 
    age_moye_client = data["DAYS_BIRTH"].mean() / 365
    tot_accord = len(df_results[df_results["probabilite_accord"]==1])
    tot_refus = len(df_results[df_results["probabilite_accord"]==0])
    global_amt_endettement_mean = data["APP_CREDIT_PERC"].mean()
    # calcul de l'explainer global
    explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])
    data_transformed = model_pipeline.named_steps['preprocessor'].transform(data_input)
    shap_expl = explainer(data_transformed)
    shap_values = shap_expl.values
    features_shap = dict(zip(data_input.columns, shap_values[0].tolist()))
    stats = {
        "score_moy": score_moy,
        "taux_accord": taux_accord,
        "risk_moy_fn": risk_moy_fn,
        "drift": drift,
        "seuil_decisionnel": seuil_decisionnel,
        "nb_clients": nb_clients,
        "age_moye_client": age_moye_client,
        "tot_accord": tot_accord,
        "tot_refus": tot_refus,
        "global_amt_endettement_mean": global_amt_endettement_mean,
        "features_shap": features_shap
    }
    return {
        "stats": stats
}






"""
Pour tester en local l'API, lancer le server local :
url = "http://127.0.0.1:8000/predict"
```bash
uvicorn api:app --reload
```

lien url de l'api sur le cloud :
https://client-scoring-model.onrender.com/predict

"""