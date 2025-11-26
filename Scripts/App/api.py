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
from loguru import logger
from Scripts.App.utils import features_mapping, compute_metrics
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
import os
import uuid
import numpy as np
import traceback

# //////////////////////////////////////////////////
# loading des data
# //////////////////////////////////////////////////


df =  pd.read_csv("./Data/Data_cleaned/application_test_final.csv")
df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
df_sample = df.sample(n=10000, random_state=42)  # échantillon pour accélérer le calcul des métriques

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

# /////////////////////////////////////////////////
# définition de l'explainabilité avec SHAP
# /////////////////////////////////////////////////


# explainer global avec SHAP

df_transformed = preprocessor.transform(df_sample)  # échantillon pour accélérer le calcul
global_explainer = shap.TreeExplainer(model)
# sauvegarde de l'explainer global
with open("./Scripts/App/global_shap_explainer.pkl", "wb") as f:
    pickle.dump(global_explainer, f)



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
@asynccontextmanager
async def lifespan(app: FastAPI):
    global CACHED_METRICS
    print("Pré-calcul des métriques globales...")
    try:
        CACHED_METRICS = compute_metrics(
            df=df.sample(n=5000, random_state=42),  # pour accélérer le calcul au démarrage de l'api
            model_pipeline=model_pipeline,
            explainer=global_explainer,
            features_mapping = features_mapping,
            sample_size=len(df)
        )
        print("Métriques pré-calculées.")
    except Exception as e:
        print(f"Erreur lors du pré-calcul des métriques : {e}")
    yield
app = FastAPI(lifespan=lifespan)



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
    input_data = input_data.replace({None: np.nan, np.inf: np.nan, -np.inf: np.nan})
    # 3- faire la prédiction avec le pipeline chargé
    try:
        input_data = input_data.astype(df.dtypes.to_dict())
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la conversion des types de données : {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}
    client_ids = input_data['SK_ID_CURR'].astype(str).tolist()  
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)[:,1]  # probabilité d'être un mauvais payeur (classe 1)
    prediction_proba_seuil = (prediction_proba>=0.3).astype(int) # inclus la notion de stringence avec un seuil pour minimiser les FN
    try:
        prediction = prediction.astype(int)
        prediction_proba = prediction_proba.astype(float)
        prediction_proba_seuil = prediction_proba_seuil.astype(int)
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la conversion des types de prédiction : {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}
    # 4- explainabilité avec SHAP
    try:
        with open("./Scripts/App/global_shap_explainer.pkl", "rb") as f:
            global_explainer = pickle.load(f)
    except Exception as e:
        logger.error("Erreur lors du preprocessing avant SHAP : %s", e)
        logger.error(traceback.format_exc())
        return {"error": "Preprocessing error before SHAP", "detail": str(e)}
    try:
        data_transformed = model_pipeline.named_steps['preprocessor'].transform(input_data)
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la transformation des données d'entrée : {e}")
        return {"error": "Data transformation error", "detail": str(e)}
    try:
        shap_values_all = global_explainer.shap_values(data_transformed)
        # pour les modèles de classification, shap_values_all est une liste avec une entrée par classe
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la conversion des valeurs SHAP en array numpy : {e}")
        return {"error": "SHAP values conversion error"}
    # on extrait seulement les valeurs numériques
    # on affiche les 5 features les plus importantes pour chaque individu
    explanation = []
    for i in range(len(input_data)):
        features_shap = dict(zip(input_data.columns, shap_values_all[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        features_shap_mapped = {features_mapping(f): v for f, v in features_shap.items()}
        try:
            features_shap_mapped = {k: float(v) for k, v in features_shap_mapped.items()}
        except Exception as e:
            print(traceback.print_exc())
            print(e)
            logger.error(f"Erreur lors de la conversion des valeurs SHAP en float : {e}")
            return {"error": "SHAP values float conversion error"}
        top_5_features = sorted(features_shap_mapped.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        top_real_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        explanation.append(top_5_features)
        try:
            explanation = [ [(feat, float(f"{val:.4f}")) for feat, val in exp] for exp in explanation]
        except Exception as e:
            print(traceback.print_exc())
            print(e)
            logger.error(f"Erreur lors de la conversion des valeurs SHAP explicatives : {e}")
            return {"error": "SHAP explanation conversion error"}
    shap_plot_local = f"./Metrics/shap_local_{client_ids}.png"
    os.makedirs("./Metrics", exist_ok=True)
    # plot SHAP local pour le premier individu
    base_value = global_explainer.expected_value # base value pour la classe prédite
    shap_values = shap_values_all[0]  # valeurs SHAP pour le premier individu et la classe prédite
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    mapped_features = [features_mapping(col) for col in feature_names]
    try:
        mapped_features = [str(feat) for feat in mapped_features]   
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la conversion des noms de features pour le plot SHAP : {e}")
        return {"error": "Feature names conversion error for SHAP plot"}
    plt.figure(figsize=(10,6))
    shap.force_plot(base_value,
                    shap_values,
                    features=data_transformed[0],
                    feature_names=mapped_features,
                    matplotlib=True,
                    show=False
                    )
    plt.savefig(shap_plot_local,
                bbox_inches = 'tight',
                dpi=150)
    plt.close()
    try:
        shap_plot_local = str(shap_plot_local)
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        logger.error(f"Erreur lors de la conversion du chemin du plot SHAP en string : {e}")
        return {"error": "SHAP plot path conversion error"}   
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probabilite_1": prediction_proba.tolist(),
        "prediction_seuil" : prediction_proba_seuil.tolist(),
        "top_features": explanation,
        "top_real_names" : top_real_features,
        "shap_plot": shap_plot_local 
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
    try:
        logger.info("Requête GET reçue pour les métriques globales.")
    except Exception as e:
        logger.error(f"Erreur lors de la réception de la requête GET pour les métriques : {e}")
    if refresh or CACHED_METRICS is None:
        print("Recalcul des métriques à la demande.")
        CACHED_METRICS = compute_metrics(df=df.sample(n=10000),  # pour accélérer le calcul
                                         model_pipeline=model_pipeline,
                                          explainer=global_explainer,
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