"""
Script qui va inférer sur le modèle entrainé.

Objetcif : 

1- générer l'ensemble des predictions sur les données de test
2- enregistrer dans un fichier unique les éléments clés caractéristiques et statistiques de la BD et du modèle
3- utiliser ce fichier pour enrichir le dashboard


"""


# /////////////////////////////////
# 1- Import des librairies
# /////////////////////////////////

import pandas as pd
import numpy as np
import pickle
from loguru import logger
import shap
import json
import tqdm
import requests
import os



# ////////////////////////////////////////////
# 2- déclaration des paramètres
# ////////////////////////////////////////////

data_dir = "./Data/application_test_final.csv"
model_dir = "./Scripts/pipeline_final.pkl"
url_api = "https://client-scoring-model.onrender.com/predict"
output_dir = "./Data"




# ////////////////////////////////////////////
# 3- loading des data
# ////////////////////////////////////////////

data = pd.read_csv(data_dir)
data_input = data.replace({np.nan: None, np.inf: None, -np.inf: None})

data_json = data_input.to_dict(orient="records")



# ////////////////////////////////////////////
# 4- génération des prédictions
# ////////////////////////////////////////////

results = []

for _, raw in tqdm.tqdm(data_input.iterrows(), total=len(data_input)):
    response = requests.post(url_api, json=data_json)
    if (response.status_code) != 200:
        logger.error(f"erreur API pour le client {raw['SK_ID_CURR']}, status_code : {response.status_code}")
    result=response.json()
    results.append({
        "ID_Client": raw["SK_ID_CURR"],
        "Score": result["prediction"][0],
        "probabilite_accord": result["probabilite_1"][0],
        "top_features": result["top_features"][0]
    }
    )

# //////////////////////////////////////////////
# Enregistrement dans un fichier unique
# //////////////////////////////////////////////

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir, "results.csv"),index=False)


# //////////////////////////////////////////////////
# 5- Calcul des statistiques globaux
# //////////////////////////////////////////////////

"""
Indicateurs de performance du modèle :

- score moyen global de prediction
- taux d'accord moyen 
- risque moyen par client d'un FN
- drift
- seuil de décisionnel
- top_features explainer

Indicateurs caractéristiques de la BD clients :

- nbre total de clients
- âge moyen
- nbre total accord / refus
- taux d'endettement global



"""


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

# enregistrement des stats dans un fichier

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
    "global_amt_endettement_mean": global_amt_endettement_mean
}

output_path = os.path.join(output_dir, "stats.json")
with open(output_path, "w") as f:
    json.dump(stats, f, indent=4)
    f.close()


# //////////////////////////////////////////////////////////////
# 5- calcul de l'explainer du modèle
# //////////////////////////////////////////////////////////////


# chargement du pipeline de prédiction
with open(model_dir, "rb") as f:
    model_pipeline = pickle.load(f)

# calcul de l'explainer

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])
data_transformed = model_pipeline.named_steps['preprocessor'].transform(data_input)
shap_expl = explainer(data_transformed)
shap_values = shap_expl.values
features_shap = dict(zip(data_input.columns, shap_values[0].tolist()))


# ///////////////////////////////////////////////////////
# 6- mapping des features 
# ///////////////////////////////////////////////////////

feature_mapping = {
    # --- Sources externes / score socio-économique ---
    "EXT_SOURCE_1": "Source externe de risque 1 (score socio-économique)",
    "EXT_SOURCE_2": "Source externe de risque 2 (score socio-économique)",
    "EXT_SOURCE_3": "Source externe de risque 3 (score socio-économique)",

    # --- Informations sur les crédits précédents ---
    "DAYS_CREDIT_mean": "Âge moyen des crédits précédents (jours)",
    "DAYS_CREDIT_min": "Ancienneté minimale d’un crédit précédent (jours)",
    "DAYS_CREDIT_max": "Ancienneté maximale d’un crédit précédent (jours)",
    "DAYS_CREDIT_std": "Variabilité des durées de crédit précédents",
    "DAYS_CREDIT_ENDDATE_mean": "Durée moyenne restante des crédits (jours)",
    "DAYS_CREDIT_ENDDATE_min": "Durée restante minimale (jours)",
    "DAYS_CREDIT_ENDDATE_max": "Durée restante maximale (jours)",
    "DAYS_CREDIT_UPDATE_mean": "Délai moyen depuis la dernière mise à jour crédit (jours)",
    "CREDIT_ACTIVE_Active_mean": "Proportion de crédits actifs",
    "CREDIT_ACTIVE_Closed_mean": "Proportion de crédits clôturés",

    # --- Données sur les cartes de crédit ---
    "CREDITCARD_CREDIT_UTILIZATION_MEAN": "Taux moyen d’utilisation du crédit (cartes)",
    "CREDITCARD_CREDIT_UTILIZATION_MAX": "Taux max d’utilisation du crédit (cartes)",
    "CREDITCARD_CREDIT_UTILIZATION_STD": "Variabilité de l’utilisation du crédit (cartes)",
    "CREDITCARD_CNT_DRAWINGS_ATM_CURRENT_MEAN": "Nombre moyen de retraits aux distributeurs (cartes)",
    "CREDITCARD_CNT_DRAWINGS_ATM_CURRENT_MAX": "Nombre max de retraits aux distributeurs (cartes)",
    "CREDITCARD_CNT_DRAWINGS_CURRENT_MEAN": "Nombre moyen de transactions par carte",
    "CREDITCARD_CNT_DRAWINGS_CURRENT_MAX": "Nombre max de transactions par carte",
    "CREDITCARD_AMT_INST_MIN_REGULARITY_MEAN": "Régularité moyenne des paiements sur carte",

    # --- Informations personnelles ---
    "DAYS_BIRTH": "Âge du client (jours, négatif → plus âgé)",
    "CODE_GENDER_M": "Genre : Homme (1/0)",
    "CODE_GENDER_F": "Genre : Femme (1/0)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (jours)",
    "DAYS_ID_PUBLISH": "Ancienneté du document d’identité (jours)",
    "DAYS_REGISTRATION": "Ancienneté d’enregistrement au domicile (jours)",
    "DAYS_LAST_PHONE_CHANGE": "Délai depuis le dernier changement de téléphone (jours)",

    # --- Données de logement ---
    "HOUSETYPE_MODE_block of flats": "Type de logement : immeuble collectif",
    "HOUSETYPE_MODE_nan": "Type de logement inconnu",
    "NAME_HOUSING_TYPE_With parents": "Type de logement : avec les parents",
    "NAME_HOUSING_TYPE_House / apartment": "Type de logement : maison/appartement",
    "TOTALAREA_MODE": "Surface totale du logement",
    "LIVINGAREA_AVG": "Surface moyenne habitable",
    "LIVINGAREA_MEDI": "Surface médiane habitable",
    "LIVINGAREA_MODE": "Surface mode habitable",
    "APARTMENTS_AVG": "Surface moyenne d’appartement",
    "APARTMENTS_MEDI": "Surface médiane d’appartement",
    "APARTMENTS_MODE": "Surface modale d’appartement",
    "FLOORSMAX_AVG": "Nombre d’étages moyen",
    "FLOORSMAX_MEDI": "Nombre d’étages médian",
    "FLOORSMAX_MODE": "Nombre d’étages mode",
    "FLOORSMIN_AVG": "Étage minimum moyen",
    "FLOORSMIN_MEDI": "Étage minimum médian",
    "FLOORSMIN_MODE": "Étage minimum mode",
    "ELEVATORS_AVG": "Présence moyenne d’ascenseurs",
    "ELEVATORS_MEDI": "Présence médiane d’ascenseurs",
    "ELEVATORS_MODE": "Présence d’ascenseurs (mode)",
    "WALLSMATERIAL_MODE_Panel": "Matériau des murs : panneau",
    "WALLSMATERIAL_MODE_nan": "Matériau des murs : inconnu",
    "FONDKAPREMONT_MODE_nan": "Type de fonds de rénovation inconnu",
    "REGION_POPULATION_RELATIVE": "Population relative de la région (densité)",

    # --- Variables financières ---
    "AMT_CREDIT": "Montant du crédit demandé (€)",
    "AMT_GOODS_PRICE": "Valeur du bien financé (€)",
    "AMT_ANNUITY_mean_y": "Mensualité moyenne (€)",
    "AMT_ANNUITY_min": "Mensualité minimale (€)",
    "AMT_ANNUITY_max_y": "Mensualité maximale (€)",
    "APP_CREDIT_PERC_mean": "Ratio crédit/revenu moyen",
    "APP_CREDIT_PERC_max": "Ratio crédit/revenu maximal",
    "RATE_DOWN_PAYMENT_mean": "Taux d’apport moyen",
    "RATE_DOWN_PAYMENT_max": "Taux d’apport maximal",
    "PAYMENT_DIFF_MEAN": "Écart moyen entre paiement attendu et réel",
    "PAYMENT_DIFF_SUM": "Somme des écarts de paiement",
    "CNT_PAYMENT_mean": "Nombre moyen d’échéances de crédit",
    "CNT_PAYMENT_sum": "Nombre total d’échéances",
    "AMT_PAYMENT_MIN": "Paiement minimum observé (€)",

    # --- Indicateurs comportementaux ---
    "EMPLOYABILITY_STABILITY": "Stabilité de l’emploi (score interne)",
    "DAYS_DECISION_mean": "Délai moyen de décision de crédit (jours)",
    "DAYS_DECISION_min": "Délai minimal de décision (jours)",
    "HOUR_APPR_PROCESS_START_mean": "Heure moyenne de traitement de la demande",
    "HOUR_APPR_PROCESS_START_min": "Heure minimale de traitement",
    "HOUR_APPR_PROCESS_START_max": "Heure maximale de traitement",
    "MONTHS_BALANCE_min": "Ancienneté minimale d’un compte (mois)",
    "MONTHS_BALANCE_size": "Taille de l’historique de compte (mois)",

    # --- Données sociales ---
    "DEF_30_CNT_SOCIAL_CIRCLE": "Nombre d’amis ayant eu un défaut à 30j",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Nombre d’amis ayant eu un défaut à 60j",
    "REG_CITY_NOT_WORK_CITY": "Client vit dans une autre ville que son travail",
    "REG_CITY_NOT_LIVE_CITY": "Client vit dans une autre ville que l’enregistrement",
    "LIVE_CITY_NOT_WORK_CITY": "Travaille dans une autre ville que son logement",
    "REGION_RATING_CLIENT_W_CITY": "Note de risque de la région (client + ville)",
    "REGION_RATING_CLIENT": "Note de risque de la région (client seul)",

    # --- Informations démographiques et professionnelles ---
    "NAME_INCOME_TYPE_Working": "Type de revenu : salarié actif",
    "NAME_INCOME_TYPE_Pensioner": "Type de revenu : retraité",
    "NAME_EDUCATION_TYPE_Higher education": "Niveau d’étude : supérieur",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Niveau d’étude : secondaire",
    "OCCUPATION_TYPE_Laborers": "Métier : ouvrier",
    "OCCUPATION_TYPE_Drivers": "Métier : conducteur",
    "OCCUPATION_TYPE_Low-skill Laborers": "Métier : ouvrier peu qualifié",
    "OCCUPATION_TYPE_nan": "Métier inconnu",
    "ORGANIZATION_TYPE_XNA": "Organisation inconnue",
    "ORGANIZATION_TYPE_Self-employed": "Travailleur indépendant",

    # --- Autres variables administratives ---
    "OWN_CAR_AGE": "Âge du véhicule possédé (années)",
    "NAME_CONTRACT_TYPE": "Type de contrat de crédit",
    "NAME_FAMILY_STATUS_Single / not married": "Statut familial : célibataire",
    "FLAG_EMP_PHONE": "Téléphone professionnel déclaré",
    "FLAG_DOCUMENT_3": "Document 3 fourni (pièce d’identité)",
    "FLAG_DOCUMENT_6": "Document 6 fourni (revenus/employeur)",
    "FLAG_WORK_PHONE": "Téléphone professionnel actif"
}


df_shap = pd.DataFrame.from_dict(features_shap, orient="index", columns=["SHAP_value"])
df_shap['feature name'] = df_shap.index
df_shap['feature name'] = df_shap['feature name'].map(feature_mapping)

# tri par influence relative

df_shap["abs_value"] = df_shap["SHAP_value"].abs()
df_shap = df_shap.sort_values(by="abs_value", ascending=False)
top_10 = df_shap.head(10)

# sauvegarde des 10_top features_shap
with open("features_shap.json", "w") as f:
    json.dump(top_10.to_dict(orient="records"), f)
    f.close()

