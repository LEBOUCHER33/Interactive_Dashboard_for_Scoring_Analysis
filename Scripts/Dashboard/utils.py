"""
Script pour implémenter les fonctions utilitaires de l'API

1- fonction de calcul des predictions sur le dataset de test et des principales indicateurs

2- fonction de mapping pour définir un dictionnaire de correspondance entre les features du df
et des noms descriptifs explicites
"""

# ////////////////////////////////
# import des lib
# ////////////////////////////////

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# ///////////////////////////////
# dictionnaire de mapping
# ///////////////////////////////

feat_mapping = {
# --- Sources externes / score socio-économique ---
    "EXT_SOURCE_1": "Source externe de risque 1 (score socio-économique)",
    "EXT_SOURCE_2": "Source externe de risque 2 (score socio-économique)",
    "EXT_SOURCE_3": "Source externe de risque 3 (score socio-économique)",
    # --- Informations sur les crédits précédents ---
    "DAYS_CREDIT_mean": "Ancienneté moyenne des crédits précédents (jours)",
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
    "DAYS_BIRTH": "Âge du client",
    "CODE_GENDER_M": "Genre : Homme (1/0)",
    "CODE_GENDER_F": "Genre : Femme (1/0)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle",
    "DAYS_ID_PUBLISH": "Ancienneté du document d’identité (jours)",
    "DAYS_REGISTRATION": "Ancienneté d’enregistrement au domicile (jours)",
    "DAYS_LAST_PHONE_CHANGE": "Délai depuis le dernier changement de téléphone (jours)",
    # --- Données de logement ---
    "HOUSETYPE_MODE_block of flats": "Type de logement : immeuble collectif (indicateur valeur immo)",
    "HOUSETYPE_MODE_nan": "Type de logement inconnu (indicateur valeur immo)",
    "NAME_HOUSING_TYPE_With parents": "Type de logement : hebergement chez parents (indicateur valeur immo)",
    "NAME_HOUSING_TYPE_House / apartment": "Type de logement : maison/appartement (indicateur valeur immo)",
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
    "ELEVATORS_AVG": "ascenseurs / indicateur valeur immo (moy)",
    "ELEVATORS_MEDI": "ascenseurs / indicateur valeur immo (med)",
    "ELEVATORS_MODE": "ascenseurs / indicateur valeur immo (mode)",
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

# ///////////////////////////////////////
# Fonction de mapping des features
# ///////////////////////////////////////

def features_mapping(feature : str, feature_mapping : dict = feat_mapping) -> str:
    """
    _Summary_: Fonction qui mappe les noms des variables en noms explicites
    _Args_:
        - feature : str
        - feature_mapping : dict
    _Returns_:
        - les features mappées
    """
    if feature is None:
        return ""
    return str(feature_mapping.get(feature, feature if feature is not None else ""))



def clean_feature_name(name: str) -> str:
    """Retire les préfixes du pipeline pour retrouver le nom brut."""
    if name.startswith("remainder__"):
        return name.replace("remainder__", "")
    if "__" in name:
        return name.split("__")[-1]
    return name

# /////////////////////////////////////////////
# 1- Fonction de calcul des predictions
# /////////////////////////////////////////////


output_dir =  "./Scripts/App"

def compute_metrics (df : pd.DataFrame, model_pipeline : object, explainer : object,
                     features_mapping = features_mapping, sample_size : int = 1000):
    """
    _Summary_ : Calcul des indicateurs clés d'un dataframe d'une BD clients
    _Args_:
        df (pd.DataFrame): dataframe complet (ou échantillonage) de la BD clients à tester_
        model_pipeline (object): _pipeline de prédiction entrainé_
        explainer (object): _objet shap déjà initialisé_
        features_mapping (dict): _fonction de mapping des features_
        sample_size (int, optional): _taille de l'échantillon à utiliser pour le calcul des indicateurs.
                                    Par défaut à 10000_
    _Returns_:
        _dict_:
            - métriques principales calculées
            - top features SHAP
    """
    # data
    if sample_size and sample_size <= len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()
    df_sample = df_sample.replace({np.nan: None, np.inf: None, -np.inf: None})
    # calcul des predictions
    prediction = model_pipeline.predict(df_sample) # score (0, 1)
    prediction_proba = model_pipeline.predict_proba(df_sample)[:,1] # proba d'être 1
    prediction_proba_seuil = (prediction_proba>=0.3).astype(int) # proba d'être 1 avec un seuil plus stringent
    # explainabilite
    data_transformed = model_pipeline.named_steps['preprocessor'].transform(df_sample)
    shap_values = explainer.shap_values(data_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    if feature_names.startswith("remainder__"):
        feature_names = feature_names[len("remainder__"):]

    mapped_feature_names = [features_mapping(col) for col in feature_names]
    df_sample_reset = df_sample.reset_index(drop=True)
    client_dict = {}
    for i in range(len(df_sample_reset)):
        raw_id = df_sample_reset["SK_ID_CURR"].iloc[i]
        try:
            current_client_id = str(int(float(raw_id)))
        except Exception:
            current_client_id = str(raw_id).split('.')[0]
        vals = shap_values[i] if not isinstance(shap_values, list) else shap_values[1][i]
        features_shap = dict(zip(feature_names, vals))
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features_serializable = [
    (   raw,                                      # nom brut envoyé au front
        features_mapping(clean_feature_name(raw)),# nom joli corrigé
        float(val)                                 # valeur
    )
    for raw, val in top_5_features                 # <-- ton input original
]
        client_dict[current_client_id] = {
            "client_id": int(current_client_id),
            "prediction": int(prediction[i]),
            "prediction_proba": float(prediction_proba[i]),
            "prediction_proba_seuil": int(prediction_proba_seuil[i]),
            "top_features": top_features_serializable
        }
    os.makedirs("./Scripts/App/assets", exist_ok=True)
    shap_plot_path = "./Scripts/App/assets/global_shap.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, 
                      features=data_transformed,
                      feature_names=mapped_feature_names,
                      show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    # resultats
    metrics_stats = {
        "score_moy": float(prediction.mean()), # Conversion float numpy -> float python
        "taux_refus": float(prediction_proba_seuil.mean()),
        "taux_accord": float(1 - prediction_proba_seuil.mean()),
        "risk_moy_fn": round(1753/61503, 3),
        "drift": 0.17,
        "seuil_decisionnel": 0.3,
        "nb_clients": int(len(df_sample)),
        "age_moye_client": round(float(abs(df_sample["DAYS_BIRTH"].mean() / 365)), 1),
        "nb_accord": int((prediction == 0).sum()),
        "nb_refus": int((prediction == 1).sum()),
        "shap_plot_path": shap_plot_path
    }
    return {
        "metrics":metrics_stats, 
        "client": client_dict
    }


