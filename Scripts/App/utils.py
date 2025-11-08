"""
Script pour implémenter les fonctions utilitaires de l'API

1- fonction de mapping pour définir un dictionnaire de correspondance entre les features du df
et des noms descriptifs explicites 
"""



# ///////////////////////////////////////
# Fonction de mapping des features
# ///////////////////////////////////////

def features_mapping(features_list):
    """
    _Summary_: Fonction qui prend en entrée une liste de features et applique un dictionnaire de mapping
    _Args_: 
    _Returns_:
    """
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
    return [feature_mapping.get(f, f) for f in features_list]




