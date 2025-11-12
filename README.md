# scoring_dashboard_interactive

## Objectif 

Développement d'un dashboard interactif pour accéder à un outil de scoring décisionnel d'octroi de crédit bancaire.
Ce dashboard devra permettre au personnel de la banque la restitution métier de l'appication de scoring ainsi que son explicabilité.


## Workflow

1- configurer l'environnement de travail :
    - créer d'un repo git pour le versionning du projet
    - créer et activer un environnement virtuel
    - installer les dépendances nécessaires dans le venv
    - récupérer les éléments clés pour le fonctionnement de l'api de scoring :
        - le modèle de scoring entraîné
        - les data de test pour inférer sur le modèle
        - les scripts relatifs au développement de l'api et aux tests de son fonctionnement
        - les éléments nécessaires à son deploiement sur le cloud : Dockerfile, requirements.txt
    - déployer l'api sur une solution cloud (Render)
    - tester le fonctionnement de l'api déployée via l'url publique

2- développer un dashboard interactif :
    - définir les spécifications du dashboard
    - élaborer la maquette du dashboard avec excalidraw
    - implémenter les scripts :
        -update des scripts python relatifs à l api
        -implementation des scripts Streamlit de développement du dashboard
    - organiser les éléments du dashboard et les graphs
    - intégrer l'api
    - tester le dashboard en local
    - déployer sur le cloud (Render)

3- assurer CI/CD et la veille technologique



## Spécifications du dashboard

Objectifs métier :

- filtrer et rechercher des dossiers clients
- pouvoir accéder rapidement aux predictions du modèle de scoring
- visualiser les scores et les probabilité de solvabilité des clients
- comprendre et justifier les décisions = explicabilité



Résultats attendus :

- résultat du scoring = décision + probabilité
- variables explicatives majeures du modèle 
- explicabilité du score



## Maquette du dashboard

1- Vue statistique globale du modèle (page_1/top bar):

    - nbre total des demandes et répartition des taux acceptation / refus
    - indicateurs de performance du modèle : drift / taux d'erreur / score_moyen_client
    - feature importance globale


2- identification du client (page_2side bar):

    - sélection d'un client via un ID 
    - score de solvabilité du client + décision
    - affichage des caractéristiques principales du profil client (âge, revenus, dettes)
    - affichage des caractéristiques principales de la demande (montant, échéance, annuité)

       

3- visualisation de la prédiction (page_2/central bar) :
    
    - explicabilité locale : visualisation des principales features influentes sur la prédiction
    - graphs du profil de risque du client : 
        - visualisation du profil client comparativement aux données globales (possibilité de sélectionner parmi les 5 features principales) / graph de distribution de la variable
        - évaluation de la capacité d'endettement du client comparativement à la moyenne









