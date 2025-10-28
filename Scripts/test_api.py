"""
Script de test de l'API 

Objectif : Tests fonctionnels sur l'api 

    1- test endpoint racine : démarrage de l'api
    2- test endpoint "/predic" :
        - format des données d'entrée
        - entrée individu seul vs multiples
        - valeur de prediction
        - valeurs des probabilités

"""
from fastapi.testclient import TestClient

# TestClient est une classe de FastAPI qui permet de simuler des requêtes HTTP pour tester les endpoints de l'API


from api import app  # on importe l'instance de l'API créée dans api.py



# on crée un client de test pour envoyer des requêtes à l'API
client = TestClient(app)

# on crée des données de test
data_sample = [
    {
        "AMT_CREDIT": 200000,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_2": 0.65,
        "FLAG_DOCUMENT_3": 1,
        "INCOME_TOTAL": 180000
    }
]

data_mult = [
    {
        "AMT_CREDIT": 20000,
        "DAYS_BIRTH": -100,
        "EXT_SOURCE_2": 0.65,
        "FLAG_DOCUMENT_3": 0,
        "INCOME_TOTAL": 180000
    },
    {
        "AMT_CREDIT": 2000,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_2": 0.65,
        "FLAG_DOCUMENT_3": 1,
        "INCOME_TOTAL": 0
    }
]

# 1- Test sur le fonctionnement de l'api

def test_read_root():
    response = client.get("/")  # on envoie une requête GET à l'endpoint racine
    assert response.status_code == 200  # on vérifie que le code de statut est 200 (OK)
    assert response.json() == {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}  # on vérifie que la réponse est correcte


# 2- Tests sur l'endpoint de prédiction

def test_endpoint_predict():

    # check si accepte les requêtes post et le format d'entrée json
    response = client.post("/predict", json=data_sample)
    assert response.status_code == 200 

    # check si on a tous les outputs
    pred = response.json() 
    assert "prediction"  in pred
    assert "probabilite_1" in pred
    assert "top_features" in pred

    # check les types des outputs
    assert isinstance(pred["prediction"], list)
    assert isinstance(pred["probabilite_1"], list)
    assert isinstance(pred["top_features"], list)

    # check les valeurs des outputs
    assert len(pred["top_features"][0]) == 5  # top 5 features
    assert pred["prediction"][0] in [0,1] # valeurs des probas 
    assert 0<= pred["probabilite_1"][0] <=1 # valeurs des targets
