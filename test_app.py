from fastapi.testclient import TestClient
from main import app
import datetime


# ct stores current time
ct = datetime.datetime.now()
tst = ct.isoformat()

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)

        # asserting the correct response is received
        assert response.status_code == 200
        json_data = response.json()
        assert json_data['flower_class'] == "Iris Virginica"
        ts = json_data['timestamp']
        assert ts > tst
        #assert response.json() == {"flower_class": "Iris Virginica", "timestamp": ts}

# test to check if Iris Setosa is classified correctly
def test_pred_IrisSetosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 4.9,
        "sepal_width": 3,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)

        # asserting the correct response is received
        assert response.status_code == 200
        json_data = response.json()
        assert json_data['flower_class'] == "Iris Setosa"
        ts = json_data['timestamp']
        assert ts > tst
        #assert response.json() == {"flower_class": "Iris Setosa", "timestamp": ts}

# test to check if Iris Versicolour is classified correctly
def test_pred_IrisVersicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 6.4,
        "sepal_width": 3.2,
        "petal_length": 4.5,
        "petal_width": 1.5,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)

        # asserting the correct response is received
        assert response.status_code == 200
        json_data = response.json()
        assert json_data['flower_class'] == "Iris Versicolour"
        ts = json_data['timestamp']
        assert ts > tst
        #assert response.json() == {"flower_class": "Iris Versicolour", "timestamp": ts}
