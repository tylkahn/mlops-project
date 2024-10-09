from fastapi import FastAPI
import numpy as np
import uvicorn
import pickle
from pydantic import BaseModel

app = FastAPI(
    title="Spotify Model",
    description="A model for predicting popularity of songs on Spotify.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for evaluating songs on Spotify.'}

class request_body(BaseModel):
    params : list

@app.on_event('startup')
def load_artifacts():
    global model
    with open('../../models/new_model.pkl', 'rb') as f:
        model = pickle.load(f)


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    X = data.params
    predictions = model.predict(X)
    np.savetxt('results.csv', predictions, delimiter=',', fmt='%.2f')
    return {'Predictions': predictions[0]}