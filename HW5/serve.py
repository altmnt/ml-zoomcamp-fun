from fastapi import FastAPI
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
import uvicorn

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

categorical = ['lead_source']
numeric = ['number_of_courses_viewed', 'annual_income']
pipeline = load_model('pipeline_v2.bin')

def predict(df):
    df[categorical] = df[categorical].fillna('NA')
    df[numeric] = df[numeric].fillna(0)

    dicts = df[categorical + numeric].to_dict(orient='records')
    y_pred = pipeline.predict_proba(dicts)[:, 1]

    return y_pred

app = FastAPI()


@app.post('/predict')
def predict_endpoint(data: dict):
    df = pd.DataFrame([data])
    predictions = predict(df)
    return {'predictions': predictions.tolist()}


if __name__ == '__main__':
    
    uvicorn.run(app, host="0.0.0.0", port=9696)


